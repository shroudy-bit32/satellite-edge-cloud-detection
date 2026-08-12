"""Hafif U-Net: piksel bazli bulut segmentasyonu (genisletilmis hedef).

Tasarim karari: klasik U-Net (~31M parametre, ~124 MB) plandaki "birkac MB"
kisitini kat kat asar. Bunun yerine siniflandiricidaki MobileNetV3 omurgasini
encoder olarak kullanip uzerine derinlemesine-ayrilabilir (depthwise separable)
konvolusyonlardan olusan ince bir decoder koyuyoruz.

Kazanci iki katli:
  - Boyut/hiz kisiti korunur (~2M parametre).
  - Encoder ImageNet on-egitimli; ayrica siniflandirici ile ayni omurga oldugu
    icin iki gorev arasinda dogrudan karsilastirma yapilabilir.
"""

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from src import config


class SeparableConv(nn.Module):
    """Depthwise + pointwise konvolusyon. Standart conv'a gore ~8x daha az parametre."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=1, groups=in_ch, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class DecoderBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            SeparableConv(in_ch + skip_ch, out_ch),
            SeparableConv(out_ch, out_ch),
        )

    def forward(self, x, skip=None):
        # ConvTranspose yerine bilinear upsample: parametre yok, checkerboard
        # artefakti yok ve ONNX/INT8 tarafinda sorunsuz.
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        if skip is not None:
            # Tek pikselik boyut farklarina karsi (tek sayili girdi boyutlari)
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class MobileUNet(nn.Module):
    """MobileNetV3 encoder + hafif decoder. Cikti: (B, 1, H, W) logit haritasi."""

    def __init__(
        self,
        encoder_name: str = config.MODEL_NAME,
        in_channels: int = config.IN_CHANNELS,
        pretrained: bool = config.PRETRAINED,
        decoder_channels: tuple = (96, 64, 48, 32),
    ):
        super().__init__()
        self.encoder = timm.create_model(
            encoder_name,
            pretrained=pretrained,
            in_chans=in_channels,
            features_only=True,          # cok olcekli ara ozellikler
            out_indices=(0, 1, 2, 3, 4),  # stride 2,4,8,16,32
        )
        enc_ch = self.encoder.feature_info.channels()  # ornek: [16,16,24,48,576]

        self.decoder = nn.ModuleList([
            DecoderBlock(enc_ch[4], enc_ch[3], decoder_channels[0]),   # 1/32 -> 1/16
            DecoderBlock(decoder_channels[0], enc_ch[2], decoder_channels[1]),  # -> 1/8
            DecoderBlock(decoder_channels[1], enc_ch[1], decoder_channels[2]),  # -> 1/4
            DecoderBlock(decoder_channels[2], enc_ch[0], decoder_channels[3]),  # -> 1/2
        ])
        self.head = nn.Sequential(
            SeparableConv(decoder_channels[3], decoder_channels[3]),
            nn.Conv2d(decoder_channels[3], 1, 1),
        )

    def forward(self, x):
        input_size = x.shape[-2:]
        f0, f1, f2, f3, f4 = self.encoder(x)

        d = self.decoder[0](f4, f3)
        d = self.decoder[1](d, f2)
        d = self.decoder[2](d, f1)
        d = self.decoder[3](d, f0)
        d = self.head(d)

        # Encoder stride 2 ile basladigi icin son bir upsample gerekiyor.
        return F.interpolate(d, size=input_size, mode="bilinear", align_corners=False)


def build_unet(
    encoder_name: str = config.MODEL_NAME,
    in_channels: int = config.IN_CHANNELS,
    pretrained: bool = config.PRETRAINED,
) -> nn.Module:
    return MobileUNet(encoder_name, in_channels, pretrained)


@torch.no_grad()
def mask_to_decision(logits: torch.Tensor, pixel_threshold: float = 0.5,
                     cloud_fraction_threshold: float = config.CLOUD_PIXEL_THRESHOLD) -> torch.Tensor:
    """Segmentasyon maskesinden goruntu-seviyesi "bulutlu mu" karari uretir.

    Bu koprii onemli: U-Net de siniflandirici ile ayni fayda analizine
    (veri indirme kazanci) sokulabilsin diye ayni esik mantigini kullanir.
    """
    probs = torch.sigmoid(logits)
    cloud_fraction = (probs >= pixel_threshold).float().mean(dim=(-2, -1)).squeeze(1)
    return (cloud_fraction >= cloud_fraction_threshold).float()


if __name__ == "__main__":
    m = build_unet(pretrained=False)
    x = torch.randn(2, config.IN_CHANNELS, config.PATCH_SIZE, config.PATCH_SIZE)
    y = m(x)
    params = sum(p.numel() for p in m.parameters())
    print(f"encoder          {config.MODEL_NAME}")
    print(f"parametre        {params:,}")
    print(f"FP32 boyut       {params * 4 / 1024**2:.2f} MB")
    print(f"girdi  {tuple(x.shape)}")
    print(f"cikti  {tuple(y.shape)}")
    assert y.shape[-2:] == x.shape[-2:], "cikti cozunurlugu girdiyle ayni olmali"
