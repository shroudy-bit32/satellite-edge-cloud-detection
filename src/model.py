"""MobileNet tabanli ikili bulut siniflandiricisi.

Coklu bant sorunu: ImageNet on-egitimli MobileNet 3 kanal bekler, bizim girdimiz
6-13 bant. timm'in in_chans parametresi ilk conv agirligini kanal boyunca
uyarlar (3 kanalin ortalamasini alip yeni kanallara kopyalar ve olcekler),
bu yuzden elle mudahale gerekmez.
"""

import timm
import torch
import torch.nn as nn

from src import config


def build_model(
    model_name: str = config.MODEL_NAME,
    in_channels: int = config.IN_CHANNELS,
    pretrained: bool = config.PRETRAINED,
) -> nn.Module:
    """Tek logit ureten ikili siniflandirici dondurur.

    Tek logit + BCEWithLogitsLoss kullaniyoruz (2 sinif + softmax yerine):
    karar esigini egitim sonrasi serbestce ayarlayabilmek icin.
    """
    model = timm.create_model(
        model_name,
        pretrained=pretrained,
        in_chans=in_channels,
        num_classes=1,
    )
    return model


@torch.no_grad()
def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_size_mb(model: nn.Module) -> float:
    """Agirliklarin FP32 olarak diskte kaplayacagi tahmini boyut."""
    params = sum(p.numel() for p in model.parameters())
    buffers = sum(b.numel() for b in model.buffers())
    return (params + buffers) * 4 / 1024**2


if __name__ == "__main__":
    m = build_model(pretrained=False)
    x = torch.randn(2, config.IN_CHANNELS, config.PATCH_SIZE, config.PATCH_SIZE)
    y = m(x)
    print(f"model            {config.MODEL_NAME}")
    print(f"girdi kanali     {config.IN_CHANNELS}")
    print(f"parametre        {count_parameters(m):,}")
    print(f"FP32 boyut       {model_size_mb(m):.2f} MB")
    print(f"cikti sekli      {tuple(y.shape)}")
