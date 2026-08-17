"""Kismi indirme: U-Net maskesiyle karenin yalnizca temiz bloklarini indirmek.

Ikili karar bir kareyi ya tamamen indirir ya tamamen atar. Dogru sekilde atilan
bulutlu karelerin ICINDEKI temiz pikseller de kaybolur - bu bir hata degil,
yontemin YAPISAL maliyetidir ve ikili siniflandiriciyla kapatilamaz, cunku hangi
bolgenin temiz oldugunu bilmek gerekir. U-Net tam olarak bunu uretir.

Bu betik o kazancin GERCEK degerini olcer: blok boyutu kucultuldukce ne kadar
temiz alan kurtariliyor ve karsiliginda ne kadar bayt odeniyor.

Iki maliyet kalemi bilerek hesaba katilir:
  1. Blok haritasi: hangi bloklarin gonderildigi (kare basina bit maskesi).
  2. Sikistirma verimliliginin dusmesi: her blok BAGIMSIZ bir iletim birimi
     olarak sikistirilir. Kucuk bloklar daha az baglam tasidigi icin daha kotu
     sikisir - kismi indirmenin asil gizli maliyeti budur ve ancak gercekten
     sikistirarak olculebilir.

Blok boyutu = kare boyutu (64) olan satir, MEVCUT ikili davranisin ta kendisidir
ve referans noktasidir.

Kullanim:
    CLOUD_PATCH_SIZE=64 CLOUD_PATCH_DIR=data/patches_t64 \
      python -m src.partial_downlink --checkpoint releases/v3.2/unet.pt --tag unet_t64
"""

import argparse
import json
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from src import config
from src.export import load_from_checkpoint


def compressed_size(block: np.ndarray, level: int = 6) -> int:
    """Blogun bagimsiz iletim birimi olarak sikistirilmis bayt sayisi.

    Gercek bir uydu CCSDS 122 / JPEG2000 kullanir; zlib bir VEKILDIR. Mutlak
    bayt sayilari bu yuzden gercek misyonu temsil etmez. Ama karsilastirma ayni
    kodekle yapildigi icin BLOK BOYUTUNUN sikistirmaya etkisi - olcmek
    istedigimiz sey - gecerli kalir.
    """
    return len(zlib.compress(np.ascontiguousarray(block).tobytes(), level))


def evaluate_policy(prob_maps: np.ndarray, true_masks: np.ndarray,
                    images: np.ndarray, block: int,
                    pixel_threshold: float, cloud_fraction_threshold: float) -> dict:
    """Bir blok boyutu icin indirme politikasini degerlendirir.

    Karar TAHMIN EDILEN maskeden verilir (uyduda elde olan bilgi budur);
    kayip muhasebesi GERCEK maskeden yapilir (yerde bilinen dogru budur).
    Bu ayrim kritik: ikisini karistirmak modeli kendi hatasindan muaf tutar.
    """
    n, _, h, w = images.shape
    nb = h // block

    pred_cloud = (prob_maps >= pixel_threshold)          # (n, h, w) bool
    true_cloud = true_masks.astype(bool)                 # (n, h, w) bool

    # (n, nb, nb, block, block) goruntusune bolerek blok bazli oranlar
    def blockify(a):
        return a.reshape(n, nb, block, nb, block).transpose(0, 1, 3, 2, 4)

    pred_frac = blockify(pred_cloud).mean(axis=(3, 4))   # (n, nb, nb)
    keep = pred_frac < cloud_fraction_threshold          # indirilecek bloklar

    true_blocks = blockify(true_cloud)
    clean_per_block = (~true_blocks).sum(axis=(3, 4))    # gercek temiz piksel
    cloud_per_block = true_blocks.sum(axis=(3, 4))

    total_px = n * h * w
    total_clean = int((~true_cloud).sum())
    total_cloud = int(true_cloud.sum())

    kept_px = int(keep.sum()) * block * block
    kept_clean = int(clean_per_block[keep].sum())
    kept_cloud = int(cloud_per_block[keep].sum())

    # --- Baytlar ---
    # Her indirilen blok bagimsiz sikistirilir; blok haritasi her kare icin
    # nb*nb bit = ceil(nb*nb/8) bayt olarak eklenir.
    img_blocks = images.reshape(n, images.shape[1], nb, block, nb, block) \
                       .transpose(0, 2, 4, 1, 3, 5)      # (n, nb, nb, C, b, b)
    payload = 0
    for i in range(n):
        for by in range(nb):
            for bx in range(nb):
                if keep[i, by, bx]:
                    payload += compressed_size(img_blocks[i, by, bx])
    overhead = n * int(np.ceil(nb * nb / 8))

    return {
        "blok": f"{block}x{block}",
        "blok_sayisi_kare_basina": nb * nb,
        "indirilen_blok_%": round(100 * float(keep.mean()), 2),
        "indirilen_alan_%": round(100 * kept_px / total_px, 2),
        "veri_azalmasi_alan_%": round(100 * (1 - kept_px / total_px), 2),
        "korunan_temiz_alan_%": round(100 * kept_clean / total_clean, 3),
        "kaybedilen_temiz_alan_%": round(100 * (1 - kept_clean / total_clean), 3),
        "bosuna_inen_bulut_alani_%": round(100 * kept_cloud / total_cloud, 3),
        "yuk_bayt": int(payload),
        "harita_bayt": int(overhead),
        "toplam_bayt": int(payload + overhead),
    }


@torch.no_grad()
def predict_masks(checkpoint: Path, index: pd.DataFrame, batch_size: int = 64):
    model, task = load_from_checkpoint(checkpoint)
    if task != "segmentation":
        raise SystemExit(f"bu checkpoint '{task}' gorevine ait, U-Net bekleniyordu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    probs, imgs, masks = [], [], []
    paths = index["path"].tolist()
    mpaths = index["mask_path"].tolist()

    for i in tqdm(range(0, len(paths), batch_size), desc="U-Net cikarimi"):
        batch = np.stack([np.load(p).astype(np.float32) for p in paths[i:i + batch_size]])
        out = model(torch.from_numpy(batch).to(device))
        probs.append(torch.sigmoid(out).squeeze(1).cpu().numpy().astype(np.float32))
        imgs.append(batch.astype(np.float16))
        masks.append(np.stack([np.load(p) for p in mpaths[i:i + batch_size]]))

    return np.concatenate(probs), np.concatenate(imgs), np.concatenate(masks)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--block-sizes", type=int, nargs="*", default=None,
                   help="varsayilan: kare boyutunun tum bolenleri (64 32 16 8)")
    p.add_argument("--pixel-threshold", type=float, default=0.5)
    p.add_argument("--cloud-fraction-threshold", type=float,
                   default=config.CLOUD_PIXEL_THRESHOLD)
    p.add_argument("--max-patches", type=int, default=None,
                   help="hizli deneme icin kare sayisini sinirla")
    args = p.parse_args()

    index = pd.read_csv(config.PATCH_INDEX)
    index = index[index["split"] == args.split].reset_index(drop=True)
    if args.max_patches:
        index = index.head(args.max_patches)
    if "mask_path" not in index.columns:
        raise SystemExit("index.csv'de mask_path yok - kismi indirme maske gerektirir")

    print(f"{args.split} bolumu: {len(index)} kare, kare boyutu {config.PATCH_SIZE}")

    probs, imgs, masks = predict_masks(args.checkpoint, index)
    size = imgs.shape[-1]

    blocks = args.block_sizes or [b for b in (size, size // 2, size // 4, size // 8) if b >= 4]
    for b in blocks:
        if size % b:
            raise SystemExit(f"blok {b}, kare boyutu {size}'i tam bolmuyor")

    rows = []
    for b in blocks:
        print(f"\nblok {b}x{b} degerlendiriliyor...")
        rows.append(evaluate_policy(probs, masks, imgs, b,
                                    args.pixel_threshold, args.cloud_fraction_threshold))

    df = pd.DataFrame(rows)

    # Bayt eksenini referansa (en buyuk blok = mevcut ikili davranis) gore normalize et
    base_bytes = df["toplam_bayt"].iloc[0]
    df["bayt_referansa_gore"] = (df["toplam_bayt"] / base_bytes).round(4)

    # Tum sahnenin sikistirilmis boyutu: "hicbir sey eleme" senaryosu
    full = sum(compressed_size(imgs[i]) for i in range(len(imgs)))
    df["veri_azalmasi_bayt_%"] = (100 * (1 - df["toplam_bayt"] / full)).round(2)

    print("\n=== Kismi indirme odunlesimi ===")
    cols = ["blok", "indirilen_alan_%", "veri_azalmasi_alan_%", "korunan_temiz_alan_%",
            "kaybedilen_temiz_alan_%", "veri_azalmasi_bayt_%", "bayt_referansa_gore"]
    print(df[cols].to_string(index=False))

    out_csv = config.REPORTS / f"{args.tag}_partial_downlink.csv"
    df.to_csv(out_csv, index=False)
    meta = {
        "tag": args.tag, "split": args.split, "kare": int(len(index)),
        "kare_boyutu": int(size),
        "pixel_threshold": args.pixel_threshold,
        "cloud_fraction_threshold": args.cloud_fraction_threshold,
        "tam_sahne_sikistirilmis_bayt": int(full),
        "kodek": "zlib level 6 (VEKIL - gercek misyon CCSDS 122/JPEG2000 kullanir)",
        "rows": rows,
    }
    (config.REPORTS / f"{args.tag}_partial_downlink.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nsonuclar: {out_csv}")


if __name__ == "__main__":
    main()
