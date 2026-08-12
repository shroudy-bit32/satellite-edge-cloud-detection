"""SPARCS (Landsat 8) -> harici dogrulama kareleri.

PDF'te SPARCS "doğrulama amaçlı küçük ölçekli etiketli set" olarak tanimli.
Egitimde KULLANILMAZ; farkli bir uyduda genelleme olcmek icindir.

Kullanim:
    python -m src.prepare_sparcs --zip "C:/Users/Cemil/Downloads/sparcs_data_L8.zip"

Format (readme.txt'ten dogrulandi):
    <scene>_data.tif   -> 1000x1000, 10 bant, uint16, L8 B1-B7,B9,B10,B11
    <scene>_labels.tif -> 1000x1000, uint8, kodlar 0-6
"""

import argparse
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from tqdm import tqdm

from src import config
from src.prepare_data import tile_positions
from src.preprocess import normalize_reflectance


def resolve_band_indices() -> list:
    """config.BANDS icin _data.tif kanal indeksleri (0 tabanli).

    Bir bandin Landsat 8 karsiligi yoksa erken ve anlasilir sekilde hata verir -
    sessizce yanlis bant kullanmaktan iyidir.
    """
    indices = []
    for s2_band in config.BANDS:
        l8_band = config.S2_TO_L8_BAND.get(s2_band)
        if l8_band is None:
            raise SystemExit(
                f"'{s2_band}' bandinin Landsat 8 karsiligi yok. SPARCS dogrulamasi "
                f"icin config.BANDS yalnizca su bantlari icerebilir: "
                f"{sorted(config.S2_TO_L8_BAND)}"
            )
        if l8_band not in config.L8_DATA_BANDS:
            raise SystemExit(f"'{l8_band}' _data.tif icinde yok (pankromatik B8 haric)")
        indices.append(config.L8_DATA_BANDS.index(l8_band))
    return indices


def dn_to_reflectance(arr: np.ndarray) -> np.ndarray:
    """Landsat 8 Collection 1 DN -> TOA reflektans."""
    return arr * config.L8_REFLECTANCE_MULT + config.L8_REFLECTANCE_ADD


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--zip", dest="zip_path", type=Path, required=True)
    p.add_argument("--out", type=Path, default=None, help="varsayilan: data/patches_sparcs")
    p.add_argument("--patch-size", type=int, default=config.PATCH_SIZE)
    p.add_argument("--shadow-as-cloud", action="store_true", default=config.SHADOW_AS_CLOUD)
    p.add_argument("--max-scenes", type=int, default=None)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    if not args.zip_path.exists():
        raise SystemExit(f"bulunamadi: {args.zip_path}")

    band_indices = resolve_band_indices()
    print(f"bant eslemesi: {list(zip(config.BANDS, [config.L8_DATA_BANDS[i] for i in band_indices]))}")

    out_dir = args.out or (config.PROJECT_ROOT / "data" / "patches_sparcs")

    with zipfile.ZipFile(args.zip_path) as z:
        names = z.namelist()
    data_files = sorted(n for n in names if n.endswith("_data.tif"))
    scenes = [n[: -len("_data.tif")] for n in data_files]
    scenes = [s for s in scenes if f"{s}_labels.tif" in names]
    if args.max_scenes:
        scenes = scenes[: args.max_scenes]

    patches_per_scene = len(tile_positions(1000, args.patch_size)) ** 2
    print(f"{len(scenes)} sahne x {patches_per_scene} kare = {len(scenes) * patches_per_scene} kare")

    if args.dry_run:
        print("--dry-run: dosya yazilmadi")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "images").mkdir(exist_ok=True)
    (out_dir / "masks").mkdir(exist_ok=True)

    rows = []
    for scene in tqdm(scenes, desc="SPARCS sahneleri"):
        # GDAL'in vsizip surucusu: zip'i acmadan dogrudan oku
        with rasterio.open(f"/vsizip/{args.zip_path}/{scene}_data.tif") as src:
            data = src.read().astype(np.float32)
        with rasterio.open(f"/vsizip/{args.zip_path}/{scene}_labels.tif") as src:
            labels = src.read(1)

        img = normalize_reflectance(dn_to_reflectance(data[band_indices])).astype(np.float16)

        cloud = labels == config.SPARCS_CLOUD_CLASS
        if args.shadow_as_cloud:
            cloud = cloud | np.isin(labels, config.SPARCS_SHADOW_CLASSES)
        cloud = cloud.astype(np.uint8)
        snow = (labels == config.SPARCS_SNOW_CLASS).astype(np.uint8)

        for y in tile_positions(img.shape[1], args.patch_size):
            for x in tile_positions(img.shape[2], args.patch_size):
                patch = img[:, y:y + args.patch_size, x:x + args.patch_size]
                patch_mask = cloud[y:y + args.patch_size, x:x + args.patch_size]
                snow_fraction = float(snow[y:y + args.patch_size, x:x + args.patch_size].mean())
                cloud_fraction = float(patch_mask.mean())

                name = f"{scene}_y{y:04d}_x{x:04d}"
                img_path = out_dir / "images" / f"{name}.npy"
                mask_path = out_dir / "masks" / f"{name}.npy"
                np.save(img_path, patch)
                np.save(mask_path, patch_mask)

                rows.append({
                    "path": str(img_path),
                    "mask_path": str(mask_path),
                    "label": int(cloud_fraction >= config.CLOUD_PIXEL_THRESHOLD),
                    "split": "test",  # SPARCS tamami harici dogrulama
                    "scene": scene,
                    "cloud_fraction": round(cloud_fraction, 6),
                    # Kar/buz orani: en zor karisiklik. Rapor icin ayri analiz sagliyor.
                    "snow_fraction": round(snow_fraction, 6),
                    "source": "sparcs_l8",
                })

    index = pd.DataFrame(rows)
    index.to_csv(out_dir / "index.csv", index=False)

    print(f"\n{len(index):,} kare yazildi -> {out_dir / 'index.csv'}")
    print("\nsinif dagilimi (0=kullanilabilir, 1=bulutlu):")
    print(index["label"].value_counts().sort_index().to_string())
    snowy = index[index["snow_fraction"] > 0.2]
    print(f"\nkar/buz orani >%20 olan kare: {len(snowy)} "
          f"(bunlarin %{100 * snowy['label'].mean():.1f}'i bulutlu etiketli)")

    meta = {
        "source": str(args.zip_path),
        "scenes": len(scenes),
        "patches": len(index),
        "patch_size": args.patch_size,
        "bands_s2": config.BANDS,
        "bands_l8": [config.L8_DATA_BANDS[i] for i in band_indices],
        "shadow_as_cloud": bool(args.shadow_as_cloud),
        "reflectance_clip": config.REFLECTANCE_CLIP,
        "caveat": "Gunes yuksekligi duzeltmesi uygulanmadi (MTL dosyalari arsivde yok); "
                  "Sentinel-2 ile arasinda sistematik reflektans kaymasi bulunur.",
    }
    (out_dir / "prepare_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"meta: {out_dir / 'prepare_meta.json'}")


if __name__ == "__main__":
    main()
