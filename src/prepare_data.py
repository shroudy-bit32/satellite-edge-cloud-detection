"""Sentinel-2 Cloud Mask Catalogue -> egitilebilir kareler (Gunler 4-6).

Zip arsivlerini ACMADAN, sahneleri tek tek bellege okuyup isler. subscenes.zip
acilmis halde 26.6 GB; bu yolla anlik bellek kullanimi ~52 MB (tek sahne).

Kullanim:
    python -m src.prepare_data --source "C:/Users/Cemil/Downloads" --dry-run
    python -m src.prepare_data --source "C:/Users/Cemil/Downloads"
    python -m src.prepare_data --source "C:/Users/Cemil/Downloads" --max-scenes 20 --tag mini

Veri formati (README'den):
    subscenes/<scene>.npy  -> (1022, 1022, 13) float32, TOA reflektans (/10000 uygulanmis)
    masks/<scene>.npy      -> (1022, 1022, 3)  bool, one-hot [CLEAR, CLOUD, CLOUD_SHADOW]
    Bant sirasi numerik: B01..B08, B8A, B09, B10, B11, B12
"""

import argparse
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from src import config
from src.preprocess import normalize_reflectance

SCENE_SIZE = 1022
TAG_COLUMNS = [
    "forest/jungle", "snow/ice", "agricultural", "urban/developed", "coastal",
    "hills/mountains", "desert/barren", "shrublands/plains", "wetland/bog/marsh",
    "open_water", "enclosed_water", "thin", "thick", "low", "high", "isolated",
    "extended", "cumulus", "cumulonimbus", "altocumulus/stratocumulus", "cirrus",
    "haze/fog", "ice_clouds", "contrails",
]


def tile_positions(total: int, patch: int) -> list:
    """Kare baslangic konumlari; son kare kenara hizalanir.

    1022 / 256 = 3.99 -> sadece stride ile 3 kare cikar ve goruntunun %25'i
    kullanilmaz. Son kareyi kenara yapistirinca 4 kare olur ve tum alan
    kapsanir (son karede kismi ortusme olur, kabul edilebilir).
    """
    if patch >= total:
        return [0]
    positions = list(range(0, total - patch + 1, patch))
    if positions[-1] + patch < total:
        positions.append(total - patch)
    return positions


def assign_splits(tags: pd.DataFrame, val_fraction: float, seed: int) -> pd.Series:
    """Sahne bazli bolumleme.

    KARE BAZLI DEGIL, SAHNE BAZLI: ayni sahneden gelen kareler birbirine cok
    benzer. Kareleri rastgele bolmek egitim verisinin test setine sizmasina
    (data leakage) ve dogrulugun sisirilmesine yol acar.

    Test = CALIBRATION + VALIDATION (README onerisi, insan seviyesi karsilastirmasi)
    MAIN -> train / val, bulut orani ve kar/buz etiketine gore tabakalanmis.
    """
    from sklearn.model_selection import train_test_split

    split = pd.Series("train", index=tags.index)
    is_test = tags["dataset"].isin(["CALIBRATION", "VALIDATION"])
    split[is_test] = "test"

    main = tags[~is_test]
    # Tabakalama anahtari: bulut orani dilimi + kar/buz (en zor karisiklik)
    cloud_bucket = pd.cut(main["cloud_percent"], bins=[-0.1, 5, 35, 65, 95, 100.1],
                          labels=["0-5", "5-35", "35-65", "65-95", "95-100"])
    strata = cloud_bucket.astype(str) + "_" + main["snow/ice"].astype(str)

    # Tek uyeli tabakalar stratify'i patlatir; onlari birlestir
    counts = strata.value_counts()
    strata = strata.where(strata.map(counts) >= 2, "other")

    train_idx, val_idx = train_test_split(
        main.index, test_size=val_fraction, random_state=seed, stratify=strata
    )
    split[val_idx] = "val"
    return split


def process_scene(subscene: np.ndarray, mask: np.ndarray, patch_size: int,
                  shadow_as_cloud: bool):
    """Bir sahneyi karelere boler. Uretici: (patch_chw, cloud_mask, cloud_fraction)."""
    # (H, W, 13) -> secilmis bantlar -> (C, H, W)
    img = subscene[..., config.BAND_INDICES].transpose(2, 0, 1)
    img = normalize_reflectance(img).astype(np.float16)

    cloud = mask[..., config.MASK_CLASS_CLOUD]
    if shadow_as_cloud:
        cloud = cloud | mask[..., config.MASK_CLASS_SHADOW]
    cloud = cloud.astype(np.uint8)

    ys = tile_positions(img.shape[1], patch_size)
    xs = tile_positions(img.shape[2], patch_size)

    for y in ys:
        for x in xs:
            patch = img[:, y:y + patch_size, x:x + patch_size]
            patch_mask = cloud[y:y + patch_size, x:x + patch_size]
            yield patch, patch_mask, float(patch_mask.mean()), (y, x)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, required=True,
                   help="subscenes.zip, masks.zip ve classification_tags.csv iceren dizin")
    p.add_argument("--out", type=Path, default=None, help="varsayilan: data/patches")
    p.add_argument("--patch-size", type=int, default=config.PATCH_SIZE)
    p.add_argument("--val-fraction", type=float, default=config.SPLIT_VAL_FRACTION)
    p.add_argument("--shadow-as-cloud", action="store_true", default=config.SHADOW_AS_CLOUD)
    p.add_argument("--max-scenes", type=int, default=None,
                   help="hizli deneme icin sahne sayisini sinirla")
    p.add_argument("--max-difficulty", type=int, default=5,
                   help="bu zorluktan yuksek sahneleri atla (5=hicbirini atlama)")
    p.add_argument("--seed", type=int, default=config.SEED)
    p.add_argument("--dry-run", action="store_true",
                   help="dosya yazmaz, sadece bolumleme ve boyut tahmini gosterir")
    args = p.parse_args()

    subscenes_zip = args.source / "subscenes.zip"
    masks_zip = args.source / "masks.zip"
    tags_csv = args.source / "classification_tags.csv"
    for f in (subscenes_zip, masks_zip, tags_csv):
        if not f.exists():
            raise SystemExit(f"bulunamadi: {f}")

    out_dir = args.out or config.DATA_PATCHES
    tags = pd.read_csv(tags_csv, index_col="index")
    print(f"{len(tags)} sahne etiketi okundu")

    if args.max_difficulty < 5:
        before = len(tags)
        tags = tags[tags["difficulty"] <= args.max_difficulty]
        print(f"zorluk filtresi: {before} -> {len(tags)} sahne")

    tags["split"] = assign_splits(tags, args.val_fraction, args.seed)
    if args.max_scenes:
        # Her bolumden oransal ornek al, bolumlerden biri bos kalmasin.
        # groupby().apply() kullanilmiyor: pandas 3.0'da gruplama kolonunu
        # sonuctan dusuruyor ve 'split' kayboluyor.
        # head() DEGIL sample(): sahneler arsivde tarih sirali, ilk N sahne
        # cografi/mevsimsel olarak taraflidir (deneme kosusunda val bolumu
        # %100 bulutlu cikabiliyor).
        total = len(tags)
        keep = []
        for _, group in tags.groupby("split"):
            n_keep = min(len(group), max(1, round(args.max_scenes * len(group) / total)))
            keep.extend(group.sample(n=n_keep, random_state=args.seed).index)
        tags = tags.loc[keep]
        print(f"--max-scenes: {len(tags)} sahne ile devam")

    print("\nsahne bazli bolumleme:")
    print(tags.groupby("split").agg(
        sahne=("scene", "count"),
        ort_bulut_yuzdesi=("cloud_percent", "mean"),
        kar_buz=("snow/ice", "sum"),
    ).to_string())

    patches_per_scene = len(tile_positions(SCENE_SIZE, args.patch_size)) ** 2
    total_patches = len(tags) * patches_per_scene
    bytes_per_patch = config.IN_CHANNELS * args.patch_size**2 * 2  # float16
    print(f"\nsahne basina {patches_per_scene} kare -> toplam {total_patches:,} kare")
    print(f"tahmini disk: {total_patches * bytes_per_patch / 1024**3:.2f} GB "
          f"(+ maskeler ~{total_patches * args.patch_size**2 / 1024**3:.2f} GB)")

    if args.dry_run:
        print("\n--dry-run: dosya yazilmadi")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "images").mkdir(exist_ok=True)
    (out_dir / "masks").mkdir(exist_ok=True)

    rows = []
    skipped = []

    with zipfile.ZipFile(subscenes_zip) as sz, zipfile.ZipFile(masks_zip) as mz:
        subscene_names = {Path(n).stem: n for n in sz.namelist() if n.endswith(".npy")}
        mask_names = {Path(n).stem: n for n in mz.namelist() if n.endswith(".npy")}

        for idx, row in tqdm(list(tags.iterrows()), desc="sahneler"):
            scene = row["scene"]
            if scene not in subscene_names or scene not in mask_names:
                skipped.append((scene, "zip icinde yok"))
                continue

            # Zip'ten bellege oku (diske acmadan)
            subscene = np.load(io.BytesIO(sz.read(subscene_names[scene])))
            mask = np.load(io.BytesIO(mz.read(mask_names[scene])))

            if subscene.shape[:2] != mask.shape[:2]:
                skipped.append((scene, f"sekil uyusmuyor {subscene.shape} vs {mask.shape}"))
                continue

            for patch, patch_mask, cloud_fraction, (y, x) in process_scene(
                subscene, mask, args.patch_size, args.shadow_as_cloud
            ):
                name = f"{scene}_y{y:04d}_x{x:04d}"
                img_path = out_dir / "images" / f"{name}.npy"
                mask_path = out_dir / "masks" / f"{name}.npy"
                np.save(img_path, patch)
                np.save(mask_path, patch_mask)

                rows.append({
                    "path": str(img_path),
                    "mask_path": str(mask_path),
                    "label": int(cloud_fraction >= config.CLOUD_PIXEL_THRESHOLD),
                    "split": row["split"],
                    "scene": scene,
                    "cloud_fraction": round(cloud_fraction, 6),
                    "difficulty": int(row["difficulty"]),
                    "dataset": row["dataset"],
                    **{t: int(row[t]) for t in TAG_COLUMNS if t in row},
                })

    index = pd.DataFrame(rows)
    index_path = out_dir / "index.csv"
    index.to_csv(index_path, index=False)

    print(f"\n{len(index):,} kare yazildi -> {index_path}")
    if skipped:
        print(f"atlanan {len(skipped)} sahne: {skipped[:5]}")

    print("\nbolum x sinif dagilimi (0=kullanilabilir, 1=bulutlu):")
    print(index.pivot_table(index="split", columns="label", values="path",
                            aggfunc="count", fill_value=0).to_string())

    meta = {
        "source": str(args.source),
        "patch_size": args.patch_size,
        "bands": config.BANDS,
        "cloud_pixel_threshold": config.CLOUD_PIXEL_THRESHOLD,
        "shadow_as_cloud": bool(args.shadow_as_cloud),
        "reflectance_clip": config.REFLECTANCE_CLIP,
        "val_fraction": args.val_fraction,
        "seed": args.seed,
        "scenes": int(tags.shape[0]),
        "patches": int(len(index)),
        "test_split_note": "CALIBRATION + VALIDATION (README onerisi)",
    }
    (out_dir / "prepare_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"\nmeta: {out_dir / 'prepare_meta.json'}")


if __name__ == "__main__":
    main()
