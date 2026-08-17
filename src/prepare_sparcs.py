"""SPARCS (Landsat 8) -> harici dogrulama kareleri.

PDF'te SPARCS "doğrulama amaçlı küçük ölçekli etiketli set" olarak tanimli.
Egitimde KULLANILMAZ; farkli bir uyduda genelleme olcmek icindir.

Kullanim:
    python -m src.prepare_sparcs --zip datas/l8cloudmasks.zip

Format:
    <scene>_data.tif   -> 1000x1000, 10 bant, uint16, L8 B1-B7,B9,B10,B11
    <scene>_mask.png   -> 1000x1000, palet indeksli, sinif kodlari 0-6
    <scene>_mtl.txt    -> Landsat metadatasi (SUN_ELEVATION burada)

IKI FARKLI PAKETLEME VAR - ikisi de desteklenir:
  * l8cloudmasks.zip (USGS ozgun dagitimi): 80 sahne, etiketler *_mask.png,
    MTL dosyalari MEVCUT -> --sun-correction kullanilabilir.
  * sparcs_data_L8.zip (turetilmis paketleme): etiketler *_labels.tif olarak
    GeoTIFF'e cevrilmis, MTL dosyalari YOK, 78 kullanilabilir sahne.
    v1-v3.2 arasindaki yayimlanmis SPARCS sayilari BU paketlemeyle uretildi.

Sahne sayisi farkli oldugu icin (80 vs 78) iki paketlemenin sonuclari birebir
ayni cikmaz: 256x256'da 1280 vs 1248 kare.
"""

import argparse
import io
import json
import re
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


# SPARCS iki farkli paketlemede dagitiliyor. USGS'in l8cloudmasks.zip arsivinde
# etiketler palet indeksli <sahne>_mask.png dosyasinda; bazi dagitimlarda
# <sahne>_labels.tif olarak duruyor. Ikisi de ayni 0-6 sinif kodlarini tasir
# (config.SPARCS_* ile eslesir), bu yuzden ikisi de kabul edilir.
LABEL_SUFFIXES = ("_mask.png", "_labels.tif")


def find_label_name(scene: str, names: set) -> str | None:
    """Sahnenin etiket dosyasinin arsiv icindeki adini dondurur, yoksa None."""
    for suffix in LABEL_SUFFIXES:
        if scene + suffix in names:
            return scene + suffix
    return None


def read_labels(zip_path: Path, label_name: str) -> np.ndarray:
    """Etiket maskesini (H, W) sinif kodu dizisi olarak okur.

    PNG'ler PIL ile okunur: palet indeksli bir PNG'de bize gereken sey RENK
    degil PALET INDEKSIDIR (sinif kodu). GDAL bazi yapilandirmalarda paleti
    RGB'ye acabilecegi ve sinif kodlarini kaybettirebilecegi icin PIL tercih
    edilir - sessizce yanlis etiket uretmekten iyidir.
    """
    if label_name.endswith(".png"):
        from PIL import Image

        with zipfile.ZipFile(zip_path) as z:
            raw = z.read(label_name)
        with Image.open(io.BytesIO(raw)) as im:
            if im.mode != "P":
                raise SystemExit(
                    f"{label_name}: palet indeksli PNG bekleniyordu, '{im.mode}' bulundu. "
                    f"Sinif kodlari palet indeksinde tasiniyor; baska bir modda "
                    f"okumak yanlis etiket uretir."
                )
            return np.array(im, dtype=np.uint8)

    with rasterio.open(f"/vsizip/{zip_path}/{label_name}") as src:
        return src.read(1)


def read_sun_elevation(zip_path: Path, scene: str, names: set) -> float | None:
    """MTL dosyasindan SUN_ELEVATION (derece) okur; MTL yoksa None.

    Sahne adindaki kare eki atilarak MTL adi bulunur:
        sending/LC80010812013365LGN00_18 -> sending/LC80010812013365LGN00_mtl.txt
    """
    mtl_name = re.sub(r"_\d+$", "", scene) + "_mtl.txt"
    if mtl_name not in names:
        return None
    with zipfile.ZipFile(zip_path) as z:
        text = z.read(mtl_name).decode("utf-8", errors="replace")
    match = re.search(r"SUN_ELEVATION\s*=\s*([-\d.]+)", text)
    return float(match.group(1)) if match else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--zip", dest="zip_path", type=Path, required=True)
    p.add_argument("--out", type=Path, default=None, help="varsayilan: data/patches_sparcs")
    p.add_argument("--patch-size", type=int, default=config.PATCH_SIZE)
    p.add_argument("--shadow-as-cloud", action="store_true", default=config.SHADOW_AS_CLOUD)
    p.add_argument("--max-scenes", type=int, default=None)
    p.add_argument("--sun-correction", action="store_true",
                   help="MTL'deki SUN_ELEVATION ile reflektansi duzelt "
                        "(Sentinel-2 ile ayni olcege getirir). Yayimlanan SPARCS "
                        "sayilari BU DUZELTME OLMADAN olculmustur; ikisini "
                        "karsilastirmak icin ayri dizinlere yazin.")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    if not args.zip_path.exists():
        raise SystemExit(f"bulunamadi: {args.zip_path}")

    band_indices = resolve_band_indices()
    print(f"bant eslemesi: {list(zip(config.BANDS, [config.L8_DATA_BANDS[i] for i in band_indices]))}")

    out_dir = args.out or (config.PROJECT_ROOT / "data" / "patches_sparcs")

    with zipfile.ZipFile(args.zip_path) as z:
        names = z.namelist()
    name_set = set(names)
    data_files = sorted(n for n in names if n.endswith("_data.tif"))
    candidates = [(n[: -len("_data.tif")]) for n in data_files]

    scenes = [(s, lbl) for s in candidates if (lbl := find_label_name(s, name_set))]
    missing = [s for s in candidates if find_label_name(s, name_set) is None]

    if not scenes:
        raise SystemExit(
            f"{args.zip_path} icinde etiket dosyasi bulunamadi.\n"
            f"{len(data_files)} adet _data.tif var ama hicbirinin etiketi yok.\n"
            f"Beklenen adlar: <sahne>_mask.png veya <sahne>_labels.tif"
        )
    if missing:
        print(f"UYARI: {len(missing)} sahnenin etiket dosyasi yok, atlandi")

    label_kind = scenes[0][1].rsplit("_", 1)[-1]
    print(f"etiket bicimi: *_{label_kind}")

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
    sun_corrected = 0
    for scene, label_name in tqdm(scenes, desc="SPARCS sahneleri"):
        # scene arsiv ICINDEKI tam yoldur (orn. "sending/LC8001...._18") ve
        # /vsizip okumalarinda oyle kullanilmalidir. Dosya adi ve index icin
        # dizin oneki atilir; aksi halde cikti yolu images/sending/... olur ve
        # var olmayan bir alt dizine yazmaya calisir.
        scene_id = scene.rsplit("/", 1)[-1]
        # GDAL'in vsizip surucusu: zip'i acmadan dogrudan oku
        with rasterio.open(f"/vsizip/{args.zip_path}/{scene}_data.tif") as src:
            data = src.read().astype(np.float32)
        labels = read_labels(args.zip_path, label_name)

        refl = dn_to_reflectance(data[band_indices])

        if args.sun_correction:
            sun_elev = read_sun_elevation(args.zip_path, scene, name_set)
            if sun_elev is not None:
                # Sentinel-2 L1C reflektanslari gunes acisina gore normalize
                # edilmistir, Landsat 8 DN'leri degildir. sin(gunes yuksekligi)
                # ile bolmek iki veri setini ayni radyometrik olcege getirir ve
                # raporun "SPARCS karsilastirmasi kotumser" cekincesini kaldirir.
                refl = refl / np.sin(np.deg2rad(sun_elev))
                sun_corrected += 1

        img = normalize_reflectance(refl).astype(np.float16)

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

                name = f"{scene_id}_y{y:04d}_x{x:04d}"
                img_path = out_dir / "images" / f"{name}.npy"
                mask_path = out_dir / "masks" / f"{name}.npy"
                np.save(img_path, patch)
                np.save(mask_path, patch_mask)

                rows.append({
                    "path": str(img_path),
                    "mask_path": str(mask_path),
                    "label": int(cloud_fraction >= config.CLOUD_PIXEL_THRESHOLD),
                    "split": "test",  # SPARCS tamami harici dogrulama
                    "scene": scene_id,
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
        "label_source": label_kind,
        "sun_correction": bool(args.sun_correction),
        "sun_corrected_scenes": sun_corrected,
        "caveat": (
            f"Gunes yuksekligi duzeltmesi {sun_corrected}/{len(scenes)} sahnede "
            f"MTL'deki SUN_ELEVATION ile uygulandi; reflektans olcegi Sentinel-2 "
            f"ile hizalandi."
            if args.sun_correction else
            "Gunes yuksekligi duzeltmesi uygulanmadi (--sun-correction verilmedi); "
            "Sentinel-2 ile arasinda sistematik reflektans kaymasi bulunur."
        ),
    }
    (out_dir / "prepare_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"meta: {out_dir / 'prepare_meta.json'}")


if __name__ == "__main__":
    main()
