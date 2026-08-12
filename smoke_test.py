"""Sentetik veriyle uctan uca hat testi: egitim -> ONNX -> INT8 -> benchmark.

Gercek veri inerken hattin calistigini dogrulamak icin. Sentetik "bulutlu"
kareler daha parlak ve daha duz (dusuk varyans) uretilir - model bunu kolayca
ogrenir, amac dogruluk degil hattin kirik olmadigini gostermek.

Kullanim:  python smoke_test.py
"""

import shutil
import sys

import numpy as np
import pandas as pd

from src import config

SMOKE_DIR = config.PROJECT_ROOT / "data" / "patches_smoke"
N_PER_CLASS = {"train": 120, "val": 40, "test": 40}


def make_synthetic_patches():
    if SMOKE_DIR.exists():
        shutil.rmtree(SMOKE_DIR)
    SMOKE_DIR.mkdir(parents=True)

    rng = np.random.default_rng(config.SEED)
    rows = []
    size = 64

    for split, n in N_PER_CLASS.items():
        for label in (0, 1):
            for i in range(n):
                # Taban sahne: dusuk reflektans, yuksek doku ("kullanilabilir")
                arr = rng.normal(0.25, 0.18, (config.IN_CHANNELS, size, size))
                mask = np.zeros((size, size), dtype=np.float32)

                if label == 1:
                    # Rastgele konumlu dairesel "bulut" lekesi: yuksek reflektans,
                    # dusuk doku. Maske de bu lekeden turetilir, boylece
                    # segmentasyon ve siniflandirma etiketleri tutarli olur.
                    cy, cx = rng.integers(16, size - 16, 2)
                    radius = rng.integers(18, 26)
                    yy, xx = np.ogrid[:size, :size]
                    blob = (yy - cy) ** 2 + (xx - cx) ** 2 <= radius**2
                    mask[blob] = 1.0
                    arr[:, blob] = rng.normal(0.75, 0.05, (config.IN_CHANNELS, int(blob.sum())))

                arr = np.clip(arr, 0, 1).astype(np.float32)

                path = SMOKE_DIR / f"{split}_{label}_{i:04d}.npy"
                mask_path = SMOKE_DIR / f"{split}_{label}_{i:04d}_mask.npy"
                np.save(path, arr)
                np.save(mask_path, mask)
                rows.append({"path": str(path), "mask_path": str(mask_path),
                             "label": label, "split": split})

    index = SMOKE_DIR / "index.csv"
    pd.DataFrame(rows).to_csv(index, index=False)
    print(f"{len(rows)} sentetik patch uretildi -> {SMOKE_DIR}")
    return index


def run(module_main, argv):
    old = sys.argv
    sys.argv = argv
    try:
        module_main()
    finally:
        sys.argv = old


def main():
    index = make_synthetic_patches()

    # Tum modulleri sentetik index'e yonlendir. dataset.py bu degeri
    # cagri aninda okudugu icin monkeypatch yeterli.
    config.PATCH_INDEX = index
    original_patch_size = config.PATCH_SIZE
    config.PATCH_SIZE = 64  # ONNX export dummy girdisi de kucuk olsun

    from src import benchmark, export, train, train_seg

    print("\n--- 1/5 siniflandirici egitimi (+ optimizasyonlar) ---")
    run(train.main, ["train", "--epochs", "3", "--batch-size", "16",
                     "--workers", "0", "--no-pretrained", "--tag", "smoke",
                     "--balanced", "--ema", "--warmup-epochs", "1"])

    print("\n--- 2/5 siniflandirici export + INT8 ---")
    run(export.main, ["export", "--checkpoint", str(config.CHECKPOINTS / "smoke_best.pt"),
                      "--tag", "smoke", "--calib-samples", "40"])

    print("\n--- 3/5 siniflandirici benchmark ---")
    run(benchmark.main, ["benchmark", "--tag", "smoke"])

    print("\n--- 4/5 U-Net segmentasyon egitimi ---")
    run(train_seg.main, ["train_seg", "--epochs", "3", "--batch-size", "8",
                         "--workers", "0", "--no-pretrained", "--tag", "smoke_unet"])

    print("\n--- 5/5 U-Net export + INT8 + benchmark ---")
    run(export.main, ["export", "--checkpoint", str(config.CHECKPOINTS / "smoke_unet_best.pt"),
                      "--tag", "smoke_unet", "--calib-samples", "40"])
    run(benchmark.main, ["benchmark", "--tag", "smoke_unet", "--task", "segmentation"])

    config.PATCH_SIZE = original_patch_size
    print("\nHAT CALISIYOR (siniflandirma + segmentasyon).")


if __name__ == "__main__":
    main()
