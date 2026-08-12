"""Hiperparametre taramasi (PDF Gunler 7-12: "hiperparametre denemeleri").

Onceki ablasyonlar mimari ve duzenleme seceneklerini taradi (epoch, EMA,
dengeli ornekleme, omurga). Bu betik klasik hiperparametreleri tarar:
ogrenme orani, agirlik duzenlemesi, batch boyutu, etiket yumusatma.

Rastgele arama kullanilir; ayni butcede izgara aramasindan daha verimlidir
cunku onemsiz boyutlarda bosa deneme yapmaz.

Kullanim:
    python -m src.hparam_sweep --trials 8 --epochs 10
    python -m src.hparam_sweep --trials 8 --epochs 10 --model mobilenetv2_050
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src import config

# Arama uzayi: her biri log-uniform veya secim
SEARCH_SPACE = {
    "lr": ("loguniform", 5e-5, 3e-3),
    "weight_decay": ("loguniform", 1e-6, 1e-2),
    "batch_size": ("choice", [32, 64, 128]),
    "label_smoothing": ("choice", [0.0, 0.0, 0.05, 0.1]),  # 0 iki kez: daha olasi
}

# Segmentasyon: label_smoothing yok, batch daha kucuk (bellek), bce_weight eklendi
SEG_SEARCH_SPACE = {
    "lr": ("loguniform", 5e-5, 3e-3),
    "weight_decay": ("loguniform", 1e-6, 1e-2),
    "batch_size": ("choice", [8, 16, 32]),
    "bce_weight": ("choice", [0.3, 0.5, 0.5, 0.7]),  # Dice ile BCE dengesi
}


def sample_config(rng: np.random.Generator, space: dict) -> dict:
    cfg = {}
    for name, spec in space.items():
        kind = spec[0]
        if kind == "loguniform":
            lo, hi = spec[1], spec[2]
            cfg[name] = float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
        elif kind == "choice":
            options = spec[1]
            value = options[int(rng.integers(len(options)))]
            cfg[name] = value
    return cfg


def run_trial(tag: str, cfg: dict, args) -> dict | None:
    if args.task == "seg":
        cmd = [sys.executable, "-m", "src.train_seg",
               "--tag", tag, "--encoder", args.model, "--epochs", str(args.epochs),
               "--lr", f"{cfg['lr']:.6g}", "--weight-decay", f"{cfg['weight_decay']:.6g}",
               "--batch-size", str(cfg["batch_size"]),
               "--bce-weight", str(cfg["bce_weight"]),
               "--workers", str(args.workers)]
    else:
        cmd = [sys.executable, "-m", "src.train",
               "--tag", tag, "--model", args.model, "--epochs", str(args.epochs),
               "--lr", f"{cfg['lr']:.6g}", "--weight-decay", f"{cfg['weight_decay']:.6g}",
               "--batch-size", str(cfg["batch_size"]),
               "--label-smoothing", str(cfg["label_smoothing"]),
               "--workers", str(args.workers)]
        if args.ema:
            cmd.append("--ema")

    # encoding acikca verilmeli: Windows'ta varsayilan sistem kodlamasi (cp1254)
    # tqdm'in UTF-8 ilerleme cubugunu cozemiyor ve okuma is parcaciginda
    # UnicodeDecodeError firlatiyor. Bu, hata mesajlarinin kaybolmasina yol acar.
    result = subprocess.run(cmd, cwd=config.PROJECT_ROOT, capture_output=True,
                            text=True, encoding="utf-8", errors="replace")
    if result.returncode != 0:
        tail = result.stderr.strip().splitlines()[-3:]
        print(f"  BASARISIZ: {' | '.join(tail)}")
        return None

    summary_path = config.REPORTS / f"{tag}_train_summary.json"
    if not summary_path.exists():
        print("  ozet dosyasi olusmadi")
        return None
    return json.loads(summary_path.read_text(encoding="utf-8"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--trials", type=int, default=8)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--model", default="mobilenetv2_100")
    p.add_argument("--ema", action="store_true", default=True)
    p.add_argument("--workers", type=int, default=config.NUM_WORKERS)
    p.add_argument("--seed", type=int, default=config.SEED)
    p.add_argument("--prefix", default="hp")
    p.add_argument("--task", default="cls", choices=["cls", "seg"])
    args = p.parse_args()

    space = SEG_SEARCH_SPACE if args.task == "seg" else SEARCH_SPACE
    # Segmentasyonda model secimi IoU'ya gore, siniflandirmada F1'e gore
    val_key, test_keys = ("best_val_iou", ["iou", "dice", "image_level_accuracy"]) \
        if args.task == "seg" else ("best_val_f1", ["f1", "accuracy", "precision",
                                                    "recall", "roc_auc"])

    rng = np.random.default_rng(args.seed)
    print(f"{args.trials} deneme x {args.epochs} epoch, gorev {args.task}, model {args.model}")
    print(f"bantlar: {len(config.BANDS)} {config.BANDS}\n")

    rows = []
    for i in range(args.trials):
        cfg = sample_config(rng, space)
        tag = f"{args.prefix}{i:02d}"
        extra = " ".join(f"{k}={v}" for k, v in cfg.items()
                         if k not in ("lr", "weight_decay", "batch_size"))
        print(f"[{i + 1}/{args.trials}] {tag}: lr={cfg['lr']:.2e} "
              f"wd={cfg['weight_decay']:.2e} bs={cfg['batch_size']} {extra}", flush=True)

        summary = run_trial(tag, cfg, args)
        if summary is None:
            continue

        t = summary["test"]
        row = {"tag": tag, **cfg, "val": round(summary[val_key], 4)}
        if "threshold" in summary:
            row["threshold"] = round(summary["threshold"], 4)
        row.update({f"test_{k}": round(t[k], 4) for k in test_keys if k in t})
        rows.append(row)

        primary = f"test_{test_keys[0]}"
        print(f"    -> val {row['val']:.4f} | {primary} {row[primary]:.4f}", flush=True)

    if not rows:
        raise SystemExit("hicbir deneme tamamlanamadi")

    df = pd.DataFrame(rows)
    # Model secimi VAL uzerinden yapilmali; test yalnizca raporlama icin.
    df = df.sort_values("val", ascending=False).reset_index(drop=True)

    print("\n" + "=" * 110)
    print(df.to_string(index=False))

    out = config.REPORTS / f"{args.prefix}_sweep.csv"
    df.to_csv(out, index=False)

    best = df.iloc[0]
    primary = f"test_{test_keys[0]}"
    print(f"\nEn iyi (val {val_key}'e gore): {best['tag']}")
    print("  " + "  ".join(f"{k}={best[k]:.3e}" if isinstance(best[k], float) and best[k] < 0.01
                           else f"{k}={best[k]}" for k in space))
    print(f"  val {best['val']:.4f} -> {primary} {best[primary]:.4f}")

    print(f"\nval araligi: {df['val'].min():.4f} - {df['val'].max():.4f} "
          f"(yayilim {df['val'].max() - df['val'].min():.4f})")
    print("YORUM: yayilim kucukse hiperparametreler bu problemde belirleyici degil.")
    print(f"\ntablo: {out}")

    (config.REPORTS / f"{args.prefix}_best.json").write_text(
        json.dumps({k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                    for k, v in best.to_dict().items()}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
