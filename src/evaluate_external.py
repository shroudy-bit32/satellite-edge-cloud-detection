"""Harici veri setinde degerlendirme (SPARCS / Landsat 8).

Egitimde gorulmemis, FARKLI BIR UYDUDAN gelen veride genelleme olcer.

Kullanim:
    python -m src.evaluate_external --checkpoint outputs/checkpoints/baseline_best.pt
    python -m src.evaluate_external --onnx outputs/onnx/baseline_int8.onnx --tag baseline

Rapor icin uc kirilim uretir:
  1. Genel metrikler (S2CMC test sonuclariyla karsilastirilir)
  2. Kar/buz agirlikli karelerde metrikler - en zor karisiklik
  3. Veri indirme kazanci (farkli sensorde de gecerli mi?)
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

from src import config
from src.benchmark import downlink_analysis, make_session


def predict_torch(checkpoint: Path, index: pd.DataFrame) -> np.ndarray:
    from src.export import load_from_checkpoint

    model, task = load_from_checkpoint(checkpoint)
    if task != "classification":
        raise SystemExit("bu betik siniflandirici bekliyor (U-Net icin --task seg eklenmeli)")

    probs = []
    with torch.no_grad():
        for path in index["path"]:
            x = torch.from_numpy(np.load(path).astype(np.float32)).unsqueeze(0)
            probs.append(torch.sigmoid(model(x).squeeze()).item())
    return np.array(probs)


def predict_onnx(onnx_path: Path, index: pd.DataFrame) -> np.ndarray:
    session = make_session(onnx_path)
    name = session.get_inputs()[0].name
    probs = []
    for path in index["path"]:
        x = np.load(path).astype(np.float32)[None]
        logit = float(session.run(None, {name: x})[0].squeeze())
        probs.append(1.0 / (1.0 + np.exp(-logit)))
    return np.array(probs)


def score(probs: np.ndarray, targets: np.ndarray, threshold: float) -> dict:
    preds = (probs >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        targets, preds, average="binary", zero_division=0
    )
    out = {
        "n": int(len(targets)),
        "accuracy": float((preds == targets).mean()),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }
    if len(set(targets.tolist())) > 1:
        out["roc_auc"] = float(roc_auc_score(targets, probs))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument("--onnx", type=Path, default=None)
    p.add_argument("--index", type=Path,
                   default=config.PROJECT_ROOT / "data" / "patches_sparcs" / "index.csv")
    p.add_argument("--tag", default="baseline", help="esigi hangi egitim ozetinden alsin")
    p.add_argument("--threshold", type=float, default=None)
    p.add_argument("--snow-threshold", type=float, default=0.2,
                   help="bu orandan fazla kar/buz iceren kareler ayri raporlanir")
    args = p.parse_args()

    if (args.checkpoint is None) == (args.onnx is None):
        raise SystemExit("--checkpoint VEYA --onnx verilmeli (ikisi birden degil)")
    if not args.index.exists():
        raise SystemExit(f"bulunamadi: {args.index}\nonce 'python -m src.prepare_sparcs' calistir")

    # Esik S2CMC egitiminden gelir. Harici sette yeniden ayarlamak
    # "test setine bakip ayar yapmak" olur ve sonucu gecersiz kilar.
    threshold = args.threshold
    if threshold is None:
        summary = config.REPORTS / f"{args.tag}_train_summary.json"
        if not summary.exists():
            raise SystemExit(f"{summary} yok - --threshold verin")
        threshold = json.loads(summary.read_text(encoding="utf-8"))["threshold"]
    print(f"karar esigi (S2CMC egitiminden): {threshold:.4f}")

    index = pd.read_csv(args.index)
    targets = index["label"].to_numpy()
    print(f"{len(index)} kare, {index['scene'].nunique()} sahne, "
          f"%{100 * targets.mean():.1f} bulutlu\n")

    model_name = str(args.checkpoint or args.onnx)
    probs = (predict_torch(args.checkpoint, index) if args.checkpoint
             else predict_onnx(args.onnx, index))

    results = {"model": model_name, "index": str(args.index), "threshold": threshold}

    print("=== Genel ===")
    results["overall"] = score(probs, targets, threshold)
    for k, v in results["overall"].items():
        print(f"  {k:<12} {v if isinstance(v, int) else f'{v:.4f}'}")

    if "snow_fraction" in index.columns:
        snowy = index["snow_fraction"].to_numpy() > args.snow_threshold
        if snowy.sum() > 0:
            print(f"\n=== Kar/buz agirlikli kareler (>%{100 * args.snow_threshold:.0f}) ===")
            results["snowy"] = score(probs[snowy], targets[snowy], threshold)
            for k, v in results["snowy"].items():
                print(f"  {k:<12} {v if isinstance(v, int) else f'{v:.4f}'}")

            # Kar/buz karelerinde yanlis pozitif = kari bulut sanmak.
            # Bu, uyduda kullanilabilir kar/buz goruntusunun silinmesi demek.
            preds = (probs[snowy] >= threshold).astype(int)
            clean_snow = targets[snowy] == 0
            if clean_snow.sum():
                fp_rate = float(preds[clean_snow].mean())
                results["snowy"]["kari_bulut_sanma_orani"] = fp_rate
                print(f"  temiz kar/buz karelerinde yanlis eleme: %{100 * fp_rate:.2f}")

        print(f"\n=== Kar/buz olmayan kareler ===")
        results["non_snowy"] = score(probs[~snowy], targets[~snowy], threshold)
        for k, v in results["non_snowy"].items():
            print(f"  {k:<12} {v if isinstance(v, int) else f'{v:.4f}'}")

    print("\n=== Veri indirme kazanci (harici sensor) ===")
    results["downlink"] = downlink_analysis(probs, targets, threshold)
    for k, v in results["downlink"].items():
        print(f"  {k:<38} {v}")

    out = config.REPORTS / f"{args.tag}_external_sparcs.json"
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nsonuclar: {out}")

    meta_path = args.index.parent / "prepare_meta.json"
    if meta_path.exists():
        caveat = json.loads(meta_path.read_text(encoding="utf-8")).get("caveat")
        if caveat:
            print(f"\nUYARI: {caveat}")


if __name__ == "__main__":
    main()
