"""Karar esigini BELIRSIZ BANTTA secip calisma noktalarini karsilastirir.

Gerekce: karelerin ~%73'u ya tamamen temiz ya tamamen kapali; bu kolay
orneklerin kararlari esikten bagimsiz. Esigi onlarla birlikte ayarlamak,
esigi asil onemli oldugu belirsiz bolgeden uzaklastiriyor.

Belirsiz bantta ayarlanan esik, U-Net karar katmani deneyinde yanlis elemeyi
%5.94'ten %0.24'e dusurmustu. Bu betik ayni fikri SINIFLANDIRICIYA uygular.

Cikti: birden fazla calisma noktasi iceren operating_points.json. Esik artik
kodda sabit degil, modelle birlikte tasinan ve yerden guncellenebilir bir
parametre.

Kullanim:
    python -m src.tune_operating_point --checkpoint outputs/checkpoints/c6_tuned_best.pt --tag c6_tuned
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import precision_recall_curve, precision_recall_fscore_support, roc_auc_score

from src import config
from src.export import load_from_checkpoint


@torch.no_grad()
def predict(checkpoint: Path, index: pd.DataFrame, batch_size: int = 64) -> np.ndarray:
    model, task = load_from_checkpoint(checkpoint)
    if task != "classification":
        raise SystemExit("siniflandirici checkpoint'i gerekli")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    probs = []
    paths = index["path"].tolist()
    for i in range(0, len(paths), batch_size):
        batch = np.stack([np.load(p).astype(np.float32) for p in paths[i:i + batch_size]])
        probs.append(torch.sigmoid(model(torch.from_numpy(batch).to(device)).squeeze(1)).cpu().numpy())
    return np.concatenate(probs)


def tune(scores: np.ndarray, targets: np.ndarray, min_precision: float | None) -> float:
    """min_precision verilirse o kisit altinda recall'u maksimize eder,
    None verilirse dogrudan en iyi F1'i secer.

    DIKKAT: min_precision=0 GECERLI BIR "en iyi F1" istegi DEGILDIR - her esik
    kisiti saglar, algoritma recall'u maksimize edip tabana iner ve her seyi
    bulutlu ilan eder. Dengeli nokta icin None kullanilmalidir.
    """
    precisions, recalls, thresholds = precision_recall_curve(targets, scores)
    precisions, recalls = precisions[:-1], recalls[:-1]
    f1s = 2 * precisions * recalls / np.clip(precisions + recalls, 1e-9, None)

    if min_precision is None:
        idx = int(np.argmax(f1s))
    else:
        feasible = precisions >= min_precision
        idx = int(np.argmax(np.where(feasible, recalls, -1.0))) if feasible.any() \
            else int(np.argmax(f1s))
    t = float(thresholds[idx])
    below = scores[scores < t]
    return float((t + below.max()) / 2) if below.size else max(t - 1e-6, 0.0)


def assess(scores: np.ndarray, targets: np.ndarray, threshold: float) -> dict:
    preds = (scores >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        targets, preds, average="binary", zero_division=0)

    n = len(targets)
    tp = int(((preds == 1) & (targets == 1)).sum())
    fp = int(((preds == 1) & (targets == 0)).sum())
    usable = int((targets == 0).sum())

    return {"accuracy": round(float((preds == targets).mean()), 4),
            "precision": round(float(precision), 4),
            "recall": round(float(recall), 4),
            "f1": round(float(f1), 4),
            "veri_azalmasi_%": round(100 * (tp + fp) / n, 2),
            "kaybedilen_kullanilabilir_%": round(100 * fp / max(usable, 1), 3)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--ambiguous-low", type=float, default=0.02)
    p.add_argument("--ambiguous-high", type=float, default=0.98)
    p.add_argument("--sparcs-index", type=Path,
                   default=config.PROJECT_ROOT / "data" / "patches_sparcs" / "index.csv")
    args = p.parse_args()

    index = pd.read_csv(config.PATCH_INDEX)
    val = index[index["split"] == "val"].reset_index(drop=True)
    test = index[index["split"] == "test"].reset_index(drop=True)

    print("tahminler uretiliyor...")
    val_scores = predict(args.checkpoint, val)
    test_scores = predict(args.checkpoint, test)
    y_val = val["label"].to_numpy()
    y_test = test["label"].to_numpy()

    amb = ((val["cloud_fraction"] > args.ambiguous_low) &
           (val["cloud_fraction"] < args.ambiguous_high)).to_numpy()
    print(f"val: {len(val)} kare, belirsiz bant {amb.sum()} kare (%{100 * amb.mean():.1f})\n")

    # --- Calisma noktalari ---
    points = {}

    summary_path = config.REPORTS / f"{args.tag}_train_summary.json"
    if summary_path.exists():
        points["mevcut (tum val, precision>=0.99)"] = json.loads(
            summary_path.read_text(encoding="utf-8"))["threshold"]

    points["belirsiz bant (precision>=0.99)"] = tune(val_scores[amb], y_val[amb], 0.99)
    points["belirsiz bant (precision>=0.995)"] = tune(val_scores[amb], y_val[amb], 0.995)
    points["dengeli (tum val, en iyi F1)"] = tune(val_scores, y_val, None)

    # --- S2CMC test ---
    rows = []
    for name, t in points.items():
        rows.append({"calisma noktasi": name, "esik": round(t, 4), **assess(test_scores, y_test, t)})
    df_test = pd.DataFrame(rows)
    print("=== S2CMC test seti ===")
    print(df_test.to_string(index=False))

    print(f"\nROC-AUC (esikten bagimsiz): {roc_auc_score(y_test, test_scores):.4f}")

    # --- SPARCS harici ---
    df_sparcs = None
    if args.sparcs_index.exists():
        sparcs = pd.read_csv(args.sparcs_index)
        sparcs_scores = predict(args.checkpoint, sparcs)
        y_sparcs = sparcs["label"].to_numpy()
        snowy = sparcs["snow_fraction"].to_numpy() > 0.2

        rows = []
        for name, t in points.items():
            r = {"calisma noktasi": name, "esik": round(t, 4),
                 **assess(sparcs_scores, y_sparcs, t)}
            preds = (sparcs_scores >= t).astype(int)
            clean_snow = snowy & (y_sparcs == 0)
            r["kar_buz_yanlis_eleme_%"] = round(100 * float(preds[clean_snow].mean()), 2)
            rows.append(r)
        df_sparcs = pd.DataFrame(rows)
        print("\n=== SPARCS (harici sensor) ===")
        print(df_sparcs.to_string(index=False))
        print(f"\nSPARCS ROC-AUC: {roc_auc_score(y_sparcs, sparcs_scores):.4f}")

    # --- Kaydet ---
    out = {"model": str(args.checkpoint), "tag": args.tag,
           "ambiguous_band": [args.ambiguous_low, args.ambiguous_high],
           "operating_points": {
               name: {"threshold": round(t, 6),
                      "s2cmc_test": assess(test_scores, y_test, t)}
               for name, t in points.items()},
           "note": "Esik kodda sabit degildir; burada tanimli noktalardan biri "
                   "secilerek kullanilir ve yerden guncellenebilir."}
    if df_sparcs is not None:
        for name in points:
            row = df_sparcs[df_sparcs["calisma noktasi"] == name].iloc[0].to_dict()
            out["operating_points"][name]["sparcs"] = {
                k: v for k, v in row.items() if k not in ("calisma noktasi", "esik")}

    path = config.REPORTS / f"{args.tag}_operating_points.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    df_test.to_csv(config.REPORTS / f"{args.tag}_operating_points.csv", index=False)
    print(f"\ncalisma noktalari: {path}")


if __name__ == "__main__":
    main()
