"""Bulut turu ve yuzey tipine gore basarim analizi (PDF: basarisizlik modlari).

Veri setinin README'si etiketlerin amacini soyle tanimliyor: "belirli
kosullardaki basarimi test etmek icin". Bu betik tam olarak onu yapar.

Onemli sinir: etiketler SAHNE bazlidir, piksel bazli degil. "thin" etiketli bir
sahnede ince bulut BULUNDUGUNU bilir, hangi piksellerin ince oldugunu bilmeyiz.
Dolayisiyla bu analiz "ince bulut iceren sahnelerde basarim" olcer, "ince bulut
piksellerinde basarim" degil.

Kullanim:
    python -m src.tag_analysis --checkpoint outputs/checkpoints/c100_best.pt --tag c100
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

from src import config
from src.export import load_from_checkpoint

CLOUD_TAGS = ["thin", "thick", "low", "high", "isolated", "extended",
              "cumulus", "cumulonimbus", "altocumulus/stratocumulus", "cirrus",
              "haze/fog", "ice_clouds", "contrails"]
SURFACE_TAGS = ["forest/jungle", "snow/ice", "agricultural", "urban/developed",
                "coastal", "hills/mountains", "desert/barren", "shrublands/plains",
                "wetland/bog/marsh", "open_water", "enclosed_water"]


@torch.no_grad()
def predict(checkpoint: Path, index: pd.DataFrame, batch_size: int = 64) -> np.ndarray:
    model, task = load_from_checkpoint(checkpoint)
    if task != "classification":
        raise SystemExit("bu analiz siniflandirici checkpoint'i bekliyor")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    probs = []
    paths = index["path"].tolist()
    for i in range(0, len(paths), batch_size):
        batch = np.stack([np.load(p).astype(np.float32) for p in paths[i:i + batch_size]])
        x = torch.from_numpy(batch).to(device)
        probs.append(torch.sigmoid(model(x).squeeze(1)).cpu().numpy())
    return np.concatenate(probs)


def score_subset(probs, targets, threshold) -> dict:
    if len(targets) == 0:
        return None
    preds = (probs >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        targets, preds, average="binary", zero_division=0
    )
    out = {"n": len(targets), "bulutlu_%": round(100 * targets.mean(), 1),
           "accuracy": round(float((preds == targets).mean()), 4),
           "precision": round(float(precision), 4),
           "recall": round(float(recall), 4),
           "f1": round(float(f1), 4)}
    if len(set(targets.tolist())) > 1:
        out["roc_auc"] = round(float(roc_auc_score(targets, probs)), 4)
        # Temiz karelerde yanlis eleme orani: geri donusu olmayan veri kaybi
        clear = targets == 0
        out["yanlis_eleme_%"] = round(100 * float(preds[clear].mean()), 2)
    return out


def analyse_tags(index: pd.DataFrame, probs: np.ndarray, threshold: float,
                 tags: list, title: str) -> pd.DataFrame:
    targets = index["label"].to_numpy()
    rows = []
    for tag in tags:
        if tag not in index.columns:
            continue
        mask = index[tag].to_numpy() == 1
        if mask.sum() < 20:  # cok az ornek varsa metrik anlamsiz
            continue
        res = score_subset(probs[mask], targets[mask], threshold)
        if res:
            rows.append({"etiket": tag, **res})

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    overall = score_subset(probs, targets, threshold)
    df = pd.concat([pd.DataFrame([{"etiket": "TUMU", **overall}]), df], ignore_index=True)
    print(f"\n=== {title} ===")
    print(df.to_string(index=False))
    return df


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--tag", default="c100")
    p.add_argument("--split", default="test")
    p.add_argument("--threshold", type=float, default=None)
    args = p.parse_args()

    threshold = args.threshold
    if threshold is None:
        summary = config.REPORTS / f"{args.tag}_train_summary.json"
        threshold = json.loads(summary.read_text(encoding="utf-8"))["threshold"]

    index = pd.read_csv(config.PATCH_INDEX)
    index = index[index["split"] == args.split].reset_index(drop=True)
    print(f"{len(index)} kare ({args.split}), karar esigi {threshold:.4f}")

    probs = predict(args.checkpoint, index)

    cloud_df = analyse_tags(index, probs, threshold, CLOUD_TAGS, "Bulut turune gore")
    surface_df = analyse_tags(index, probs, threshold, SURFACE_TAGS, "Yuzey tipine gore")

    # Zorluk seviyesine gore
    rows = []
    targets = index["label"].to_numpy()
    for d in sorted(index["difficulty"].unique()):
        mask = index["difficulty"].to_numpy() == d
        res = score_subset(probs[mask], targets[mask], threshold)
        if res:
            rows.append({"zorluk": int(d), **res})
    diff_df = pd.DataFrame(rows)
    print("\n=== Anotasyon zorluguna gore ===")
    print(diff_df.to_string(index=False))

    out = config.REPORTS / f"{args.tag}_tag_analysis.csv"
    combined = pd.concat([
        cloud_df.assign(kategori="bulut turu"),
        surface_df.assign(kategori="yuzey tipi"),
        diff_df.rename(columns={"zorluk": "etiket"}).assign(kategori="zorluk"),
    ], ignore_index=True)
    combined.to_csv(out, index=False)
    print(f"\ntablo: {out}")

    # En zayif kirilimlar
    valid = combined[combined["etiket"] != "TUMU"].dropna(subset=["f1"])
    if len(valid):
        worst = valid.nsmallest(3, "f1")
        print("\nEn dusuk F1'e sahip kirilimlar:")
        for _, r in worst.iterrows():
            print(f"  {r['etiket']} ({r['kategori']}): F1 {r['f1']:.4f}, "
                  f"n={r['n']}, yanlis eleme %{r.get('yanlis_eleme_%', float('nan'))}")


if __name__ == "__main__":
    main()
