"""U-Net maskesinden karar uretme: sekil oznitelikleri ve esik stratejileri.

Iki hipotezi test eder:

A) SEKIL OZNITELIKLERI: maskenin konturu icindeki doluluk (solidity), bagli
   bilesen sayisi ve en buyuk yapinin orani, tek basina bulut oranindan daha
   iyi bir karar verir mi? Ozellikle PRECISION'i artirir mi (yanlis eleme
   geri donusu olmayan veri kaybi oldugu icin kritik metrik).

B) UC DEGERLERI DISLAMA: karelerin %73'u ya tamamen temiz ya tamamen kapali
   ve kararlari belirsiz degil. Esigi yalnizca BELIRSIZ bantta (kismi bulutlu
   kareler) ayarlamak, esigi asil onemli oldugu yere odaklar mi?

Kullanim:
    python -m src.decision_layer --unet outputs/checkpoints/unet050_best.pt
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from scipy import ndimage
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, precision_recall_fscore_support

from src import config
from src.dataset import CloudSegDataset
from src.export import load_from_checkpoint


def shape_features(mask: np.ndarray) -> dict:
    """Ikili maskeden geometrik oznitelikler. Hepsi maskenin kendisinden gelir,
    ek etiket veya bant gerektirmez - yorungede hesaplanabilir."""
    total = mask.size
    cloud_px = int(mask.sum())
    if cloud_px == 0:
        return {"cloud_fraction": 0.0, "solidity": 0.0, "n_components": 0,
                "largest_fraction": 0.0, "filled_ratio": 0.0, "compactness": 0.0}

    m8 = mask.astype(np.uint8)
    contours, _ = cv2.findContours(m8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hull_area = sum(cv2.contourArea(cv2.convexHull(c)) for c in contours)
    perimeter = sum(cv2.arcLength(c, True) for c in contours)

    labeled, n_comp = ndimage.label(m8)
    largest = int(ndimage.sum(m8, labeled, range(1, n_comp + 1)).max()) if n_comp else 0
    filled = int(ndimage.binary_fill_holes(m8.astype(bool)).sum())

    return {
        "cloud_fraction": cloud_px / total,
        "solidity": min(cloud_px / hull_area, 1.0) if hull_area > 0 else 0.0,
        "n_components": n_comp,
        "largest_fraction": largest / total,
        "filled_ratio": cloud_px / filled if filled > 0 else 0.0,
        "compactness": (4 * np.pi * cloud_px) / perimeter**2 if perimeter > 0 else 0.0,
    }


@torch.no_grad()
def collect(checkpoint: Path, split: str, pixel_threshold: float = 0.5) -> pd.DataFrame:
    model, task = load_from_checkpoint(checkpoint)
    if task != "segmentation":
        raise SystemExit("U-Net checkpoint'i gerekli")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    ds = CloudSegDataset(config.PATCH_INDEX, split, augment=False)
    loader = torch.utils.data.DataLoader(ds, batch_size=16, num_workers=0)

    rows = []
    for x, y in loader:
        probs = torch.sigmoid(model(x.to(device)).float()).cpu().numpy()
        masks = (probs >= pixel_threshold).astype(np.uint8)[:, 0]
        truth = y.numpy()[:, 0]
        for m, t in zip(masks, truth):
            feats = shape_features(m)
            feats["true_fraction"] = float(t.mean())
            rows.append(feats)
    return pd.DataFrame(rows)


def evaluate(decision: np.ndarray, truth: np.ndarray) -> dict:
    precision, recall, f1, _ = precision_recall_fscore_support(
        truth, decision, average="binary", zero_division=0
    )
    clear = truth == 0
    return {"accuracy": round(float((decision == truth).mean()), 4),
            "precision": round(float(precision), 4),
            "recall": round(float(recall), 4),
            "f1": round(float(f1), 4),
            "yanlis_eleme_%": round(100 * float(decision[clear].mean()), 2)}


def tune_on(scores: np.ndarray, truth: np.ndarray, min_precision: float = 0.99) -> float:
    """Verilen alt kume uzerinde precision kisitli esik secimi."""
    precisions, recalls, thresholds = precision_recall_curve(truth, scores)
    precisions, recalls = precisions[:-1], recalls[:-1]
    feasible = precisions >= min_precision
    if feasible.any():
        idx = int(np.argmax(np.where(feasible, recalls, -1.0)))
    else:
        f1s = 2 * precisions * recalls / np.clip(precisions + recalls, 1e-9, None)
        idx = int(np.argmax(f1s))
    return float(thresholds[idx])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--unet", type=Path, default=config.CHECKPOINTS / "unet050_best.pt")
    p.add_argument("--cloud-threshold", type=float, default=config.CLOUD_PIXEL_THRESHOLD)
    p.add_argument("--ambiguous-low", type=float, default=0.02)
    p.add_argument("--ambiguous-high", type=float, default=0.98)
    p.add_argument("--tag", default="unet050")
    args = p.parse_args()

    print("val ve test bolumlerinde maske uretiliyor...")
    val = collect(args.unet, "val")
    test = collect(args.unet, "test")
    print(f"val {len(val)} kare, test {len(test)} kare\n")

    for df in (val, test):
        df["label"] = (df["true_fraction"] >= args.cloud_threshold).astype(int)

    y_val, y_test = val["label"].to_numpy(), test["label"].to_numpy()
    results = {}

    # --- Referans: ham bulut orani, sabit esik ---
    results["1. bulut orani, sabit esik (%30)"] = evaluate(
        (test["cloud_fraction"] >= args.cloud_threshold).astype(int), y_test)

    # --- Esik val uzerinde ayarlanir (tum kareler) ---
    t_all = tune_on(val["cloud_fraction"].to_numpy(), y_val)
    results[f"2. bulut orani, val'de ayarli esik ({t_all:.3f})"] = evaluate(
        (test["cloud_fraction"] >= t_all).astype(int), y_test)

    # --- HIPOTEZ B: esik yalnizca belirsiz bantta ayarlanir ---
    amb = ((val["true_fraction"] > args.ambiguous_low) &
           (val["true_fraction"] < args.ambiguous_high)).to_numpy()
    print(f"belirsiz bant: val'in {amb.sum()}/{len(val)} karesi "
          f"(%{100 * amb.mean():.1f})")
    if amb.sum() > 30 and len(set(y_val[amb].tolist())) > 1:
        t_amb = tune_on(val["cloud_fraction"].to_numpy()[amb], y_val[amb])
        results[f"3. esik BELIRSIZ bantta ayarli ({t_amb:.3f})"] = evaluate(
            (test["cloud_fraction"] >= t_amb).astype(int), y_test)
    else:
        print("  belirsiz bantta yeterli ornek yok, atlandi")

    # --- HIPOTEZ A: sekil oznitelikleri ile lojistik regresyon ---
    FEATURES = ["cloud_fraction", "solidity", "n_components", "largest_fraction",
                "filled_ratio", "compactness"]
    Xv, Xt = val[FEATURES].to_numpy(), test[FEATURES].to_numpy()

    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf.fit(Xv, y_val)
    scores_val = clf.predict_proba(Xv)[:, 1]
    scores_test = clf.predict_proba(Xt)[:, 1]
    t_shape = tune_on(scores_val, y_val)
    results["4. sekil oznitelikleri (lojistik regresyon)"] = evaluate(
        (scores_test >= t_shape).astype(int), y_test)

    # Sadece bulut orani ile ayni modeli kur: sekil gercekten katki veriyor mu?
    clf_base = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf_base.fit(val[["cloud_fraction"]].to_numpy(), y_val)
    sb_val = clf_base.predict_proba(val[["cloud_fraction"]].to_numpy())[:, 1]
    sb_test = clf_base.predict_proba(test[["cloud_fraction"]].to_numpy())[:, 1]
    t_base = tune_on(sb_val, y_val)
    results["5. yalnizca bulut orani (kontrol)"] = evaluate(
        (sb_test >= t_base).astype(int), y_test)

    df_res = pd.DataFrame(results).T
    print("\n" + "=" * 92)
    print(df_res.to_string())

    print("\n--- lojistik regresyon katsayilari (standartlastirilmamis) ---")
    for name, coef in zip(FEATURES, clf.coef_[0]):
        print(f"  {name:<20} {coef:+.4f}")

    out = config.REPORTS / f"{args.tag}_decision_layer.csv"
    df_res.to_csv(out)
    print(f"\ntablo: {out}")

    shape_f1 = results["4. sekil oznitelikleri (lojistik regresyon)"]["f1"]
    base_f1 = results["5. yalnizca bulut orani (kontrol)"]["f1"]
    shape_p = results["4. sekil oznitelikleri (lojistik regresyon)"]["precision"]
    base_p = results["5. yalnizca bulut orani (kontrol)"]["precision"]
    print(f"\nSEKIL OZNITELIKLERININ KATKISI: F1 {shape_f1 - base_f1:+.4f}, "
          f"precision {shape_p - base_p:+.4f}")


if __name__ == "__main__":
    main()
