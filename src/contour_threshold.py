"""Bulut esigini kontur geometrisinden turetmeyi dene (uc degerler haric).

Fikir: "bulutlu" tanimini ham piksel sayisi yerine maskenin KENDI KONTURU
uzerinden kur, ve esigi ararken uc degerleri (tamamen temiz / tamamen kapali
kareler) disla - cunku onlar dagilimi ezip yapiyi gizliyor.

Onceki analizde `cloud_fraction` dagilimina TUM kareler dahil edilerek
bakilmisti; karelerin %73'u uclarda oldugu icin ortadaki yapi gorunmuyordu.
Bu betik yalnizca kismi bulutlu kareleri alip GEOMETRIK olculere bakar.

Uc aday olcu:
  cloud_fraction  : bulut pikseli / toplam           (mevcut tanim)
  filled_fraction : delikleri doldurulmus alan / toplam
                    (gozenekli bulut ortusu, kapladigi bolgeyi gercekte
                     piksel sayisindan fazla engeller)
  hull_fraction   : konveks kabuk alani / toplam
                    (bulutun yayildigi bolgenin tamami)

Kullanim:
    python -m src.contour_threshold
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy import ndimage

from src import config


def mask_measures(mask: np.ndarray) -> dict:
    total = mask.size
    cloud_px = int(mask.sum())
    if cloud_px == 0:
        return {"cloud_fraction": 0.0, "filled_fraction": 0.0,
                "hull_fraction": 0.0, "solidity": 0.0}

    m8 = mask.astype(np.uint8)
    filled = int(ndimage.binary_fill_holes(m8.astype(bool)).sum())

    contours, _ = cv2.findContours(m8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hull_area = sum(cv2.contourArea(cv2.convexHull(c)) for c in contours)

    return {
        "cloud_fraction": cloud_px / total,
        "filled_fraction": filled / total,
        "hull_fraction": min(hull_area / total, 1.0),
        "solidity": min(cloud_px / hull_area, 1.0) if hull_area > 0 else 0.0,
    }


def valley_search(values: np.ndarray, bins: int = 25, lo: float = 0.02,
                  hi: float = 0.98) -> dict:
    """Dagilimda dogal bir kesme noktasi (vadi) var mi?"""
    sel = values[(values > lo) & (values < hi)]
    if len(sel) < 50:
        return {"n": len(sel), "flatness": float("nan"), "valley": float("nan")}

    hist, edges = np.histogram(sel, bins=bins, range=(lo, hi))
    centers = (edges[:-1] + edges[1:]) / 2
    return {
        "n": int(len(sel)),
        "flatness": float(hist.min() / max(hist.max(), 1)),
        "valley": float(centers[int(np.argmin(hist))]),
        "peak": float(centers[int(np.argmax(hist))]),
        "hist_min": int(hist.min()),
        "hist_max": int(hist.max()),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sample", type=int, default=3000)
    p.add_argument("--ambiguous-low", type=float, default=0.02)
    p.add_argument("--ambiguous-high", type=float, default=0.98)
    p.add_argument("--splits", nargs="*", default=["train", "val"],
                   help="test bolumu ANALIZE DAHIL EDILMEZ (sizinti olmasin)")
    args = p.parse_args()

    idx = pd.read_csv(config.PATCH_INDEX)
    idx = idx[idx["split"].isin(args.splits)]
    sample = idx.sample(n=min(args.sample, len(idx)), random_state=config.SEED)
    print(f"{len(sample)} kare ({', '.join(args.splits)} bolumlerinden)")
    print("test bolumu bilerek disarida: esik tanimi test verisinden turetilemez\n")

    rows = [mask_measures(np.load(path)) for path in sample["mask_path"]]
    df = pd.DataFrame(rows)

    # UC DEGERLERI DISLA - kullanicinin onerisi
    amb = df[(df["cloud_fraction"] > args.ambiguous_low) &
             (df["cloud_fraction"] < args.ambiguous_high)].copy()
    print(f"uc degerler cikarildi: {len(df)} -> {len(amb)} kare "
          f"(%{100 * len(amb) / len(df):.1f} kaldi)\n")

    print("=== Her olcunun belirsiz bant icindeki dagilimi ===")
    print(f"{'olcu':<18} {'ortalama':>9} {'medyan':>8} {'std':>8} "
          f"{'duzluk':>8} {'vadi':>7} {'tepe':>7}")
    print("-" * 72)

    results = {}
    for measure in ["cloud_fraction", "filled_fraction", "hull_fraction", "solidity"]:
        v = amb[measure].to_numpy()
        stats = valley_search(v, lo=args.ambiguous_low, hi=args.ambiguous_high)
        results[measure] = stats
        print(f"{measure:<18} {v.mean():>9.4f} {np.median(v):>8.4f} {v.std():>8.4f} "
              f"{stats['flatness']:>8.3f} {stats['valley']:>7.3f} {stats['peak']:>7.3f}")

    print("\nDUZLUK YORUMU: 0'a yakin = belirgin vadi (dogal esik VAR)")
    print("               1'e yakin = duz dagilim (dogal esik YOK)")

    best = min(results.items(), key=lambda kv: kv[1]["flatness"]
               if np.isfinite(kv[1]["flatness"]) else 9)
    print(f"\nen belirgin yapiya sahip olcu: {best[0]} (duzluk {best[1]['flatness']:.3f})")
    if best[1]["flatness"] < 0.25:
        print(f"  -> vadi %{100 * best[1]['valley']:.0f} civarinda; esik adayi olabilir")
    else:
        print("  -> hicbir olcude belirgin vadi yok; esik veriden turetilemiyor")

    # Olculer birbirinden ne kadar farkli karar veriyor?
    print(f"\n=== Etiketleme farki (esik %{100 * config.CLOUD_PIXEL_THRESHOLD:.0f}) ===")
    base = (df["cloud_fraction"] >= config.CLOUD_PIXEL_THRESHOLD).to_numpy()
    for measure in ["filled_fraction", "hull_fraction"]:
        alt = (df[measure] >= config.CLOUD_PIXEL_THRESHOLD).to_numpy()
        changed = int((alt != base).sum())
        newly = int((alt & ~base).sum())
        print(f"  {measure:<18} {changed:>5} kare farkli etiketlenir "
              f"(%{100 * changed / len(df):.2f}); {newly} tanesi YENIDEN bulutlu olur")

    print(f"\n=== Olculer arasi korelasyon (belirsiz bant) ===")
    print(amb[["cloud_fraction", "filled_fraction", "hull_fraction", "solidity"]]
          .corr().round(4).to_string())

    out = config.REPORTS / "contour_threshold_analysis.json"
    out.write_text(json.dumps({"ambiguous_band": [args.ambiguous_low, args.ambiguous_high],
                               "n_total": len(df), "n_ambiguous": len(amb),
                               "measures": results}, indent=2), encoding="utf-8")
    print(f"\nsonuc: {out}")


if __name__ == "__main__":
    main()
