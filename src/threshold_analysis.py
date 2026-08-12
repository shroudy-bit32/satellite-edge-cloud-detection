"""Bulut esiginin (CLOUD_PIXEL_THRESHOLD) sonuclara etkisi.

Neden gerekli: %30 degeri proje icinde secildi, dayanagi yoktu. Bu deger hem
etiketleme kuralini hem karar kuralini belirledigi icin TUM sonuclari kaydirir.
Referans misyon CloudScout ayni karari %70 esikle veriyor.

Bu betik esigi degistirdiginde ne olduğunu olcer. Onemli ayrinti: esik
degisince ETIKETLER de degisir, yani her esik icin gorev yeniden tanimlanir.
Modeli yeniden egitmeden, egitilmis modelin bulut orani tahminini kullanarak
her esikte kazanc/kayip hesaplanir.

Kullanim:
    python -m src.threshold_analysis --unet outputs/checkpoints/unet_best.pt
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src import config
from src.dataset import CloudSegDataset
from src.export import load_from_checkpoint


@torch.no_grad()
def predicted_cloud_fractions(checkpoint: Path, split: str = "test",
                              pixel_threshold: float = 0.5, batch_size: int = 16):
    """U-Net ile her kare icin tahmini ve gercek bulut piksel oranini dondurur.

    U-Net kullaniyoruz cunku bulut ORANI uretebiliyor; ikili siniflandirici
    yalnizca sabit bir esige gore karar veriyor ve esik taramasi yapilamiyor.
    """
    model, task = load_from_checkpoint(checkpoint)
    if task != "segmentation":
        raise SystemExit("bu analiz U-Net checkpoint'i gerektirir")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    ds = CloudSegDataset(config.PATCH_INDEX, split, augment=False)
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, num_workers=0)

    pred, true = [], []
    for x, y in loader:
        x = x.to(device)
        probs = torch.sigmoid(model(x).float())
        pred.append(((probs >= pixel_threshold).float().mean(dim=(-2, -1))).squeeze(1).cpu().numpy())
        true.append(y.mean(dim=(-2, -1)).squeeze(1).numpy())
    return np.concatenate(pred), np.concatenate(true)


def analyse(pred_fraction: np.ndarray, true_fraction: np.ndarray, thresholds) -> pd.DataFrame:
    rows = []
    n = len(true_fraction)
    for t in thresholds:
        # Esik hem gercek etiketi hem karari tanimlar
        truth = true_fraction >= t
        decision = pred_fraction >= t

        tp = int((decision & truth).sum())
        fp = int((decision & ~truth).sum())
        fn = int((~decision & truth).sum())
        usable = int((~truth).sum())

        filtered = tp + fp

        # ESAS METRIK: elenen goruntulerde kaybedilen TEMIZ PIKSEL alani.
        #
        # Etiket bazli "kaybedilen kullanilabilir goruntu" orani esikler arasi
        # karsilastirma icin YANILTICIDIR: esik degisince "kullanilabilir"in
        # tanimi da degisir. %5 esiginde %5 bulutlu bir goruntu "bulutlu"
        # sayilip elenir ve bu "dogru karar" olarak kazanc yazilir - oysa o
        # goruntunun %95'i bilimsel olarak kullanilabilir veridir.
        #
        # Temiz piksel alani ise esikten bagimsiz, mutlak bir buyukluktur:
        # gercekte ne kadar kullanilabilir veri cope gitti?
        clear_area = 1.0 - true_fraction
        lost_clear = float(clear_area[decision].sum())
        total_clear = float(clear_area.sum())

        rows.append({
            "esik_%": round(100 * t),
            "gercekte_bulutlu_%": round(100 * truth.mean(), 2),
            "elenen_%": round(100 * filtered / n, 2),
            "kaybedilen_temiz_alan_%": round(100 * lost_clear / max(total_clear, 1e-9), 2),
            "kaybedilen_kullanilabilir_goruntu_%": round(100 * fp / max(usable, 1), 3),
            "kacan_bulutlu": fn,
            "precision": round(tp / max(tp + fp, 1), 4),
            "recall": round(tp / max(tp + fn, 1), 4),
        })
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--unet", type=Path, default=config.CHECKPOINTS / "unet_best.pt")
    p.add_argument("--split", default="test")
    p.add_argument("--tag", default="unet")
    p.add_argument("--thresholds", type=float, nargs="*",
                   default=[0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.70, 0.90])
    args = p.parse_args()

    pred, true = predicted_cloud_fractions(args.unet, args.split)
    print(f"{len(true)} kare, ortalama gercek bulut orani %{100 * true.mean():.1f}")
    print(f"tahmin-gercek bulut orani korelasyonu: {np.corrcoef(pred, true)[0, 1]:.4f}\n")

    df = analyse(pred, true, args.thresholds)
    print(df.to_string(index=False))

    out = config.REPORTS / f"{args.tag}_threshold_analysis.csv"
    df.to_csv(out, index=False)

    print("\nOKUMA NOTU: 'en iyi esik' diye tek bir cevap YOKTUR. Esik, bant")
    print("genisligi tasarrufu ile feda edilen temiz veri arasindaki degis-tokusu")
    print("belirleyen OPERASYONEL bir politika parametresidir. Asagidaki tablo")
    print("bu degis-tokusu gosterir; secim misyon onceligine gore yapilmalidir.\n")
    print(f"{'esik':>6} {'kazanilan bant genisligi':>26} {'feda edilen temiz veri':>24}")
    for _, r in df.iterrows():
        print(f"%{r['esik_%']:>5.0f} {r['elenen_%']:>25.2f}% {r['kaybedilen_temiz_alan_%']:>23.2f}%")

    print(f"\nproje varsayilani: %{100 * config.CLOUD_PIXEL_THRESHOLD:.0f}")
    print("CloudScout referansi: %70 (muhafazakar: yalnizca neredeyse tamamen kapali sahneler elenir)")
    print(f"\ntablo: {out}")

    (config.REPORTS / f"{args.tag}_threshold_analysis.json").write_text(
        json.dumps({"split": args.split, "rows": df.to_dict("records")}, indent=2),
        encoding="utf-8")


if __name__ == "__main__":
    main()
