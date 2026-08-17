"""Harici veri setinde degerlendirme (SPARCS / Landsat 8).

Egitimde gorulmemis, FARKLI BIR UYDUDAN gelen veride genelleme olcer.

Kullanim:
    # Siniflandirici
    python -m src.evaluate_external --checkpoint outputs/checkpoints/c6_tuned_best.pt --tag c6_tuned
    python -m src.evaluate_external --onnx outputs/onnx/c6_tuned_int8.onnx --tag c6_tuned

    # U-Net (maskeden karar uretilerek ayni kirilimlar)
    python -m src.evaluate_external --task segmentation \
        --checkpoint outputs/checkpoints/unet_t64_best.pt --tag unet_t64

Rapor icin uc kirilim uretir:
  1. Genel metrikler (S2CMC test sonuclariyla karsilastirilir)
  2. Kar/buz agirlikli karelerde metrikler - en zor karisiklik
  3. Veri indirme kazanci (farkli sensorde de gecerli mi?)

Segmentasyon gorevinde ayrica piksel bazli IoU/Dice hesaplanir: SPARCS piksel
maskeleri de sakladigi icin (prepare_sparcs.py masks/ altina yazar) tek kosuda
hem karar kalitesi hem segmentasyon kalitesi olculebilir.
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
        raise SystemExit(f"bu checkpoint '{task}' gorevine ait; U-Net icin "
                         f"--task segmentation kullanin")

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


def predict_seg(checkpoint: Path, index: pd.DataFrame,
                pixel_threshold: float = 0.5) -> tuple:
    """U-Net ile karar skoru uretir: her kare icin BULUT PIKSEL ORANI.

    Bu oran, siniflandiricinin dondurdugu olasilikla ayni role sahiptir - esikle
    karsilastirilip ikili karara cevrilir. Boylece score(), downlink_analysis()
    ve kar/buz kirilimi hic degismeden calisir ve iki model AYNI olcutlerle
    karsilastirilabilir. Fark: esik bir olasilik degil, bulut pikseli oranidir
    (config.CLOUD_PIXEL_THRESHOLD) - yani unet.mask_to_decision ile ayni mantik.

    Maske sutunu varsa piksel bazli sayimlar da toplanir; ayni kosuda hem karar
    kalitesi hem segmentasyon kalitesi (IoU/Dice) olculur.
    """
    from src.export import load_from_checkpoint
    from src.losses import segmentation_metrics

    model, task = load_from_checkpoint(checkpoint)
    if task != "segmentation":
        raise SystemExit(f"bu checkpoint '{task}' gorevine ait, U-Net bekleniyordu")

    has_masks = "mask_path" in index.columns
    fractions, counts = [], []

    with torch.no_grad():
        for row in index.itertuples():
            x = torch.from_numpy(np.load(row.path).astype(np.float32)).unsqueeze(0)
            logits = model(x)
            mask = (torch.sigmoid(logits) >= pixel_threshold).float()
            fractions.append(float(mask.mean().item()))

            if has_masks:
                target = torch.from_numpy(
                    np.load(row.mask_path).astype(np.float32)
                ).unsqueeze(0).unsqueeze(0)
                counts.append(
                    segmentation_metrics(logits, target, pixel_threshold)["_counts"]
                )

    return np.array(fractions), (counts if has_masks else None)


def predict_seg_onnx(onnx_path: Path, index: pd.DataFrame,
                     pixel_threshold: float = 0.5) -> tuple:
    """predict_seg'in ONNX karsiligi.

    DIKKAT: ONNX U-Net STATIK sekille ihrac edilmistir (INT8 kuantizasyon
    gerekliligi). SPARCS kareleri modelin egitildigi boyutla ayni hazirlanmis
    olmalidir, orn. v3.2 icin:
        python -m src.prepare_sparcs --zip <...> --patch-size 64
    """
    session = make_session(onnx_path)
    name = session.get_inputs()[0].name

    has_masks = "mask_path" in index.columns
    fractions, counts = [], []

    for row in index.itertuples():
        x = np.load(row.path).astype(np.float32)[None]
        logits = session.run(None, {name: x})[0]
        mask = (1.0 / (1.0 + np.exp(-logits)) >= pixel_threshold).astype(np.float32)
        fractions.append(float(mask.mean()))

        if has_masks:
            target = np.load(row.mask_path).astype(np.float32).reshape(mask.shape)
            counts.append((
                float((mask * target).sum()),
                float((mask * (1 - target)).sum()),
                float(((1 - mask) * target).sum()),
                float(((1 - mask) * (1 - target)).sum()),
            ))

    return np.array(fractions), (counts if has_masks else None)


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
    p.add_argument("--task", default="classification",
                   choices=["classification", "segmentation"])
    p.add_argument("--threshold", type=float, default=None,
                   help="karar esigi; segmentasyonda bulut pikseli orani esigi "
                        "(varsayilan config.CLOUD_PIXEL_THRESHOLD)")
    p.add_argument("--pixel-threshold", type=float, default=0.5,
                   help="segmentasyon: maskeyi ikili yapan piksel esigi "
                        "(yayimlanan IoU bu esikte olculmustur)")
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
        if args.task == "segmentation":
            # U-Net egitim ozetinde karar esigi YOKTUR: maskeden goruntu
            # kararina gecis esigi bir egitim ciktisi degil, politika
            # parametresidir (bkz. unet.mask_to_decision).
            threshold = config.CLOUD_PIXEL_THRESHOLD
            print(f"karar esigi = bulut pikseli orani {threshold:.4f} "
                  f"(config.CLOUD_PIXEL_THRESHOLD), piksel esigi "
                  f"{args.pixel_threshold:.2f}")
        else:
            summary = config.REPORTS / f"{args.tag}_train_summary.json"
            if not summary.exists():
                raise SystemExit(f"{summary} yok - --threshold verin")
            threshold = json.loads(summary.read_text(encoding="utf-8"))["threshold"]
            print(f"karar esigi (S2CMC egitiminden): {threshold:.4f}")
    else:
        print(f"karar esigi (elle verildi): {threshold:.4f}")

    index = pd.read_csv(args.index)
    targets = index["label"].to_numpy()
    print(f"{len(index)} kare, {index['scene'].nunique()} sahne, "
          f"%{100 * targets.mean():.1f} bulutlu\n")

    model_name = str(args.checkpoint or args.onnx)
    pixel_counts = None

    if args.task == "segmentation":
        if args.checkpoint:
            probs, pixel_counts = predict_seg(args.checkpoint, index, args.pixel_threshold)
        else:
            probs, pixel_counts = predict_seg_onnx(args.onnx, index, args.pixel_threshold)
    else:
        probs = (predict_torch(args.checkpoint, index) if args.checkpoint
                 else predict_onnx(args.onnx, index))

    results = {"model": model_name, "task": args.task,
               "index": str(args.index), "threshold": threshold}
    if args.task == "segmentation":
        results["pixel_threshold"] = args.pixel_threshold

    print("=== Genel ===")
    results["overall"] = score(probs, targets, threshold)
    for k, v in results["overall"].items():
        print(f"  {k:<12} {v if isinstance(v, int) else f'{v:.4f}'}")

    if pixel_counts:
        from src.losses import accumulate_metrics

        # Piksel bazli metrikler kare boyutundan bagimsizdir (ayni pikseller
        # degerlendirilir), bu yuzden S2CMC IoU'suyla dogrudan karsilastirilir.
        print("\n=== Piksel bazli segmentasyon (harici sensor) ===")
        results["pixel"] = accumulate_metrics(pixel_counts)
        for k, v in results["pixel"].items():
            print(f"  {k:<16} {v:.4f}")

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
