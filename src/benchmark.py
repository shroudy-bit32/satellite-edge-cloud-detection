"""Karsilastirma tablosu ve veri indirme kazanci analizi (Gunler 13-18).

Plandaki iki cikti da buradan uretilir:
  1. Dogruluk / boyut / CPU cikarim suresi karsilastirmasi (FP32 vs INT8)
  2. "Bu filtre uyduda calissaydi indirilen veri hacmi %X azalirdi" hesabi

Kullanim:
    python -m src.benchmark --tag baseline --threshold 0.62
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pandas as pd
import torch
from sklearn.metrics import precision_recall_fscore_support

from src import config
from src.dataset import CloudPatchDataset, CloudSegDataset
from src.losses import accumulate_metrics
from src.model import build_model


def model_disk_size_mb(path: Path) -> float:
    """Modelin diskteki gercek boyutu.

    ONNX agirliklari yan bir .data dosyasinda tutabilir; sadece .onnx dosyasina
    bakmak boyutu oldugundan cok kucuk gosterir.
    """
    total = path.stat().st_size
    sidecar = path.with_suffix(path.suffix + ".data")
    if sidecar.exists():
        total += sidecar.stat().st_size
    return total / 1024**2


def make_session(onnx_path: Path, threads: int = config.BENCHMARK_THREADS) -> ort.InferenceSession:
    """Sabit is parcacigi sayisiyla CPU oturumu.

    Thread sayisini sabitlemek sart: aksi halde olculen sure makinenin cekirdek
    sayisina gore degisir ve "standart CPU" hedefi anlamsizlasir.
    """
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    opts.inter_op_num_threads = 1
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    return ort.InferenceSession(str(onnx_path), opts, providers=["CPUExecutionProvider"])


def measure_latency(session: ort.InferenceSession) -> dict:
    """Tek goruntu (batch=1) cikarim suresi - uydudaki gercek senaryo."""
    name = session.get_inputs()[0].name
    x = np.random.rand(1, config.IN_CHANNELS, config.PATCH_SIZE, config.PATCH_SIZE).astype(np.float32)

    for _ in range(config.BENCHMARK_WARMUP):
        session.run(None, {name: x})

    times = []
    for _ in range(config.BENCHMARK_RUNS):
        t0 = time.perf_counter()
        session.run(None, {name: x})
        times.append((time.perf_counter() - t0) * 1000)

    times = np.array(times)
    return {
        "latency_mean_ms": float(times.mean()),
        "latency_p50_ms": float(np.percentile(times, 50)),
        "latency_p95_ms": float(np.percentile(times, 95)),
    }


def predict_onnx(session: ort.InferenceSession, split: str = "test") -> tuple:
    ds = CloudPatchDataset(config.PATCH_INDEX, split, augment=False)
    name = session.get_inputs()[0].name
    probs, targets = [], []
    for i in range(len(ds)):
        x, y = ds[i]
        logit = session.run(None, {name: x.unsqueeze(0).numpy()})[0]
        probs.append(1.0 / (1.0 + np.exp(-float(logit.squeeze()))))
        targets.append(float(y))
    return np.array(probs), np.array(targets)


@torch.no_grad()
def predict_torch(checkpoint: Path, split: str = "test") -> tuple:
    ckpt = torch.load(checkpoint, map_location="cpu")
    model = build_model(ckpt.get("model_name", config.MODEL_NAME),
                        in_channels=ckpt.get("in_channels", config.IN_CHANNELS),
                        pretrained=False)
    model.load_state_dict(ckpt["model"])
    model.eval()

    ds = CloudPatchDataset(config.PATCH_INDEX, split, augment=False)
    loader = torch.utils.data.DataLoader(ds, batch_size=64, num_workers=0)
    probs, targets = [], []
    for x, y in loader:
        probs.append(torch.sigmoid(model(x).squeeze(1)).numpy())
        targets.append(y.numpy())
    return np.concatenate(probs), np.concatenate(targets)


def score(probs, targets, threshold: float) -> dict:
    preds = (probs >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        targets, preds, average="binary", zero_division=0
    )
    return {
        "accuracy": float((preds == targets).mean()),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def downlink_analysis(probs, targets, threshold: float) -> dict:
    """Veri indirme kazanci.

    Model uyduda calisip "bulutlu" dediklerini elerse:
      kazanc  = elenen goruntu orani
      kayip   = yanlislikla elenen KULLANILABILIR goruntu orani (yanlis pozitif)

    Kayip metrigi kritik: bilimsel veri kaybi geri donusu olmayan bir maliyet.
    """
    preds = (probs >= threshold).astype(int)
    total = len(targets)

    filtered = int(preds.sum())
    tp = int(((preds == 1) & (targets == 1)).sum())   # dogru elenen bulutlu
    fp = int(((preds == 1) & (targets == 0)).sum())   # YANLIS elenen kullanilabilir
    fn = int(((preds == 0) & (targets == 1)).sum())   # kacan bulutlu (bosuna indirilir)

    cloudy_total = int((targets == 1).sum())
    usable_total = int((targets == 0).sum())

    return {
        "test_goruntu_sayisi": total,
        "gercek_bulutlu_orani_%": round(100 * cloudy_total / total, 2),
        "elenen_goruntu_orani_%": round(100 * filtered / total, 2),
        "veri_indirme_azalmasi_%": round(100 * filtered / total, 2),
        "dogru_elenen_bulutlu": tp,
        "yanlis_elenen_kullanilabilir": fp,
        "kaybedilen_kullanilabilir_veri_%": round(100 * fp / max(usable_total, 1), 3),
        "kacan_bulutlu_(bosuna_indirilen)": fn,
        "teorik_ust_sinir_%": round(100 * cloudy_total / total, 2),
    }


def predict_onnx_seg(session: ort.InferenceSession, split: str = "test", threshold: float = 0.5):
    """Segmentasyon: piksel sayimlarini toplar (tum maskeleri bellekte tutmadan)."""
    ds = CloudSegDataset(config.PATCH_INDEX, split, augment=False)
    name = session.get_inputs()[0].name
    counts = []
    for i in range(len(ds)):
        x, y = ds[i]
        logits = session.run(None, {name: x.unsqueeze(0).numpy()})[0]
        preds = (1.0 / (1.0 + np.exp(-logits)) >= threshold).astype(np.float32).squeeze(0)
        target = y.numpy()
        counts.append((
            float((preds * target).sum()),
            float((preds * (1 - target)).sum()),
            float(((1 - preds) * target).sum()),
            float(((1 - preds) * (1 - target)).sum()),
        ))
    return accumulate_metrics(counts)


def benchmark_segmentation(tag: str, threshold: float = 0.5):
    fp32 = config.ONNX_DIR / f"{tag}_fp32.onnx"
    int8 = config.ONNX_DIR / f"{tag}_int8.onnx"

    rows = []
    for label, path in (("ONNX FP32", fp32), ("ONNX INT8", int8)):
        if not path.exists():
            print(f"atlandi: {path} yok")
            continue
        session = make_session(path)
        metrics = predict_onnx_seg(session, threshold=threshold)
        rows.append({"model": label,
                     "boyut_MB": round(model_disk_size_mb(path), 2),
                     "iou": round(metrics["iou"], 4),
                     "dice": round(metrics["dice"], 4),
                     "precision": round(metrics["precision"], 4),
                     "recall": round(metrics["recall"], 4),
                     **measure_latency(session)})

    df = pd.DataFrame(rows)
    df["hedef_boyut_OK"] = df["boyut_MB"] <= config.TARGET_MODEL_SIZE_MB
    df["hedef_sure_OK"] = df["latency_mean_ms"] <= config.TARGET_LATENCY_MS

    print("\n=== Segmentasyon karsilastirma tablosu ===")
    print(df.to_string(index=False))

    csv_path = config.REPORTS / f"{tag}_benchmark.csv"
    df.to_csv(csv_path, index=False)
    print(f"\ntablo: {csv_path}")
    return df


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", default="baseline")
    p.add_argument("--task", default="classification",
                   choices=["classification", "segmentation"])
    p.add_argument("--threshold", type=float, default=None,
                   help="varsayilan: train_summary.json icindeki ayarlanmis esik")
    args = p.parse_args()

    if args.task == "segmentation":
        benchmark_segmentation(args.tag, args.threshold or 0.5)
        return

    summary_path = config.REPORTS / f"{args.tag}_train_summary.json"
    threshold = args.threshold
    if threshold is None:
        if not summary_path.exists():
            raise SystemExit(f"{summary_path} yok - once egitimi calistir veya --threshold ver")
        threshold = json.loads(summary_path.read_text(encoding="utf-8"))["threshold"]
    print(f"karar esigi: {threshold:.3f}\n")

    checkpoint = config.CHECKPOINTS / f"{args.tag}_best.pt"
    fp32 = config.ONNX_DIR / f"{args.tag}_fp32.onnx"
    int8 = config.ONNX_DIR / f"{args.tag}_int8.onnx"

    rows = []
    probs_by_model = {}

    # PyTorch FP32 (referans dogruluk)
    probs, targets = predict_torch(checkpoint)
    probs_by_model["PyTorch FP32"] = probs
    rows.append({"model": "PyTorch FP32",
                 "boyut_MB": round(checkpoint.stat().st_size / 1024**2, 2),
                 **score(probs, targets, threshold),
                 "latency_mean_ms": float("nan"), "latency_p50_ms": float("nan"),
                 "latency_p95_ms": float("nan")})

    # ONNX FP32 / INT8
    reference_probs = None
    for label, path in (("ONNX FP32", fp32), ("ONNX INT8", int8)):
        if not path.exists():
            print(f"atlandi: {path} yok")
            continue
        session = make_session(path)
        probs, targets = predict_onnx(session)
        if label == "ONNX FP32":
            reference_probs = probs
        probs_by_model[label] = probs
        rows.append({"model": label,
                     "boyut_MB": round(model_disk_size_mb(path), 2),
                     **score(probs, targets, threshold),
                     **measure_latency(session)})

        if label == "ONNX INT8" and reference_probs is not None:
            drift = float(np.abs(probs - reference_probs).mean())
            print(f"INT8 vs FP32 ortalama olasilik sapmasi: {drift:.5f}")

    df = pd.DataFrame(rows)
    df["hedef_boyut_OK"] = df["boyut_MB"] <= config.TARGET_MODEL_SIZE_MB
    df["hedef_sure_OK"] = df["latency_mean_ms"] <= config.TARGET_LATENCY_MS

    print("\n=== Karsilastirma tablosu ===")
    print(df.to_string(index=False))

    csv_path = config.REPORTS / f"{args.tag}_benchmark.csv"
    df.to_csv(csv_path, index=False)

    # Fayda analizi HER model icin ayri hesaplanir ve hangi modele ait oldugu
    # acikca yazilir. Tek bir modelin (orn. INT8) sonucunu etiketsiz basmak
    # yanilticiydi: kuantizasyon bozulursa tablo sebepsiz sacma gorunuyor.
    analyses = {}
    for label, model_probs in probs_by_model.items():
        analyses[label] = downlink_analysis(model_probs, targets, threshold)
        print(f"\n=== Veri indirme kazanci - {label} ===")
        for k, v in analyses[label].items():
            print(f"  {k:<38} {v}")

    json_path = config.REPORTS / f"{args.tag}_downlink_analysis.json"
    json_path.write_text(json.dumps({"threshold": threshold, "models": analyses}, indent=2),
                         encoding="utf-8")
    print(f"\ntablo: {csv_path}\nanaliz: {json_path}")


if __name__ == "__main__":
    main()
