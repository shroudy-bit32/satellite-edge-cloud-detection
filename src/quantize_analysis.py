"""INT8 kuantizasyon yapilandirma taramasi (Gunler 13-16).

Sorun: MobileNetV3'un varsayilan statik kuantizasyonu dogrulugu cokertiyor
(F1 0.936 -> 0.257). Bu betik olasi sebepleri TEK TEK izole eder, tahmin etmez.

MobileNetV3'un bilinen kuantizasyon zorluklari:
  - hard-swish aktivasyonlari: genis ve asimetrik deger araligi
  - squeeze-excite bloklari: carpma islemi iki kuantizasyon hatasini carpiyor
  - depthwise konvolusyonlar: kanal basina cok farkli olcekler

Kullanim:
    python -m src.quantize_analysis --checkpoint outputs/checkpoints/baseline_best.pt
    python -m src.quantize_analysis --checkpoint ... --split test --full
"""

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import pandas as pd
from onnxruntime.quantization import (
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quantize_dynamic,
    quantize_static,
)
from onnxruntime.quantization.shape_inference import quant_pre_process
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

from src import config
from src.dataset import CloudPatchDataset
from src.export import PatchCalibrationReader, export_onnx

CALIB_METHODS = {
    "minmax": CalibrationMethod.MinMax,
    "percentile": CalibrationMethod.Percentile,
    "entropy": CalibrationMethod.Entropy,
}


def build_configs() -> list:
    """Her yapilandirma tek bir hipotezi test eder."""
    return [
        # Mevcut varsayilan - referans (bozuk oldugunu biliyoruz)
        dict(name="qdq_s8_perch_percentile", act=QuantType.QInt8, weight=QuantType.QInt8,
             per_channel=True, calib="percentile", reduce_range=False, fmt=QuantFormat.QDQ),

        # Hipotez: isaretli aktivasyon kuantizasyonu hard-swish'e uymuyor
        dict(name="qdq_u8_perch_percentile", act=QuantType.QUInt8, weight=QuantType.QInt8,
             per_channel=True, calib="percentile", reduce_range=False, fmt=QuantFormat.QDQ),

        # Hipotez: percentile kirpma cok agresif, gercek araligi kaciriyor
        dict(name="qdq_u8_perch_minmax", act=QuantType.QUInt8, weight=QuantType.QInt8,
             per_channel=True, calib="minmax", reduce_range=False, fmt=QuantFormat.QDQ),
        dict(name="qdq_u8_perch_entropy", act=QuantType.QUInt8, weight=QuantType.QInt8,
             per_channel=True, calib="entropy", reduce_range=False, fmt=QuantFormat.QDQ),

        # Hipotez: kanal basina agirlik kuantizasyonu depthwise'da sorun cikariyor
        dict(name="qdq_u8_pertensor_percentile", act=QuantType.QUInt8, weight=QuantType.QInt8,
             per_channel=False, calib="percentile", reduce_range=False, fmt=QuantFormat.QDQ),

        # Hipotez: 8 bitin tamami tasma yaratiyor, 7 bite dusurmek stabilize eder
        dict(name="qdq_u8_perch_percentile_rr", act=QuantType.QUInt8, weight=QuantType.QInt8,
             per_channel=True, calib="percentile", reduce_range=True, fmt=QuantFormat.QDQ),

        # Hipotez: QDQ yerine QOperator formati farkli davraniyor
        dict(name="qop_u8_perch_percentile", act=QuantType.QUInt8, weight=QuantType.QInt8,
             per_channel=True, calib="percentile", reduce_range=False, fmt=QuantFormat.QOperator),
    ]


def evaluate_session(session: ort.InferenceSession, dataset, threshold: float) -> dict:
    name = session.get_inputs()[0].name
    probs, targets = [], []
    for i in range(len(dataset)):
        x, y = dataset[i]
        logit = float(session.run(None, {name: x.unsqueeze(0).numpy()})[0].squeeze())
        probs.append(1.0 / (1.0 + np.exp(-logit)))
        targets.append(float(y))

    probs, targets = np.array(probs), np.array(targets)
    preds = (probs >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        targets, preds, average="binary", zero_division=0
    )
    return {
        "accuracy": float((preds == targets).mean()),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": float(roc_auc_score(targets, probs)) if len(set(targets.tolist())) > 1 else float("nan"),
        "_probs": probs,
    }


def measure_latency(session: ort.InferenceSession, runs: int = 30) -> float:
    name = session.get_inputs()[0].name
    x = np.random.rand(1, config.IN_CHANNELS, config.PATCH_SIZE, config.PATCH_SIZE).astype(np.float32)
    for _ in range(5):
        session.run(None, {name: x})
    t0 = time.perf_counter()
    for _ in range(runs):
        session.run(None, {name: x})
    return (time.perf_counter() - t0) / runs * 1000


def make_session(path: Path) -> ort.InferenceSession:
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = config.BENCHMARK_THREADS
    opts.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])


def count_quantized_nodes(path: Path) -> tuple:
    model = onnx.load(str(path))
    ops = [n.op_type for n in model.graph.node]
    quantized = sum(1 for o in ops if o in ("QuantizeLinear", "DequantizeLinear",
                                            "QLinearConv", "QLinearMatMul"))
    return len(ops), quantized


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--tag", default="baseline")
    p.add_argument("--split", default="val", help="degerlendirme bolumu")
    p.add_argument("--subset", type=int, default=300,
                   help="hizli tarama icin kare sayisi (0 = tumu)")
    p.add_argument("--calib-samples", type=int, default=200)
    p.add_argument("--threshold", type=float, default=None)
    args = p.parse_args()

    threshold = args.threshold
    if threshold is None:
        summary = config.REPORTS / f"{args.tag}_train_summary.json"
        threshold = json.loads(summary.read_text(encoding="utf-8"))["threshold"]
    print(f"karar esigi: {threshold:.4f}")

    dataset = CloudPatchDataset(config.PATCH_INDEX, args.split, augment=False)
    if args.subset and args.subset < len(dataset):
        # Sinif dengesini koruyarak alt ornekle
        df = dataset.df.groupby("label", group_keys=False).sample(
            n=args.subset // 2, random_state=config.SEED
        )
        dataset.df = df.reset_index(drop=True)
    print(f"degerlendirme: {args.split} bolumunden {len(dataset)} kare "
          f"(%{100 * dataset.df['label'].mean():.1f} bulutlu)\n")

    work = config.ONNX_DIR / "quant_sweep"
    work.mkdir(exist_ok=True)
    fp32 = work / "fp32.onnx"
    export_onnx(args.checkpoint, fp32)

    prep = work / "fp32_prep.onnx"
    quant_pre_process(str(fp32), str(prep), skip_symbolic_shape=False)

    input_name = make_session(prep).get_inputs()[0].name

    rows = []

    # Referans: FP32
    session = make_session(fp32)
    ref = evaluate_session(session, dataset, threshold)
    ref_probs = ref.pop("_probs")
    total_nodes, _ = count_quantized_nodes(fp32)
    rows.append({"yapilandirma": "FP32 (referans)", "boyut_MB": round(fp32.stat().st_size / 1024**2, 2),
                 **ref, "latency_ms": round(measure_latency(session), 2),
                 "sapma": 0.0, "kuantize_dugum": 0, "toplam_dugum": total_nodes})
    print(f"FP32 referans: F1 {ref['f1']:.4f}  AUC {ref['roc_auc']:.4f}\n")

    # Dinamik kuantizasyon (Conv'lari kapsamaz, karsilastirma icin)
    dyn = work / "dynamic.onnx"
    quantize_dynamic(str(prep), str(dyn), weight_type=QuantType.QInt8)
    session = make_session(dyn)
    res = evaluate_session(session, dataset, threshold)
    probs = res.pop("_probs")
    total, q = count_quantized_nodes(dyn)
    rows.append({"yapilandirma": "dinamik (sadece MatMul)",
                 "boyut_MB": round(dyn.stat().st_size / 1024**2, 2), **res,
                 "latency_ms": round(measure_latency(session), 2),
                 "sapma": round(float(np.abs(probs - ref_probs).mean()), 5),
                 "kuantize_dugum": q, "toplam_dugum": total})
    print(f"dinamik: F1 {res['f1']:.4f}")

    for cfg in build_configs():
        out = work / f"{cfg['name']}.onnx"
        try:
            quantize_static(
                str(prep), str(out),
                calibration_data_reader=PatchCalibrationReader(input_name, args.calib_samples),
                quant_format=cfg["fmt"],
                activation_type=cfg["act"],
                weight_type=cfg["weight"],
                per_channel=cfg["per_channel"],
                calibrate_method=CALIB_METHODS[cfg["calib"]],
                reduce_range=cfg["reduce_range"],
            )
        except Exception as exc:
            print(f"{cfg['name']}: BASARISIZ - {type(exc).__name__}: {exc}")
            rows.append({"yapilandirma": cfg["name"], "boyut_MB": float("nan"),
                         "f1": float("nan"), "hata": str(exc)[:80]})
            continue

        session = make_session(out)
        res = evaluate_session(session, dataset, threshold)
        probs = res.pop("_probs")
        total, q = count_quantized_nodes(out)
        rows.append({"yapilandirma": cfg["name"],
                     "boyut_MB": round(out.stat().st_size / 1024**2, 2), **res,
                     "latency_ms": round(measure_latency(session), 2),
                     "sapma": round(float(np.abs(probs - ref_probs).mean()), 5),
                     "kuantize_dugum": q, "toplam_dugum": total})
        print(f"{cfg['name']}: F1 {res['f1']:.4f}  AUC {res['roc_auc']:.4f}  "
              f"sapma {rows[-1]['sapma']:.5f}")

    df = pd.DataFrame(rows)
    print("\n" + "=" * 100)
    print(df.to_string(index=False))

    out_csv = config.REPORTS / f"{args.tag}_quantization_sweep.csv"
    df.to_csv(out_csv, index=False)
    print(f"\ntablo: {out_csv}")

    valid = df[df["f1"].notna() & (df["yapilandirma"] != "FP32 (referans)")]
    if len(valid):
        best = valid.loc[valid["f1"].idxmax()]
        print(f"\nen iyi kuantize yapilandirma: {best['yapilandirma']}")
        print(f"  F1 {best['f1']:.4f} (FP32: {ref['f1']:.4f}, kayip "
              f"{100 * (ref['f1'] - best['f1']) / ref['f1']:.2f}%)")
        print(f"  boyut {best['boyut_MB']} MB, sure {best['latency_ms']} ms")

    shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
