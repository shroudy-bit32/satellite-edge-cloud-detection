"""INT8 bozulmasinin KAYNAGINI bul: op tipi bazli ikiye bolme.

Yapilandirma taramasi (quantize_analysis.py) hicbir parametrenin sorunu
cozmedigini gosterdi. Ayrica AUC 0.35 (rastgeleden kotu) kademeli hassasiyet
kaybi degil, sistematik bozulma isareti.

Bu betik hangi OP TIPININ bozulmaya yol actigini izole eder: her seferinde
farkli bir op alt kumesini kuantize eder, gerisini FP32 birakir.

Kullanim:
    python -m src.quantize_bisect --checkpoint outputs/checkpoints/baseline_best.pt
"""

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
import onnx
import pandas as pd
from onnxruntime.quantization import CalibrationMethod, QuantFormat, QuantType, quantize_static
from onnxruntime.quantization.shape_inference import quant_pre_process

from src import config
from src.dataset import CloudPatchDataset
from src.export import PatchCalibrationReader, export_onnx
from src.quantize_analysis import evaluate_session, make_session, measure_latency

# Her deney bir alt kumeyi kuantize eder; gerisi FP32 kalir.
OP_SUBSETS = [
    ("sadece Conv", ["Conv"]),
    ("Conv + Gemm", ["Conv", "Gemm"]),
    ("Conv + Gemm + Add", ["Conv", "Gemm", "Add"]),
    ("Conv + Gemm + Mul", ["Conv", "Gemm", "Mul"]),          # SE bloklari Mul kullanir
    ("Conv + Gemm + GlobalAveragePool", ["Conv", "Gemm", "GlobalAveragePool"]),
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--tag", default="baseline")
    p.add_argument("--split", default="val")
    p.add_argument("--subset", type=int, default=300)
    p.add_argument("--calib-samples", type=int, default=200)
    args = p.parse_args()

    summary = config.REPORTS / f"{args.tag}_train_summary.json"
    threshold = json.loads(summary.read_text(encoding="utf-8"))["threshold"]

    dataset = CloudPatchDataset(config.PATCH_INDEX, args.split, augment=False)
    if args.subset and args.subset < len(dataset):
        dataset.df = dataset.df.groupby("label", group_keys=False).sample(
            n=args.subset // 2, random_state=config.SEED
        ).reset_index(drop=True)
    print(f"{len(dataset)} kare ile degerlendirme, esik {threshold:.4f}\n")

    work = config.ONNX_DIR / "quant_bisect"
    work.mkdir(exist_ok=True)
    fp32 = work / "fp32.onnx"
    export_onnx(args.checkpoint, fp32)
    prep = work / "prep.onnx"
    quant_pre_process(str(fp32), str(prep), skip_symbolic_shape=False)

    # Grafikteki op dagilimi: neyi kuantize edebiliyoruz, onu bilelim
    graph = onnx.load(str(prep)).graph
    op_counts = Counter(n.op_type for n in graph.node)
    print("ONNX grafigindeki op tipleri:")
    for op, cnt in op_counts.most_common():
        print(f"  {op:<24} {cnt}")
    print()

    input_name = make_session(prep).get_inputs()[0].name

    session = make_session(fp32)
    ref = evaluate_session(session, dataset, threshold)
    ref_probs = ref.pop("_probs")
    rows = [{"kuantize edilen": "hicbiri (FP32)",
             "boyut_MB": round(fp32.stat().st_size / 1024**2, 2), **ref,
             "latency_ms": round(measure_latency(session), 2), "sapma": 0.0}]
    print(f"FP32 referans: F1 {ref['f1']:.4f}  AUC {ref['roc_auc']:.4f}\n")

    for name, op_types in OP_SUBSETS:
        out = work / f"{name.replace(' ', '_').replace('+', '')}.onnx"
        try:
            quantize_static(
                str(prep), str(out),
                calibration_data_reader=PatchCalibrationReader(input_name, args.calib_samples),
                quant_format=QuantFormat.QDQ,
                activation_type=QuantType.QUInt8,   # taramada 3.7x hizli cikti
                weight_type=QuantType.QInt8,
                per_channel=True,
                calibrate_method=CalibrationMethod.Percentile,
                op_types_to_quantize=op_types,
            )
        except Exception as exc:
            print(f"{name}: BASARISIZ - {type(exc).__name__}: {exc}")
            continue

        session = make_session(out)
        res = evaluate_session(session, dataset, threshold)
        probs = res.pop("_probs")
        drift = float(np.abs(probs - ref_probs).mean())
        rows.append({"kuantize edilen": name,
                     "boyut_MB": round(out.stat().st_size / 1024**2, 2), **res,
                     "latency_ms": round(measure_latency(session), 2),
                     "sapma": round(drift, 5)})
        print(f"{name:<32} F1 {res['f1']:.4f}  AUC {res['roc_auc']:.4f}  "
              f"sapma {drift:.5f}  boyut {rows[-1]['boyut_MB']} MB")

    df = pd.DataFrame(rows)
    print("\n" + "=" * 100)
    print(df.to_string(index=False))
    out_csv = config.REPORTS / f"{args.tag}_quantization_bisect.csv"
    df.to_csv(out_csv, index=False)
    print(f"\ntablo: {out_csv}")

    shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
