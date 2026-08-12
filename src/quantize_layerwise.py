"""Conv katmanlarini KADEMELI kuantize ederek bozulmanin kaynagini bul.

Op tipi bazli ikiye bolme (quantize_bisect.py) sorunun Conv katmanlarinda
oldugunu gosterdi. Bu betik daha da daraltir: ilk k Conv'u kuantize edip
gerisini FP32 birakir, k'yi artirir.

Iki olasi sonuc, iki farkli cozum:
  - Belirli bir k'de ANI cokus  -> tek bir sorunlu katman var, o katman
                                   FP32 birakilarak sorun cozulur
  - Kademeli bozulma            -> hata katmanlar boyunca birikiyor; PTQ bu
                                   mimaride yetersiz, QAT veya farkli omurga
                                   gerekir

Kullanim:
    python -m src.quantize_layerwise --checkpoint outputs/checkpoints/baseline_best.pt
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import onnx
import pandas as pd
from onnxruntime.quantization import CalibrationMethod, QuantFormat, QuantType, quantize_static
from onnxruntime.quantization.shape_inference import quant_pre_process

from src import config
from src.dataset import CloudPatchDataset
from src.export import PatchCalibrationReader, export_onnx
from src.quantize_analysis import evaluate_session, make_session


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--tag", default="baseline")
    p.add_argument("--split", default="val")
    p.add_argument("--subset", type=int, default=200)
    p.add_argument("--calib-samples", type=int, default=100)
    p.add_argument("--steps", type=int, nargs="*", default=None,
                   help="kac Conv kuantize edilsin (varsayilan otomatik)")
    args = p.parse_args()

    threshold = json.loads(
        (config.REPORTS / f"{args.tag}_train_summary.json").read_text(encoding="utf-8")
    )["threshold"]

    dataset = CloudPatchDataset(config.PATCH_INDEX, args.split, augment=False)
    if args.subset and args.subset < len(dataset):
        dataset.df = dataset.df.groupby("label", group_keys=False).sample(
            n=args.subset // 2, random_state=config.SEED
        ).reset_index(drop=True)

    work = config.ONNX_DIR / "quant_layerwise"
    work.mkdir(exist_ok=True)
    fp32 = work / "fp32.onnx"
    export_onnx(args.checkpoint, fp32)
    prep = work / "prep.onnx"
    quant_pre_process(str(fp32), str(prep), skip_symbolic_shape=False)

    graph = onnx.load(str(prep)).graph
    # Topolojik sirada Conv dugum adlari (ONNX grafigi zaten topolojik sirali)
    conv_nodes = [n.name for n in graph.node if n.op_type == "Conv"]
    print(f"{len(conv_nodes)} Conv dugumu bulundu")

    steps = args.steps or [0, 1, 2, 4, 8, 16, 24, 32, 40, 48, len(conv_nodes)]
    steps = sorted({min(s, len(conv_nodes)) for s in steps})

    input_name = make_session(prep).get_inputs()[0].name

    session = make_session(fp32)
    ref = evaluate_session(session, dataset, threshold)
    ref_probs = ref.pop("_probs")
    print(f"FP32 referans: F1 {ref['f1']:.4f}  AUC {ref['roc_auc']:.4f}\n")

    rows = []
    for k in steps:
        if k == 0:
            rows.append({"kuantize_conv": 0, "boyut_MB": round(fp32.stat().st_size / 1024**2, 2),
                         **ref, "sapma": 0.0})
            continue

        out = work / f"first{k}.onnx"
        exclude = conv_nodes[k:]  # ilk k disindaki Conv'lari haric tut
        quantize_static(
            str(prep), str(out),
            calibration_data_reader=PatchCalibrationReader(input_name, args.calib_samples),
            quant_format=QuantFormat.QDQ,
            activation_type=QuantType.QUInt8,
            weight_type=QuantType.QInt8,
            per_channel=True,
            calibrate_method=CalibrationMethod.Percentile,
            op_types_to_quantize=["Conv"],
            nodes_to_exclude=exclude,
        )

        res = evaluate_session(make_session(out), dataset, threshold)
        probs = res.pop("_probs")
        drift = float(np.abs(probs - ref_probs).mean())
        rows.append({"kuantize_conv": k, "boyut_MB": round(out.stat().st_size / 1024**2, 2),
                     **res, "sapma": round(drift, 5)})
        print(f"ilk {k:>2} Conv kuantize -> F1 {res['f1']:.4f}  AUC {res['roc_auc']:.4f}  "
              f"sapma {drift:.5f}  boyut {rows[-1]['boyut_MB']} MB")
        out.unlink(missing_ok=True)

    df = pd.DataFrame(rows)
    print("\n" + "=" * 90)
    print(df.to_string(index=False))

    # Ani cokus var mi? Ardisik adimlar arasindaki en buyuk F1 dususu
    df_sorted = df.sort_values("kuantize_conv")
    drops = df_sorted["f1"].diff()
    if drops.notna().any():
        worst = drops.idxmin()
        print(f"\nen buyuk tek adimlik dusus: {drops.min():.4f} "
              f"({df_sorted.loc[worst, 'kuantize_conv']}. adimda)")
        total_drop = df_sorted["f1"].iloc[0] - df_sorted["f1"].iloc[-1]
        share = abs(drops.min()) / total_drop if total_drop else 0
        print(f"toplam dususun %{100 * share:.1f}'i bu tek adimda gerceklesiyor")
        print("YORUM:", "tek sorunlu katman var, karma hassasiyet cozer"
              if share > 0.5 else "hata katmanlar boyunca birikiyor, PTQ yetersiz")

    out_csv = config.REPORTS / f"{args.tag}_quantization_layerwise.csv"
    df.to_csv(out_csv, index=False)
    print(f"\ntablo: {out_csv}")

    shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
