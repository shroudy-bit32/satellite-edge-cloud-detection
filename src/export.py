"""PyTorch -> ONNX -> INT8 kuantizasyon (Gunler 13-16).

Kullanim:
    python -m src.export --checkpoint outputs/checkpoints/baseline_best.pt

Neden STATIK kuantizasyon:
  quantize_dynamic sadece MatMul/Gemm katmanlarini INT8'e cevirir. MobileNet
  agirligin neredeyse tamamini Conv katmanlarinda tasir, dolayisiyla dinamik
  kuantizasyon ne boyut ne hiz kazanci verir. Statik kuantizasyon Conv'lari da
  kapsar ama kalibrasyon verisi ister - asagidaki CalibrationDataReader bunu saglar.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from onnxruntime.quantization import (
    CalibrationDataReader,
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quantize_static,
)
from onnxruntime.quantization.shape_inference import quant_pre_process

from src import config
from src.dataset import CloudPatchDataset, CloudSegDataset
from src.model import build_model
from src.unet import build_unet


def load_from_checkpoint(checkpoint: Path):
    """Checkpoint'ten dogru mimariyi kurar (siniflandirici veya U-Net)."""
    ckpt = torch.load(checkpoint, map_location="cpu")
    task = ckpt.get("task", "classification")
    factory = build_unet if task == "segmentation" else build_model
    model = factory(ckpt.get("model_name", config.MODEL_NAME),
                    in_channels=ckpt.get("in_channels", config.IN_CHANNELS),
                    pretrained=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, task


class PatchCalibrationReader(CalibrationDataReader):
    """Kalibrasyon icin val bolumunden ornek akisi.

    Ornekler egitim dagilimini temsil etmeli: hem bulutlu hem temiz kareler
    karisik gelsin, yoksa aktivasyon araliklari yanlis olcuklenir.
    """

    def __init__(self, input_name: str, num_samples: int = 200, task: str = "classification"):
        ds_cls = CloudSegDataset if task == "segmentation" else CloudPatchDataset
        ds = ds_cls(config.PATCH_INDEX, "val", augment=False)
        rng = np.random.default_rng(config.SEED)
        idx = rng.permutation(len(ds))[:num_samples]
        self.input_name = input_name
        self.samples = iter([ds[int(i)][0].unsqueeze(0).numpy() for i in idx])

    def get_next(self):
        item = next(self.samples, None)
        return None if item is None else {self.input_name: item}


def export_onnx(checkpoint: Path, out_path: Path, opset: int = 18, dynamic_batch: bool = False) -> Path:
    """ONNX'e cikar.

    Varsayilan STATIK sekil (batch=1, sabit HxW). Nedeni:
      - Uyduda cikarim zaten kare kare, batch=1. Deployment gercekligi bu.
      - Dinamik eksenlerle onnxruntime'in sekil cikarimi tamamlanamiyor ve
        statik kuantizasyon "Incomplete symbolic shape inference" ile patliyor.
    Toplu degerlendirme icin dynamic_batch=True kullanilabilir ama o model
    kuantize edilemez.
    """
    model, task = load_from_checkpoint(checkpoint)
    output_name = "mask_logits" if task == "segmentation" else "logit"

    dummy = torch.randn(1, config.IN_CHANNELS, config.PATCH_SIZE, config.PATCH_SIZE)
    torch.onnx.export(
        model, dummy, str(out_path),
        input_names=["input"], output_names=[output_name],
        dynamic_axes={"input": {0: "batch"}, output_name: {0: "batch"}} if dynamic_batch else None,
        opset_version=opset,
    )

    # Torch'un dynamo exporter'i agirliklari yan dosyaya (.onnx.data) yazabiliyor.
    # Bu iki soruna yol acar: model tek basina tasinamaz ve dosya boyutu
    # olcumu yanilticidir. Tek dosyaya geri katliyoruz.
    inline_external_data(out_path)

    print(f"FP32 ONNX  {out_path}  ({out_path.stat().st_size / 1024**2:.2f} MB)")
    return out_path


def inline_external_data(onnx_path: Path) -> None:
    """Yan .data dosyasindaki agirliklari .onnx icine geri gomer."""
    sidecar = onnx_path.with_suffix(onnx_path.suffix + ".data")
    if not sidecar.exists():
        return
    import onnx

    model = onnx.load(str(onnx_path), load_external_data=True)
    onnx.save_model(model, str(onnx_path), save_as_external_data=False)
    sidecar.unlink()


def quantize_int8(fp32_path: Path, int8_path: Path, num_calib: int = 200,
                  task: str = "classification", calibrate_method: str = "percentile") -> Path:
    """Statik INT8 kuantizasyon.

    calibrate_method:
      minmax     - aktivasyon araliginin uc degerlerini kullanir; tek bir aykiri
                   piksel (ornegin kar/buz parlamasi) tum olcegi bozabilir
      percentile - araligi %99.999 yuzdelige gore keser, aykiri degerlere
                   dayanikli. MobileNet gibi hassas modellerde INT8 dogruluk
                   kaybini belirgin azaltir; VARSAYILAN
      entropy    - KL diverjansini minimize eder, en yavas kalibrasyon
    """
    import onnxruntime as ort

    # Kuantizasyon oncesi sekil cikarimi + optimizasyon; atlanirsa bazi
    # Conv katmanlari kuantize edilemez ve sessizce FP32 kalir.
    prep_path = fp32_path.with_name(fp32_path.stem + "_prep.onnx")
    quant_pre_process(str(fp32_path), str(prep_path), skip_symbolic_shape=False)

    input_name = ort.InferenceSession(
        str(prep_path), providers=["CPUExecutionProvider"]
    ).get_inputs()[0].name

    methods = {"minmax": CalibrationMethod.MinMax,
               "percentile": CalibrationMethod.Percentile,
               "entropy": CalibrationMethod.Entropy}

    quantize_static(
        str(prep_path), str(int8_path),
        calibration_data_reader=PatchCalibrationReader(input_name, num_calib, task),
        quant_format=QuantFormat.QDQ,
        # Aktivasyonlar QUInt8: x86'da isaretli int8'in yerel destegi yok ve
        # ek donusum gerektiriyor. Olculdu: s8 26.1 ms -> u8 10.8 ms (2.4x),
        # dogruluk farki ihmal edilebilir.
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        # per_channel ZORUNLU. BN katlamasi sonrasi agirliklarin kanal bazli
        # araligi mertebelerce degisiyor; tensor basina tek olcek modeli
        # cokertiyor (olculdu: F1 0.928 -> 0.325).
        per_channel=True,
        calibrate_method=methods[calibrate_method],
    )
    prep_path.unlink(missing_ok=True)
    print(f"INT8 ONNX  {int8_path}  ({int8_path.stat().st_size / 1024**2:.2f} MB)")
    return int8_path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--tag", default=None, help="cikti dosya adi oneki (varsayilan: checkpoint adi)")
    p.add_argument("--calib-samples", type=int, default=200)
    p.add_argument("--skip-int8", action="store_true")
    p.add_argument("--dynamic-batch", action="store_true",
                   help="dinamik batch ekseni (kuantize EDILEMEZ)")
    p.add_argument("--opset", type=int, default=18)
    p.add_argument("--calibrate-method", default="percentile",
                   choices=["minmax", "percentile", "entropy"])
    args = p.parse_args()

    tag = args.tag or args.checkpoint.stem.replace("_best", "")
    fp32 = config.ONNX_DIR / f"{tag}_fp32.onnx"
    int8 = config.ONNX_DIR / f"{tag}_int8.onnx"

    task = torch.load(args.checkpoint, map_location="cpu").get("task", "classification")
    print(f"gorev: {task}")

    export_onnx(args.checkpoint, fp32, opset=args.opset, dynamic_batch=args.dynamic_batch)
    if not args.skip_int8:
        if args.dynamic_batch:
            raise SystemExit("dinamik batch ile statik kuantizasyon calismaz; --skip-int8 kullan")
        quantize_int8(fp32, int8, args.calib_samples, task, args.calibrate_method)
        ratio = fp32.stat().st_size / int8.stat().st_size
        print(f"\nboyut kazanci  {ratio:.2f}x")


if __name__ == "__main__":
    main()
