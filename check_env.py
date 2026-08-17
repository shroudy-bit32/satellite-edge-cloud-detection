"""Ortam doğrulama: GPU, kütüphaneler ve CPU çıkarım hattı çalışıyor mu?

Çalıştır:  python check_env.py
"""

import importlib
import platform
import sys

PACKAGES = [
    "numpy", "scipy", "pandas", "matplotlib", "tqdm", "yaml",
    "rasterio", "tifffile", "PIL", "cv2",
    "timm", "torchmetrics", "sklearn", "albumentations",
    "onnx", "onnxruntime",
    "gradio", "pytorch_grad_cam",
]


def check_packages():
    missing = []
    for name in PACKAGES:
        try:
            mod = importlib.import_module(name)
            version = getattr(mod, "__version__", "?")
            print(f"  [OK]     {name:<18} {version}")
        except ImportError:
            missing.append(name)
            print(f"  [EKSIK]  {name}")
    return missing


def check_torch():
    import torch

    print(f"  torch            {torch.__version__}")
    print(f"  CUDA derlemesi   {torch.version.cuda}")
    available = torch.cuda.is_available()
    print(f"  cuda.is_available {available}")
    if not available:
        print("  !! GPU gorunmuyor - CPU-only wheel kurulmus olabilir.")
        return False

    print(f"  GPU              {torch.cuda.get_device_name(0)}")
    total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"  VRAM             {total_gb:.1f} GB")

    # Gercekten hesap yapabiliyor mu?
    a = torch.randn(2048, 2048, device="cuda")
    (a @ a).sum().item()
    torch.cuda.synchronize()
    print("  matmul testi     OK")
    return True


def check_onnx_cpu_pipeline():
    """MobileNet -> ONNX -> onnxruntime CPU. Projenin omurgasi bu hat."""
    import numpy as np
    import onnxruntime as ort
    import timm
    import torch

    model = timm.create_model("mobilenetv3_small_100", pretrained=False, num_classes=2)
    model.eval()
    dummy = torch.randn(1, 3, 224, 224)

    path = "_env_check.onnx"
    torch.onnx.export(
        model, dummy, path,
        input_names=["input"], output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        opset_version=17,
    )
    print("  ONNX export      OK")

    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    out = sess.run(None, {"input": dummy.numpy()})[0]
    print(f"  ORT CPU cikarim  OK  cikti sekli={out.shape}")

    # Kuantizasyon API'si erisilebilir mi? (13-16. gunler icin)
    from onnxruntime.quantization import quantize_dynamic  # noqa: F401
    print("  quantize API     OK")

    import os
    os.remove(path)
    _ = np  # kullanilmadi uyarisi olmasin


def main():
    print(f"Python  {sys.version.split()[0]}  ({platform.machine()})")
    print(f"Yorumlayici: {sys.executable}\n")

    print("Paketler:")
    missing = check_packages()

    print("\nPyTorch / GPU:")
    gpu_ok = check_torch()

    print("\nONNX hatti:")
    if missing:
        print("  atlandi (eksik paket var)")
    else:
        check_onnx_cpu_pipeline()

    print("\n" + "=" * 50)
    if missing:
        print(f"EKSIK PAKETLER: {', '.join(missing)}")
        print("  pip install -r requirements.txt")
    elif not gpu_ok:
        print("Paketler tamam ama GPU gorunmuyor.")
    else:
        print("Ortam hazir.")


if __name__ == "__main__":
    main()
