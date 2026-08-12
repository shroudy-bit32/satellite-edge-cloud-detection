"""Cikarim sirasindaki TEPE BELLEK kullanimi.

Diskteki model boyutu, uydu uzerindeki bellek ihtiyacinin yalnizca bir parcasi.
Cikarim sirasinda ara katman aktivasyonlari icin de bellek gerekir ve 256x256
girdide bu, agirliklardan buyuk olabilir. PDF'in "uydu uzerindeki sinirli
bellegi temsil eder" kisiti asil bunu kastediyor.

Her model AYRI BIR SURECTE olculur; aksi halde onceki modelin ayirdigi bellek
sonrakinin olcumune karisir.

Kullanim:
    python -m src.memory_profile
    python -m src.memory_profile --sizes 128 256 512
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

from src import config

WORKER = """
import json, sys
import numpy as np, onnxruntime as ort, psutil

model_path, size, channels, runs = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
proc = psutil.Process()

def rss_mb():
    return proc.memory_info().rss / 1024**2

def peak_mb():
    info = proc.memory_info()
    return getattr(info, "peak_wset", info.rss) / 1024**2

base = rss_mb()

opts = ort.SessionOptions()
opts.intra_op_num_threads = 1
opts.inter_op_num_threads = 1
sess = ort.InferenceSession(model_path, opts, providers=["CPUExecutionProvider"])
after_load = rss_mb()

name = sess.get_inputs()[0].name
x = np.random.rand(1, channels, size, size).astype(np.float32)
for _ in range(runs):
    sess.run(None, {name: x})

print(json.dumps({
    "baseline_MB": round(base, 2),
    "after_load_MB": round(after_load, 2),
    "weights_MB": round(after_load - base, 2),
    "peak_MB": round(peak_mb(), 2),
    "inference_overhead_MB": round(peak_mb() - after_load, 2),
}))
"""


def measure(model_path: Path, size: int, channels: int, runs: int = 20) -> dict | None:
    script = config.PROJECT_ROOT / "_mem_worker.py"
    script.write_text(WORKER, encoding="utf-8")
    try:
        r = subprocess.run([sys.executable, str(script), str(model_path), str(size),
                            str(channels), str(runs)],
                           capture_output=True, text=True, encoding="utf-8",
                           errors="replace", cwd=config.PROJECT_ROOT)
        if r.returncode != 0:
            print(f"  hata: {(r.stderr or '').strip().splitlines()[-1:]}")
            return None
        return json.loads(r.stdout.strip().splitlines()[-1])
    finally:
        script.unlink(missing_ok=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="*", default=None,
                   help="ONNX yollari; varsayilan: v2 surumundeki dort model")
    p.add_argument("--sizes", type=int, nargs="*", default=[128, 256, 512])
    p.add_argument("--channels", type=int, default=config.IN_CHANNELS)
    p.add_argument("--runs", type=int, default=20)
    args = p.parse_args()

    if args.models:
        paths = [Path(m) for m in args.models]
        # Ayni dosya adi birden fazla yerde olabilir (releases/v1/unet_int8.onnx,
        # releases/v2/unet_int8.onnx ...). Cakisma varsa ust dizini de ada kat.
        stems = [p.stem for p in paths]
        models = [((f"{p.parent.name}/{p.stem}" if stems.count(p.stem) > 1 else p.stem), p)
                  for p in paths]
    else:
        rel = config.PROJECT_ROOT / "releases" / "v2"
        models = [(n, rel / f"{n}.onnx") for n in
                  ["classifier_fp32", "classifier_int8", "unet_fp32", "unet_int8"]]

    rows = []
    for name, path in models:
        if not path.exists():
            print(f"atlandi (yok): {path}")
            continue
        disk = path.stat().st_size / 1024**2
        for size in args.sizes:
            print(f"olculuyor: {name} @ {size}x{size} ...", flush=True)
            m = measure(path, size, args.channels, args.runs)
            if m is None:
                continue
            rows.append({"model": name, "girdi": f"{size}x{size}",
                         "disk_MB": round(disk, 2),
                         "agirlik_MB": m["weights_MB"],
                         "cikarim_ek_MB": m["inference_overhead_MB"],
                         "tepe_bellek_MB": m["peak_MB"],
                         "python_taban_MB": m["baseline_MB"]})

    if not rows:
        raise SystemExit("olcum yapilamadi")

    df = pd.DataFrame(rows)
    print("\n" + "=" * 96)
    print(df.to_string(index=False))

    print("\nOKUMA NOTU:")
    print("  disk_MB        : ONNX dosyasinin boyutu")
    print("  agirlik_MB     : oturum acilinca artan bellek (agirliklar + ORT yapilari)")
    print("  cikarim_ek_MB  : cikarim sirasinda EKSTRA ayrilan bellek (aktivasyonlar)")
    print("  tepe_bellek_MB : surecin toplam tepe kullanimi (Python yorumlayici dahil)")
    print("  python_taban_MB: yalnizca yorumlayici + kutuphaneler; uydu yaziliminda olmaz")

    print("\nGERCEK MODEL MALIYETI = agirlik_MB + cikarim_ek_MB")
    for name in df["model"].unique():
        sub = df[df["model"] == name]
        worst = sub.loc[sub["girdi"] == "256x256"]
        if len(worst):
            w = worst.iloc[0]
            print(f"  {name:<18} {w['agirlik_MB'] + w['cikarim_ek_MB']:>7.2f} MB "
                  f"(256x256 girdide)")

    out = config.REPORTS / "memory_profile.csv"
    df.to_csv(out, index=False)
    print(f"\ntablo: {out}")


if __name__ == "__main__":
    main()
