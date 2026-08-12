"""Mevcut model surumunu dondurup releases/ altina kaydeder.

Amac: sonraki deneyler mevcut calisan surumu bozmasin. Her surum kendi
checkpoint'leri, ONNX dosyalari, config kopyasi ve olculmus sayilariyla
birlikte saklanir.

Kullanim:
    python -m src.snapshot_release --version v1 --note "MobileNetV2, ilk calisan INT8"
"""

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

from src import config

def artifacts_for(clf: str, unet: str) -> list:
    """(kaynak, surum icindeki ad) ciftleri."""
    return [
        (config.CHECKPOINTS / f"{clf}_best.pt", "classifier.pt"),
        (config.CHECKPOINTS / f"{unet}_best.pt", "unet.pt"),
        (config.ONNX_DIR / f"{clf}_fp32.onnx", "classifier_fp32.onnx"),
        (config.ONNX_DIR / f"{clf}_int8.onnx", "classifier_int8.onnx"),
        (config.ONNX_DIR / f"{unet}_fp32.onnx", "unet_fp32.onnx"),
        (config.ONNX_DIR / f"{unet}_int8.onnx", "unet_int8.onnx"),
        (config.REPORTS / f"{clf}_train_summary.json", "classifier_train_summary.json"),
        (config.REPORTS / f"{unet}_train_summary.json", "unet_train_summary.json"),
        (config.REPORTS / f"{clf}_benchmark.csv", "classifier_benchmark.csv"),
        (config.REPORTS / f"{unet}_benchmark.csv", "unet_benchmark.csv"),
        (config.REPORTS / f"{clf}_downlink_analysis.json", "downlink_analysis.json"),
        (config.REPORTS / f"{clf}_external_sparcs.json", "external_sparcs.json"),
        (config.REPORTS / f"{clf}_tag_analysis.csv", "tag_analysis.csv"),
        (config.REPORTS / f"{unet}_threshold_analysis.csv", "threshold_analysis.csv"),
        (config.REPORTS / f"{unet}_decision_layer.csv", "decision_layer.csv"),
        (config.REPORTS / f"{clf}_operating_points.json", "operating_points.json"),
        (config.REPORTS / "memory_profile.csv", "memory_profile.csv"),
        (config.REPORTS / "tiling_tradeoff.csv", "tiling_tradeoff.csv"),
        (config.REPORTS / "tiling_latency_clean.csv", "tiling_latency_clean.csv"),
        (config.REPORTS / "TEKNIK_RAPOR.md", "TEKNIK_RAPOR.md"),
        (config.PROJECT_ROOT / "src" / "config.py", "config_snapshot.py"),
    ]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--version", required=True)
    p.add_argument("--note", default="")
    p.add_argument("--stamp", default=None)
    p.add_argument("--classifier-tag", default="v2")
    p.add_argument("--unet-tag", default="unet")
    args = p.parse_args()

    out = config.PROJECT_ROOT / "releases" / args.version
    if out.exists():
        raise SystemExit(f"{out} zaten var - farkli bir surum adi kullanin")
    out.mkdir(parents=True)

    copied, missing = [], []
    for src, name in artifacts_for(args.classifier_tag, args.unet_tag):
        if src.exists():
            shutil.copy2(src, out / name)
            copied.append((name, round(src.stat().st_size / 1024**2, 3)))
        else:
            missing.append(str(src))

    # Olculmus sayilari manifest'e goem
    def read(path, key=None):
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        return data.get(key) if key else data

    clf = read(config.REPORTS / f"{args.classifier_tag}_train_summary.json")
    unet = read(config.REPORTS / f"{args.unet_tag}_train_summary.json")
    downlink = read(config.REPORTS / f"{args.classifier_tag}_downlink_analysis.json")
    sparcs = read(config.REPORTS / f"{args.classifier_tag}_external_sparcs.json")

    manifest = {
        "version": args.version,
        "created": args.stamp or datetime.now().strftime("%Y-%m-%d %H:%M"),
        "note": args.note,
        "source_tags": {"classifier": args.classifier_tag, "unet": args.unet_tag},
        "hyperparameters": (clf or {}).get("hyperparameters"),
        "classifier": {
            "backbone": clf.get("model") if clf else None,
            "threshold": clf.get("threshold") if clf else None,
            "test": clf.get("test") if clf else None,
        },
        "unet": {
            "encoder": unet.get("encoder") if unet else None,
            "test": unet.get("test") if unet else None,
        },
        "downlink": downlink.get("models") if downlink else None,
        "external_sparcs": {k: v for k, v in (sparcs or {}).items()
                            if k in ("overall", "snowy", "non_snowy")},
        "data": {
            "bands": config.BANDS,
            "patch_size": config.PATCH_SIZE,
            "cloud_pixel_threshold": config.CLOUD_PIXEL_THRESHOLD,
            "reflectance_clip": config.REFLECTANCE_CLIP,
        },
        "files": [{"name": n, "MB": s} for n, s in copied],
    }
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False),
                                       encoding="utf-8")

    lines = [f"# Surum {args.version}", "", args.note, "",
             f"Olusturulma: {manifest['created']}", "", "## Olculmus sonuclar", ""]
    if clf:
        t = clf["test"]
        lines += [f"**Siniflandirici** ({clf.get('model')}), esik {clf['threshold']:.4f}",
                  f"- test: accuracy {t['accuracy']:.4f}, precision {t['precision']:.4f}, "
                  f"recall {t['recall']:.4f}, F1 {t['f1']:.4f}", ""]
    if unet:
        t = unet["test"]
        lines += [f"**U-Net** ({unet.get('encoder')})",
                  f"- test: IoU {t['iou']:.4f}, Dice {t['dice']:.4f}, "
                  f"goruntu-seviyesi dogruluk {t['image_level_accuracy']:.4f}", ""]
    if downlink and "models" in downlink:
        lines += ["**Veri indirme kazanci:**", ""]
        for model, a in downlink["models"].items():
            lines += [f"- {model}: %{a['veri_indirme_azalmasi_%']} azalma, "
                      f"%{a['kaybedilen_kullanilabilir_veri_%']} kullanilabilir veri kaybi"]
        lines += [""]
    lines += ["## Dosyalar", ""] + [f"- `{n}` ({s} MB)" for n, s in copied]
    if missing:
        lines += ["", "## Eksik", ""] + [f"- {m}" for m in missing]

    (out / "README.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"surum kaydedildi: {out}")
    print(f"  {len(copied)} dosya kopyalandi")
    if missing:
        print(f"  {len(missing)} dosya bulunamadi:")
        for m in missing:
            print(f"    {m}")


if __name__ == "__main__":
    main()
