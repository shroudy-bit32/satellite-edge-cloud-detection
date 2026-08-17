"""Demo arayuzu: goruntu verildiginde karar + dikkat/isi haritasi (Gunler 17-18).

Kullanim:
    python app.py                                  # web arayuzu
    python app.py --image data/patches/x.npy       # betik modu (arayuzsuz)
    python app.py --classifier outputs/checkpoints/baseline_best.pt \
                  --segmenter  outputs/checkpoints/unet_best.pt

Girdi olarak .npy patch veya ham .tif GeoTIFF kabul eder.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from src import config
from src.inference import (
    CloudClassifier,
    CloudSegmenter,
    overlay_heatmap,
    read_image,
    to_rgb_preview,
)

# Dagitilan surumun modelleri (config.CURRENT_RELEASE).
# Deney checkpoint'leri yerine dondurulmus surum kullanilir; boylece demo,
# raporlanan sayilarla ayni modeli gosterir.
DEFAULT_CLASSIFIER = config.CURRENT / "classifier.pt"
DEFAULT_SEGMENTER = config.CURRENT / "unet.pt"

# Veri seti indirilmeden arayuzu denemek icin uretilen sentetik ornek.
SAMPLE_PATH = config.OUTPUTS / "demo_sample.npy"


def make_sample_patch(path: Path = SAMPLE_PATH, size: int = 256) -> Path:
    """Sentetik ornek kare uretir (6 bant, parlak ve duz bir 'bulut' lekesi).

    DIKKAT: bu GERCEK Sentinel-2 verisi DEGILDIR. Modelin bu kare icin verdigi
    karar bilimsel olarak anlamsizdir; tek amaci veri seti indirilmeden arayuzun
    ve cikarim hattinin calistigini gosterebilmektir. Raporlanan tum sayilar
    data/patches altindaki gercek karelerden uretilmistir (bkz. README bolum 12).

    Sentetik kare smoke_test.py ile ayni mantikla kurulur: bulut pikselleri
    yuksek reflektans + dusuk doku, zemin dusuk reflektans + yuksek doku.
    """
    rng = np.random.default_rng(config.SEED)

    arr = rng.normal(0.25, 0.18, (config.IN_CHANNELS, size, size))

    cy, cx = size // 2 - size // 8, size // 2 + size // 10
    radius = size // 3
    yy, xx = np.ogrid[:size, :size]
    blob = (yy - cy) ** 2 + (xx - cx) ** 2 <= radius**2
    arr[:, blob] = rng.normal(0.75, 0.05, (config.IN_CHANNELS, int(blob.sum())))

    arr = np.clip(arr, 0, 1).astype(np.float32)

    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, arr)
    return path


def load_threshold(tag: str | None = None) -> float:
    """Karar esigini dagitilan surumden oku.

    Once surumun operating_points.json dosyasina bakilir (birden fazla calisma
    noktasi tanimlar); yoksa egitim ozetindeki esige, o da yoksa config'e duser.
    Esik kodda sabit degildir - surumle birlikte tasinan bir parametredir.
    """
    points = config.CURRENT / "operating_points.json"
    if points.exists():
        data = json.loads(points.read_text(encoding="utf-8"))["operating_points"]
        # Dagitim icin onerilen nokta: en yuksek precision kisitli olan
        for name in ("belirsiz bant (precision>=0.995)", "mevcut (tum val, precision>=0.99)"):
            if name in data:
                return float(data[name]["threshold"])
        return float(next(iter(data.values()))["threshold"])

    if tag:
        path = config.REPORTS / f"{tag}_train_summary.json"
        if path.exists():
            return float(json.loads(path.read_text(encoding="utf-8"))["threshold"])
    return config.DECISION_THRESHOLD


def analyze(image_path, classifier: CloudClassifier, segmenter: CloudSegmenter | None):
    """Tek goruntuyu isler; arayuz ve betik modu ayni yolu kullanir."""
    arr = read_image(image_path)
    rgb = to_rgb_preview(arr)

    result = classifier.predict(arr)
    cam = classifier.heatmap(arr)
    heatmap_img = overlay_heatmap(rgb, cam)

    mask_img, seg_info = None, None
    if segmenter is not None:
        seg = segmenter.predict(arr)
        # Maskeyi kirmizi kanalda vurgula
        mask_rgb = rgb.copy()
        cloud = seg["mask"] > 0.5
        mask_rgb[cloud] = (0.5 * mask_rgb[cloud] + 0.5 * np.array([255, 0, 0])).astype(np.uint8)
        mask_img = mask_rgb
        seg_info = seg

    return arr, rgb, result, heatmap_img, mask_img, seg_info


def format_report(result: dict, seg_info: dict | None, shape: tuple) -> str:
    lines = [
        f"### Karar: {result['decision']}",
        "",
        f"- Bulut olasiligi: **{result['probability']:.4f}**",
        f"- Karar esigi: {result['threshold']:.4f}",
        f"- Goruntu boyutu: {shape[1]}x{shape[2]}, {shape[0]} bant",
    ]
    if seg_info is not None:
        lines += [
            "",
            "### Segmentasyon (U-Net)",
            f"- Bulut piksel orani: **%{100 * seg_info['cloud_fraction']:.2f}**",
            f"- Maske esigi (%{100 * config.CLOUD_PIXEL_THRESHOLD:.0f}) uzerinden karar: "
            f"{'BULUTLU' if seg_info['is_cloudy'] else 'KULLANILABILIR'}",
        ]
    if result["is_cloudy"]:
        lines += ["", "> Bu goruntu uyduda elenir; yer istasyonuna indirilmez."]
    else:
        lines += ["", "> Bu goruntu indirme kuyruguna alinir."]
    return "\n".join(lines)


def run_cli(image_path: str, classifier, segmenter):
    arr, rgb, result, heatmap_img, mask_img, seg_info = analyze(image_path, classifier, segmenter)

    print(f"\ngoruntu: {image_path}")
    print(f"karar:   {result['decision']}")
    print(f"olasilik {result['probability']:.4f} (esik {result['threshold']:.4f})")
    if seg_info is not None:
        print(f"bulut piksel orani: %{100 * seg_info['cloud_fraction']:.2f}")

    # Gorseli diske yaz (arayuzsuz calisirken de ciktiyi gorebilmek icin)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = [(rgb, "RGB onizleme"), (heatmap_img, "Grad-CAM dikkat haritasi")]
    if mask_img is not None:
        panels.append((mask_img, "U-Net bulut maskesi"))

    fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 5))
    axes = np.atleast_1d(axes)
    for ax, (img, title) in zip(axes, panels):
        ax.imshow(img)
        ax.set_title(title)
        ax.axis("off")
    fig.suptitle(f"{result['decision']}  |  olasilik {result['probability']:.4f}", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.94))  # suptitle panel basliklarinin uzerine binmesin

    out = config.REPORTS / f"demo_{Path(image_path).stem}.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"gorsel: {out}")


def build_interface(classifier, segmenter):
    import gradio as gr

    def process(file_obj):
        if file_obj is None:
            return None, None, None, "Once bir goruntu yukleyin."
        try:
            arr, rgb, result, heatmap_img, mask_img, seg_info = analyze(
                file_obj.name, classifier, segmenter
            )
        except Exception as exc:  # arayuz cokmesin, hatayi kullaniciya goster
            return None, None, None, f"**Hata:** {exc}"
        return rgb, heatmap_img, mask_img, format_report(result, seg_info, arr.shape)

    with gr.Blocks(title="Uydu Ustu Bulut Filtresi") as demo:
        gr.Markdown(
            "# Uydu Ustu Bulut Filtresi\n"
            "Uctaki hafif model, bulutlu goruntuleri yer istasyonuna indirmeden eler. "
            "Bir Sentinel-2 karesi (.npy) veya GeoTIFF (.tif) yukleyin."
        )
        with gr.Row():
            with gr.Column(scale=1):
                file_input = gr.File(label="Goruntu", file_types=[".npy", ".tif", ".tiff"])
                run_btn = gr.Button("Analiz et", variant="primary")
                report = gr.Markdown()
            with gr.Column(scale=2):
                with gr.Row():
                    rgb_out = gr.Image(label="RGB onizleme")
                    cam_out = gr.Image(label="Grad-CAM dikkat haritasi")
                mask_out = gr.Image(label="U-Net bulut maskesi"
                                    if segmenter else "U-Net maskesi (model yuklenmedi)")

        # Elde gercek kare yoksa arayuz denenemez; sentetik ornek tek tikla
        # yuklenebilsin diye ornek olarak sunulur. Etiketi bilerek uyarici.
        if SAMPLE_PATH.exists():
            gr.Examples(
                examples=[[str(SAMPLE_PATH)]],
                inputs=file_input,
                label="Sentetik ornek (gercek uydu verisi DEGIL - yalnizca hat testi)",
            )

        run_btn.click(process, inputs=file_input, outputs=[rgb_out, cam_out, mask_out, report])
        file_input.change(process, inputs=file_input, outputs=[rgb_out, cam_out, mask_out, report])

    return demo


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--classifier", type=Path, default=DEFAULT_CLASSIFIER)
    p.add_argument("--segmenter", type=Path, default=DEFAULT_SEGMENTER)
    p.add_argument("--tag", default=None, help="esigi hangi egitim ozetinden alsin "
                                              "(varsayilan: surumun operating_points.json'i)")
    p.add_argument("--image", default=None, help="verilirse arayuz acilmaz, tek goruntu islenir")
    p.add_argument("--sample", action="store_true",
                   help="sentetik ornek kare uret ve onu isle (veri seti gerekmez, "
                        "sonuc bilimsel olarak anlamsizdir - yalnizca hat testi)")
    p.add_argument("--share", action="store_true")
    args = p.parse_args()

    if args.sample:
        args.image = str(make_sample_patch())
        print(f"sentetik ornek uretildi: {args.image}")
        print("UYARI: gercek uydu verisi degil, yalnizca arayuz/hat testi icin")

    if not args.classifier.exists():
        raise SystemExit(f"siniflandirici checkpoint yok: {args.classifier}\n"
                         f"once 'python -m src.train' calistirin")

    classifier = CloudClassifier(args.classifier, threshold=load_threshold(args.tag))
    print(f"siniflandirici yuklendi: {args.classifier} (esik {classifier.threshold:.4f})")

    segmenter = None
    if args.segmenter.exists():
        segmenter = CloudSegmenter(args.segmenter)
        print(f"U-Net yuklendi: {args.segmenter}")
    else:
        print(f"U-Net bulunamadi ({args.segmenter}), segmentasyon paneli bos kalacak")

    if args.image:
        run_cli(args.image, classifier, segmenter)
    else:
        # Elde gercek kare olmasa da arayuz denenebilir olsun.
        if not SAMPLE_PATH.exists():
            make_sample_patch()
        build_interface(classifier, segmenter).launch(share=args.share)


if __name__ == "__main__":
    main()
