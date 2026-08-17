# Surum v3.1

v3 + 128x128 karelemeli U-Net. Aktivasyon bellegi 15.81 -> 6.68 MB (2.4x az), IoU maliyeti 0.0020, sahne suresi %7 fazla. Siniflandirici degismedi (c6_tuned, 256x256).

Olusturulma: 2026-08-02 20:48

## Olculmus sonuclar

**Siniflandirici** (mobilenetv2_100), esik 0.5065
- test: accuracy 0.9563, precision 0.9751, recall 0.9462, F1 0.9605

**U-Net** (mobilenetv2_050)
- test: IoU 0.8856, Dice 0.9394, goruntu-seviyesi dogruluk 0.9448

**Veri indirme kazanci:**

- PyTorch FP32: %54.48 azalma, %3.088 kullanilabilir veri kaybi
- ONNX FP32: %54.48 azalma, %3.088 kullanilabilir veri kaybi
- ONNX INT8: %55.21 azalma, %4.038 kullanilabilir veri kaybi

## Dosyalar

- `classifier.pt` (8.728 MB)
- `unet.pt` (2.278 MB)
- `classifier_fp32.onnx` (8.656 MB)
- `classifier_int8.onnx` (2.586 MB)
- `unet_fp32.onnx` (2.351 MB)
- `unet_int8.onnx` (0.957 MB)
- `classifier_train_summary.json` (0.001 MB)
- `unet_train_summary.json` (0.001 MB)
- `classifier_benchmark.csv` (0.0 MB)
- `unet_benchmark.csv` (0.0 MB)
- `downlink_analysis.json` (0.001 MB)
- `external_sparcs.json` (0.001 MB)
- `tag_analysis.csv` (0.002 MB)
- `operating_points.json` (0.002 MB)
- `memory_profile.csv` (0.0 MB)
- `tiling_tradeoff.csv` (0.0 MB)
- `tiling_latency_clean.csv` (0.0 MB)
- `TEKNIK_RAPOR.md` (0.031 MB)
- `config_snapshot.py` (0.006 MB)

## Eksik

- C:\Users\Cemil\PycharmProjects\InternProject\outputs\reports\unet_t128_threshold_analysis.csv
- C:\Users\Cemil\PycharmProjects\InternProject\outputs\reports\unet_t128_decision_layer.csv