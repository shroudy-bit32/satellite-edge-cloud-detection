# Surum v1

MobileNetV2 tabanli ilk calisan surum: siniflandirici + U-Net, INT8 kuantizasyon calisiyor, SPARCS harici dogrulama tamam.

Olusturulma: 2026-07-31 17:53

## Olculmus sonuclar

**Siniflandirici** (mobilenetv2_100), esik 0.9266
- test: accuracy 0.9302, precision 0.9896, recall 0.8850, F1 0.9344

**U-Net** (mobilenetv2_100)
- test: IoU 0.8746, Dice 0.9331, goruntu-seviyesi dogruluk 0.9479

**Veri indirme kazanci:**

- PyTorch FP32: %50.21 azalma, %1.188 kullanilabilir veri kaybi
- ONNX FP32: %50.21 azalma, %1.188 kullanilabilir veri kaybi
- ONNX INT8: %50.73 azalma, %1.663 kullanilabilir veri kaybi

## Dosyalar

- `classifier.pt` (8.726 MB)
- `unet.pt` (7.51 MB)
- `classifier_fp32.onnx` (8.656 MB)
- `classifier_int8.onnx` (2.586 MB)
- `unet_fp32.onnx` (7.494 MB)
- `unet_int8.onnx` (2.365 MB)
- `classifier_train_summary.json` (0.001 MB)
- `unet_train_summary.json` (0.001 MB)
- `classifier_benchmark.csv` (0.001 MB)
- `unet_benchmark.csv` (0.0 MB)
- `downlink_analysis.json` (0.001 MB)
- `external_sparcs.json` (0.001 MB)
- `TEKNIK_RAPOR.md` (0.018 MB)
- `config_snapshot.py` (0.005 MB)