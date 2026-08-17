# Surumler

**Dagitilan surum: v3.2**

`src/config.py` icindeki `CURRENT_RELEASE` bu surume isaret eder; `app.py` ve demo
otomatik olarak buradaki modelleri kullanir. `CLOUD_RELEASE` ortam degiskeniyle
gecici olarak baska bir surume gecilebilir:

```
CLOUD_RELEASE=v3 python app.py
```

## Surum matrisi

| surum | kare | U-Net encoder | IoU | bellek | ms/sahne | precision | temiz alan kaybi (%30 esik) |
|---|---|---|---|---|---|---|---|
| v1 | 256x256 | mnv2_100 | 0.8663 | 27.27 MB | 129 | - | - |
| v2 | 256x256 | mnv2_050 | 0.8855 | 26.69 MB | 129 | 0.9264 | %16.17 |
| v3 | 256x256 | mnv2_050 | 0.8844 | 26.47 MB | 129 | 0.9531 | %16.17 |
| v3.1 | 128x128 | mnv2_050 | 0.8824 | 17.82 MB | 138 | 0.9442 | %13.78 |
| **v3.2** | **64x64** | **mnv2_050** | **0.8807** | **14.87 MB** | **175** | **0.9771** | **%8.13** |

Siniflandirici v2'den itibaren degismemistir (`c6_tuned`, mobilenetv2_100, 6 bant,
INT8 2.59 MB, 5.76 ms/kare, ROC-AUC 0.9883). Farklar U-Net tarafindadir.

## Olcut bazinda en iyiler

- **En yuksek IoU:** v2 (0.8855) - v3 ile fark gozlenen degiskenlik icinde
- **En kisa sure:** v1/v2/v3 (129 ms/sahne)
- **En dusuk bellek:** v3.2 (14.87 MB)
- **En yuksek precision:** v3.2 (0.9771)
- **En az bilimsel veri kaybi:** v3.2 (%8.13)
- **En dengeli:** v3.1 (bellek 2.4x az, sure %7 fazla)

## Her surumun icerigi

Her dizin sunlari icerir:

- `classifier.pt`, `unet.pt` - PyTorch checkpoint'leri
- `classifier_fp32/int8.onnx`, `unet_fp32/int8.onnx` - dagitilabilir modeller
- `operating_points.json` - olculmus calisma noktalari (esik kodda sabit degildir)
- `MANIFEST.json`, `README.md` - surum bilgisi ve olculmus sonuclar
- `config_snapshot.py` - surumun uretildigi yapilandirma
- `TEKNIK_RAPOR.md` - o ana kadarki tam teknik rapor
- cesitli analiz tablolari (benchmark, bellek, esik, etiket bazli basarim)
