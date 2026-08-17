# Surum v3.2

64x64 karelemeli U-Net + iki siniflandirici secenegi.
**Dagitilan surum budur** (`config.CURRENT_RELEASE`).

v3'e gore U-Net bellegi %44 azaldi (26.47 -> 14.87 MB), bilimsel veri kaybi
yariya indi (%16.17 -> %8.13), IoU maliyeti 0.0037.

## Modeller

| dosya | kare | boyut (INT8) | bellek | ms/kare | ms/sahne |
|---|---|---|---|---|---|
| `classifier_int8.onnx` | 256x256 | 2.59 MB | 22.12 MB | 5.76 | 92 |
| `classifier_64_int8.onnx` | 64x64 | 2.59 MB | 15.74 MB | 1.89 | 484 |
| `unet_int8.onnx` | 64x64 | 0.96 MB | 14.87 MB | 0.69 | 175 |

## Hangi siniflandirici kullanilmali: 256x256

**`classifier.pt` / `classifier_int8.onnx` onerilir.** 64x64 varyanti pakete
dahildir ama dagitim icin uygun degildir:

| olcut | 256x256 | 64x64 | kazanan |
|---|---|---|---|
| S2CMC ROC-AUC | 0.9883 | 0.9872 | esit |
| **SPARCS ROC-AUC** | **0.9644** | **0.8668** | 256x256 |
| SPARCS recall (varsayilan esikte) | 0.9760 | 0.0034 | 256x256 |
| sahne basina sure | 92 ms | 484 ms | 256x256 |
| bellek | 22.12 MB | 15.74 MB | 64x64 |

64x64 siniflandirici farkli sensore genellemiyor: SPARCS'ta 1248 karenin
yalnizca birini eliyor. Bu yalnizca esik sorunu degil - siralama kalitesi de
dusuyor (AUC 0.9644 -> 0.8668). Kucuk pencerede model baglamdan yoksun kalip
ince dokusal istatistiklere dayanmak zorunda kaliyor; bunlar sensorler arasinda
en cok degisen ozelliklerdir.

Ayrica sahne basina 5.3 kat yavastir: 64x64'te bir sahne 256 kare demektir.

**U-Net'te ayni sorun gorulmedi.** Muhtemel sebep gorev farkidir: segmentasyon
her pikselde yerel karar verir ve baglama az bagimlidir; siniflandirma tum
kareyi tek bir sayiya indirdigi icin baglam kaybina duyarlidir.

## Calisma noktalari

`operating_points.json` (256x256 siniflandirici icin) dort nokta tanimlar.
Dagitim icin onerilen: **belirsiz bant (precision>=0.995)**, esik 0.7947.

| nokta | veri azalmasi | kaybedilen kullanilabilir veri | SPARCS kaybi |
|---|---|---|---|
| dengeli | %56.15 | %5.226 | %29.08 |
| mevcut | %54.48 | %3.088 | %19.98 |
| belirsiz bant >=0.99 | %51.15 | %1.188 | %12.34 |
| **belirsiz bant >=0.995** | **%48.65** | **%0.475** | **%7.95** |

Esik kodda sabit degildir; bu dosyadan okunur ve yerden guncellenebilir.

## U-Net (64x64)

| metrik | deger |
|---|---|
| IoU (INT8) | 0.8807 |
| Dice | 0.9392 |
| precision | 0.9771 |
| goruntu-seviyesi dogruluk | 0.9480 |
| temiz alan kaybi (%30 esik) | %8.13 |

Anotatorler arasi uyum (ayni sahnelerde): Dice 0.9597. Model insan seviyesinin
~2 puan altindadir.

## Kisitlar

- Modeller STATIK sekille ihrac edilmistir (INT8 kuantizasyon gerekliligi);
  girdi boyutu calisma aninda degistirilemez.
- Bellek olcumleri ONNX Runtime'a ozgudur. Toplamin ~10.5 MB'i ORT'nin taban
  ayak izidir ve hicbir ayarla (arena/bellek deseni kapatma dahil) dusmez;
  gomulu bir cikarim motoru belirgin sekilde daha az kullanirdi.
- 64x64 siniflandiricinin S2CMC sonuclari 256x256 ile DOGRUDAN karsilastirilamaz:
  kare boyutu degisince etiket tanimi (bulut orani hangi pencerede olculuyor)
  da degisir. Karsilastirilabilir olcutler veri azalmasi, temiz alan kaybi ve
  SPARCS sonuclaridir.

## Dosyalar

Ikili modeller (`.pt`, `.onnx`), egitim ozetleri, benchmark tablolari, calisma
noktalari, bellek profili, kareleme odunlesimi, etiket bazli basarim analizi,
esik duyarlilik analizi, karar katmani deneyi, tam teknik rapor ve surumun
uretildigi `config_snapshot.py`.
