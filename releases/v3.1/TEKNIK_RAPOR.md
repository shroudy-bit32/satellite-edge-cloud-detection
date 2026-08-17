# Uydu Uzerinde Ucta Yapay Zeka ile Bulutlu Goruntulerin Filtrelenmesi
## Teknik Rapor

Tarih: 2026-08-03  
Referans misyon: ESA Phi-Sat-1 / CloudScout

---

## 1. Ozet

Yer gozlem uydularinin cektigi goruntulerin onemli bir bolumu bulutlarla kapli
oldugundan bilimsel olarak kullanilamaz; bunlarin yere indirilmesi sinirli
haberlesme bant genisligini bosa harcar. Bu calismada, bulutlu goruntuleri
uydu uzerindeyken tespit edip eleyecek hafif bir yapay zeka modeli gelistirildi,
INT8'e kuantize edildi ve saglayacagi bant genisligi kazanci sayisal olarak
olculdu. Calisma tamamen yazilim ortaminda yurutulmustur; hedef donanim uzerinde
calistirma kapsam disidir.

Referans misyon ve ilgili calismalar icin bkz. **bolum 14**.

## 2. Veri

### 2.1 Sentinel-2 Cloud Mask Catalogue (ana veri seti)

- Kaynak: 513 alt-sahne, 1022x1022 piksel, 20 m cozunurluk, 13 bant
- Kullanilan bantlar: B02, B03, B04, B08, B10, B11
- Kare boyutu: 256x256, sahne basina 16 kare
- Etiketleme kurali: bulut pikseli orani >= %30 ise 'bulutlu'
- Bulut golgesi ikili gorevde temiz sayildi (SHADOW_AS_CLOUD=False)

**Bolum dagilimi (kare sayisi):**

| split   |   kullanilabilir (0) |   bulutlu (1) |   toplam |
|:--------|---------------------:|--------------:|---------:|
| test    |                  421 |           539 |      960 |
| train   |                 2729 |          3431 |     6160 |
| val     |                  477 |           611 |     1088 |

**Sahne bazli zorluk (veri setinin kendi 1-5 skalasi):**

| split   |   sahne |   ortalama zorluk |
|:--------|--------:|------------------:|
| test    |      60 |              2.65 |
| train   |     385 |              2.03 |
| val     |      68 |              2    |

> Bolumleme SAHNE bazlidir, kare bazli degil: ayni sahnenin kareleri
> birbirine cok benzedigi icin kare bazli bolme test setine sizinti
> yaratir ve dogrulugu yapay olarak yukseltir.

> Test bolumu, veri setinin README'sinin onerisiyle CALIBRATION + VALIDATION
> sahnelerinden olusur. Bu sahneler icin anotatorler arasi uyum yayimlanmis
> oldugundan model performansi insan seviyesiyle karsilastirilabilir.
> Test setinin ortalama zorlugu egitim setinden yuksektir (tasarim geregi).

### 2.2 SPARCS (harici dogrulama)

- 78 Landsat-8 sahnesi, 1248 kare
- Bant eslemesi (Sentinel-2 -> Landsat-8): B02->B2, B03->B3, B04->B4, B08->B5, B10->B9, B11->B6
- Egitimde KULLANILMADI; yalnizca farkli sensore genelleme olcumu icin

> **Uyari:** Gunes yuksekligi duzeltmesi uygulanmadi (MTL dosyalari arsivde yok); Sentinel-2 ile arasinda sistematik reflektans kaymasi bulunur.

### 2.3 95-Cloud (kapsam disi)

95-Cloud veri seti yalnizca 4 bant icerir (R, G, B, NIR); cirrus (B10) ve
SWIR (B11) bantlari yoktur. Bu iki bant bulut ayriminin en guclu sinyalleri
oldugundan 6 bantli model bu veri setiyle beslenemez. Ayri bir 4 bantli model
varyanti gerektirdiginden kapsam disi birakilmistir.

## 3. Modeller

### 3.1 Ikili siniflandirici (temel hedef)

MobileNetV2 omurgasi, tek logit cikisli. Coklu bant girdisi icin ilk
konvolusyon agirligi kanal boyunca uyarlanir (timm `in_chans`).

### 3.2 U-Net segmentasyon (genisletilmis hedef)

Ayni MobileNetV2 omurgasi encoder olarak, uzerine derinlemesine-ayrilabilir
konvolusyonlardan olusan ince bir decoder. Klasik U-Net (~31M parametre,
~124 MB) boyut kisitini kat kat astigi icin tercih edilmedi.

## 4. Sonuclar

### 4.1 Siniflandirici

**Test metrikleri (PyTorch FP32):**

| metrik    |   deger |
|:----------|--------:|
| accuracy  |  0.9563 |
| precision |  0.9751 |
| recall    |  0.9462 |
| f1        |  0.9605 |
| roc_auc   |  0.9883 |
| ap        |  0.9921 |

Karar esigi: 0.5065 (dogrulama seti uzerinde precision >= 0.99 kisitiyla secildi)

**Karsilastirma tablosu (PDF bolum 3):**

| model        |   boyut_MB |   accuracy |   precision |   recall |       f1 |   latency_mean_ms |   latency_p50_ms |   latency_p95_ms | hedef_boyut_OK   | hedef_sure_OK   |
|:-------------|-----------:|-----------:|------------:|---------:|---------:|------------------:|-----------------:|-----------------:|:-----------------|:----------------|
| PyTorch FP32 |       8.73 |   0.95625  |    0.975143 | 0.946197 | 0.960452 |         nan       |        nan       |        nan       | False            | False           |
| ONNX FP32    |       8.66 |   0.95625  |    0.975143 | 0.946197 | 0.960452 |           9.76796 |          9.68675 |         10.7786  | False            | True            |
| ONNX INT8    |       2.59 |   0.955208 |    0.967925 | 0.951763 | 0.959775 |           5.75752 |          5.6637  |          6.52437 | True             | True            |

### 4.2 U-Net segmentasyon

**Test metrikleri (PyTorch FP32):**

| metrik               |   deger |
|:---------------------|--------:|
| iou                  |  0.8896 |
| dice                 |  0.9415 |
| precision            |  0.9697 |
| recall               |  0.915  |
| pixel_accuracy       |  0.9412 |
| image_level_accuracy |  0.948  |

**Karsilastirma tablosu (PDF bolum 3):**

| model     |   boyut_MB |    iou |   dice |   precision |   recall |   latency_mean_ms |   latency_p50_ms |   latency_p95_ms | hedef_boyut_OK   | hedef_sure_OK   |
|:----------|-----------:|-------:|-------:|------------:|---------:|------------------:|-----------------:|-----------------:|:-----------------|:----------------|
| ONNX FP32 |       2.35 | 0.8892 | 0.9413 |      0.9704 |   0.914  |          1.0365   |          1.14235 |          1.40902 | True             | True            |
| ONNX INT8 |       0.96 | 0.8807 | 0.9366 |      0.9771 |   0.8993 |          0.721243 |          0.593   |          1.36436 | True             | True            |

## 5. Veri indirme kazanci (fayda analizi)

PDF'in istedigi "bu filtre uyduda calissaydi indirilen veri hacmi %X azalirdi"
hesabi. Test seti uzerinden, karar esigi dogrulama setinde sabitlenerek.

| model        |   veri azalmasi % |   kaybedilen kullanilabilir % |   bosuna indirilen bulutlu |   teorik ust sinir % |
|:-------------|------------------:|------------------------------:|---------------------------:|---------------------:|
| PyTorch FP32 |             54.48 |                         3.088 |                         29 |                56.15 |
| ONNX FP32    |             54.48 |                         3.088 |                         29 |                56.15 |
| ONNX INT8    |             55.21 |                         4.038 |                         26 |                56.15 |

> Yanlis eleme (kullanilabilir goruntunun atilmasi) geri donusu olmayan
> bilimsel veri kaybidir. Karar esigi bu nedenle F1'i degil, yuksek
> precision'i hedefleyecek sekilde secilmistir.

## 6. Harici dogrulama: SPARCS (Landsat-8)

Farkli bir uydudan, egitimde hic gorulmemis veri. Karar esigi Sentinel-2
egitiminden alinmis, bu veri uzerinde YENIDEN AYARLANMAMISTIR.

| kirilim                  |    n |   accuracy |   precision |   recall |     f1 |   roc_auc |   kari_bulut_sanma_orani |
|:-------------------------|-----:|-----------:|------------:|---------:|-------:|----------:|-------------------------:|
| genel                    | 1248 |     0.8421 |      0.5996 |   0.9795 | 0.7438 |    0.9645 |                 nan      |
| kar/buz olmayan          | 1152 |     0.8733 |      0.6643 |   0.9788 | 0.7914 |    0.9731 |                 nan      |
| kar/buz agirlikli (>%20) |   96 |     0.4688 |      0.15   |   1      | 0.2609 |    0.8646 |                   0.5862 |

> **Basarisizlik modu:** temiz kar/buz karelerinin %58.6'i yanlislikla eleniyor. Kar ve bulut spektral olarak benzer; ayrica Landsat-8 reflektanslarina gunes yuksekligi duzeltmesi uygulanamadigi icin sistematik bir kayma vardir.

## 7. Kuantizasyon calismasi

INT8 kuantizasyon ilk denemede basarisiz oldu ve kok neden sistematik
eleme ile arandi. Bu bolum hangi aciklamalarin hangi olcumle elendigini
kaydeder; elenen hipotezler sonucun kendisi kadar bilgilendiricidir.

### 7.1 MobileNetV3 ile yapilandirma taramasi (basarisiz)

| yapilandirma                |   boyut_MB |   accuracy |   precision |     recall |        f1 |   roc_auc |   latency_ms |   sapma |   kuantize_dugum |   toplam_dugum |
|:----------------------------|-----------:|-----------:|------------:|-----------:|----------:|----------:|-------------:|--------:|-----------------:|---------------:|
| FP32 (referans)             |       6.04 |   0.983333 |    0.980132 | 0.986667   | 0.983389  |  0.996044 |         9.23 | 0       |                0 |            122 |
| dinamik (sadece MatMul)     |       1.55 |   0.6      |    0.556818 | 0.98       | 0.710145  |  0.739    |        95.43 | 0.39074 |                0 |            464 |
| qdq_s8_perch_percentile     |       1.71 |   0.573333 |    0.761905 | 0.213333   | 0.333333  |  0.702111 |        24.59 | 0.42835 |              364 |            491 |
| qdq_u8_perch_percentile     |       1.71 |   0.566667 |    0.75     | 0.2        | 0.315789  |  0.701689 |         6.71 | 0.42824 |              364 |            491 |
| qdq_u8_perch_minmax         |       1.71 |   0.5      |    0.5      | 0.00666667 | 0.0131579 |  0.356067 |         6.76 | 0.51011 |              364 |            491 |
| qdq_u8_perch_entropy        |       1.71 |   0.5      |    0.5      | 0.00666667 | 0.0131579 |  0.356067 |         6.76 | 0.51011 |              364 |            491 |
| qdq_u8_pertensor_percentile |       1.58 |   0.39     |    0.365854 | 0.3        | 0.32967   |  0.340178 |         6.48 | 0.56193 |              364 |            491 |
| qdq_u8_perch_percentile_rr  |       1.71 |   0.623333 |    0.836364 | 0.306667   | 0.44878   |  0.735978 |         6.67 | 0.38686 |              364 |            491 |
| qop_u8_perch_percentile     |       1.59 |   0.566667 |    0.75     | 0.2        | 0.315789  |  0.701689 |         6.77 | 0.42824 |              131 |            205 |

Yedi farkli yapilandirmanin hicbiri dogrulugu kurtarmadi.

### 7.2 Op tipi bazli izolasyon

| kuantize edilen                 |   boyut_MB |   accuracy |   precision |   recall |       f1 |   roc_auc |   latency_ms |   sapma |
|:--------------------------------|-----------:|-----------:|------------:|---------:|---------:|----------:|-------------:|--------:|
| hicbiri (FP32)                  |       6.04 |   0.983333 |    0.980132 | 0.986667 | 0.983389 |  0.996044 |         5.53 | 0       |
| sadece Conv                     |       1.7  |   0.553333 |    0.578431 | 0.393333 | 0.468254 |  0.635289 |         7.45 | 0.44225 |
| Conv + Gemm                     |       1.7  |   0.553333 |    0.576923 | 0.4      | 0.472441 |  0.630533 |         7.45 | 0.44096 |
| Conv + Gemm + Add               |       1.7  |   0.553333 |    0.576923 | 0.4      | 0.472441 |  0.630533 |         7.47 | 0.44096 |
| Conv + Gemm + Mul               |       1.72 |   0.543333 |    0.591549 | 0.28     | 0.38009  |  0.628067 |         8.17 | 0.45326 |
| Conv + Gemm + GlobalAveragePool |       1.7  |   0.553333 |    0.576923 | 0.4      | 0.472441 |  0.630533 |         8.69 | 0.44096 |

Yalnizca Conv katmanlarini kuantize etmek bile bozulmayi uretti;
squeeze-excite / hard-swish bloklari tek basina sorumlu degil.

### 7.3 Katman bazli kademeli kuantizasyon

|   kuantize_conv |   boyut_MB |   accuracy |   precision |   recall |       f1 |   roc_auc |   sapma |
|----------------:|-----------:|-----------:|------------:|---------:|---------:|----------:|--------:|
|               0 |       6.04 |      0.98  |    0.970588 |     0.99 | 0.980198 |    0.9949 | 0       |
|               1 |       5.8  |      0.595 |    0.756757 |     0.28 | 0.408759 |    0.7972 | 0.42372 |
|               2 |       5.8  |      0.63  |    0.685714 |     0.48 | 0.564706 |    0.6796 | 0.39564 |
|               4 |       5.8  |      0.615 |    0.661972 |     0.47 | 0.549708 |    0.6734 | 0.39553 |
|               8 |       5.8  |      0.68  |    0.704545 |     0.62 | 0.659574 |    0.7443 | 0.33975 |
|              16 |       5.76 |      0.66  |    0.677778 |     0.61 | 0.642105 |    0.7096 | 0.36845 |
|              24 |       5.54 |      0.545 |    0.56     |     0.42 | 0.48     |    0.6257 | 0.45601 |
|              32 |       5.41 |      0.55  |    0.573529 |     0.39 | 0.464286 |    0.6487 | 0.44392 |
|              40 |       5.2  |      0.59  |    0.628571 |     0.44 | 0.517647 |    0.6654 | 0.41672 |
|              48 |       4.14 |      0.555 |    0.57971  |     0.4  | 0.473373 |    0.6434 | 0.43535 |
|              53 |       1.7  |      0.555 |    0.575342 |     0.42 | 0.485549 |    0.6439 | 0.43396 |

Tek bir sorumlu katman bulunamadi; bozulma yaygin.

### 7.4 MobileNetV2 ile ayni tarama (basarili)

| yapilandirma                |   boyut_MB |   accuracy |   precision |   recall |       f1 |   roc_auc |   latency_ms |   sapma |   kuantize_dugum |   toplam_dugum |
|:----------------------------|-----------:|-----------:|------------:|---------:|---------:|----------:|-------------:|--------:|-----------------:|---------------:|
| FP32 (referans)             |       8.66 |   0.95     |    0.992701 | 0.906667 | 0.947735 |  0.987778 |        17.51 | 0       |                0 |            100 |
| dinamik (sadece MatMul)     |       2.33 |   0.46     |    0.433333 | 0.26     | 0.325    |  0.555022 |       291.86 | 0.42812 |                0 |            417 |
| qdq_s8_perch_percentile     |       2.59 |   0.936667 |    0.964539 | 0.906667 | 0.934708 |  0.985067 |        26.1  | 0.04704 |              236 |            301 |
| qdq_u8_perch_percentile     |       2.59 |   0.93     |    0.957447 | 0.9      | 0.927835 |  0.985267 |        10.82 | 0.04802 |              236 |            301 |
| qdq_u8_perch_minmax         |       2.59 |   0.926667 |    0.957143 | 0.893333 | 0.924138 |  0.983111 |        10.7  | 0.05068 |              236 |            301 |
| qdq_u8_perch_entropy        |       2.59 |   0.926667 |    0.957143 | 0.893333 | 0.924138 |  0.983111 |        11    | 0.05068 |              236 |            301 |
| qdq_u8_pertensor_percentile |       2.37 |   0.446667 |    0.416667 | 0.266667 | 0.325203 |  0.549822 |        10.44 | 0.42565 |              236 |            301 |
| qdq_u8_perch_percentile_rr  |       2.59 |   0.913333 |    0.913333 | 0.913333 | 0.913333 |  0.966333 |        10.31 | 0.13791 |              236 |            301 |
| qop_u8_perch_percentile     |       2.3  |   0.93     |    0.957447 | 0.9      | 0.927835 |  0.985267 |        10.17 | 0.04802 |               56 |             69 |

**Sonuclar:**

1. MobileNetV2 ayni yapilandirmalarla sorunsuz kuantize oluyor (dogruluk kaybi ~%1.4).
2. `per_channel=True` her iki mimaride de ZORUNLU; tensor basina olcekleme modeli cokertiyor.
3. Aktivasyonlarda `QUInt8`, `QInt8`'e gore x86'da ~2.4 kat hizli; dogruluk farki ihmal edilebilir.
4. MobileNetV3'un neden basarisiz oldugu KESIN OLARAK aciklanamadi. Agirlik dinamik araligi hipotezi olculdu ve elendi (V2'nin araligi daha genis oldugu halde V2 sorunsuz kuantize oluyor). En olasi aciklama hard-swish aktivasyon dagilimlaridir; dogrulanmamistir.

## 8. Hiperparametre ve optimizasyon denemeleri (ablasyon)

Tum kosular MobileNetV3 omurgasiyla, tek tohumla yapilmistir.

| kosu                      |   val F1 |   esik |   test acc |   precision |   recall |   test F1 |   ROC-AUC |
|:--------------------------|---------:|-------:|-----------:|------------:|---------:|----------:|----------:|
| 15 epoch (sade)           |   0.9777 | 0.624  |     0.926  |      0.9398 |   0.9276 |    0.9337 |    0.9834 |
| 15 ep + EMA d=0.999       |   0.7314 | 0.8319 |     0.5771 |      0.571  |   0.9926 |    0.7249 |    0.6482 |
| 15 ep + EMA d=0.99        |   0.9786 | 0.7083 |     0.9344 |      0.9542 |   0.9276 |    0.9407 |    0.9845 |
| 15 ep + EMA d=0.95        |   0.975  | 0.3162 |     0.9375 |      0.9493 |   0.9388 |    0.944  |    0.9828 |
| 15 ep + dengeli ornekleme |   0.9717 | 0.7533 |     0.924  |      0.9569 |   0.9054 |    0.9304 |    0.9851 |
| 10 epoch (sade)           |   0.9827 | 0.3713 |     0.9323 |      0.9158 |   0.9685 |    0.9414 |    0.9877 |
| 30 epoch (sade)           |   0.9835 | 0.3663 |     0.9292 |      0.9469 |   0.9258 |    0.9362 |    0.9834 |

**Cikarimlar:**

- 30 epoch, 10 epoch'a gore test basarimini artirmadi; egitim kaybi 0.038'den
  0.006'ya duserken dogrulama F1'i yatay kaldi (asiri ogrenme).
- EMA, decay degeri egitim uzunluguna uygun secilmezse modeli cokertiyor:
  0.999^1440 = 0.236, yani nihai agirliklarin %24'u egitilmemis baslangic degeri.
  Kural: decay^(toplam_adim) ihmal edilebilir olmali.
- Sade kosular ile dengeli ornekleme arasindaki farklar (F1 0.930-0.941) tek
  tohumlu kosulardan geldigi icin **istatistiksel olarak anlamli sayilamaz**.
  Kesin siralama icin coklu tohum gerekir.

## 9. Hiperparametre taramasi ve bant secimi

### Rastgele arama (8 deneme, 10 epoch)

| tag   |          lr |   weight_decay |   batch_size |   label_smoothing |   val_f1 |   threshold |   test_f1 |   test_acc |   precision |   recall |   roc_auc |
|:------|------------:|---------------:|-------------:|------------------:|---------:|------------:|----------:|-----------:|------------:|---------:|----------:|
| hp06  | 0.00148123  |    0.000336246 |           32 |               0.1 |   0.9708 |      0.5443 |    0.9407 |     0.9354 |      0.9704 |   0.9128 |    0.9847 |
| hp00  | 0.001189    |    5.69526e-05 |           64 |               0.1 |   0.9702 |      0.6546 |    0.962  |     0.9583 |      0.9845 |   0.9406 |    0.9851 |
| hp02  | 0.00112821  |    0.00139398  |           64 |               0   |   0.9701 |      0.7283 |    0.9504 |     0.9458 |      0.9784 |   0.9239 |    0.99   |
| hp03  | 0.0003161   |    3.04223e-05 |           32 |               0.1 |   0.9685 |      0.6372 |    0.9297 |     0.925  |      0.9814 |   0.8831 |    0.9682 |
| hp01  | 0.000868952 |    2.38073e-06 |           64 |               0.1 |   0.9633 |      0.603  |    0.9521 |     0.9479 |      0.9842 |   0.9221 |    0.9794 |
| hp04  | 0.000698007 |    0.00195455  |           64 |               0   |   0.9578 |      0.53   |    0.9534 |     0.9479 |      0.9587 |   0.9481 |    0.9806 |
| hp07  | 0.000213487 |    0.0076347   |           64 |               0.1 |   0.9535 |      0.5255 |    0.9333 |     0.9271 |      0.9589 |   0.9091 |    0.9698 |
| hp05  | 0.000126775 |    0.000165325 |          128 |               0   |   0.8584 |      0.9691 |    0.8769 |     0.8646 |      0.8956 |   0.859  |    0.9408 |

Secim DOGRULAMA seti uzerinden yapildi; test yalnizca raporlama icin.

**Bulgu:** val F1 yayilimi ~0.11. Bu, daha once olculen mimari/duzenleme
farklarindan (~0.008) bir mertebe buyuk. Ogrenme orani baskin faktor:
proje varsayilani 3e-4 iken en iyi degerler 1.1-1.5e-3 araliginda cikti.

### Bant sayisi (ayni taranmis ayarlarla, 15 epoch)

| bant sayisi   |   bantlar |   test F1 |   test acc |   ROC-AUC |
|:--------------|----------:|----------:|-----------:|----------:|
| 6 bant        |         6 |    0.9605 |     0.9563 |    0.9883 |
| 9 bant        |         9 |    0.9403 |     0.9333 |    0.9743 |
| 13 bant       |        13 |    0.9223 |     0.9156 |    0.97   |

**Bulgu:** bant sayisi arttikca basarim TEKDUZE dusuyor. Ek bantlar
(kirmizi-kenar B05-B07) bulut tespitine bilgi katmiyor, buna karsilik
ImageNet on-egitimli ilk konvolusyon agirliklarini seyreltiyor ve
CPU cikarim suresini artiriyor. Boyut ise neredeyse degismiyor.

Cekince: hiperparametreler 6 bantli veri uzerinde tarandi, bu 6 banda
yapisal avantaj saglar. 13 bant hem taranmis hem elle secilen ayarla
kaybettigi icin sonuc bu cekinceden bagimsizdir; 9 bant icin yalnizca
taranmis ayar denendi.

### Bilgi damitma

- Ogretmen: mobilenetv2_140, test F1 0.9471
- Ogrenci: mobilenetv2_100, test F1 0.9540
- alpha=0.3, sicaklik=3.0

**Bulgu:** damitma dogruluk kazanci saglamadi. Sebep yontem degil kurulum:
ogretmen olarak secilen daha genis model (mobilenetv2_140), ayni ayarlarla
egitilen ogrenci mimarisinden DAHA ZAYIF cikti. Zayif ogretmenden bilgi
damitilamaz. Anlamli bir deneme icin once gercekten daha guclu bir
ogretmen kurulmali (ayri hiperparametre taramasi veya model toplulugu).

## 10. Surum karsilastirmasi: v1 -> v2

Iki surum de `releases/` altinda dondurulmustur; v2 gelistirmeleri v1'i
bozmadan olculebilsin diye.

| olcut                           | v1                  | v2                  |
|:--------------------------------|:--------------------|:--------------------|
| Siniflandirici omurgasi         | mobilenetv2_100     | mobilenetv2_100     |
| Ogrenme orani                   | 3e-4 (elle secilen) | 1.19e-03 (taranmis) |
| Etiket yumusatma                | yok                 | 0.1                 |
| Siniflandirici test F1          | 0.9344              | 0.9605              |
| Siniflandirici ROC-AUC          | 0.9810              | 0.9883              |
| U-Net encoder                   | mobilenetv2_100     | mobilenetv2_050     |
| U-Net IoU                       | 0.8746              | 0.8844              |
| U-Net Dice                      | 0.9331              | 0.9387              |
| U-Net goruntu-seviyesi dogruluk | 0.9479              | 0.9510              |

### 10.1 Neyin ise yaradigi, neyin yaramadigi

v2'ye giden yolda dort ayri iyilestirme denendi. Ikisi kazanc sagladi,
ikisi saglamadi; ikisi de raporlanmaya degerdir.

**Kazanc saglayanlar:**

1. **Hiperparametre taramasi (asil kazanc).** Proje varsayilani olan
   ogrenme orani 3e-4, rastgele aramada bulunan optimumun (~1.2e-3)
   dortte biriymis. Tarama tek basina ROC-AUC'yi 0.9813'ten 0.9884'e
   cikardi ve INT8 sonrasi F1 kaybini neredeyse sifirladi
   (v1: 0.9458 -> 0.9403; v2: 0.9605 -> 0.9598). Mimari degismedi,
   boyut degismedi, yalnizca egitim ayarlari degisti.

2. **U-Net encoder'inin KUCULTULMESI.** mobilenetv2_100 -> mobilenetv2_050
   gecisi hem boyutu ucte bire indirdi (2.37 MB -> 0.96 MB) hem dogrulugu
   ARTIRDI (IoU 0.8663 -> 0.8855). Beklenmedik gorunse de aciklanabilir:
   segmentasyon her piksel icin denetim sinyali verir (6.160 kare x 65.536
   piksel), kucuk model bu yogunlukta sinyalle asiri ogrenmeden ogrenir.

**Kazanc saglamayanlar:**

3. **Bant sayisini artirmak.** 6 -> 9 -> 13 bant gecisinde basarim TEKDUZE
   dustu (test F1 0.9605 / 0.9403 / 0.9223). Ek bantlar bilgi katmadi,
   ImageNet on-egitimli ilk konvolusyon agirliklarini seyreltti ve CPU
   suresini artirdi.

4. **Bilgi damitma.** Ogrenci ogretmeni gecti (0.9540 vs 0.9471) ama bu bir
   basari degil, kurulum hatasinin isareti: ogretmen olarak secilen daha
   genis model, ogrenciden zayif kaldi. Zayif ogretmenden bilgi damitilamaz.

### 10.2 En buyuk pratik kazanc: calisma noktasi

Modeldeki iyilesme gercek ama olculu (ROC-AUC +0.007). Operasyonel sonuca
asil etki eden degisiklik, karar esiginin NASIL secildigidir.

v1 ve v2'nin ilk hali esigi TUM dogrulama seti uzerinde seciyordu. Ancak
karelerin ~%73'u ya tamamen temiz ya tamamen kapali; bu kolay orneklerin
karari esikten bagimsizdir ve esik secimini asil onemli oldugu belirsiz
bolgeden uzaklastirir. Esigi yalnizca kismi bulutlu karelerde (bulut orani
%2-%98) ayarlamak bu carpitmayi kaldirir.

### 10.3 Calisma noktalari

Karar esigi artik kodda sabit bir sayi degildir. Model, her biri olculmus
sonuclariyla tanimlanmis birden fazla calisma noktasiyla birlikte teslim
edilir (`operating_points.json`); hangisinin kullanilacagi bir dagitim
kararidir ve yerden guncellenebilir, yeniden egitim gerektirmez.

| calisma noktasi                   |   esik |   S2CMC precision |   S2CMC F1 |   veri azalmasi % |   kayip % |   SPARCS kayip % |   SPARCS kar/buz yanlis eleme % |
|:----------------------------------|-------:|------------------:|-----------:|------------------:|----------:|-----------------:|--------------------------------:|
| mevcut (tum val, precision>=0.99) | 0.5065 |            0.9751 |     0.9605 |             54.48 |     3.088 |           19.979 |                           58.62 |
| belirsiz bant (precision>=0.99)   | 0.7181 |            0.9898 |     0.9437 |             51.15 |     1.188 |           12.343 |                           43.68 |
| belirsiz bant (precision>=0.995)  | 0.7947 |            0.9957 |     0.9245 |             48.65 |     0.475 |            7.95  |                           29.89 |
| dengeli (tum val, en iyi F1)      | 0.3608 |            0.9592 |     0.9592 |             56.15 |     5.226 |           29.079 |                           68.97 |

**Okuma:** esik yukseldikce her iki veri setinde de kaybedilen kullanilabilir
veri azalir, karsiliginda bant genisligi kazanci bir miktar duser. Belirsiz
bantta secilen esik, S2CMC ve SPARCS'ta TEKDUZE daha iyidir.

**Onemli sinir - yontem her yapilandirmada ise yaramiyor.** Uc kare
boyutunda ayri ayri denendi:

| kare | belirsiz bantta secilen esik | precision | yanlis eleme | sonuc |
|---|---|---|---|---|
| 256x256 | 0.862 (yukari) | 0.9974 | %5.94 -> %0.24 | kazanc |
| 128x128 | 0.288 (asagi) | 0.9515 | %5.84 -> %6.02 | kazanc yok |
| 64x64 | 0.891 (yukari) | 0.9916 | %1.99 -> %0.79 | kazanc |

Sebep, yontemin `precision >= 0.99` kisitini saglayan EN DUSUK esigi
aramasidir. 256x256 ve 64x64'te belirsiz bant gercekten zor oldugu icin esik
yukari itilir; 128x128'de model o alt kumede zaten yeterince iyi oldugundan
dusuk bir esik de kisiti saglar ve arama orada durur.

Sonuc: yontem uc yapilandirmanin ikisinde belirgin kazanc verdi, birinde hic
vermedi. Genel gecer kabul edilemez; her model ve kare boyutu icin ayrica
OLCULMELIDIR. Kazanc goruldugu yerlerde ise buyuktur (yanlis eleme 2.5-25 kat
azalma), bu nedenle denenmeye degerdir.

Kritik nokta farkli sensordedir: mevcut nokta SPARCS'ta kullanilabilir verinin
%19.98'ini kaybederken, belirsiz bant (precision>=0.995) noktasi bunu %7.95'e
indirir; kar/buz karelerinde yanlis eleme %58.62'den %29.89'a duser. Model ayni
modeldir - degisen yalnizca esigin nasil secildigidir.

**Dagitim onerisi:** belirsiz bant (precision>=0.995). Kullanilabilir verinin
%99.5'i korunur, bant genisligi kazanci %48.65 olur - teorik ust sinirin
(%56.15) %87'si.

## 11. Bellek profili

PDF'in "uydu uzerindeki sinirli bellegi temsil eder" kisiti yalnizca disk
boyutunu degil, cikarim sirasindaki bellek ihtiyacini da kapsar. Diskteki
model boyutu bu ihtiyacin kucuk bir parcasidir.

Her model ayri bir surecte olculmustur (onceki modelin ayirdigi bellek
sonrakinin olcumune karismasin diye).

| model          | girdi   |   disk_MB |   agirlik_MB |   cikarim_ek_MB |   model_maliyeti_MB |
|:---------------|:--------|----------:|-------------:|----------------:|--------------------:|
| c64_tuned_int8 | 64x64   |      2.59 |        11.46 |            4.28 |               15.74 |

**Sutunlar:**

- `disk_MB`: ONNX dosya boyutu
- `agirlik_MB`: oturum acilisinda artan bellek (agirliklar + ORT bellek havuzu)
- `cikarim_ek_MB`: cikarim sirasinda ek ayrilan bellek (ara aktivasyonlar)
- `model_maliyeti_MB`: ikisinin toplami; Python yorumlayicisinin ~50 MB'lik
  taban kullanimi HARIC (uydu yaziliminda bu maliyet olmaz)

**Bulgular:**

1. **Gercek bellek ihtiyaci, disk boyutunun 10-25 katidir.** INT8 siniflandirici
   diskte 2.59 MB, calisirken 22.12 MB. Yalnizca dosya boyutuna bakarak bellek
   kisiti degerlendirmek yaniltici olur.

2. **U-Net diskte kucuk, bellekte buyuktur.** Siniflandiriciya gore diskte 2.7 kat
   kucuk (0.96 MB vs 2.59 MB) ama bellekte %22 daha pahali (26.89 MB vs 22.12 MB).
   Sebep mimaridir: segmentasyon decoder'i tam cozunurlukte ara tensorler tasir,
   siniflandirici ise havuzlama ile hizla kuculur. Az parametre, buyuk aktivasyon.
   Dagitim karari bu nedenle kisita baglidir: disk darsa U-Net, RAM darsa
   siniflandirici.

3. **Model 256x256 girdiye kilitlidir.** 128 ve 512 olcumleri basarisiz oldu:
   INT8 kuantizasyonun calisabilmesi icin model STATIK sekille ihrac edilmisti.
   Farkli cozunurluk ayri bir model gerektirir.

**Cekince:** `agirlik_MB` saf agirlik degildir; ONNX Runtime'in onceden ayirdigi
bellek havuzunu icerir. Gomulu bir cikarim motoru (TFLite Micro, ozel cekirdek)
belirgin sekilde daha az kullanirdi. Bu rakamlar ORT'ye ozgu UST SINIRLARDIR,
modelin icsel gereksinimi degil.

### 11.1 Kareleme (tiling) ile bellek azaltma

Aktivasyon bellegi girdi boyutuyla olceklendigi icin, sahneyi daha kucuk
karelere bolmek bellek ihtiyacini dusurur. Bedeli: sahne basina daha cok
cikarim ve daha dar baglam.

Ayni U-Net mimarisi uc kare boyutunda ayri ayri egitildi. Karsilastirma
PIKSEL BAZLI IoU uzerinden yapildi; bu metrik ayni test piksellerini
degerlendirdigi icin kareleme boyutundan bagimsizdir.

| kare    |   sahne basina kare |    IoU |   Dice |   precision |   recall |   ms/kare |   ms/sahne |   aktivasyon_MB |   model_maliyeti_MB |
|:--------|--------------------:|-------:|-------:|------------:|---------:|----------:|-----------:|----------------:|--------------------:|
| 64x64   |                 256 | 0.8807 | 0.9366 |      0.9771 |   0.8993 |     0.685 |        175 |            4.08 |               14.87 |
| 128x128 |                  64 | 0.8824 | 0.9375 |      0.9442 |   0.931  |     2.156 |        138 |            6.68 |               17.82 |
| 256x256 |                  16 | 0.8844 | 0.9386 |      0.9531 |   0.9246 |     8.067 |        129 |           15.81 |               26.72 |

**Bulgular:**

1. **Baglam kaybi beklenenden cok daha kucuk.** Kare 16 kat kuculdugunde
   (256x256 -> 64x64) IoU yalnizca 0.0037 dusuyor. Bulut tespiti buyuk olcude
   YEREL bir gorev: bir pikselin bulut olup olmadigi yakin komsulugundan
   anlasiliyor, genis sahne baglami gerekmiyor.

2. **Bellek kazanci alan oraniyla degil, daha yavas olceklendi.** Alan 16 kat
   kuculurken aktivasyon bellegi 3.9 kat azaldi (15.81 -> 4.08 MB). Sebep:
   bellek yalnizca girdi alanina bagli degil; encoder'in derin katmanlarindaki
   kanal sayilari ve calisma zamaninin sabit yukleri de paya giriyor.

3. **Kare boyutu precision/recall dengesini kaydiriyor.** 64x64'te precision
   0.9771 / recall 0.8993, 256x256'da 0.9531 / 0.9246. Kucuk kareler modeli
   daha temkinli yapiyor - bu projede istenen yon, cunku yanlis eleme geri
   donusu olmayan veri kaybidir.

4. **128x128 dengeli secim:** bellek 2.4 kat az, sahne suresi yalnizca %7 fazla,
   IoU maliyeti 0.0020.

**Olcum notu:** ilk sure olcumleri farkli calisma zincirlerinde alindigi icin
tutarsiz cikti (128x128 modeli 64x64'ten 7 kat yavas gorunuyordu, alan orani
yalnizca 4). Sureler DONGUSEL olarak yeniden olculdu: her turda her modelden
birer ornek alinarak termal surukleme ve arka plan yuku uc modele de esit
dagitildi. Tablodaki degerler bu temiz olcumden gelmektedir.

### 11.2 Siniflandiricida kareleme - negatif sonuc

U-Net'te ise yarayan ince kareleme, siniflandiriciya uygulandiginda
basarisiz oldu. Ayni hiperparametrelerle 64x64'te egitilen siniflandirici:

| olcut                           | 256x256   | 64x64    | sonuc              |
|:--------------------------------|:----------|:---------|:-------------------|
| S2CMC test ROC-AUC              | 0.9883    | 0.9872   | esit               |
| S2CMC test F1 (INT8)            | 0.9598    | 0.9329   | karsilastirilamaz* |
| SPARCS ROC-AUC                  | 0.9644    | 0.8668   | 256x256            |
| SPARCS recall (varsayilan esik) | 0.9760    | 0.0034   | 256x256            |
| sure (ms/kare)                  | 5.76      | 1.89     | 64x64              |
| sure (ms/sahne)                 | 92        | 484      | 256x256            |
| bellek                          | 22.12 MB  | 15.74 MB | 64x64              |

\* Kare boyutu degisince etiket tanimi da degisir (bulut orani hangi
pencerede olculuyor), bu nedenle S2CMC F1'leri dogrudan karsilastirilamaz.
Karsilastirilabilir olcutler ROC-AUC, SPARCS sonuclari ve sahne basina suredir.

**Bulgu: 64x64 siniflandirici farkli sensore hic genellemiyor.** SPARCS'ta
1248 karenin yalnizca birini eliyor (recall 0.0034). Bu yalnizca esik
yerlesimi sorunu degildir - ROC-AUC da 0.9644'ten 0.8668'e dusuyor, yani
siralama kalitesi gercekten bozuluyor.

**Neden U-Net'te olmuyor da siniflandiricida oluyor:** segmentasyon her
pikselde YEREL bir karar verir ve genis baglama zaten az bagimlidir;
siniflandirma ise tum kareyi tek bir sayiya indirir. Pencere daraldikca bu
ozet, ince dokusal istatistiklere dayanmak zorunda kalir - ve dokusal
istatistikler sensorler arasinda en cok degisen ozelliklerdir. 256x256'da
model sahne genelindeki parlaklik dagilimi gibi daha dayanikli ipuclari
kullanabiliyor.

**Ayrica sahne basina 5.3 kat yavas:** kare basina hizli olmasi yaniltici,
cunku 64x64'te bir sahne 16 yerine 256 kare demektir.

Sonuc: kareleme karari GOREVE BAGLIDIR. Segmentasyonda ince kareleme ucuz,
siniflandirmada pahalidir. Model `releases/v3.2/classifier_64*` olarak
saklanmistir ama dagitim icin 256x256 varyanti onerilir.

### 11.3 Dagitim senaryolarinda bellek

Modeller ayri olculdugunde her biri ONNX Runtime'in taban ayak izini kendi
hesabina yazar. Gercek dagitimda ikisi ayni surecte calisir ve bu tabani
PAYLASIR. Asagidaki olcumler tek surecte, iki oturum birlikte yuklenerek
yapilmistir.

| senaryo                     | tepe bellek   |
|:----------------------------|:--------------|
| yalniz siniflandirici (256) | 22.14 MB      |
| yalniz U-Net (64)           | 14.75 MB      |
| ONERILEN: clf256 + unet64   | 27.78 MB      |
| alternatif: clf64 + unet64  | 20.83 MB      |

Ayri olculen degerlerin toplami 36.9 MB, birlikte olculen 27.78 MB -
aradaki ~9 MB paylasilan calisma zamani tabanidir.

### 11.4 Piyasa karsilastirmasi

| sistem | bellek | baglam |
|---|---|---|
| TFLite Micro cekirdek calisma zamani | ~16 KB | Cortex-M3 |
| TinyML MobileNet/ResNet (96x96 civari girdi) | <100 KB RAM | mikrodenetleyici |
| TinyML model + calisma zamani ikili | 100-375 KB | Nano 33 BLE ornegi |
| **bu calisma (ORT, clf256 + unet64)** | **27.78 MB** | masaustu x86 |
| bu calisma, agirlik + tepe aktivasyon (icsel) | ~14.3 MB | calisma zamani harici |
| Myriad 2 CMX SRAM (cip ustu, hizli) | 2 MB | Phi-Sat-1 |
| Myriad 2 LPDDR3 (cip disi) | 128 / 512 MB | Phi-Sat-1 |

**Okuma:**

1. **Mikrodenetleyici sinifinin cok uzagindayiz.** TinyML dagitimlari 100 KB'in
   altinda calisir; biz 27.78 MB kullaniyoruz - yaklasik 280 kat fazla. Ancak
   karsilastirma esdeger degildir: TinyML modelleri 96x96 gri tonlamali gibi
   girdilerle calisir, bizimki 6 bantli 256x256 goruntu isler.

2. **Asil fark modelde degil, calisma zamaninda.** Olculen 27.78 MB'in ~10.5 MB'i
   ONNX Runtime'in taban ayak izidir ve hicbir ayarla dusmez (arena ve bellek
   deseni kapatma denendi). TFLite Micro'nun cekirdegi 16 KB - yaklasik 650 kat
   kucuk. Ayni modeller gomulu bir motorla calistirilsa icsel gereksinim
   ~14.3 MB'e (agirlik 3.55 MB + tepe aktivasyon ~10.7 MB) inerdi.

3. **Referans misyon donanimina sigiyoruz.** Phi-Sat-1'in Myriad 2 VPU'su
   128-512 MB LPDDR3 tasir; 27.78 MB bunun %5-22'sidir. Ancak cip ustu 2 MB
   CMX SRAM'e sigmaz, yani model cip disi bellekten calisirdi - daha yavas ve
   daha cok guc tuketen bir yerlesim.

4. **Kalan tek anlamli kaldirac calisma zamanidir.** Model tarafinda kareleme
   ile bellegi %44 dusurduk (26.47 -> 14.87 MB, U-Net) ve daha fazlasi icin yer
   kalmadi. 2 MB SRAM hedefine yaklasmak icin gomulu bir cikarim motoru ve
   katman katman akis (streaming) gerekir; bu, yazilim simulasyonu kapsamini
   asar ve gelecek calisma olarak birakilmistir.

## 12. Kisitlar ve iyilestirme onerileri

### 12.1 Bilinen kisitlar

1. **Bulut esigi (%30) gerekcelendirilmedi.** Bu deger hem etiketleme
   kuralini hem karar kuralini belirliyor ve tum sonuclari kaydiriyor. Referans
   misyon CloudScout ayni karari %70 esikle veriyor. Duyarlilik analizi yapilmadi.
2. **Boyut ve sure hedefleri (5 MB / 100 ms) proje icinde secildi**, PDF bir sayi
   vermiyor. Referans misyondan turetilmeleri daha savunulabilir olurdu.
3. **Tek tohumlu kosular.** Modeller arasi kucuk farklar icin istatistiksel
   dayanak yok. Ozellikle v1->v2 siniflandirici farki (ROC-AUC +0.007) bu
   uyarinin kapsamindadir; U-Net ve calisma noktasi kazanclari ise gozlenen
   degiskenlikten belirgin sekilde buyuktur.
3b. **Esik veriden turetilemedi.** Alti ayri yol denendi ve hicbiri kararli
   bir dogal esik vermedi: bulut turu etiketleri (sahne bazli, dislayici degil,
   ve yorungede mevcut degil), minimum bulut boyutu (1 piksele iniyor),
   dagilim yapisi (uclar disinda duz), bagli bilesenler (toplam alanla 0.97
   korelasyonlu), kontur doluluk orani (etiketlerin %0.07'sini degistiriyor)
   ve uc degerleri disliyarak dagilim aramasi (vadi konumu kutu sayisi ve
   ornekleme tohumuyla savruluyor; cekirdek yogunluk tahmininde vadi derinligi
   1.04x - istatistiksel olarak anlamsiz). Esik bu nedenle operasyonel bir
   politika parametresi olarak birakilmistir.
4. **Precision kisiti test setine tasinmiyor.** Dogrulama setinde 0.99 hedefiyle
   secilen esik, daha zor olan test setinde daha dusuk precision veriyor.
5. **SPARCS karsilastirmasi kotumser.** Landsat-8 icin gunes yuksekligi duzeltmesi
   uygulanamadi (MTL dosyalari arsivde yok); olculen reflektans kaymasinin bir
   kismi model basarisizligi degil radyometrik uyumsuzluktur.
6. **CloudScout ile dogrudan karsilastirma gecerli degildir:** farkli veri seti,
   farkli gorev esigi (%70), farkli donanim sinifi (1.8 W uzay VPU'su vs masaustu
   CPU) ve farkli girdi boyutu. Boyut olarak ayni sinifta olundugu soylenebilir,
   dogruluk veya hiz ustunlugu iddia edilemez.

### 12.2 Iyilestirme onerileri

1. Bulut esigi icin duyarlilik analizi (%10/20/30/50/70) ve operasyonel gerekce.
2. Coklu tohumlu kosularla model secimini saglamlastirmak.
3. Kar/buz basarisizligi: kar agirlikli karelerle veri artirimi veya kar/buz
   ozel bir kayip agirligi.
4. Genislik carpani taramasi (mobilenetv2_050 / _075) ile boyut-dogruluk egrisi.
5. Kismi indirme senaryosu: U-Net maskesiyle bir goruntunun yalnizca temiz
   bolgelerini indirmek, ikili karardan daha yuksek kazanc saglayabilir.
6. Kuantizasyon-farkinda egitim (QAT) ile MobileNetV3'un da kullanilabilir hale
   getirilmesi.

## 13. Surum matrisi: hangi olcutte hangisi

Tum surumler `releases/` altinda dondurulmustur. Siniflandirici v2'den
itibaren degismemistir (c6_tuned); asagidaki farklar U-Net tarafindandir.

| surum   | kare    | U-Net encoder   |    IoU | bellek   |   ms/sahne | precision   | temiz alan kaybi (%30)   |
|:--------|:--------|:----------------|-------:|:---------|-----------:|:------------|:-------------------------|
| v1      | 256x256 | mnv2_100        | 0.8663 | 27.27 MB |        129 | -           | -                        |
| v2      | 256x256 | mnv2_050        | 0.8855 | 26.69 MB |        129 | 0.9264      | %16.17                   |
| v3      | 256x256 | mnv2_050        | 0.8844 | 26.47 MB |        129 | 0.9531      | %16.17                   |
| v3.1    | 128x128 | mnv2_050        | 0.8824 | 17.82 MB |        138 | 0.9442      | %13.78                   |
| v3.2    | 64x64   | mnv2_050        | 0.8807 | 14.87 MB |        175 | 0.9771      | %8.13                    |

### 13.1 Olcut bazinda en iyiler

- **En yuksek dogruluk (IoU):** v2 (0.8855). v3 ile arasindaki 0.0011'lik fark
  gozlenen kosu-arasi degiskenligin icindedir; pratikte esittirler.
- **En kisa sahne suresi:** v1/v2/v3 (129 ms). Kareleme ne kadar inceyse sahne
  basina o kadar cok cikarim yapilir.
- **En dusuk bellek:** v3.2 (14.87 MB) - v3'e gore %44 az.
- **En yuksek precision:** v3.2 (0.9771). Ince kareleme modeli daha temkinli yapar.
- **En az bilimsel veri kaybi:** v3.2 (%8.13 temiz alan) - v3'un yarisi.
- **En dengeli:** v3.1 (128x128). Bellek 2.4 kat az, sure yalnizca %7 fazla,
  IoU maliyeti 0.0020.

### 13.2 Dagitim karari: v3.2

**v3.2 ana surum olarak secilmistir.** Gerekce:

1. Sure butcesi bol: 175 ms/sahne, referans misyon CloudScout'un 325 ms'inin
   yarisi. Kaybedilen 46 ms, kazanilan bellek ve veri korumasi karsisinda ucuz.
2. Bellek en kritik kisit: uydu uzerindeki calisma bellegi, disk alanindan cok
   daha darda. v3.2 bu eksende %44 kazandiriyor.
3. Bilimsel veri kaybi yariya iniyor (%16.17 -> %8.13). Yanlis eleme geri donusu
   olmayan bir maliyet oldugu icin bu, projenin oncelik siralamasinda en ustte.
4. Dogruluk maliyeti ihmal edilebilir: IoU -0.0037.

Sure kisiti daralirsa v3.1 (138 ms), bellek kisiti gevserse v3 (129 ms)
tercih edilebilir. Ucu de `releases/` altinda hazir durmaktadir.

## 14. Ilgili calismalar (literatur taramasi)

### 14.1 Referans misyon: Phi-Sat-1 / CloudScout

Bu calismanin dogrudan referansi, ESA'nin **Phi-Sat-1** misyonudur (Eylul 2020,
6U CubeSat, FSSCat ikilisinin parcasi). Yorungede derin ogrenme calistiran ilk
gosterim misyonudur: hiperspektral-termal HyperScout-2 kamerasindan gelen
goruntuleri **Intel Movidius Myriad 2** VPU uzerinde isleyen **CloudScout**
adli evrisimli sinir agi, bulutlu goruntuleri yere indirmeden eliyor.

Yayimlanan degerler:

| olcut | CloudScout |
|---|---|
| model ayak izi | 2.1 MB |
| dogruluk | %92 |
| cikarim suresi | 325 ms (sahne basina) |
| guc | 1.8 W |
| bant genisligi tasarrufu | ~%30 |
| donanim | Myriad 2 VPU, 2 MB CMX SRAM + 128/512 MB LPDDR3 |

Bu calisma ayni problemi yazilim ortaminda ele alir; hedef donanim uzerinde
calistirma PDF geregi kapsam disidir. CloudScout ile DOGRUDAN karsilastirma
gecerli degildir (farkli veri seti, farkli gorev esigi, farkli donanim sinifi),
ancak boyut ve tasarruf mertebeleri baglam saglar.

### 14.2 Veri setleri

**Sentinel-2 Cloud Mask Catalogue** (Francis, Mrziglod, Sidiropoulos; ESA PhiLab)
bu calismanin ana veri setidir. 2018 Sentinel-2 L1C arsivinden rastgele
orneklenen 513 alt-sahne, IRIS aracıyla yari-otomatik (dinamik Random Forest +
elle duzeltme) etiketlenmistir. Ayirt edici ozelligi, iki anotatorun bagimsiz
etiketledigi 60 sahne uzerinden **insan seviyesi referansi** sunmasidir
(CLOUD F1 %95.97, piksel dogrulugu %94.98). Bu calismada test bolumu bilerek
o 60 sahneden olusturulmustur.

**SPARCS** (Hughes & Kennedy, Remote Sensing 2019, 11(21):2591) 80 Landsat-8
alt-sahnesinden olusan elle etiketlenmis bir dogrulama setidir. Bu calismada
yalnizca harici dogrulama icin, egitime hic sokulmadan kullanilmistir.

**95-Cloud** yalnizca 4 bant (R,G,B,NIR) icerdigi ve cirrus/SWIR bantlari
bulunmadigi icin kapsam disi birakilmistir.

**CloudSEN12** (Nature Scientific Data, 2022) 49.400 kareyle bu alandaki en
genis veri setlerinden biridir ve sekiz bulut tespit algoritmasinin ciktisini
birlikte sunar. Bu calismada kullanilmamistir ancak alanin karsilastirma
zemini olarak dikkate degerdir.

### 14.3 Bulut tespit yontemleri

| yontem | tur | bildirilen basarim |
|---|---|---|
| Sen2Cor | esik/kural tabanli | IoU 0.4698 |
| Fmask | esik/kural tabanli | IoU 0.5713, F1 0.832 |
| CD-FM3SF | derin ogrenme | IoU 0.8363 |
| Swin-Unet | derin ogrenme (transformer) | F1 0.891 |
| UNetMobV2 | derin ogrenme | CloudSEN12'de en iyi |
| **bu calisma (U-Net, INT8)** | **derin ogrenme** | **IoU 0.8807, Dice 0.9392** |

Rakamlar FARKLI VERI SETLERINDEN gelmektedir; dogrudan karsilastirilamazlar.
Gecerli tek karsilastirma, ayni veri setindeki insan seviyesi referansidir:
bu calismanin Dice degeri 0.9392, anotatorler arasi uyum 0.9597 - fark 2.05
puandir.

### 14.4 Hafif mimariler ve kuantizasyon

MobileNet ailesi (derinlemesine-ayrilabilir konvolusyonlar) ucta cikarim icin
standart secimdir ve PDF de bu aileyi isaret eder. Ancak **MobileNetV3'un
egitim sonrasi kuantizasyona (PTQ) direncli oldugu** bu calismada deneysel
olarak dogrulanmistir: yedi farkli yapilandirmanin hicbiri dogrulugu
koruyamamis, ayni yapilandirmalar MobileNetV2'de sorunsuz calismistir
(bkz. bolum 7).

**TensorFlow Lite Micro** (arXiv:2010.08678) gomulu cikarim icin referans
cercevedir; cekirdek calisma zamani ~16 KB'dir ve TinyML modelleri <100 KB RAM
ile calisabilmektedir. Bu calismada kullanilan ONNX Runtime'in taban ayak izi
~10.5 MB olcumustur - yaklasik 650 kat buyuk. Bu fark, bolum 11.4'teki bellek
karsilastirmasinin ana aciklamasidir.

### 14.5 Kaynaklar

- Giuffrida ve ark., *CloudScout: A Deep Neural Network for On-Board Cloud
  Detection on Hyperspectral Images*, Remote Sensing, 2020
- ESA, *Phi-Sat-1 Mission*, eoPortal
- Francis, Mrziglod, Sidiropoulos, *Sentinel-2 Cloud Mask Catalogue*, Zenodo
  (record 4172871)
- Hughes & Kennedy, *High Quality Cloud Masking of Landsat 8 Imagery Using
  Convolutional Neural Networks*, Remote Sensing 11(21):2591, 2019
- Aybar ve ark., *CloudSEN12*, Nature Scientific Data, 2022
- David ve ark., *TensorFlow Lite Micro: Embedded Machine Learning on TinyML
  Systems*, arXiv:2010.08678
