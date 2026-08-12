# Uydu Üzerinde Bulut Tespiti ve Downlink Optimizasyonu

Sentinel-2 görüntülerindeki bulutlu kareleri **uydu üzerinde**, yere indirilmeden
önce tespit edip eleyen hafif bir model hattı. Amaç sınırlı haberleşme bant
genişliğini boşa harcamamak; kısıt ise uydu bilgisayarının belleği ve işlem gücü.

Referans misyon: ESA Φ-sat-1 / CloudScout. Çalışma tamamen yazılım ortamında
yürütülmüştür; hedef donanım üzerinde çalıştırma kapsam dışıdır.

---

## Özet

İki model üretildi ve ikisi birlikte dağıtılmak üzere tasarlandı:

| | İkili sınıflandırıcı | U-Net segmentasyon |
|---|---|---|
| Omurga | `mobilenetv2_100` | `mobilenetv2_050` encoder + separable decoder |
| Girdi | 6 × 256 × 256 | 6 × 64 × 64 |
| Parametre | 2.23 M | 540.249 |
| INT8 disk | 2.59 MB | 0.96 MB |
| Çalışma belleği | 22.12 MB | 14.87 MB |
| Süre (kare) | 5.76 ms | 0.685 ms |
| Süre (sahne) | 92 ms | 175 ms |
| Ne üretir | Tek karar (indir / ele) | Piksel maskesi + bulut oranı |

**Önerilen dağıtım: yalnız U-Net (v3.2).** 0.96 MB disk, 14.75 MB tepe bellek,
175 ms/sahne. Karar kalitesi sınıflandırıcıyla başa baş (%2.38 yanlış eleme vs
%3.09), üstelik piksel maskesi de üretiyor.

İkili sınıflandırıcı, proje yönergesinde **temel hedef** olarak zorunlu
tutulduğu için birlikte teslim edilmiştir. İkisi aynı süreçte yüklendiğinde tepe
bellek 27.78 MB olur (ayrı ölçümlerin toplamı 36.9 MB; aradaki ~9 MB paylaşılan
ONNX Runtime tabanı). Gerçek bir misyonda yalnız U-Net yeterlidir — bkz. bölüm 6.

Downlink kazancı: **%48.65 azalma, %0.475 kullanılabilir veri kaybı**
(seçilen çalışma noktasında; bkz. bölüm 5).

> Bir sahne 1022×1022 pikseldir: 256×256 karelemede 16 kare, 64×64 karelemede
> 256 kare. Aşağıdaki tablolarda kare başına ve sahne başına süreler ayrı
> verilmiştir — ikisini karıştırmak yanıltıcı sonuç verir.

---

## 1. Veri

### Sentinel-2 Cloud Mask Catalogue (ana veri seti)

513 alt-sahne, 1022×1022 piksel, 20 m çözünürlük, 13 bant.
Kullanılan bantlar: **B02, B03, B04, B08, B10 (cirrus), B11 (SWIR)**.

| Bölüm | Kullanılabilir (0) | Bulutlu (1) | Toplam | Sahne | Ort. zorluk |
|---|---|---|---|---|---|
| train | 2729 | 3431 | 6160 | 385 | 2.03 |
| val | 477 | 611 | 1088 | 68 | 2.00 |
| test | 421 | 539 | 960 | 60 | 2.65 |

**Bölümleme sahne bazlıdır, kare bazlı değil.** Aynı sahnenin kareleri birbirine
çok benzediği için kare bazlı bölme test setine sızıntı yaratır ve doğruluğu
yapay olarak yükseltir.

Test bölümü, veri setinin kendi önerisiyle CALIBRATION + VALIDATION sahnelerinden
oluşuyor. Bu sahneler için anotatörler arası uyum yayımlanmış (piksel doğruluğu
%94.98, CLOUD F1 %95.97), yani sonuçlar **insan seviyesiyle** karşılaştırılabilir.
Test setinin ortalama zorluğu eğitim setinden yüksektir — tasarım gereği.

Etiketleme kuralı: bir karedeki bulut pikseli oranı ≥ %30 ise "bulutlu".
Bulut gölgesi ikili görevde temiz sayıldı: gölgeli bir görüntü bilimsel olarak
hâlâ kullanılabilir, bulutla kaplanmış değil.

### SPARCS (harici doğrulama)

78 Landsat-8 sahnesi, 1248 kare. **Eğitimde kullanılmadı**, karar eşiği bu veri
üzerinde yeniden ayarlanmadı. Bant eşlemesi dalga boyu merkezlerine göre yapıldı
(B02→B2, B03→B3, B04→B4, B08→B5, B10→B9, B11→B6).

> Güneş yüksekliği düzeltmesi uygulanamadı (MTL dosyaları arşivde yok).
> Sentinel-2 L1C reflektansları güneş açısına göre normalize edilmiştir,
> dolayısıyla iki veri seti arasında sistematik bir kayma vardır. Bölüm 7'deki
> sonuçlar bu kaymaya rağmen elde edilmiştir.

### 95-Cloud (kapsam dışı)

Yalnızca 4 bant içeriyor (R, G, B, NIR); cirrus ve SWIR yok. Bu iki bant bulut
ayrımının en güçlü sinyalleri olduğundan 6 bantlı model bu veriyle beslenemez.

### Ön işleme

TOA reflektans → `clip(0, 1.2) / 1.2`.

1.0'a kırpmak yanlış olurdu: L1C reflektansları fiziksel olarak 1'i aşabilir ve
tam da ayırt etmemiz gereken sınıfta (parlak bulut tepeleri, kar/buz) aşar.
8 sahnede ölçüldü: piksellerin %0.141'i > 1.0, p99.9 = 1.04, maksimum 1.94.
Sınır 1.5 seçilirse verinin %99'u dinamik aralığın %56'sına sıkışır ve INT8'de
girdi çözünürlüğü düşer. 1.2, p99.9'u korurken aralığın %70'ini kullanır.

Normalizasyon `src/preprocess.py`'de tek noktada tutuluyor — eğitim ile çıkarım
farklı normalizasyon uygularsa doğruluk sessizce düşer.

---

## 2. Modeller

### 2.1 İkili sınıflandırıcı

`mobilenetv2_100`, `in_chans=6`, tek logit çıkışlı.

Tek logit + `BCEWithLogitsLoss` tercih edildi (2 sınıf + softmax yerine): karar
eşiğinin eğitim sonrasında serbestçe ayarlanabilmesi gerekiyordu. Bu projede
yanlış pozitifin maliyeti yanlış negatiften çok daha yüksek, dolayısıyla eşik bir
hiperparametre değil operasyonel politika parametresidir.

ImageNet ön-eğitimli ağırlıklar 3 kanal bekler; `timm` ilk konvolüsyonun
ağırlığını kanal boyunca uyarlar.

### 2.2 U-Net

Klasik U-Net (~31M parametre, ~124 MB) boyut kısıtını kat kat aşıyor. Bunun
yerine aynı MobileNetV2 ailesinden `mobilenetv2_050` encoder + derinlemesine-
ayrılabilir konvolüsyonlardan oluşan ince bir decoder.

Decoder'da `ConvTranspose` yerine bilinear upsample: parametre yok, checkerboard
artefaktı yok, ONNX/INT8 tarafında sorunsuz.

Kayıp: `0.3 × BCE + 0.7 × soft Dice`. Bulut maskeleri dengesizdir; tek başına BCE
bulut oranı düşük karelerde "hepsi temiz" diyerek düşük kayıp alır, Dice örtüşme
oranını doğrudan optimize ettiği için bu çöküşü engeller.

---

## 3. Kuantizasyon

ONNX Runtime **statik** QDQ kuantizasyonu.

`quantize_dynamic` sadece MatMul/Gemm katmanlarını INT8'e çevirir. MobileNet
ağırlığın neredeyse tamamını Conv katmanlarında taşır, dolayısıyla dinamik
kuantizasyon ne boyut ne hız kazancı verir.

| Ayar | Değer | Gerekçe |
|---|---|---|
| Ağırlık | `QInt8`, **per-channel** | Zorunlu. BN katlaması sonrası kanal bazlı aralık mertebelerce değişiyor; tensör başına tek ölçek modeli çökertti (ölçüldü: F1 0.928 → 0.325) |
| Aktivasyon | `QUInt8` | x86'da işaretli int8'in yerel desteği yok; ölçüldü: s8 26.1 ms → u8 10.8 ms (2.4x) |
| Kalibrasyon | Percentile, 100 örnek | Aralığı %99.999 yüzdeliğe göre keser; tek bir kar/buz parlaması minmax'te tüm ölçeği bozuyor |

Kuantizasyon öncesi `quant_pre_process` (şekil çıkarımı) çağrılıyor. Atlanırsa
bazı Conv katmanları kuantize edilemez ve **sessizce FP32 kalır** — dosya küçülür,
hız gelmez.

ONNX export'u statik şekillidir (batch=1, sabit H×W). Uydu üzerinde çıkarım zaten
kare kare yapılıyor; ayrıca dinamik eksenlerle ORT'nin şekil çıkarımı
tamamlanamıyor ve statik kuantizasyon "Incomplete symbolic shape inference" ile
başarısız oluyor. **Sonuç: her kare boyutu ayrı bir modeldir.**

> INT8 ilk denemede başarısız oldu ve kök neden sistematik eleme ile arandı
> (op tipi bazlı izolasyon → katman bazlı kademeli kuantizasyon → ikili arama).
> Sonuç: MobileNetV3'ün hard-swish + SE blokları statik kuantizasyonda çöküyor,
> MobileNetV2 aynı taramayı sorunsuz geçiyor. Omurga seçimi bu ölçüme dayanır.
> Detay: teknik raporun 7. bölümü.

---

## 4. Ölçülen sonuçlar

Tüm süreler ONNX Runtime CPU, **tek iş parçacığı** (`intra_op_num_threads=1`).
Thread sayısı sabitlenmezse ölçüm makinenin çekirdek sayısına göre değişir ve
"standart CPU" hedefi anlamsızlaşır.

### 4.1 Sınıflandırıcı (test: 960 kare, eşik 0.5065)

| | PyTorch FP32 | ONNX FP32 | ONNX INT8 |
|---|---|---|---|
| Boyut | 8.73 MB | 8.66 MB | **2.59 MB** |
| Accuracy | 0.9563 | 0.9563 | 0.9552 |
| Precision | 0.9751 | 0.9751 | 0.9679 |
| Recall | 0.9462 | 0.9462 | 0.9518 |
| F1 | 0.9605 | 0.9605 | 0.9598 |
| ROC-AUC | 0.9883 | — | — |
| ms / kare (256²) | — | 9.77 | **5.76** |
| ms / sahne (16 kare, türetilmiş) | — | ~156 | **92** |

INT8 doğruluk kaybı 0.001 — pratikte bedelsiz 3.3× küçülme.

### 4.2 U-Net v3.2 (64×64 kareleme)

| | PyTorch FP32 | ONNX FP32 | ONNX INT8 |
|---|---|---|---|
| Boyut | 2.28 MB | 2.35 MB | **0.96 MB** |
| IoU | 0.8896 | 0.8892 | 0.8807 |
| Dice | 0.9415 | 0.9413 | 0.9366 |
| Precision | 0.9697 | 0.9704 | **0.9771** |
| Recall | 0.9150 | 0.9140 | 0.8993 |
| Piksel doğruluğu | 0.9412 | — | — |
| Görüntü-seviyesi doğruluk | 0.9480 | — | — |
| ms / kare (64²) | — | 1.04 | **0.72** |
| ms / sahne (256 kare, türetilmiş) | — | ~265 | ~185 |

> Süreler `benchmark.py`'nin kare başına ölçümüdür; sahne başına değerler kare
> sayısıyla çarpılarak türetilmiştir. Kareleme çalışmasında (bölüm 4.3) süreler
> üç model için döngüsel olarak yeniden ölçüldü ve INT8 için **0.685 ms/kare →
> 175 ms/sahne** bulundu. Bölüm 4.3'teki değerler, ölçüm koşulları eşitlendiği
> için sürümler arası karşılaştırmada esas alınmalıdır.

### 4.3 Kareleme çalışması

Aynı U-Net mimarisi üç kare boyutunda ayrı ayrı eğitildi. Karşılaştırma piksel
bazlı IoU üzerinden yapıldı; bu metrik aynı test piksellerini değerlendirdiği için
kareleme boyutundan bağımsızdır.

| Kare | Sahne başına | IoU | Precision | Recall | ms/kare | ms/sahne | Aktivasyon | Toplam bellek |
|---|---|---|---|---|---|---|---|---|
| 64² | 256 | 0.8807 | 0.9771 | 0.8993 | 0.685 | 175 | 4.08 MB | **14.87 MB** |
| 128² | 64 | 0.8824 | 0.9442 | 0.9310 | 2.156 | 138 | 6.68 MB | 17.82 MB |
| 256² | 16 | 0.8844 | 0.9531 | 0.9246 | 8.067 | 129 | 15.81 MB | 26.72 MB |

**Bulgular:**

1. **Bağlam kaybı beklenenden çok küçük.** Kare 16 kat küçüldüğünde IoU yalnızca
   0.0037 düşüyor. Bulut tespiti büyük ölçüde *yerel* bir görev: bir pikselin
   bulut olup olmadığı yakın komşuluğundan anlaşılıyor.
2. **Bellek alan oranıyla değil, daha yavaş ölçekleniyor.** Alan 16 kat küçülürken
   aktivasyon 3.9 kat azaldı — encoder'ın derin katmanlarındaki kanal sayıları ve
   çalışma zamanının sabit yükleri de paya giriyor.
3. **Kare boyutu precision/recall dengesini kaydırıyor.** Küçük kareler modeli
   daha temkinli yapıyor — bu projede istenen yön.

> **Ölçüm notu:** ilk süre ölçümleri farklı çalışma zincirlerinde alındığı için
> tutarsız çıktı. Süreler döngüsel olarak yeniden ölçüldü: her turda her modelden
> birer örnek alınarak termal sürüklenme ve arka plan yükü üç modele de eşit
> dağıtıldı. Tablodaki değerler bu temiz ölçümden gelmektedir.

### 4.4 Negatif sonuç: sınıflandırıcıda kareleme

U-Net'te işe yarayan ince kareleme, sınıflandırıcıya uygulandığında başarısız oldu.

| Ölçüt | 256×256 | 64×64 | Kazanan |
|---|---|---|---|
| S2CMC ROC-AUC | 0.9883 | 0.9872 | eşit |
| SPARCS ROC-AUC | 0.9644 | 0.8668 | 256² |
| SPARCS recall | 0.9760 | **0.0034** | 256² |
| ms / kare | 5.76 | 1.89 | 64² |
| ms / sahne | **92** | 484 | 256² |
| Bellek | 22.12 MB | 15.74 MB | 64² |

64×64 sınıflandırıcı farklı sensöre hiç genellemiyor: SPARCS'ta 1248 karenin
yalnızca birini eliyor. Bu sadece eşik yerleşimi sorunu değil — ROC-AUC da
düşüyor, yani sıralama kalitesi gerçekten bozuluyor.

**Neden U-Net'te olmuyor:** segmentasyon her pikselde yerel karar verir ve geniş
bağlama zaten az bağımlıdır. Sınıflandırma ise tüm kareyi tek sayıya indirir;
pencere daraldıkça bu özet ince dokusal istatistiklere dayanmak zorunda kalır ve
dokusal istatistikler sensörler arasında en çok değişen özelliklerdir.

Ayrıca kare başına hızlı olması yanıltıcı: sahne başına 5.3 kat yavaş.

**Sonuç: kareleme kararı göreve bağlıdır.**

---

## 5. Downlink kazancı ve çalışma noktası

Karar eşiği kodda sabit bir sayı değildir. Model, her biri ölçülmüş sonuçlarıyla
tanımlanmış birden fazla çalışma noktasıyla teslim edilir
(`operating_points.json`); hangisinin kullanılacağı bir dağıtım kararıdır, yerden
güncellenebilir ve yeniden eğitim gerektirmez.

### Neden eşik belirsiz bantta seçiliyor

Karelerin ~%73'ü ya tamamen temiz ya tamamen kapalıdır; bu kolay örneklerin kararı
eşikten bağımsızdır ve eşik seçimini asıl önemli olduğu bölgeden uzaklaştırır.
Eşiği yalnızca kısmi bulutlu karelerde (bulut oranı %2–%98) ayarlamak bu
çarpıtmayı kaldırır.

### Çalışma noktaları (sınıflandırıcı)

| Çalışma noktası | Eşik | Precision | F1 | Veri azalması | Kayıp | SPARCS kayıp | SPARCS kar yanlış eleme |
|---|---|---|---|---|---|---|---|
| Tüm val, precision ≥ 0.99 | 0.5065 | 0.9751 | 0.9605 | %54.48 | %3.088 | %19.98 | %58.62 |
| Belirsiz bant, precision ≥ 0.99 | 0.7181 | 0.9898 | 0.9437 | %51.15 | %1.188 | %12.34 | %43.68 |
| **Belirsiz bant, precision ≥ 0.995** | **0.7947** | **0.9957** | 0.9245 | **%48.65** | **%0.475** | **%7.95** | **%29.89** |
| Dengeli, en iyi F1 | 0.3608 | 0.9592 | 0.9592 | %56.15 | %5.226 | %29.08 | %68.97 |

**Dağıtım önerisi: belirsiz bant (precision ≥ 0.995).** Kullanılabilir verinin
%99.5'i korunur, bant genişliği kazancı %48.65 olur — teorik üst sınırın (%56.15)
%87'si. Model aynı modeldir; değişen yalnızca eşiğin nasıl seçildiğidir.

> Yanlış eleme (kullanılabilir görüntünün atılması) geri dönüşü olmayan bilimsel
> veri kaybıdır. Yanlış negatif ise yalnızca boşuna indirilen bir karedir. Bu
> asimetri nedeniyle eşik F1'i değil yüksek precision'ı hedefler.

### Yöntemin sınırı — her yapılandırmada işe yaramıyor

| Kare | Seçilen eşik | Precision | Yanlış eleme | Sonuç |
|---|---|---|---|---|
| 256×256 | 0.862 (yukarı) | 0.9974 | %5.94 → %0.24 | kazanç |
| 128×128 | 0.288 (aşağı) | 0.9515 | %5.84 → %6.02 | kazanç yok |
| 64×64 | 0.891 (yukarı) | 0.9916 | %1.99 → %0.79 | kazanç |

Sebep: yöntem `precision ≥ 0.99` kısıtını sağlayan en düşük eşiği arıyor.
128×128'de model o alt kümede zaten yeterince iyi olduğundan düşük bir eşik de
kısıtı sağlıyor ve arama orada duruyor. **Genel geçer kabul edilemez; her model ve
kare boyutu için ayrıca ölçülmelidir.** Kazanç görüldüğü yerlerde ise büyük
(yanlış elemede 2.5–25 kat azalma).

### U-Net'in karar performansı

U-Net maskesinden görüntü kararı da üretilebiliyor (`mask_to_decision`), böylece
iki model aynı fayda analizine sokulabiliyor:

| Karar kuralı | Accuracy | Precision | Recall | F1 | Yanlış eleme |
|---|---|---|---|---|---|
| Bulut oranı, sabit %30 eşik | 0.9476 | 0.9790 | 0.9237 | 0.9505 | %2.38 |
| Bulut oranı, val'de ayarlı (0.342) | 0.9430 | 0.9821 | 0.9121 | 0.9458 | %1.99 |
| **Belirsiz bantta ayarlı (0.891)** | 0.8750 | **0.9916** | 0.7772 | 0.8714 | **%0.79** |
| Şekil öznitelikleri (lojistik regresyon) | 0.9428 | 0.9811 | 0.9127 | 0.9457 | %2.10 |
| Kontrol: yalnızca bulut oranı | 0.9430 | 0.9821 | 0.9121 | 0.9458 | %1.99 |

**Negatif sonuç:** maskeden çıkarılan şekil öznitelikleri (solidity, bağlı bileşen
sayısı, kompaktlık, doluluk oranı) hiçbir katkı vermedi — F1 farkı 0.0001. Bulut
oranı tek başına aynı bilgiyi taşıyor. Hipotez ölçüldü ve elendi.

**İki modelin karşılaştırması:** ikili karar görevinde sınıflandırıcı bir miktar
önde (yüksek precision noktasında %0.475 kayıp, U-Net'te %0.79). U-Net'in
üstünlüğü karar kalitesinde değil, ürettiği çıktının biçimindedir — bkz. bölüm 6.

---

## 6. Kısmi indirme: U-Net'in asıl gerekçesi

İkili karar bir görüntüyü ya tamamen indirir ya tamamen atar. Ama atılan
görüntülerin içinde de temiz alan vardır ve bu alan geri dönüşsüz kaybedilir.

`threshold_analysis.csv` bu kaybı bulut eşiğine göre ölçüyor:

| Eşik | Elenen görüntü | **Kaybedilen temiz alan** | Kaybedilen kullanılabilir görüntü | Precision |
|---|---|---|---|---|
| %5 | %58.46 | %19.42 | %2.99 | 0.9806 |
| %10 | %56.51 | %15.92 | %2.87 | 0.9797 |
| %20 | %53.39 | %10.87 | %2.58 | 0.9790 |
| **%30** | **%51.42** | **%8.13** | **%2.38** | 0.9790 |
| %50 | %48.07 | %4.44 | %2.20 | 0.9777 |
| %70 | %45.48 | %2.62 | %2.50 | 0.9715 |

İki farklı kayıp metriği var ve karıştırılmamalı:

- **Kaybedilen kullanılabilir görüntü (%2.38):** tamamen temizken yanlışlıkla
  elenen kareler. İkili kararın *hatası*.
- **Kaybedilen temiz alan (%8.13):** doğru şekilde elenen bulutlu karelerin
  içindeki temiz pikseller. İkili kararın *yapısal maliyeti* — hata değil,
  yöntemin kendisinden kaynaklanan kayıp.

Bu ikincisi ikili sınıflandırıcıyla kapatılamaz; hangi bölgenin temiz olduğunu
bilmek gerekir. **U-Net tam olarak bunu üretiyor.** Maske sayesinde bir görüntünün
yalnızca temiz bölgelerini indirmek mümkün hale geliyor ve %8.13'lük yapısal
kaybın büyük kısmı kurtarılabilir.

Bu, U-Net'i projede tutmanın ana gerekçesidir: ikili karar kalitesinde
sınıflandırıcıyla başa baş, ama ikili kararın tavanını kaldırma potansiyeli
taşıyan tek çıktı biçimi.

**Henüz ölçülmedi.** Kısmi indirmenin gerçek kazancı, kare bazlı indirme
protokolünün ek yükü (kare koordinatları, sıkıştırma verimliliğinin düşmesi)
hesaba katılarak ölçülmelidir. Gelecek çalışma olarak bırakılmıştır.

---

## 6.1 Dağıtım önerisi: yalnız U-Net

Ölçümler tek başına U-Net'in yeterli olduğunu gösteriyor:

| | Yalnız U-Net | Yalnız sınıflandırıcı | İkisi birlikte |
|---|---|---|---|
| Disk | **0.96 MB** | 2.59 MB | 3.55 MB |
| Tepe bellek | **14.75 MB** | 22.14 MB | 27.78 MB |
| Süre / sahne | 175 ms | 92 ms | 267 ms |
| Yanlış eleme (dengeli nokta) | **%2.38** | %3.09 | — |
| Piksel maskesi | ✓ | ✗ | ✓ |

U-Net her boyut ekseninde daha ucuz, dengeli çalışma noktasında daha az
kullanılabilir veri kaybettiriyor ve kısmi indirmeye açılan tek çıktı biçimini
üretiyor. İkili sınıflandırıcı diskte daha büyük, çünkü MobileNetV2'nin 1280
boyutlu sınıflandırma başlığını taşıyor; U-Net'te o kısım yok.

**İkisi neden birlikte teslim edildi:** proje yönergesi ikili sınıflandırıcıyı
temel hedef olarak zorunlu kılıyor. Bu bir mühendislik tercihi değil, kapsam
kısıtıdır.

**Kademeli kurgu (sınıflandırıcı ön süzgeç, U-Net kalanlara) neden tercih
edilmedi:** sınıflandırıcının kendisi 92 ms tükettiği için kazandırdığı süreyi
geri alıyor. En iyi durumda %20 hız kazancına karşılık bellek 1.9 kat büyüyor.

**Tek modele geçiş için eksik iki ölçüm:**

1. **U-Net'in yüksek-precision davranışı.** Seçilen çalışma noktasında
   (precision ≥ 0.995) sınıflandırıcı %0.475 kayıpla çalışıyor; U-Net'in ölçülen
   en yakın noktası %0.79 kayıp ve recall 0.7772. U-Net için eşik taraması bu
   bölgede yapılmadı.
2. **U-Net'in SPARCS performansı.** Harici doğrulama yalnızca sınıflandırıcıya
   uygulandı. Segmentasyonun yerel karar verdiği için daha iyi genellemesi
   beklenir (bkz. 4.4), ancak bu sınanmamış bir varsayımdır.

Bu iki ölçüm tamamlanana kadar `clf256 + unet64` muhafazakâr alternatif olarak
korunmaktadır.

---

## 7. Harici doğrulama (Landsat-8 / SPARCS)

Farklı sensör, eğitimde hiç görülmemiş veri, eşik yeniden ayarlanmadı.

| Kırılım | n | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|---|
| Genel | 1248 | 0.8421 | 0.5996 | 0.9795 | 0.7438 | 0.9645 |
| Kar/buz olmayan | 1152 | 0.8733 | 0.6643 | 0.9788 | 0.7914 | 0.9731 |
| **Kar/buz ağırlıklı (>%20)** | **96** | **0.4688** | **0.1500** | **1.0000** | 0.2609 | 0.8646 |

### Bilinen başarısızlık modu: kar/buz

**Temiz kar/buz karelerinin %58.6'sı yanlışlıkla eleniyor.** Kar ve bulut görünür
+ NIR bantlarında neredeyse aynı spektral imzaya sahip; ayrım esas olarak SWIR
(B11) ve cirrus (B10) bantlarından gelmeli, ancak model bunu yeterince
öğrenememiş.

Etkisi asimetrik: recall 1.0 olduğu için hiçbir bulut kaçmıyor, ama kutup bölgesi
görüntülerinde bilimsel veri kaybı kabul edilemez seviyede.

Çalışma noktası seçimi bu kaybı belirgin şekilde azaltıyor: yanlış eleme
%58.62'den %29.89'a iniyor (bölüm 5). Model aynı model.

**Öneri:** NDSI (B03/B11 üzerinden kar indeksi) girdiye eklenmesi veya karlı
sahnelerin eğitim setinde ağırlıklandırılması.

---

## 8. Bellek profili

Disk boyutu, uydu üzerindeki bellek ihtiyacının küçük bir parçasıdır. Her model
**ayrı süreçte** ölçüldü; aksi halde önceki modelin ayırdığı bellek sonrakinin
ölçümüne karışır. Python yorumlayıcısının ~50 MB'lik tabanı hariç tutulmuştur
(uydu yazılımında bu maliyet olmaz).

| Senaryo | Tepe bellek |
|---|---|
| **ÖNERİLEN: yalnız U-Net (64²)** | **14.75 MB** |
| Yalnız sınıflandırıcı (256²) | 22.14 MB |
| Yönerge gereği teslim edilen: clf256 + unet64 | 27.78 MB |
| Alternatif: clf64 + unet64 | 20.83 MB |

Ayrı ölçülen değerlerin toplamı 36.9 MB, birlikte ölçülen 27.78 MB — aradaki
~9 MB paylaşılan çalışma zamanı tabanıdır.

**Gerçek bellek ihtiyacı disk boyutunun 10–25 katıdır.** INT8 sınıflandırıcı
diskte 2.59 MB, çalışırken 22.1 MB. Yalnızca dosya boyutuna bakarak bellek kısıtı
değerlendirmek yanıltıcı olur.

**U-Net diskte küçük, bellekte pahalıdır.** 256² girdide sınıflandırıcıdan diskte
2.7 kat küçük ama bellekte yaklaşık %20 daha pahalı (26.7 MB vs 22.1 MB) —
segmentasyon decoder'ı tam çözünürlükte ara tensörler taşır, sınıflandırıcı ise
havuzlama ile hızla küçülür. 64×64 karelemenin asıl kazancı burada: bu
dezavantajı tersine çeviriyor.

> Bellek ölçümleri koşudan koşuya ±0.2 MB oynuyor (sınıflandırıcı için 22.12 ve
> 22.14 MB olarak iki kez ölçüldü). Bu düzeydeki farklar anlamlı değildir;
> yukarıdaki değerler bir ondalığa yuvarlanmıştır.

> **Çekince:** ölçülen bellek ONNX Runtime'ın önceden ayırdığı havuzu içerir ve
> ~10.5 MB'ı hiçbir ayarla düşmeyen ORT tabanıdır (arena ve bellek deseni kapatma
> denendi). Gömülü bir motorla (TFLite Micro çekirdeği ~16 KB) içsel gereksinim
> ~14.3 MB'e inerdi. Bu rakamlar ORT'ye özgü **üst sınırlardır**.

Φ-sat-1'in Myriad 2 VPU'su 128–512 MB LPDDR3 taşır; 27.78 MB bunun %5–22'sidir.
Ancak çip üstü 2 MB CMX SRAM'e sığmaz — model çip dışı bellekten çalışırdı.

---

## 9. Sürüm matrisi

Sınıflandırıcı v2'den itibaren değişmemiştir (`c6_tuned`); aşağıdaki farklar
U-Net tarafındandır.

| Sürüm | Kare | Encoder | IoU | Bellek | ms/sahne | Precision | Temiz alan kaybı (%30) |
|---|---|---|---|---|---|---|---|
| v1 | 256² | `mobilenetv2_100` | 0.8663 | 27.27 MB | 129 | — | — |
| v2 | 256² | `mobilenetv2_050` | 0.8855 | 26.69 MB | 129 | 0.9264 | %16.17 |
| v3 | 256² | `mobilenetv2_050` | 0.8844 | 26.47 MB | 129 | 0.9531 | %16.17 |
| v3.1 | 128² | `mobilenetv2_050` | 0.8824 | 17.82 MB | 138 | 0.9442 | %13.78 |
| **v3.2** | **64²** | **`mobilenetv2_050`** | **0.8807** | **14.87 MB** | **175** | **0.9771** | **%8.13** |

### Dağıtım kararı: v3.2

1. **Süre bütçesi bol:** 175 ms/sahne, referans misyonun 325 ms'inin yarısı.
   Kaybedilen 46 ms, kazanılan bellek ve veri koruması karşısında ucuz.
2. **Bellek en kritik kısıt:** v3.2 bu eksende %44 kazandırıyor.
3. **Bilimsel veri kaybı yarıya iniyor** (%16.17 → %8.13). Yanlış eleme geri
   dönüşü olmayan bir maliyet olduğu için bu, öncelik sıralamasında en üstte.
4. **Doğruluk maliyeti ihmal edilebilir:** IoU −0.0037, gözlenen koşu-arası
   değişkenliğin içinde.

Süre kısıtı daralırsa v3.1 (138 ms), bellek kısıtı gevşerse v3 (129 ms) tercih
edilebilir. Üçü de `releases/` altında hazır durmaktadır.

---

## 10. Bilinen kısıtlar

1. **Bulut eşiği (%30) gerekçelendirilmedi.** Hem etiketleme hem karar kuralını
   belirliyor ve tüm sonuçları kaydırıyor. Referans misyon CloudScout aynı kararı
   %70 eşikle veriyor. Duyarlılık analizi yapılmadı.
2. **Boyut ve süre hedefleri (5 MB / 100 ms) proje içinde seçildi**; referans
   misyondan türetilmeleri daha savunulabilir olurdu.
3. **Tek tohumlu koşular.** Modeller arası küçük farklar için istatistiksel dayanak
   yok. U-Net ve çalışma noktası kazançları gözlenen değişkenlikten belirgin
   şekilde büyüktür; sınıflandırıcı v1→v2 farkı (ROC-AUC +0.007) değildir.
4. **Eşik veriden türetilemedi.** Altı ayrı yol denendi (bulut türü etiketleri,
   minimum bulut boyutu, dağılım yapısı, bağlı bileşenler, kontur doluluk oranı,
   uç değer dışlayarak dağılım araması) ve hiçbiri kararlı bir doğal eşik vermedi.
   Eşik bu nedenle operasyonel bir politika parametresi olarak bırakılmıştır.
5. **Precision kısıtı test setine taşınmıyor.** Validasyonda 0.99 hedefiyle seçilen
   eşik, daha zor olan test setinde daha düşük precision veriyor.
6. **SPARCS karşılaştırması kötümser** (güneş yüksekliği düzeltmesi yok);
   ölçülen kaymanın bir kısmı model başarısızlığı değil radyometrik uyumsuzluktur.
7. **CloudScout ile doğrudan karşılaştırma geçerli değildir:** farklı veri seti,
   farklı görev eşiği (%70), farklı donanım sınıfı (1.8 W uzay VPU'su vs masaüstü
   CPU). Boyut olarak aynı sınıfta olunduğu söylenebilir; doğruluk veya hız
   üstünlüğü iddia edilemez.

### İyileştirme önerileri

1. Bulut eşiği için duyarlılık analizi (%10/20/30/50/70) ve operasyonel gerekçe
2. Çoklu tohumlu koşularla model seçimini sağlamlaştırmak
3. Kar/buz başarısızlığı: kar ağırlıklı veri artırımı veya özel kayıp ağırlığı
4. Genişlik çarpanı taraması (`mobilenetv2_050` / `_075`) ile boyut-doğruluk eğrisi
5. **Kısmi indirme senaryosunun ölçülmesi** (bölüm 6)
6. Kuantizasyon-farkında eğitim (QAT) ile MobileNetV3'ün kullanılabilir kılınması

---

## 11. Proje yapısı

```
src/
  config.py               Merkezi yapılandırma (ortam değişkenleriyle override edilebilir)
  preprocess.py           Reflektans normalizasyonu (eğitim + çıkarım ortak)
  prepare_data.py         Sentinel-2 Cloud Mask Catalogue -> kare + index.csv
  prepare_sparcs.py       SPARCS (Landsat-8) -> kare, bant eşlemesiyle
  dataset.py              Dataset / DataLoader, sınıf dengeleme
  model.py                MobileNetV2 sınıflandırıcı
  unet.py                 Hafif U-Net + maske -> karar köprüsü
  losses.py               Dice + BCE, segmentasyon metrikleri
  train.py                Sınıflandırıcı eğitimi (EMA, warmup, eşik ayarı)
  train_seg.py            U-Net eğitimi (AMP, gradyan biriktirme)
  hparam_sweep.py         Hiperparametre taraması
  distill.py              Bilgi damıtma denemesi
  export.py               ONNX export + statik INT8 kuantizasyon
  benchmark.py            Doğruluk / boyut / süre tablosu + downlink analizi
  memory_profile.py       Ayrı süreçte tepe bellek ölçümü
  quantize_analysis.py    Kuantizasyon hata analizi
  quantize_layerwise.py   Katman bazlı duyarlılık
  quantize_bisect.py      Bozulan katmanı ikili aramayla bulma
  threshold_analysis.py   Bulut eşiği taraması
  contour_threshold.py    Kontur tabanlı maske sonrası işleme
  decision_layer.py       Şekil özniteliklerinden karar üretme (hipotez testi)
  tune_operating_point.py Belirsiz bantta eşik seçimi -> operating_points.json
  evaluate_external.py    SPARCS harici doğrulama
  tag_analysis.py         Sürümler arası karşılaştırma
  make_report.py          Teknik rapor üretimi
  snapshot_release.py     Sürüm dondurma -> releases/<v>/ + MANIFEST.json
smoke_test.py             Sentetik veriyle uçtan uca hat testi
reports/                  Yayınlanan sürümün ölçüm çıktıları (CSV / JSON)
teslim/                   Teknik rapor, sunum, demo görseli
```

`reports/` klasörü, README ve teknik rapordaki tüm sayıların kaynağıdır —
benchmark tabloları, çalışma noktaları, bellek profili, eşik taraması ve harici
doğrulama sonuçları. Kod çalışırken bu dosyaları `outputs/reports/` altında
üretir; depoda referans olarak kök dizinde tutulmaktadır.

Dondurulmuş `.onnx` / `.pt` ağırlıkları depoyu hafif tutmak için **Releases**
bölümünde `v*.zip` olarak paketlenmiştir. Her sürüm kendi checkpoint'leri, ONNX
dosyaları, config kopyası ve ölçülmüş sayılarıyla dondurulur — sonraki deneyler
çalışan sürümü bozmasın diye.

---

## 12. Çalıştırma

### Kurulum

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

### Hızlı sistem kontrolü

```bash
python smoke_test.py
```

Sentetik veriyle eğitim → ONNX → INT8 → benchmark hattını uçtan uca çalıştırır.
Gerçek veri veya indirilmiş ağırlık gerektirmez. Amaç doğruluk değil, hattın kırık
olmadığını göstermek.

### Yayınlanan modellerle benchmark

Releases'tan `v3.2.zip` indirilip proje köküne açılır. Benchmark betiği dosyaları
`outputs/onnx/<tag>_{fp32,int8}.onnx` adıyla arar, release paketindeki adlar
farklıdır — önce kopyalayın:

```bash
mkdir -p outputs/onnx outputs/reports

cp releases/v3.2/unet_fp32.onnx   outputs/onnx/v3.2_fp32.onnx
cp releases/v3.2/unet_int8.onnx   outputs/onnx/v3.2_int8.onnx

cp releases/v3.2/classifier_fp32.onnx          outputs/onnx/c6_tuned_fp32.onnx
cp releases/v3.2/classifier_int8.onnx          outputs/onnx/c6_tuned_int8.onnx
cp releases/v3.2/classifier_train_summary.json outputs/reports/c6_tuned_train_summary.json
```

> Aynı dosyalar depodaki `reports/` klasöründe de duruyor; release paketini
> indirmeden yalnızca eşiği görmek isterseniz
> `cp reports/c6_tuned_train_summary.json outputs/reports/` yeterlidir.

**Sınıflandırıcı** (256×256, varsayılan kare boyutu):

```bash
python -m src.benchmark --tag c6_tuned
```

Eşik `c6_tuned_train_summary.json` içinden okunur (0.5065).

**U-Net v3.2** (64×64):

```bash
CLOUD_PATCH_SIZE=64 CLOUD_PATCH_DIR=data/patches_t64 \
  python -m src.benchmark --tag v3.2 --task segmentation
```

> `CLOUD_PATCH_SIZE=64` **zorunludur.** Modeller statik şekille dışa aktarıldığı
> için her kare boyutu ayrı bir modeldir; varsayılan 256 ile çalıştırılırsa ONNX
> Runtime girdi şekli hatası verir. `--threshold` verilmezse piksel eşiği 0.5
> kullanılır — yayınlanan IoU bu eşikte ölçülmüştür.

**v3.1** (128×128): `CLOUD_PATCH_SIZE=128 CLOUD_PATCH_DIR=data/patches_t128`

Windows PowerShell'de:

```powershell
$env:CLOUD_PATCH_SIZE=64; $env:CLOUD_PATCH_DIR="data/patches_t64"
```

### Sıfırdan eğitim

```bash
# Veri hazırlama (256x256 ve 64x64 ayrı setler)
python -m src.prepare_data --source <s2cmc_dizini> --with-masks
python -m src.prepare_data --source <s2cmc_dizini> --with-masks \
  --patch-size 64 --out data/patches_t64

# Sınıflandırıcı
python -m src.train --model mobilenetv2_100 --tag c6_tuned \
  --lr 0.00119 --weight-decay 5.7e-05 --batch-size 64 \
  --label-smoothing 0.1 --balanced --ema

# U-Net (64x64)
CLOUD_PATCH_SIZE=64 CLOUD_PATCH_DIR=data/patches_t64 \
  python -m src.train_seg --encoder mobilenetv2_050 --tag unet_t64 \
  --lr 0.001128 --weight-decay 0.001394 --batch-size 32 --bce-weight 0.3

# Export + analizler
python -m src.export --checkpoint outputs/checkpoints/c6_tuned_best.pt --tag c6_tuned
python -m src.tune_operating_point --checkpoint outputs/checkpoints/c6_tuned_best.pt --tag c6_tuned
python -m src.evaluate_external --checkpoint outputs/checkpoints/c6_tuned_best.pt --tag c6_tuned
python -m src.memory_profile

# Sürüm dondurma
python -m src.snapshot_release --version v3.2 \
  --classifier-tag c6_tuned --unet-tag unet_t64 --note "..."
```

---

## 13. Yapılandırma

Kodu değiştirmeden ortam değişkenleriyle varyant çalıştırılabilir:

| Değişken | Varsayılan | Açıklama |
|---|---|---|
| `CLOUD_BANDS` | B02,B03,B04,B08,B10,B11 | `all` = 13 bandın tümü |
| `CLOUD_PATCH_DIR` | `data/patches` | Kare dizini |
| `CLOUD_PATCH_SIZE` | 256 | Kare boyutu |
| `CLOUD_THRESHOLD` | 0.30 | Bulut pikseli oranı eşiği (görüntü etiketine geçiş) |
| `CLOUD_RELEASE` | v3.2 | Aktif sürüm |

Bant sayısı model boyutunu neredeyse hiç etkilemiyor (ölçüldü: 6 bant 2.226.017
parametre, 13 bant 2.228.033 — fark %0.09). Yalnızca ilk konvolüsyonun girdi
kanalı değişiyor.

---

## 14. Teslim dosyaları

- `teslim/TEKNIK_RAPOR.md` / `.docx` — detaylı analiz raporu (kuantizasyon kök
  neden analizi, ablasyon çalışmaları, literatür taraması dahil)
- `teslim/SUNUM.pptx` — sunum
- `teslim/demo_ornek.png` — örnek çıktı
