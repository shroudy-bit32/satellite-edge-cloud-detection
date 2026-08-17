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
| Çalışma belleği | 22.14 MB | 14.87 MB |
| Süre (kare) | 5.76 ms | 0.685 ms |
| Süre (sahne) | 92 ms | 175 ms |
| Ne üretir | Tek karar (indir / ele) | Piksel maskesi + bulut oranı |

**Dağıtılan yapılandırma: `clf256 + unet64` (v3.2).** İkisi aynı süreçte
yüklendiğinde tepe bellek 27.78 MB olur (ayrı ölçümlerin toplamı 36.9 MB;
aradaki ~9 MB paylaşılan ONNX Runtime tabanı). İkili sınıflandırıcı proje
yönergesinde **temel hedef** olarak zorunlu tutulmuştur; U-Net genişletilmiş
hedeftir.

Downlink kazancı: **%48.65 azalma, %0.475 kullanılabilir veri kaybı** —
sınıflandırıcının seçilen çalışma noktasında (eşik 0.7947; bkz. bölüm 5).

**İki model iki farklı rejimde üstün — bu yüzden ikisi birlikte dağıtılıyor.**
Sınıflandırıcı eğitim dağılımında daha fazla bant genişliği kazandırıyor
(%48.65 vs %42.72, eşleşmiş precision noktasında). U-Net farklı sensörde çok
daha dayanıklı: SPARCS'ta 17 kat az kullanılabilir veri kaybı (%0.464 vs %7.95),
42 kat az kar/buz yanlış elemesi (%0.71 vs %29.89). Sensör değiştiğinde
sınıflandırıcının ROC-AUC'si 0.0238 düşerken U-Net'inki düşmüyor. Ayrıntı:
bölüm 6.2 ve 7.2.

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

> **Güneş yüksekliği düzeltmesi:** yayımlanan SPARCS sonuçları bu düzeltme
> **olmadan** ölçülmüştür. Sentinel-2 L1C reflektansları güneş açısına göre
> normalize edilmiştir, Landsat-8 DN'leri değildir; dolayısıyla iki veri seti
> arasında sistematik bir kayma vardır ve bölüm 7'deki sonuçlar bu kaymaya
> rağmen elde edilmiştir.
>
> Düzeltme artık **uygulanabilir durumdadır:** USGS'in `l8cloudmasks.zip`
> arşivi 80 sahnenin hepsi için `*_mtl.txt` dosyasını içeriyor ve
> `prepare_sparcs.py --sun-correction` bunu okuyup `1/sin(SUN_ELEVATION)` ile
> ölçekliyor. Arşivdeki güneş yüksekliği 30.1°–69.4° arasında değişiyor, yani
> düzeltme çarpanı **1.07–2.00** — bazı sahnelerde iki kat. Bu, kaymanın ihmal
> edilebilir olmadığını gösteriyor; düzeltmeli/düzeltmesiz karşılaştırma
> yapılmamıştır.

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
| Bellek | 22.14 MB | 15.74 MB | 64² |

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
iki model aynı ikili karar görevinde karşılaştırılabiliyor:

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

**İki modelin karşılaştırması:** ikili karar görevinde sınıflandırıcı önde
(yüksek precision noktasında %0.475 kayıp ve recall 0.8627; U-Net'te %0.79 kayıp
ve recall 0.7772). U-Net'in üstünlüğü karar kalitesinde değil, ürettiği çıktının
biçimindedir — bkz. bölüm 6.

> **Eksik:** bu tabloda `veri_azalmasi_%` kolonu yok. Planın istediği fayda
> metriği (indirilen veri hacminde %X azalma) U-Net için hesaplanmadı; yalnızca
> sınıflandırıcı için ölçüldü (bölüm 5, çalışma noktaları tablosu).

---

## 6. Kısmi indirme ve tek model sorusu

İkili karar bir görüntüyü ya tamamen indirir ya tamamen atar. Ama atılan
görüntülerin içinde de temiz alan vardır ve bu alan geri dönüşsüz kaybedilir.

`outputs/reports/unet_t64_threshold_analysis.csv` bu kaybı bulut eşiğine göre
ölçüyor:

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
bilmek gerekir. **U-Net tam olarak bunu üretiyor** — maske sayesinde görüntünün
yalnızca temiz bölgelerini indirmek mümkün hale geliyor.

### 6.1 Ölçüm: kısmi indirme ödünleşimi

Kareyi daha küçük bloklara bölüp yalnızca tahmin edilen maskeye göre temiz
blokları indirmek. Karar **tahmin edilen** maskeden verilir (uyduda olan bilgi
budur), kayıp muhasebesi **gerçek** maskeden yapılır. Her blok bağımsız bir
iletim birimi olarak sıkıştırılır; blok haritası (kare başına bit maskesi) da
maliyete eklenir. `blok = 64×64` satırı mevcut ikili davranıştır.

Test bölümü, 15.360 kare (`src/partial_downlink.py`):

| Blok | İndirilen alan | **Kaybedilen temiz alan** | **Bant genişliği kazancı (bayt)** | Bayt (referansa göre) |
|---|---|---|---|---|
| **64×64** (mevcut) | %48.58 | **%8.130** | **%52.50** | 1.000 |
| 32×32 | %49.17 | %6.517 | %50.28 | 1.047 |
| 16×16 | %49.82 | %5.096 | %46.51 | 1.126 |
| 8×8 | %50.46 | **%4.001** | **%38.97** | 1.285 |

**Negatif sonuç: ödünleşim elverişsiz.** Bloğu 8×8'e kadar küçültmek yapısal
kaybın yarısını kurtarıyor (%8.13 → %4.00) ama bant genişliği kazancını
13.5 puan düşürüyor (%52.50 → %38.97). Marjinal oran her adımda kötüleşiyor:

| Adım | Kurtarılan temiz alan | Kaybedilen bant genişliği kazancı | Oran |
|---|---|---|---|
| 64 → 32 | +1.61 puan | −2.22 puan | 1.38 : 1 |
| 64 → 16 | +3.03 puan | −5.99 puan | 1.97 : 1 |
| 64 → 8 | +4.13 puan | −13.53 puan | **3.28 : 1** |

Referans yapılandırmada korunan temiz alanın puanı 2.46 MB'a mal oluyor; 8×8'e
inerken ek puanlar **15.59 MB**'a, yani 6.3 kat pahalıya geliyor.

**Neden:** iki etken ters yönde çalışıyor.

1. **Kurtarılan alan küçük.** İndirilen alan yalnızca %48.58'den %50.46'ya
   çıkıyor. Sebep, U-Net maskelerinin *mekansal olarak tutarlı* olması: %30'dan
   fazla bulutlu bir 64×64 kare genellikle büyük ölçüde bulutludur, satranç
   tahtası değil. İnce bölme kurtarılacak fazla temiz ada bulamıyor.
2. **Parçalama sıkıştırmayı bozuyor.** Yük 225.9 MB'dan 290.2 MB'a çıkıyor
   (+%28). Küçük bloklar bağımsız sıkıştırıldığında bağlam kaybediyor. Raporda
   öngörülen "sıkıştırma verimliliğinin düşmesi" maliyeti ölçüldü ve kurtarılan
   alandan büyük çıktı.

> **Kodek çekincesi:** sıkıştırma vekili zlib'dir; gerçek bir misyon CCSDS 122
> veya JPEG2000 kullanır. Mutlak bayt sayıları misyonu temsil etmez. Ancak
> karşılaştırma tek kodekle yapıldığı için ölçülen şey — *blok boyutunun
> sıkıştırmaya etkisi* — geçerlidir.

**Sonucun sınırı:** bu, blok tabanlı bir protokolün ve genel amaçlı bir kodeğin
sonucudur; kısmi indirme fikrinin tümden reddi değildir. Bölge-ilgi kodlaması
(JPEG2000 ROI) ya da bulutlu bölgeleri atmak yerine kayıplı kodlamak,
parçalanma cezasını ödemeden aynı kazancı verebilir. Denenmemiştir.

**U-Net'in gerekçesi bu ölçümle değişiyor.** Kısmi indirme, U-Net'i projede
tutmanın ana gerekçesi olarak gösterilmişti; ölçüm bunu desteklemiyor. Buna
karşılık bölüm 7.2 U-Net için daha güçlü ve ölçülmüş bir gerekçe sağlıyor:
**farklı sensörde dayanıklılık** (kar/buz yanlış elemesi %58.62 → %5.07).
Gerekçe spekülatif olandan ölçülmüş olana kaymıştır.

### 6.2 Tek modele geçiş sorusu: ölçüldü, cevap rejime bağlı

Kaynak ekseninde U-Net disk ve bellekte belirgin şekilde daha ucuz; süre
tarafında değil:

| | Yalnız U-Net | Yalnız sınıflandırıcı | İkisi birlikte (dağıtılan) |
|---|---|---|---|
| Disk | **0.96 MB** | 2.59 MB | 3.55 MB |
| Tepe bellek | **14.75 MB** | 22.14 MB | 27.78 MB |
| Süre / sahne | 175 ms | **92 ms** | 267 ms |
| Piksel maskesi | ✓ | ✗ | ✓ |

Sınıflandırıcı diskte 2.7 kat büyük, çünkü encoder'ı daha geniş
(`mobilenetv2_100` vs `mobilenetv2_050` — genişlik çarpanı 1.0 vs 0.5) ve
ayrıca MobileNetV2'nin 320→1280 `conv_head` katmanını taşıyor (≈410 K
parametre, toplamın %18'i); U-Net'te o kısım yok. Baskın etken genişlik
çarpanıdır.

**Karar kalitesi eşleşmiş noktada karşılaştırılmalı.** İki modelin "dengeli"
noktaları farklı tanımlı olduğu için doğrudan kıyaslanamaz; anlamlı
karşılaştırma benzer precision hedefinde yapılır:

| | Sınıflandırıcı @0.7947 | U-Net @%30 (dağıtılan kural) | U-Net @0.891 |
|---|---|---|---|
| precision | **0.9957** | 0.9790 | 0.9916 |
| recall | 0.8627 | **0.9237** | 0.7772 |
| veri azalması | %48.65 | **%51.42** | ~%44 (türetilmiş) |
| kaybedilen kullanılabilir veri | **%0.475** | %2.375 | %0.79 |

> `veri azalması`, elenen kare oranıyla aynı büyüklüktür (kareler eşit boyutlu
> olduğu için elenen kare oranı = elenen veri hacmi oranı); sınıflandırıcının
> `downlink_analysis.json`'ında `elenen_goruntu_orani_%` ile
> `veri_indirme_azalmasi_%` birebir aynıdır. U-Net değerleri
> `unet_t64_threshold_analysis.csv`'nin `elenen_%` kolonundan gelir. Her iki
> model de %30 etiketleme kuralıyla değerlendirilmiştir; sınıflandırıcı 256×256
> karelerde, U-Net 64×64 karelerde — ikisi de alan oranı olduğu için hacim
> karşılaştırması geçerlidir, ancak karar granülerliği farklıdır.

**Okuma:** U-Net dağıtılan kuralla daha fazla veri eliyor (%51.42 vs %48.65) ama
karşılığında 5 kat daha fazla kullanılabilir veri kaybettiriyor (%2.375 vs
%0.475). Yüksek precision'a çıkarıldığında ise recall'ü 0.7772'ye düşüyor ve
bant genişliği kazancı ~%44'e iniyor. **Her iki yönde de sınıflandırıcı, korunan
veri başına daha fazla bant genişliği kazandırıyor.**

**İkisi neden birlikte teslim edildi:** proje yönergesi ikili sınıflandırıcıyı
temel hedef olarak zorunlu kılıyor. Bu bir mühendislik tercihi değil, kapsam
kısıtıdır.

**Kademeli kurgu (sınıflandırıcı ön süzgeç, U-Net kalanlara) neden tercih
edilmedi:** sınıflandırıcının kendisi 92 ms tükettiği için kazandırdığı süreyi
geri alıyor. Sahnelerin %54.48'i elenirse toplam süre
`92 + 0.4552 × 175 ≈ 172 ms` olur — yalnız U-Net'e (175 ms) göre **~%2**
kazanç, karşılığında bellek 1.9 kat büyüyor (14.75 → 27.78 MB).

**Her iki ölçüm de tamamlandı. Sonuç: tek bir kazanan yok — iki model iki farklı
rejimde üstün.**

### Eşleşmiş çalışma noktalarında karşılaştırma

Her iki model de aynı yöntemle (belirsiz bantta, precision kısıtıyla) ayarlandı:

| | Sınıflandırıcı @0.7947 | U-Net @0.8906 |
|---|---|---|
| **S2CMC** precision | 0.9957 | 0.9916 |
| **S2CMC** veri azalması | **%48.65** | %42.72 |
| **S2CMC** kaybedilen kullanılabilir | **%0.475** | %0.787 |
| **SPARCS** kaybedilen kullanılabilir | %7.95 | **%0.464** |
| **SPARCS** kar/buz yanlış eleme | %29.89 | **%0.71** |

Ve eşikten bağımsız ayırt etme kalitesi:

| ROC-AUC | S2CMC | SPARCS | Sensör değişiminde |
|---|---|---|---|
| Sınıflandırıcı | **0.9883** | 0.9645 | **−0.0238** |
| U-Net | 0.9645 | **0.9733** | **+0.0088** |

**Okuma:** sınıflandırıcı eğitildiği dağılıma daha iyi oturmuş ve orada daha
fazla bant genişliği kazandırıyor; U-Net dağılım kaydığında dayanıklı kalıyor —
harici sensörde 17 kat az kullanılabilir veri, 42 kat az kar/buz yanlış elemesi.
Kutup yörüngeli bir gözlem uydusu için bu ikincisi hafife alınamaz: yörüngeden
her sahne tipi için yeniden ayar yapılamaz.

### Ölçülmüş sınır: U-Net precision ≥ 0.995'e ulaşamıyor

Belirsiz bantta hiçbir eşik bu kısıtı sağlamıyor; çıkabildiği en yüksek nokta
0.9916. Sınıflandırıcı 0.9957'ye çıkabiliyor. Yani sınıflandırıcının üstünlüğü
"biraz daha iyi" değil, U-Net'in erişemediği bir bölgeye erişebilmesidir.

> `unet_t64_operating_points.json` içinde bu nokta `constraint_met: false` ile
> işaretlidir. `tune()` kısıt sağlanamadığında en iyi F1'e düşer; bu geri düşüş
> eskiden sessizdi ve "precision>=0.995" etiketli bir nokta 0.9788 precision
> raporluyordu. Artık hem ekrana uyarı basılıyor hem JSON'a işaretleniyor.

### Dağıtım kararı

`clf256 + unet64` artık bir uzlaşma değil, ölçüme dayalı bir tasarım: **sınıflandırıcı
eğitim dağılımında verimi maksimize ediyor, U-Net dağılım kaydığında güvenlik ağı
sağlıyor ve kısmi indirmeye açılan tek çıktı biçimini üretiyor.** Yönerge kısıtı
bu seçimle çelişmiyor, onu destekliyor.

Yalnız U-Net'e geçiş, bant genişliği kazancından ~6 puan (%48.65 → %42.72) feragat
etmeyi göze alan bir misyon için savunulabilir; yalnız sınıflandırıcıya geçiş ise
kar/buz sahnelerindeki veri kaybı nedeniyle savunulamaz.

---

## 7. Harici doğrulama (Landsat-8 / SPARCS)

Farklı sensör, eğitimde hiç görülmemiş veri, eşik yeniden ayarlanmadı.

### 7.1 Sınıflandırıcı (256×256)

| Kırılım | n | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|---|
| Genel | 1248 | 0.8421 | 0.5996 | 0.9795 | 0.7438 | 0.9645 |
| Kar/buz olmayan | 1152 | 0.8733 | 0.6643 | 0.9788 | 0.7914 | 0.9731 |
| **Kar/buz ağırlıklı (>%20)** | **96** | **0.4688** | **0.1500** | **1.0000** | 0.2609 | 0.8646 |

**Temiz kar/buz karelerinin %58.6'sı yanlışlıkla eleniyor.** Kar ve bulut görünür
+ NIR bantlarında neredeyse aynı spektral imzaya sahip; ayrım esas olarak SWIR
(B11) ve cirrus (B10) bantlarından gelmeli, ancak model bunu yeterince
öğrenememiş. Etkisi asimetrik: recall 1.0 olduğu için hiçbir bulut kaçmıyor, ama
kutup bölgesi görüntülerinde bilimsel veri kaybı kabul edilemez seviyede.
Çalışma noktası seçimi bunu %29.89'a indiriyor (bölüm 5); model aynı model.

### 7.2 U-Net (64×64) — segmentasyon daha iyi genelliyor

Maskeden `mask_to_decision` ile görüntü kararı üretilerek aynı kırılımlar
ölçüldü (20.480 kare, 80 sahne):

| Kırılım | n | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|---|
| Genel | 20480 | 0.9236 | 0.7712 | 0.9523 | **0.8522** | **0.9733** |
| Kar/buz olmayan | 19116 | 0.9220 | 0.7761 | 0.9532 | 0.8556 | 0.9737 |
| Kar/buz ağırlıklı (>%20) | 1364 | 0.9465 | 0.5897 | 0.9109 | 0.7160 | 0.9466 |

**Piksel bazlı:** IoU 0.7225, Dice 0.8389 (S2CMC'de 0.8892 / 0.9413). Farklı
sensörde IoU 0.167 düşüyor — gerçek bir genelleme açığı, ama çöküş değil.

**Doğrudan karşılaştırma:**

| | Sınıflandırıcı 256px | U-Net 64px |
|---|---|---|
| ROC-AUC | 0.9645 | **0.9733** |
| F1 | 0.7438 | **0.8522** |
| Veri azalması | **%38.22** | %28.57 |
| Kaybedilen kullanılabilir veri | %19.98 | **%8.51** |
| **Kar/buz yanlış eleme** | **%58.62** | **%5.07** |

Kar/buz yanlış elemesi **11.6 kat** azalıyor. Sınıflandırıcı ham bant genişliği
kazancında önde, U-Net korunan veri başına kazançta.

**Bu fark karelemeden gelmiyor.** İtiraz şu olurdu: U-Net 64×64'te ölçüldü, ince
kareleme zaten modeli temkinli yapıyor. Ama aynı 64×64 ızgarada ölçülmüş
sınıflandırıcı (`c64_tuned_operating_points.json`) bunu çürütüyor:

| 64×64 ızgara, aynı SPARCS | clf64 (mevcut nokta) | clf64 (dengeli) | U-Net 64 |
|---|---|---|---|
| F1 | 0.0068 | 0.5571 | **0.8522** |
| Veri azalması | %0.08 | %55.13 | %28.57 |
| Kaybedilen kullanılabilir | %0.0 | %43.41 | **%8.51** |
| Kar/buz yanlış eleme | %0.0 | **%64.37** | **%5.07** |

clf64 iki uçta da kullanılamaz: ya hiçbir şey elemiyor ya kullanılabilir verinin
%43'ünü yok ediyor. Kar/buz yanlış elemesi 64×64'te %64.37 — clf256'nın
%58.62'sinden **daha kötü**. Yani ince kareleme kar/buz sorununu çözmüyor; fark
görev türünden geliyor. Bu, bölüm 4.4'teki hipotezi doğruluyor: segmentasyon her
pikselde yerel karar verdiği için sensörler arasında en çok değişen özelliğe
(dokusal istatistikler) daha az bağımlı.

### 7.3 Negatif sonuç: güneş yüksekliği düzeltmesi yardımcı olmuyor

`l8cloudmasks.zip` 80 sahnenin hepsi için MTL dosyası içeriyor, bu yüzden
`1/sin(SUN_ELEVATION)` düzeltmesi uygulanabildi (çarpan 1.07–2.00, medyan 1.22).
Aynı U-Net, düzeltmeli ve düzeltmesiz setlerde:

| | Düzeltmesiz | Düzeltmeli |
|---|---|---|
| **ROC-AUC** | **0.9733** | **0.9732** |
| Precision | 0.7712 | 0.7156 |
| Recall | 0.9523 | 0.9728 |
| F1 | 0.8522 | 0.8246 |
| IoU | 0.7225 | 0.6936 |
| Kar/buz yanlış eleme | %5.07 | %5.86 |

**ROC-AUC pratikte değişmiyor (0.9733 → 0.9732).** Bu belirleyici: düzeltme
modelin ayırt etme yeteneğine hiçbir şey katmıyor. Yaptığı tek şey, görüntüleri
~1.22 kat parlatarak sabit eşiğin (piksel 0.5 + bulut oranı %30) skor dağılımı
üzerindeki yerini kaydırmak — recall yükseliyor, precision düşüyor.

Sonuç: **radyometrik uyumsuzluk SPARCS'taki performans açığını açıklamıyor.**
Açık gerçek bir alan kayması (spektral tepki farkları, farklı yüzey örtüsü,
farklı coğrafya); ölçek düzeltmesiyle kapanmıyor. Bu, raporun "SPARCS
karşılaştırması kötümser" çekincesini zayıflatan ölçülmüş bir sonuçtur.

**Öneri (kar/buz için hâlâ geçerli):** NDSI (B03/B11 üzerinden kar indeksi)
girdiye eklenmesi veya karlı sahnelerin eğitim setinde ağırlıklandırılması.

---

## 8. Bellek profili

Disk boyutu, uydu üzerindeki bellek ihtiyacının küçük bir parçasıdır. Her model
**ayrı süreçte** ölçüldü; aksi halde önceki modelin ayırdığı bellek sonrakinin
ölçümüne karışır. Python yorumlayıcısının ~50 MB'lik tabanı hariç tutulmuştur
(uydu yazılımında bu maliyet olmaz).

| Senaryo | Tepe bellek |
|---|---|
| **DAĞITILAN: clf256 + unet64** | **27.78 MB** |
| Yalnız U-Net (64²) — bkz. 6.2 | 14.75 MB |
| Yalnız sınıflandırıcı (256²) | 22.14 MB |
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

> **Ölçüm kaynakları farklıdır, karıştırılmamalı.** Yukarıdaki senaryo tablosu
> `outputs/reports/combined_memory.json`'dan gelir (tek süreçte, oturumlar
> birlikte yüklenerek). `outputs/reports/memory_profile.csv` ise her modeli ayrı
> süreçte ölçer ve şu an yalnızca `c64_tuned_int8` satırını (15.74 MB) içerir —
> önceki satırlar tek yapılandırmayla yapılan yeniden koşuda üzerine yazıldı.
> Bellek ölçümleri koşudan koşuya ±0.2 MB oynar; bu düzeydeki farklar anlamlı
> değildir.

> **Çekince:** ölçülen bellek ONNX Runtime'ın önceden ayırdığı havuzu içerir ve
> ~10.5 MB'ı hiçbir ayarla düşmeyen ORT **tabanıdır**. Gömülü bir motorla
> (TFLite Micro çekirdeği ~16 KB) içsel gereksinim ~14.3 MB'e inerdi. Bu
> rakamlar ORT'ye özgü **üst sınırlardır**.

**Ölçülmüş bir bellek kaldıracı: arena + bellek deseni kapatma.** Taban yük
düşmüyor ama tepe bellek düşüyor (`outputs/reports/ort_arena_test.csv`):

| Model | Varsayılan | Arena+pattern kapalı | Süre bedeli |
|---|---|---|---|
| 256² U-Net | 26.66 MB | **22.55 MB** (−%15) | 8.52 → 12.23 ms (+%43) |
| 64² U-Net | 14.68 MB | 14.32 MB (−%2) | 0.689 → 0.737 ms (+%7) |

İnce karelemede bu kaldıraç neredeyse tükenmiştir — 64×64'te kazanılacak yer
kalmamış. Dağıtılan yapılandırma bu nedenle ayarı varsayılan bırakır.

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
   %70 eşikle veriyor. Bölüm 6'daki tarama yalnızca *karar* eşiğini hazır maske
   üzerinde değiştiriyor; *etiketleme* eşiğini değiştirip modeli yeniden eğiterek
   yapılan duyarlılık analizi yapılmadı.
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
6. ~~**SPARCS karşılaştırması kötümser** (güneş yüksekliği düzeltmesi yok).~~
   **Ölçüldü ve elendi (bölüm 7.3):** düzeltme uygulandığında ROC-AUC değişmiyor
   (0.9733 → 0.9732). Radyometrik uyumsuzluk performans açığını açıklamıyor;
   açık gerçek bir alan kaymasıdır. Bu kısıt artık geçerli değildir.
7. **CloudScout ile doğrudan karşılaştırma geçerli değildir:** farklı veri seti,
   farklı görev eşiği (%70), farklı donanım sınıfı (1.8 W uzay VPU'su vs masaüstü
   CPU). Boyut olarak aynı sınıfta olunduğu söylenebilir; doğruluk veya hız
   üstünlüğü iddia edilemez.

### İyileştirme önerileri

1. Etiketleme eşiği için yeniden eğitimli duyarlılık analizi (%10/20/30/50/70) ve
   operasyonel gerekçe
2. Çoklu tohumlu koşularla model seçimini sağlamlaştırmak
3. Kar/buz başarısızlığı: kar ağırlıklı veri artırımı veya özel kayıp ağırlığı
4. Genişlik çarpanı taraması (`mobilenetv2_050` / `_075`) ile boyut-doğruluk eğrisi
5. ~~Kısmi indirme senaryosunun ölçülmesi.~~ **Ölçüldü (bölüm 6.1):** ödünleşim
   elverişsiz çıktı. Denenmemiş alternatif: bölge-ilgi kodlaması (JPEG2000 ROI).
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
  inference.py            Çıkarım sarmalayıcıları, Grad-CAM, görüntü okuma (app.py bunu kullanır)
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
  partial_downlink.py     Kısmi indirme ödünleşimi (blok boyutu vs bayt maliyeti)
  make_report.py          Teknik rapor üretimi (Markdown)
  md_to_docx.py           TEKNIK_RAPOR.md -> .docx dönüşümü (biçimi koruyarak)
  snapshot_release.py     Sürüm dondurma -> releases/<v>/ + MANIFEST.json
app.py                    Demo arayüzü: karar + Grad-CAM ısı haritası + U-Net maskesi
check_env.py              Kurulum doğrulama: paketler, GPU, CPU çıkarım hattı
smoke_test.py             Sentetik veriyle uçtan uca hat testi
outputs/
  checkpoints/            Eğitim çıktıları (.pt)
  onnx/                   Export edilen modeller (fp32 / int8)
  reports/                Ölçüm çıktıları (CSV / JSON) — rapordaki sayıların kaynağı
releases/                 Dondurulmuş sürümler: v1, v2, v3, v3.1, v3.2 (dağıtılan)
teslim/                   Teknik rapor, proje rehberi, sunum, demo görseli
```

`outputs/reports/` klasörü, README ve teknik rapordaki sayıların kaynağıdır —
benchmark tabloları, çalışma noktaları, bellek profili, eşik taraması ve harici
doğrulama sonuçları. `src/make_report.py` teknik raporu bu dosyalardan üretir.

> **Bilinen sınır:** raporun tablo gövdeleri bu dosyalardan okunur ama bazı
> yorum satırlarındaki sayılar `make_report.py` içinde sabit yazılıdır (bellek
> yorumları, sürüm matrisi, literatür karşılaştırmasındaki Dice). Bir ölçüm
> yeniden koşulduğunda bu sabitler otomatik güncellenmez.

Dağıtılan sürümün ağırlıkları `releases/v3.2/` altındadır. Klasörün tamamı
161 MB olduğu için depoya doğrudan konmak yerine **Releases** bölümünde
`v*.zip` olarak paketlenmesi önerilir. Her sürüm kendi checkpoint'leri, ONNX
dosyaları, config kopyası ve ölçülmüş sayılarıyla dondurulur — sonraki deneyler
çalışan sürümü bozmasın diye.

---

## 12. Çalıştırma

### Kurulum

```bash
# CPU sürümü — bu proje için yeterlidir
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

> **GPU gerekmez.** Yayımlanan tüm süre ve bellek ölçümleri bilinçli olarak
> **ONNX Runtime CPU, tek iş parçacığı** ile yapılmıştır (bölüm 4) — uydu
> bilgisayarını temsil eden senaryo bu. Dolayısıyla GPU'nun olmaması hiçbir
> raporlanan sayıyı etkilemez; yalnızca sıfırdan eğitimi yavaşlatır. Eğitim
> betikleri CUDA yokluğunu kendileri ele alır (`autocast` ve `GradScaler`
> CPU'da devre dışı kalır), kod değişikliği gerekmez.
>
> NVIDIA GPU varsa eğitim için:
> `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128`

### Hızlı sistem kontrolü

```bash
python check_env.py     # paketler, GPU ve CPU çıkarım hattı yerinde mi
python smoke_test.py    # sentetik veriyle uçtan uca hat testi
```

`smoke_test.py` eğitim → ONNX → INT8 → benchmark hattını sentetik veriyle
çalıştırır. Gerçek veri veya indirilmiş ağırlık gerektirmez. Amaç doğruluk değil,
hattın kırık olmadığını göstermek.

### Demo uygulaması

Görüntü verildiğinde modelin kararını, Grad-CAM dikkat haritasını ve U-Net bulut
maskesini gösterir.

```bash
python app.py                              # Gradio web arayüzü
python app.py --image <kare>.npy           # arayüzsüz, tek görüntü
python app.py --sample                     # sentetik örnekle çalıştır (veri gerekmez)
CLOUD_RELEASE=v3 python app.py             # başka bir sürümle çalıştır
```

Modeller `releases/v3.2/`'den (`config.CURRENT_RELEASE`) okunur; karar eşiği
`operating_points.json` içinden alınır — kodda sabit değildir. Girdi olarak
`.npy` kare veya ham `.tif` GeoTIFF kabul edilir. Arayüzsüz modda çıktı görseli
`outputs/reports/demo_<ad>.png` olarak yazılır (örnek: `teslim/demo_ornek.png`).

> `--sample`, `outputs/demo_sample.npy` adında sentetik bir kare üretir ve
> arayüzde tek tıkla denenebilir örnek olarak sunar. **Bu gerçek uydu verisi
> değildir**; modelin o kare için verdiği karar bilimsel olarak anlamsızdır. Tek
> amacı veri seti indirilmeden arayüzün ve çıkarım hattının çalıştığını
> göstermektir. Raporlanan tüm sayılar `data/patches` altındaki gerçek karelerden
> üretilmiştir.

Gerçek bir kareyle çalıştırmak için `data/patches/` altındaki herhangi bir `.npy`
dosyası kullanılabilir (veri hazırlama adımı için bkz. "Veri edinme").

### Veri edinme

Depoda ham veri yoktur (`data/` sürüm kontrolü dışındadır). Aşağıdakiler yalnızca
**yeniden eğitim veya yeni ölçüm** için gereklidir; yayımlanan modellerle
benchmark ve teknik rapor üretimi için veri indirmek gerekmez.

| Veri seti | Kaynak | Kullanım |
|---|---|---|
| Sentinel-2 Cloud Mask Catalogue | Zenodo kayıt **4172871** (`zenodo.org/records/4172871`) — Francis, Mrziglod, Sidiropoulos | Eğitim + doğrulama + test |
| SPARCS (Landsat-8) | USGS SPARCS doğrulama seti, `l8cloudmasks.zip` (1.5 GB) — Hughes & Kennedy, Remote Sensing 11(21):2591, 2019 | Yalnızca harici doğrulama |
| 95-Cloud | Kaggle | Kapsam dışı (4 bant; cirrus/SWIR yok) |

S2CMC arşivinden `subscenes.zip` ve `masks.zip` gerekir (açılmış hali ~26.6 GB).
`prepare_data.py` zip'leri açmadan sahne sahne okur, anlık bellek ~52 MB'ta
kalır. Beklenen dosya biçimi `src/prepare_data.py` docstring'inde tanımlıdır.

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

> `outputs/reports/c6_tuned_train_summary.json` depoda zaten duruyor, yani eşiği
> görmek için release paketini indirmeniz gerekmez. Yukarıdaki üç kopyalama
> satırından yalnızca `.onnx` dosyalarına ait olanlar zorunludur.

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
# 1) Veri hazırlama — maskeler her zaman yazılır, ayrı bir flag gerekmez
python -m src.prepare_data --source <s2cmc_dizini>                       # 256x256
python -m src.prepare_data --source <s2cmc_dizini> \
  --patch-size 64 --out data/patches_t64                                 # 64x64

# SPARCS (harici doğrulama) — evaluate_external bunu data/patches_sparcs'ta arar
python -m src.prepare_sparcs --zip <yol>/l8cloudmasks.zip

# 2) Sınıflandırıcı (dağıtılan c6_tuned'ı üreten tam komut)
python -m src.train --model mobilenetv2_100 --tag c6_tuned --epochs 15 \
  --lr 0.00119 --weight-decay 5.7e-05 --batch-size 64 \
  --label-smoothing 0.1 --ema

# 3) U-Net (64x64)
CLOUD_PATCH_SIZE=64 CLOUD_PATCH_DIR=data/patches_t64 \
  python -m src.train_seg --encoder mobilenetv2_050 --tag unet_t64 \
  --lr 0.001128 --weight-decay 0.001394 --batch-size 32 --bce-weight 0.3

# 4) Export + sınıflandırıcı analizleri
python -m src.export --checkpoint outputs/checkpoints/c6_tuned_best.pt --tag c6_tuned
python -m src.tune_operating_point --checkpoint outputs/checkpoints/c6_tuned_best.pt --tag c6_tuned
python -m src.evaluate_external --checkpoint outputs/checkpoints/c6_tuned_best.pt --tag c6_tuned
python -m src.memory_profile

# 5) U-Net analizleri (kare boyutu ortam değişkeniyle verilmeli)
CLOUD_PATCH_SIZE=64 CLOUD_PATCH_DIR=data/patches_t64 \
  python -m src.export --checkpoint outputs/checkpoints/unet_t64_best.pt --tag unet_t64
CLOUD_PATCH_SIZE=64 CLOUD_PATCH_DIR=data/patches_t64 \
  python -m src.threshold_analysis --unet outputs/checkpoints/unet_t64_best.pt --tag unet_t64
CLOUD_PATCH_SIZE=64 CLOUD_PATCH_DIR=data/patches_t64 \
  python -m src.decision_layer --unet outputs/checkpoints/unet_t64_best.pt --tag unet_t64

# U-Net harici doğrulama (SPARCS) — maskeden karar + piksel bazlı IoU
python -m src.prepare_sparcs --zip <yol>/sparcs_data_L8.zip \
  --patch-size 64 --out data/patches_sparcs_t64
python -m src.evaluate_external --task segmentation \
  --checkpoint outputs/checkpoints/unet_t64_best.pt --tag unet_t64 \
  --index data/patches_sparcs_t64/index.csv

# Kısmi indirme ödünleşimi (blok boyutu vs bayt maliyeti)
CLOUD_PATCH_SIZE=64 CLOUD_PATCH_DIR=data/patches_t64 \
  python -m src.partial_downlink --checkpoint releases/v3.2/unet.pt --tag unet_t64

# U-Net çalışma noktaları — sınıflandırıcıyla AYNI yöntemle seçilir
CLOUD_PATCH_SIZE=64 CLOUD_PATCH_DIR=data/patches_t64 \
  python -m src.tune_operating_point --task segmentation \
  --checkpoint outputs/checkpoints/unet_t64_best.pt --tag unet_t64 \
  --sparcs-index data/patches_sparcs_t64/index.csv

# 6) Rapor + sürüm dondurma
python -m src.make_report --classifier-tag c6_tuned --unet-tag unet_t64
python -m src.md_to_docx          # teslim/TEKNIK_RAPOR.md -> .docx
python -m src.snapshot_release --version v3.2 \
  --classifier-tag c6_tuned --unet-tag unet_t64 --note "..."
```

> **Dağıtılan modeli birebir üretmek için dikkat edilecekler.**
> `--epochs 15` zorunludur; `config.EPOCHS` varsayılanı 30'dur ve rapor 30
> epoch'un test başarımını artırmadığını ölçmüştür (aşırı öğrenme).
> `--balanced` **verilmemelidir**: `c6_tuned_train_summary.json`
> `"balanced_sampling": false` diyor. `--ema-decay` varsayılanı zaten 0.99'dur.
> U-Net'in epoch sayısı `unet_t64_train_summary.json`'a yazılmamıştır — bu bir
> kayıt eksiğidir, `train_seg.py` özetine eklenmelidir.

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

- `teslim/TEKNIK_RAPOR.md` / `.docx` — 14 bölümlük detaylı analiz raporu
  (kuantizasyon kök neden analizi, ablasyon çalışmaları, literatür taraması dahil)
- `teslim/PROJE_REHBERI.docx` — proje rehberi
- `teslim/SUNUM.pptx` — 15 slaytlık sunum
- `teslim/demo_ornek.png` — demo çıktısı: RGB önizleme, Grad-CAM, U-Net maskesi
- `STAJ_RAPORU.docx` (kökte) — staj raporu
- `staj_plani_uydu_yapay_zeka.pdf` (kökte) — proje yönergesi

> **Not:** `teslim/TEKNIK_RAPOR.md` 3 Ağustos tarihli ve dağıtım tavsiyesini
> `clf256 + unet64` olarak veriyor; bu README ile aynı yöndedir. Yalnız U-Net
> tartışması (bölüm 6.2) rapordan sonra eklenmiştir ve raporda yer almaz.
