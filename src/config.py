"""Merkezi yapilandirma. Tum betikler buradaki yollari ve sabitleri kullanir.

Ortam degiskenleriyle gecersiz kilinabilir (deney varyantlari icin):
    CLOUD_BANDS      bant listesi, virgulle ayrilmis; "all" = 13 bandin tumu
    CLOUD_PATCH_DIR  kare dizini (varsayilan: data/patches)
    CLOUD_THRESHOLD  bulut piksel esigi (varsayilan: 0.30)
Boylece farkli varyantlar kodu degistirmeden, birbirinin verisini bozmadan
kosulabilir.
"""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Dagitilan surum. app.py ve demo bu surumun modellerini kullanir.
# Surumler releases/ altinda dondurulmustur, uzerlerine yazilmaz.
CURRENT_RELEASE = os.environ.get("CLOUD_RELEASE", "v3.2")
RELEASES = PROJECT_ROOT / "releases"
CURRENT = RELEASES / CURRENT_RELEASE

DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PATCHES = Path(os.environ.get("CLOUD_PATCH_DIR", PROJECT_ROOT / "data" / "patches"))
OUTPUTS = PROJECT_ROOT / "outputs"
CHECKPOINTS = OUTPUTS / "checkpoints"
ONNX_DIR = OUTPUTS / "onnx"
REPORTS = OUTPUTS / "reports"
TB_LOGS = OUTPUTS / "tensorboard"

for _d in (DATA_RAW, DATA_PATCHES, OUTPUTS, CHECKPOINTS, ONNX_DIR, REPORTS, TB_LOGS):
    _d.mkdir(parents=True, exist_ok=True)

# Patch index CSV: prepare_data.py bunu uretir, dataset.py bunu okur.
# Sutunlar: path,label,split   (label: 0=kullanilabilir, 1=bulutlu)
PATCH_INDEX = DATA_PATCHES / "index.csv"

# --- Veri ---
PATCH_SIZE = int(os.environ.get("CLOUD_PATCH_SIZE", 256))

# Sentinel-2 L1C bant sirasi (Cloud Mask Catalogue subscene'lerindeki sira)
S2_BANDS = ["B01", "B02", "B03", "B04", "B05", "B06", "B07",
            "B08", "B8A", "B09", "B10", "B11", "B12"]

# Egitimde kullanilacak bantlar. Varsayilan: RGB + NIR + cirrus + SWIR.
# CLOUD_BANDS ortam degiskeniyle degistirilebilir ("all" = 13 bandin tumu).
# NOT: bant sayisi model boyutunu neredeyse hic etkilemez (olculdu: 6 bant
# 2.226.017 parametre, 13 bant 2.228.033 - fark %0.09). Yalnizca ilk
# konvolusyonun girdi kanali degisir.
_bands_env = os.environ.get("CLOUD_BANDS", "").strip()
if _bands_env.lower() == "all":
    BANDS = list(S2_BANDS)
elif _bands_env:
    BANDS = [b.strip() for b in _bands_env.split(",") if b.strip()]
else:
    BANDS = ["B02", "B03", "B04", "B08", "B10", "B11"]
BAND_INDICES = [S2_BANDS.index(b) for b in BANDS]
IN_CHANNELS = len(BANDS)

# Piksel maskesinden goruntu-seviyesi etikete gecis esigi.
# Karedeki bulut pikseli orani bu degeri asarsa "bulutlu" (label=1).
# RAPORDA GEREKCELENDIR: bu esik veri indirme kazanci ile veri kaybi arasindaki dengeyi belirler.
CLOUD_PIXEL_THRESHOLD = float(os.environ.get("CLOUD_THRESHOLD", 0.30))

# Sentinel-2 L1C reflektans olcegi (ham GeoTIFF DN -> reflektans).
# DIKKAT: Cloud Mask Catalogue'daki .npy dosyalarina bu bolme ZATEN uygulanmis,
# tekrar bolunmemeli.
S2_REFLECTANCE_SCALE = 10000.0

# Reflektans kirpma siniri. README: degerler 1'i asabilir (fiziksel olarak
# gecerli), bu yuzden 1.0'a kirpmak parlak bulut tepelerini duzlestirir.
#
# Deger 8 sahne uzerinde olculerek secildi (4 kar/buz+bulutlu, 4 normal):
#   piksellerin %0.141'i > 1.0,  %0.005'i > 1.5,  gozlenen maksimum 1.94
#   p99 = 0.838,  p99.9 = 1.040
# 1.5 secilirse verinin %99'u [0, 0.56] araliginda sikisir - dinamik aralikigin
# %44'u bosa gider ve INT8 kuantizasyonunda girdi cozunurlugu duser.
# 1.2, p99.9'u (1.04) korurken aralikigin %70'ini kullanir: dengeli secim.
REFLECTANCE_CLIP = 1.2

# Maskedeki sinif kanallari (one-hot): 0=CLEAR, 1=CLOUD, 2=CLOUD_SHADOW
MASK_CLASS_CLEAR, MASK_CLASS_CLOUD, MASK_CLASS_SHADOW = 0, 1, 2

# Bulut golgesi ikili gorevde bulut sayilsin mi?
# Varsayilan False: golgeli bir goruntu bilimsel olarak halen kullanilabilir,
# bulutla kaplanmis degil. Raporda gerekcelendirilmeli.
SHADOW_AS_CLOUD = False

# --- SPARCS (Landsat 8) harici dogrulama ---
# _data.tif 10 bant icerir: L8 B1..B7, B9, B10, B11 (pankromatik B8 yok).
# Bant kimlikleri readme + istatistiklerle dogrulandi (kar/buz sahnesinde
# SWIR bantlari dusuk, cirrus bandi dar aralikli).
L8_DATA_BANDS = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B9", "B10", "B11"]

# Sentinel-2 -> Landsat 8 dalga boyu eslemesi (nm cinsinden merkez dalga boylari)
S2_TO_L8_BAND = {
    "B01": "B1",   # 443  -> 443   coastal aerosol
    "B02": "B2",   # 490  -> 482   blue
    "B03": "B3",   # 560  -> 562   green
    "B04": "B4",   # 665  -> 655   red
    "B08": "B5",   # 842  -> 865   NIR
    "B10": "B9",   # 1375 -> 1375  cirrus
    "B11": "B6",   # 1610 -> 1610  SWIR1
    "B12": "B7",   # 2190 -> 2200  SWIR2
}

# Landsat 8 Collection 1 DN -> TOA reflektans (tum OLI bantlari icin sabit).
# UYARI: gunes yuksekligi duzeltmesi UYGULANMAZ - MTL dosyalari bu arsivde yok.
# Sentinel-2 L1C reflektanslari gunes acisina gore normalize edilmistir, bu
# yuzden SPARCS ile arada sistematik bir kayma kalir. Raporda belirtilmeli.
L8_REFLECTANCE_MULT = 2.0e-5
L8_REFLECTANCE_ADD = -0.1

# SPARCS etiket kodlari (readme.txt):
#   0 Cloud Shadow, 1 Cloud Shadow over Water, 2 Water,
#   3 Ice/Snow, 4 Land, 5 Clouds, 6 Flooded
SPARCS_CLOUD_CLASS = 5
SPARCS_SHADOW_CLASSES = (0, 1)
SPARCS_SNOW_CLASS = 3

# MAIN bolumunden ayrilacak dogrulama orani.
# Test bolumu README onerisiyle CALIBRATION + VALIDATION sahnelerinden olusur:
# boylece model performansi anotator-arasi uyum (piksel dogrulugu %94.98,
# CLOUD F1 %95.97) ile karsilastirilabilir - insan seviyesi referansi.
SPLIT_VAL_FRACTION = 0.15

# --- Model ---
MODEL_NAME = "mobilenetv3_small_100"
PRETRAINED = True

# --- Egitim ---
BATCH_SIZE = 64
EPOCHS = 30
LR = 3e-4
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 4
SEED = 42

# --- Cikarim / degerlendirme ---
# Karar esigi. 0.5 varsayilan; tune_threshold ile val uzerinde optimize edilir.
# DIKKAT: yanlis pozitif = kullanilabilir goruntunun uyduda silinmesi (geri donusu yok).
# Bu yuzden yuksek precision tarafinda calisiriz.
DECISION_THRESHOLD = 0.5

# Uyduyu temsil eden "standart CPU" senaryosu: tek is parcacigi.
BENCHMARK_THREADS = 1
BENCHMARK_WARMUP = 10
BENCHMARK_RUNS = 100

# Plandaki kisitlar (rapor tablosunda hedef sutunu olarak kullanilir)
TARGET_MODEL_SIZE_MB = 5.0
TARGET_LATENCY_MS = 100.0
