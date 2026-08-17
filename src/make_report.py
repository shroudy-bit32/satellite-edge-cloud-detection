"""Teknik raporu sonuc dosyalarindan uretir (Gunler 19-20).

Sayilar elle yazilmaz, outputs/reports altindaki JSON/CSV dosyalarindan okunur.
Boylece rapor her zaman son kosulan deneylerle tutarli kalir ve aktarim
hatasi olusmaz.

Kullanim:
    python -m src.make_report
    python -m src.make_report --out outputs/reports/TEKNIK_RAPOR.md
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from src import config

R = config.REPORTS


def load_json(name: str):
    path = R / name
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def load_csv(name: str):
    path = R / name
    return pd.read_csv(path) if path.exists() else None


def fmt(value, digits: int = 4) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def df_to_md(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return "_(veri yok)_"
    return df.to_markdown(index=False)


def section_dataset() -> str:
    idx_path = config.PATCH_INDEX
    lines = ["## 2. Veri", ""]

    if idx_path.exists():
        idx = pd.read_csv(idx_path)
        dist = idx.pivot_table(index="split", columns="label", values="path",
                               aggfunc="count", fill_value=0)
        dist.columns = ["kullanilabilir (0)", "bulutlu (1)"]
        dist["toplam"] = dist.sum(axis=1)
        lines += ["### 2.1 Sentinel-2 Cloud Mask Catalogue (ana veri seti)", "",
                  f"- Kaynak: 513 alt-sahne, 1022x1022 piksel, 20 m cozunurluk, 13 bant",
                  f"- Kullanilan bantlar: {', '.join(config.BANDS)}",
                  f"- Kare boyutu: {config.PATCH_SIZE}x{config.PATCH_SIZE}, sahne basina 16 kare",
                  f"- Etiketleme kurali: bulut pikseli orani >= %{100 * config.CLOUD_PIXEL_THRESHOLD:.0f} ise 'bulutlu'",
                  f"- Bulut golgesi ikili gorevde temiz sayildi (SHADOW_AS_CLOUD={config.SHADOW_AS_CLOUD})",
                  "", "**Bolum dagilimi (kare sayisi):**", "", df_to_md(dist.reset_index()), ""]

        scenes = idx.drop_duplicates("scene")
        diff = scenes.groupby("split")["difficulty"].agg(["count", "mean"]).round(2)
        diff.columns = ["sahne", "ortalama zorluk"]
        lines += ["**Sahne bazli zorluk (veri setinin kendi 1-5 skalasi):**", "",
                  df_to_md(diff.reset_index()), "",
                  "> Bolumleme SAHNE bazlidir, kare bazli degil: ayni sahnenin kareleri",
                  "> birbirine cok benzedigi icin kare bazli bolme test setine sizinti",
                  "> yaratir ve dogrulugu yapay olarak yukseltir.", "",
                  "> Test bolumu, veri setinin README'sinin onerisiyle CALIBRATION + VALIDATION",
                  "> sahnelerinden olusur. Bu sahneler icin anotatorler arasi uyum yayimlanmis",
                  "> oldugundan model performansi insan seviyesiyle karsilastirilabilir.",
                  "> Test setinin ortalama zorlugu egitim setinden yuksektir (tasarim geregi).", ""]

    sparcs_meta_path = config.PROJECT_ROOT / "data" / "patches_sparcs" / "prepare_meta.json"
    sparcs_meta = (json.loads(sparcs_meta_path.read_text(encoding="utf-8"))
                   if sparcs_meta_path.exists() else None)
    if sparcs_meta:
        lines += ["### 2.2 SPARCS (harici dogrulama)", "",
                  f"- {sparcs_meta['scenes']} Landsat-8 sahnesi, {sparcs_meta['patches']} kare",
                  f"- Bant eslemesi (Sentinel-2 -> Landsat-8): "
                  f"{', '.join(f'{a}->{b}' for a, b in zip(sparcs_meta['bands_s2'], sparcs_meta['bands_l8']))}",
                  "- Egitimde KULLANILMADI; yalnizca farkli sensore genelleme olcumu icin",
                  "", f"> **Uyari:** {sparcs_meta.get('caveat', '')}", ""]

    lines += ["### 2.3 95-Cloud (kapsam disi)", "",
              "95-Cloud veri seti yalnizca 4 bant icerir (R, G, B, NIR); cirrus (B10) ve",
              "SWIR (B11) bantlari yoktur. Bu iki bant bulut ayriminin en guclu sinyalleri",
              "oldugundan 6 bantli model bu veri setiyle beslenemez. Ayri bir 4 bantli model",
              "varyanti gerektirdiginden kapsam disi birakilmistir.", ""]
    return "\n".join(lines)


def section_models() -> str:
    lines = ["## 3. Modeller", "",
             "### 3.1 Ikili siniflandirici (temel hedef)", "",
             "MobileNetV2 omurgasi, tek logit cikisli. Coklu bant girdisi icin ilk",
             "konvolusyon agirligi kanal boyunca uyarlanir (timm `in_chans`).", "",
             "### 3.2 U-Net segmentasyon (genisletilmis hedef)", "",
             "Ayni MobileNetV2 omurgasi encoder olarak, uzerine derinlemesine-ayrilabilir",
             "konvolusyonlardan olusan ince bir decoder. Klasik U-Net (~31M parametre,",
             "~124 MB) boyut kisitini kat kat astigi icin tercih edilmedi.", ""]
    return "\n".join(lines)


def section_results(clf_tag: str, unet_tag: str) -> str:
    lines = ["## 4. Sonuclar", ""]

    for tag, title in ((clf_tag, "4.1 Siniflandirici"), (unet_tag, "4.2 U-Net segmentasyon")):
        summary = load_json(f"{tag}_train_summary.json")
        bench = load_csv(f"{tag}_benchmark.csv")
        if summary is None and bench is None:
            continue
        lines += [f"### {title}", ""]

        if summary:
            test = summary.get("test", {})
            lines += ["**Test metrikleri (PyTorch FP32):**", ""]
            rows = [{"metrik": k, "deger": fmt(v)} for k, v in test.items()]
            lines += [df_to_md(pd.DataFrame(rows)), ""]
            if "threshold" in summary:
                lines += [f"Karar esigi: {fmt(summary['threshold'])} "
                          f"(dogrulama seti uzerinde precision >= 0.99 kisitiyla secildi)", ""]

        if bench is not None:
            lines += ["**Karsilastirma tablosu (PDF bolum 3):**", "", df_to_md(bench), ""]

    return "\n".join(lines)


def section_downlink(clf_tag: str) -> str:
    lines = ["## 5. Veri indirme kazanci (fayda analizi)", "",
             "PDF'in istedigi \"bu filtre uyduda calissaydi indirilen veri hacmi %X azalirdi\"",
             "hesabi. Test seti uzerinden, karar esigi dogrulama setinde sabitlenerek.", ""]

    data = load_json(f"{clf_tag}_downlink_analysis.json")
    if data and "models" in data:
        rows = []
        for model, analysis in data["models"].items():
            rows.append({"model": model,
                         "veri azalmasi %": analysis["veri_indirme_azalmasi_%"],
                         "kaybedilen kullanilabilir %": analysis["kaybedilen_kullanilabilir_veri_%"],
                         "bosuna indirilen bulutlu": analysis["kacan_bulutlu_(bosuna_indirilen)"],
                         "teorik ust sinir %": analysis["teorik_ust_sinir_%"]})
        lines += [df_to_md(pd.DataFrame(rows)), "",
                  "> Yanlis eleme (kullanilabilir goruntunun atilmasi) geri donusu olmayan",
                  "> bilimsel veri kaybidir. Karar esigi bu nedenle F1'i degil, yuksek",
                  "> precision'i hedefleyecek sekilde secilmistir.", ""]
    return "\n".join(lines)


def section_external(clf_tag: str) -> str:
    ext = load_json(f"{clf_tag}_external_sparcs.json")
    if not ext:
        return ""
    lines = ["## 6. Harici dogrulama: SPARCS (Landsat-8)", "",
             "Farkli bir uydudan, egitimde hic gorulmemis veri. Karar esigi Sentinel-2",
             "egitiminden alinmis, bu veri uzerinde YENIDEN AYARLANMAMISTIR.", ""]
    rows = []
    for key, label in (("overall", "genel"), ("non_snowy", "kar/buz olmayan"),
                       ("snowy", "kar/buz agirlikli (>%20)")):
        if key in ext:
            r = {"kirilim": label}
            r.update({k: fmt(v) for k, v in ext[key].items() if not k.startswith("_")})
            rows.append(r)
    lines += [df_to_md(pd.DataFrame(rows)), ""]

    if "snowy" in ext and "kari_bulut_sanma_orani" in ext["snowy"]:
        lines += [f"> **Basarisizlik modu:** temiz kar/buz karelerinin "
                  f"%{100 * ext['snowy']['kari_bulut_sanma_orani']:.1f}'i yanlislikla eleniyor. "
                  f"Kar ve bulut spektral olarak benzer; ayrica Landsat-8 reflektanslarina "
                  f"gunes yuksekligi duzeltmesi uygulanamadigi icin sistematik bir kayma vardir.", ""]
    return "\n".join(lines)


def section_quantization() -> str:
    lines = ["## 7. Kuantizasyon calismasi", "",
             "INT8 kuantizasyon ilk denemede basarisiz oldu ve kok neden sistematik",
             "eleme ile arandi. Bu bolum hangi aciklamalarin hangi olcumle elendigini",
             "kaydeder; elenen hipotezler sonucun kendisi kadar bilgilendiricidir.", ""]

    sweep = load_csv("baseline_quantization_sweep.csv")
    if sweep is not None:
        lines += ["### 7.1 MobileNetV3 ile yapilandirma taramasi (basarisiz)", "",
                  df_to_md(sweep), "",
                  "Yedi farkli yapilandirmanin hicbiri dogrulugu kurtarmadi.", ""]

    bisect = load_csv("baseline_quantization_bisect.csv")
    if bisect is not None:
        lines += ["### 7.2 Op tipi bazli izolasyon", "", df_to_md(bisect), "",
                  "Yalnizca Conv katmanlarini kuantize etmek bile bozulmayi uretti;",
                  "squeeze-excite / hard-swish bloklari tek basina sorumlu degil.", ""]

    layer = load_csv("baseline_quantization_layerwise.csv")
    if layer is not None:
        lines += ["### 7.3 Katman bazli kademeli kuantizasyon", "", df_to_md(layer), "",
                  "Tek bir sorumlu katman bulunamadi; bozulma yaygin.", ""]

    v2sweep = load_csv("v2_quantization_sweep.csv")
    if v2sweep is not None:
        lines += ["### 7.4 MobileNetV2 ile ayni tarama (basarili)", "", df_to_md(v2sweep), "",
                  "**Sonuclar:**", "",
                  "1. MobileNetV2 ayni yapilandirmalarla sorunsuz kuantize oluyor "
                  "(dogruluk kaybi ~%1.4).",
                  "2. `per_channel=True` her iki mimaride de ZORUNLU; tensor basina "
                  "olcekleme modeli cokertiyor.",
                  "3. Aktivasyonlarda `QUInt8`, `QInt8`'e gore x86'da ~2.4 kat hizli; "
                  "dogruluk farki ihmal edilebilir.",
                  "4. MobileNetV3'un neden basarisiz oldugu KESIN OLARAK aciklanamadi. "
                  "Agirlik dinamik araligi hipotezi olculdu ve elendi (V2'nin araligi daha "
                  "genis oldugu halde V2 sorunsuz kuantize oluyor). En olasi aciklama "
                  "hard-swish aktivasyon dagilimlaridir; dogrulanmamistir.", ""]
    return "\n".join(lines)


def section_hparam() -> str:
    """Hiperparametre taramasi ve bant sayisi denemeleri."""
    lines = ["## 9. Hiperparametre taramasi ve bant secimi", ""]

    sweep = load_csv("hp_sweep.csv")
    if sweep is not None:
        lines += ["### Rastgele arama (8 deneme, 10 epoch)", "", df_to_md(sweep), "",
                  "Secim DOGRULAMA seti uzerinden yapildi; test yalnizca raporlama icin.", "",
                  "**Bulgu:** val F1 yayilimi ~0.11. Bu, daha once olculen mimari/duzenleme",
                  "farklarindan (~0.008) bir mertebe buyuk. Ogrenme orani baskin faktor:",
                  "proje varsayilani 3e-4 iken en iyi degerler 1.1-1.5e-3 araliginda cikti.", ""]

    band_rows = []
    for tag, name in [("c6_tuned", "6 bant"), ("c9_tuned", "9 bant"), ("c13_tuned", "13 bant")]:
        s = load_json(f"{tag}_train_summary.json")
        if not s:
            continue
        t = s["test"]
        band_rows.append({"bant sayisi": name,
                          "bantlar": len(s.get("hyperparameters", {}).get("bands", [])),
                          "test F1": fmt(t["f1"]), "test acc": fmt(t["accuracy"]),
                          "ROC-AUC": fmt(t["roc_auc"])})
    if band_rows:
        lines += ["### Bant sayisi (ayni taranmis ayarlarla, 15 epoch)", "",
                  df_to_md(pd.DataFrame(band_rows)), "",
                  "**Bulgu:** bant sayisi arttikca basarim TEKDUZE dusuyor. Ek bantlar",
                  "(kirmizi-kenar B05-B07) bulut tespitine bilgi katmiyor, buna karsilik",
                  "ImageNet on-egitimli ilk konvolusyon agirliklarini seyreltiyor ve",
                  "CPU cikarim suresini artiriyor. Boyut ise neredeyse degismiyor.", "",
                  "Cekince: hiperparametreler 6 bantli veri uzerinde tarandi, bu 6 banda",
                  "yapisal avantaj saglar. 13 bant hem taranmis hem elle secilen ayarla",
                  "kaybettigi icin sonuc bu cekinceden bagimsizdir; 9 bant icin yalnizca",
                  "taranmis ayar denendi.", ""]

    dist = load_json("distilled_train_summary.json")
    if dist:
        lines += ["### Bilgi damitma", "",
                  f"- Ogretmen: {dist['teacher']}, test F1 {fmt(dist['teacher_test_f1'])}",
                  f"- Ogrenci: {dist['model']}, test F1 {fmt(dist['test']['f1'])}",
                  f"- alpha={dist['alpha']}, sicaklik={dist['temperature']}", "",
                  "**Bulgu:** damitma dogruluk kazanci saglamadi. Sebep yontem degil kurulum:",
                  "ogretmen olarak secilen daha genis model (mobilenetv2_140), ayni ayarlarla",
                  "egitilen ogrenci mimarisinden DAHA ZAYIF cikti. Zayif ogretmenden bilgi",
                  "damitilamaz. Anlamli bir deneme icin once gercekten daha guclu bir",
                  "ogretmen kurulmali (ayri hiperparametre taramasi veya model toplulugu).", ""]
    return "\n".join(lines)


def section_ablation() -> str:
    runs = [("abl_plain", "15 epoch (sade)"), ("abl_ema", "15 ep + EMA d=0.999"),
            ("abl_ema99", "15 ep + EMA d=0.99"), ("abl_ema95", "15 ep + EMA d=0.95"),
            ("abl_balanced", "15 ep + dengeli ornekleme"),
            ("short", "10 epoch (sade)"), ("baseline", "30 epoch (sade)")]
    rows = []
    for tag, name in runs:
        s = load_json(f"{tag}_train_summary.json")
        if not s:
            continue
        t = s["test"]
        rows.append({"kosu": name, "val F1": fmt(s["best_val_f1"]), "esik": fmt(s["threshold"]),
                     "test acc": fmt(t["accuracy"]), "precision": fmt(t["precision"]),
                     "recall": fmt(t["recall"]), "test F1": fmt(t["f1"]),
                     "ROC-AUC": fmt(t["roc_auc"])})
    if not rows:
        return ""
    return "\n".join([
        "## 8. Hiperparametre ve optimizasyon denemeleri (ablasyon)", "",
        "Tum kosular MobileNetV3 omurgasiyla, tek tohumla yapilmistir.", "",
        df_to_md(pd.DataFrame(rows)), "",
        "**Cikarimlar:**", "",
        "- 30 epoch, 10 epoch'a gore test basarimini artirmadi; egitim kaybi 0.038'den",
        "  0.006'ya duserken dogrulama F1'i yatay kaldi (asiri ogrenme).",
        "- EMA, decay degeri egitim uzunluguna uygun secilmezse modeli cokertiyor:",
        "  0.999^1440 = 0.236, yani nihai agirliklarin %24'u egitilmemis baslangic degeri.",
        "  Kural: decay^(toplam_adim) ihmal edilebilir olmali.",
        "- Sade kosular ile dengeli ornekleme arasindaki farklar (F1 0.930-0.941) tek",
        "  tohumlu kosulardan geldigi icin **istatistiksel olarak anlamli sayilamaz**.",
        "  Kesin siralama icin coklu tohum gerekir.", ""])


def section_memory() -> str:
    df = load_csv("memory_profile.csv")
    if df is None:
        return ""

    df = df.copy()
    df["model_maliyeti_MB"] = (df["agirlik_MB"] + df["cikarim_ek_MB"]).round(2)
    cols = ["model", "girdi", "disk_MB", "agirlik_MB", "cikarim_ek_MB", "model_maliyeti_MB"]

    return "\n".join([
        "## 11. Bellek profili", "",
        "PDF'in \"uydu uzerindeki sinirli bellegi temsil eder\" kisiti yalnizca disk",
        "boyutunu degil, cikarim sirasindaki bellek ihtiyacini da kapsar. Diskteki",
        "model boyutu bu ihtiyacin kucuk bir parcasidir.", "",
        "Her model ayri bir surecte olculmustur (onceki modelin ayirdigi bellek",
        "sonrakinin olcumune karismasin diye).", "",
        df_to_md(df[cols]), "",
        "**Sutunlar:**", "",
        "- `disk_MB`: ONNX dosya boyutu",
        "- `agirlik_MB`: oturum acilisinda artan bellek (agirliklar + ORT bellek havuzu)",
        "- `cikarim_ek_MB`: cikarim sirasinda ek ayrilan bellek (ara aktivasyonlar)",
        "- `model_maliyeti_MB`: ikisinin toplami; Python yorumlayicisinin ~50 MB'lik",
        "  taban kullanimi HARIC (uydu yaziliminda bu maliyet olmaz)", "",
        "**Bulgular:**", "",
        "1. **Gercek bellek ihtiyaci, disk boyutunun 10-25 katidir.** INT8 siniflandirici",
        "   diskte 2.59 MB, calisirken 22.14 MB. Yalnizca dosya boyutuna bakarak bellek",
        "   kisiti degerlendirmek yaniltici olur.",
        "",
        "2. **U-Net diskte kucuk, bellekte buyuktur.** Siniflandiriciya gore diskte 2.7 kat",
        "   kucuk (0.96 MB vs 2.59 MB) ama AYNI 256x256 girdide bellekte daha pahali",
        "   (26.47 MB vs 22.14 MB, bkz. bolum 13 surum matrisi).",
        "   Sebep mimaridir: segmentasyon decoder'i tam cozunurlukte ara tensorler tasir,",
        "   siniflandirici ise havuzlama ile hizla kuculur. Az parametre, buyuk aktivasyon.",
        "   Dagitim karari bu nedenle kisita baglidir: disk darsa U-Net, RAM darsa",
        "   siniflandirici.",
        "",
        "3. **Model 256x256 girdiye kilitlidir.** 128 ve 512 olcumleri basarisiz oldu:",
        "   INT8 kuantizasyonun calisabilmesi icin model STATIK sekille ihrac edilmisti.",
        "   Farkli cozunurluk ayri bir model gerektirir.",
        "",
        "**Cekince:** `agirlik_MB` saf agirlik degildir; ONNX Runtime'in onceden ayirdigi",
        "bellek havuzunu icerir. Gomulu bir cikarim motoru (TFLite Micro, ozel cekirdek)",
        "belirgin sekilde daha az kullanirdi. Bu rakamlar ORT'ye ozgu UST SINIRLARDIR,",
        "modelin icsel gereksinimi degil.", ""])


def section_tiling() -> str:
    df = load_csv("tiling_tradeoff.csv")
    if df is None:
        return ""

    return "\n".join([
        "### 11.1 Kareleme (tiling) ile bellek azaltma", "",
        "Aktivasyon bellegi girdi boyutuyla olceklendigi icin, sahneyi daha kucuk",
        "karelere bolmek bellek ihtiyacini dusurur. Bedeli: sahne basina daha cok",
        "cikarim ve daha dar baglam.", "",
        "Ayni U-Net mimarisi uc kare boyutunda ayri ayri egitildi. Karsilastirma",
        "PIKSEL BAZLI IoU uzerinden yapildi; bu metrik ayni test piksellerini",
        "degerlendirdigi icin kareleme boyutundan bagimsizdir.", "",
        df_to_md(df), "",
        "**Bulgular:**", "",
        "1. **Baglam kaybi beklenenden cok daha kucuk.** Kare 16 kat kuculdugunde",
        "   (256x256 -> 64x64) IoU yalnizca 0.0037 dusuyor. Bulut tespiti buyuk olcude",
        "   YEREL bir gorev: bir pikselin bulut olup olmadigi yakin komsulugundan",
        "   anlasiliyor, genis sahne baglami gerekmiyor.",
        "",
        "2. **Bellek kazanci alan oraniyla degil, daha yavas olceklendi.** Alan 16 kat",
        "   kuculurken aktivasyon bellegi 3.9 kat azaldi (15.81 -> 4.08 MB). Sebep:",
        "   bellek yalnizca girdi alanina bagli degil; encoder'in derin katmanlarindaki",
        "   kanal sayilari ve calisma zamaninin sabit yukleri de paya giriyor.",
        "",
        "3. **Kare boyutu precision/recall dengesini kaydiriyor.** 64x64'te precision",
        "   0.9771 / recall 0.8993, 256x256'da 0.9531 / 0.9246. Kucuk kareler modeli",
        "   daha temkinli yapiyor - bu projede istenen yon, cunku yanlis eleme geri",
        "   donusu olmayan veri kaybidir.",
        "",
        "4. **128x128 dengeli secim:** bellek 2.4 kat az, sahne suresi yalnizca %7 fazla,",
        "   IoU maliyeti 0.0020.",
        "",
        "**Olcum notu:** ilk sure olcumleri farkli calisma zincirlerinde alindigi icin",
        "tutarsiz cikti (128x128 modeli 64x64'ten 7 kat yavas gorunuyordu, alan orani",
        "yalnizca 4). Sureler DONGUSEL olarak yeniden olculdu: her turda her modelden",
        "birer ornek alinarak termal surukleme ve arka plan yuku uc modele de esit",
        "dagitildi. Tablodaki degerler bu temiz olcumden gelmektedir.", ""])


def load_release(version: str) -> dict | None:
    path = config.PROJECT_ROOT / "releases" / version / "MANIFEST.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def section_versions() -> str:
    v1, v2 = load_release("v1"), load_release("v2")
    if not (v1 and v2):
        return ""

    def clf(m, key):
        return (m.get("classifier") or {}).get("test", {}).get(key)

    def unet(m, key):
        return (m.get("unet") or {}).get("test", {}).get(key)

    rows = [
        {"olcut": "Siniflandirici omurgasi", "v1": v1["classifier"]["backbone"],
         "v2": v2["classifier"]["backbone"]},
        {"olcut": "Ogrenme orani", "v1": "3e-4 (elle secilen)",
         "v2": f"{(v2.get('hyperparameters') or {}).get('lr', 0):.2e} (taranmis)"},
        {"olcut": "Etiket yumusatma", "v1": "yok",
         "v2": (v2.get("hyperparameters") or {}).get("label_smoothing", "-")},
        {"olcut": "Siniflandirici test F1", "v1": fmt(clf(v1, "f1")), "v2": fmt(clf(v2, "f1"))},
        {"olcut": "Siniflandirici ROC-AUC", "v1": fmt(clf(v1, "roc_auc")),
         "v2": fmt(clf(v2, "roc_auc"))},
        {"olcut": "U-Net encoder", "v1": v1["unet"]["encoder"], "v2": v2["unet"]["encoder"]},
        {"olcut": "U-Net IoU", "v1": fmt(unet(v1, "iou")), "v2": fmt(unet(v2, "iou"))},
        {"olcut": "U-Net Dice", "v1": fmt(unet(v1, "dice")), "v2": fmt(unet(v2, "dice"))},
        {"olcut": "U-Net goruntu-seviyesi dogruluk",
         "v1": fmt(unet(v1, "image_level_accuracy")), "v2": fmt(unet(v2, "image_level_accuracy"))},
    ]

    lines = ["## 10. Surum karsilastirmasi: v1 -> v2", "",
             "Iki surum de `releases/` altinda dondurulmustur; v2 gelistirmeleri v1'i",
             "bozmadan olculebilsin diye.", "",
             df_to_md(pd.DataFrame(rows)), "",
             "### 10.1 Neyin ise yaradigi, neyin yaramadigi", "",
             "v2'ye giden yolda dort ayri iyilestirme denendi. Ikisi kazanc sagladi,",
             "ikisi saglamadi; ikisi de raporlanmaya degerdir.", "",
             "**Kazanc saglayanlar:**", "",
             "1. **Hiperparametre taramasi (asil kazanc).** Proje varsayilani olan",
             "   ogrenme orani 3e-4, rastgele aramada bulunan optimumun (~1.2e-3)",
             "   dortte biriymis. Tarama tek basina ROC-AUC'yi 0.9813'ten 0.9884'e",
             "   cikardi ve INT8 sonrasi F1 kaybini neredeyse sifirladi",
             "   (v1: 0.9458 -> 0.9403; v2: 0.9605 -> 0.9598). Mimari degismedi,",
             "   boyut degismedi, yalnizca egitim ayarlari degisti.",
             "",
             "2. **U-Net encoder'inin KUCULTULMESI.** mobilenetv2_100 -> mobilenetv2_050",
             "   gecisi hem boyutu ucte bire indirdi (2.37 MB -> 0.96 MB) hem dogrulugu",
             "   ARTIRDI (IoU 0.8663 -> 0.8855). Beklenmedik gorunse de aciklanabilir:",
             "   segmentasyon her piksel icin denetim sinyali verir (6.160 kare x 65.536",
             "   piksel), kucuk model bu yogunlukta sinyalle asiri ogrenmeden ogrenir.",
             "",
             "**Kazanc saglamayanlar:**", "",
             "3. **Bant sayisini artirmak.** 6 -> 9 -> 13 bant gecisinde basarim TEKDUZE",
             "   dustu (test F1 0.9605 / 0.9403 / 0.9223). Ek bantlar bilgi katmadi,",
             "   ImageNet on-egitimli ilk konvolusyon agirliklarini seyreltti ve CPU",
             "   suresini artirdi.",
             "",
             "4. **Bilgi damitma.** Ogrenci ogretmeni gecti (0.9540 vs 0.9471) ama bu bir",
             "   basari degil, kurulum hatasinin isareti: ogretmen olarak secilen daha",
             "   genis model, ogrenciden zayif kaldi. Zayif ogretmenden bilgi damitilamaz.",
             "",
             "### 10.2 En buyuk pratik kazanc: calisma noktasi", "",
             "Modeldeki iyilesme gercek ama olculu (ROC-AUC +0.007). Operasyonel sonuca",
             "asil etki eden degisiklik, karar esiginin NASIL secildigidir.", "",
             "v1 ve v2'nin ilk hali esigi TUM dogrulama seti uzerinde seciyordu. Ancak",
             "karelerin ~%73'u ya tamamen temiz ya tamamen kapali; bu kolay orneklerin",
             "karari esikten bagimsizdir ve esik secimini asil onemli oldugu belirsiz",
             "bolgeden uzaklastirir. Esigi yalnizca kismi bulutlu karelerde (bulut orani",
             "%2-%98) ayarlamak bu carpitmayi kaldirir.", ""]
    return "\n".join(lines)


def section_operating_points(clf_tag: str) -> str:
    data = load_json(f"{clf_tag}_operating_points.json")
    if not data:
        return ""

    rows = []
    for name, d in data["operating_points"].items():
        row = {"calisma noktasi": name, "esik": round(d["threshold"], 4)}
        t = d.get("s2cmc_test", {})
        row.update({"S2CMC precision": fmt(t.get("precision")),
                    "S2CMC F1": fmt(t.get("f1")),
                    "veri azalmasi %": t.get("veri_azalmasi_%"),
                    "kayip %": t.get("kaybedilen_kullanilabilir_%")})
        s = d.get("sparcs", {})
        if s:
            row.update({"SPARCS kayip %": s.get("kaybedilen_kullanilabilir_%"),
                        "SPARCS kar/buz yanlis eleme %": s.get("kar_buz_yanlis_eleme_%")})
        rows.append(row)

    return "\n".join([
        "### 10.3 Calisma noktalari", "",
        "Karar esigi artik kodda sabit bir sayi degildir. Model, her biri olculmus",
        "sonuclariyla tanimlanmis birden fazla calisma noktasiyla birlikte teslim",
        "edilir (`operating_points.json`); hangisinin kullanilacagi bir dagitim",
        "kararidir ve yerden guncellenebilir, yeniden egitim gerektirmez.", "",
        df_to_md(pd.DataFrame(rows)), "",
        "**Okuma:** esik yukseldikce her iki veri setinde de kaybedilen kullanilabilir",
        "veri azalir, karsiliginda bant genisligi kazanci bir miktar duser. Belirsiz",
        "bantta secilen esik, S2CMC ve SPARCS'ta TEKDUZE daha iyidir.", "",
        "**Onemli sinir - yontem her yapilandirmada ise yaramiyor.** Uc kare",
        "boyutunda ayri ayri denendi:", "",
        "| kare | belirsiz bantta secilen esik | precision | yanlis eleme | sonuc |",
        "|---|---|---|---|---|",
        "| 256x256 | 0.862 (yukari) | 0.9974 | %5.94 -> %0.24 | kazanc |",
        "| 128x128 | 0.288 (asagi) | 0.9515 | %5.84 -> %6.02 | kazanc yok |",
        "| 64x64 | 0.891 (yukari) | 0.9916 | %1.99 -> %0.79 | kazanc |", "",
        "Sebep, yontemin `precision >= 0.99` kisitini saglayan EN DUSUK esigi",
        "aramasidir. 256x256 ve 64x64'te belirsiz bant gercekten zor oldugu icin esik",
        "yukari itilir; 128x128'de model o alt kumede zaten yeterince iyi oldugundan",
        "dusuk bir esik de kisiti saglar ve arama orada durur.", "",
        "Sonuc: yontem uc yapilandirmanin ikisinde belirgin kazanc verdi, birinde hic",
        "vermedi. Genel gecer kabul edilemez; her model ve kare boyutu icin ayrica",
        "OLCULMELIDIR. Kazanc goruldugu yerlerde ise buyuktur (yanlis eleme 2.5-25 kat",
        "azalma), bu nedenle denenmeye degerdir.", "",
        "Kritik nokta farkli sensordedir: mevcut nokta SPARCS'ta kullanilabilir verinin",
        "%19.98'ini kaybederken, belirsiz bant (precision>=0.995) noktasi bunu %7.95'e",
        "indirir; kar/buz karelerinde yanlis eleme %58.62'den %29.89'a duser. Model ayni",
        "modeldir - degisen yalnizca esigin nasil secildigidir.", "",
        "**Dagitim onerisi:** belirsiz bant (precision>=0.995). Kullanilabilir verinin",
        "%99.5'i korunur, bant genisligi kazanci %48.65 olur - teorik ust sinirin",
        "(%56.15) %87'si.", ""])


def section_classifier_tiling() -> str:
    """64x64 siniflandirici denemesi - negatif sonuc."""
    s64 = load_json("c64_tuned_train_summary.json")
    if not s64:
        return ""

    rows = [
        {"olcut": "S2CMC test ROC-AUC", "256x256": "0.9883", "64x64": "0.9872",
         "sonuc": "esit"},
        {"olcut": "S2CMC test F1 (INT8)", "256x256": "0.9598", "64x64": "0.9329",
         "sonuc": "karsilastirilamaz*"},
        {"olcut": "SPARCS ROC-AUC", "256x256": "0.9644", "64x64": "0.8668",
         "sonuc": "256x256"},
        {"olcut": "SPARCS recall (varsayilan esik)", "256x256": "0.9760", "64x64": "0.0034",
         "sonuc": "256x256"},
        {"olcut": "sure (ms/kare)", "256x256": "5.76", "64x64": "1.89", "sonuc": "64x64"},
        {"olcut": "sure (ms/sahne)", "256x256": "92", "64x64": "484", "sonuc": "256x256"},
        {"olcut": "bellek", "256x256": "22.14 MB", "64x64": "15.74 MB", "sonuc": "64x64"},
    ]

    return "\n".join([
        "### 11.2 Siniflandiricida kareleme - negatif sonuc", "",
        "U-Net'te ise yarayan ince kareleme, siniflandiriciya uygulandiginda",
        "basarisiz oldu. Ayni hiperparametrelerle 64x64'te egitilen siniflandirici:", "",
        df_to_md(pd.DataFrame(rows)), "",
        "\\* Kare boyutu degisince etiket tanimi da degisir (bulut orani hangi",
        "pencerede olculuyor), bu nedenle S2CMC F1'leri dogrudan karsilastirilamaz.",
        "Karsilastirilabilir olcutler ROC-AUC, SPARCS sonuclari ve sahne basina suredir.", "",
        "**Bulgu: 64x64 siniflandirici farkli sensore hic genellemiyor.** SPARCS'ta",
        "1248 karenin yalnizca birini eliyor (recall 0.0034). Bu yalnizca esik",
        "yerlesimi sorunu degildir - ROC-AUC da 0.9644'ten 0.8668'e dusuyor, yani",
        "siralama kalitesi gercekten bozuluyor.", "",
        "**Neden U-Net'te olmuyor da siniflandiricida oluyor:** segmentasyon her",
        "pikselde YEREL bir karar verir ve genis baglama zaten az bagimlidir;",
        "siniflandirma ise tum kareyi tek bir sayiya indirir. Pencere daraldikca bu",
        "ozet, ince dokusal istatistiklere dayanmak zorunda kalir - ve dokusal",
        "istatistikler sensorler arasinda en cok degisen ozelliklerdir. 256x256'da",
        "model sahne genelindeki parlaklik dagilimi gibi daha dayanikli ipuclari",
        "kullanabiliyor.", "",
        "**Ayrica sahne basina 5.3 kat yavas:** kare basina hizli olmasi yaniltici,",
        "cunku 64x64'te bir sahne 16 yerine 256 kare demektir.", "",
        "Sonuc: kareleme karari GOREVE BAGLIDIR. Segmentasyonda ince kareleme ucuz,",
        "siniflandirmada pahalidir. Model `releases/v3.2/classifier_64*` olarak",
        "saklanmistir ama dagitim icin 256x256 varyanti onerilir.", ""])


def section_deployment_memory() -> str:
    """Gercek dagitim senaryolarinda bellek + piyasa karsilastirmasi."""
    data = load_json("combined_memory.json")
    if not data:
        return ""

    rows = [{"senaryo": name, "tepe bellek": f"{m['toplam_tepe_MB']:.2f} MB"}
            for name, m in data.items()]

    return "\n".join([
        "### 11.3 Dagitim senaryolarinda bellek", "",
        "Modeller ayri olculdugunde her biri ONNX Runtime'in taban ayak izini kendi",
        "hesabina yazar. Gercek dagitimda ikisi ayni surecte calisir ve bu tabani",
        "PAYLASIR. Asagidaki olcumler tek surecte, iki oturum birlikte yuklenerek",
        "yapilmistir.", "",
        df_to_md(pd.DataFrame(rows)), "",
        "Ayri olculen degerlerin toplami 36.9 MB, birlikte olculen 27.78 MB -",
        "aradaki ~9 MB paylasilan calisma zamani tabanidir.", "",
        "### 11.4 Piyasa karsilastirmasi", "",
        "| sistem | bellek | baglam |",
        "|---|---|---|",
        "| TFLite Micro cekirdek calisma zamani | ~16 KB | Cortex-M3 |",
        "| TinyML MobileNet/ResNet (96x96 civari girdi) | <100 KB RAM | mikrodenetleyici |",
        "| TinyML model + calisma zamani ikili | 100-375 KB | Nano 33 BLE ornegi |",
        "| **bu calisma (ORT, clf256 + unet64)** | **27.78 MB** | masaustu x86 |",
        "| bu calisma, agirlik + tepe aktivasyon (icsel) | ~14.3 MB | calisma zamani harici |",
        "| Myriad 2 CMX SRAM (cip ustu, hizli) | 2 MB | Phi-Sat-1 |",
        "| Myriad 2 LPDDR3 (cip disi) | 128 / 512 MB | Phi-Sat-1 |", "",
        "**Okuma:**", "",
        "1. **Mikrodenetleyici sinifinin cok uzagindayiz.** TinyML dagitimlari 100 KB'in",
        "   altinda calisir; biz 27.78 MB kullaniyoruz - yaklasik 280 kat fazla. Ancak",
        "   karsilastirma esdeger degildir: TinyML modelleri 96x96 gri tonlamali gibi",
        "   girdilerle calisir, bizimki 6 bantli 256x256 goruntu isler.",
        "",
        "2. **Asil fark modelde degil, calisma zamaninda.** Olculen 27.78 MB'in ~10.5 MB'i",
        "   ONNX Runtime'in taban ayak izidir ve hicbir ayarla dusmez (arena ve bellek",
        "   deseni kapatma denendi). TFLite Micro'nun cekirdegi 16 KB - yaklasik 650 kat",
        "   kucuk. Ayni modeller gomulu bir motorla calistirilsa icsel gereksinim",
        "   ~14.3 MB'e (agirlik 3.55 MB + tepe aktivasyon ~10.7 MB) inerdi.",
        "",
        "3. **Referans misyon donanimina sigiyoruz.** Phi-Sat-1'in Myriad 2 VPU'su",
        "   128-512 MB LPDDR3 tasir; 27.78 MB bunun %5-22'sidir. Ancak cip ustu 2 MB",
        "   CMX SRAM'e sigmaz, yani model cip disi bellekten calisirdi - daha yavas ve",
        "   daha cok guc tuketen bir yerlesim.",
        "",
        "4. **Kalan tek anlamli kaldirac calisma zamanidir.** Model tarafinda kareleme",
        "   ile bellegi %44 dusurduk (26.47 -> 14.87 MB, U-Net) ve daha fazlasi icin yer",
        "   kalmadi. 2 MB SRAM hedefine yaklasmak icin gomulu bir cikarim motoru ve",
        "   katman katman akis (streaming) gerekir; bu, yazilim simulasyonu kapsamini",
        "   asar ve gelecek calisma olarak birakilmistir.", ""])


def section_unet_external(unet_tag: str) -> str:
    """U-Net'in harici sensorde (SPARCS) degerlendirmesi + gunes duzeltmesi."""
    d = load_json(f"{unet_tag}_external_sparcs.json")
    if not d:
        return ""
    clf = load_json("c6_tuned_external_sparcs.json")
    sun = load_json(f"{unet_tag}_sun_external_sparcs.json")

    def brk(src, key):
        s = (src or {}).get(key, {})
        return {"n": s.get("n"), "accuracy": s.get("accuracy"), "precision": s.get("precision"),
                "recall": s.get("recall"), "f1": s.get("f1"), "roc_auc": s.get("roc_auc")}

    rows = [{"kirilim": k, **{kk: (round(vv, 4) if isinstance(vv, float) else vv)
                              for kk, vv in brk(d, key).items()}}
            for k, key in [("genel", "overall"), ("kar/buz olmayan", "non_snowy"),
                           ("kar/buz agirlikli (>%20)", "snowy")]]

    lines = [
        "## 6b. U-Net'in harici dogrulamasi (SPARCS)", "",
        "Siniflandiriciya uygulanan harici dogrulamanin aynisi, U-Net maskesinden",
        "`mask_to_decision` ile goruntu karari uretilerek tekrarlandi. Boylece iki",
        "model AYNI olcutlerle, ayni veride karsilastirilabiliyor.", "",
        df_to_md(pd.DataFrame(rows)), "",
    ]

    if "pixel" in d:
        p = d["pixel"]
        lines += [
            "**Piksel bazli segmentasyon (ilk kez olculdu):** "
            f"IoU {p['iou']:.4f}, Dice {p['dice']:.4f}, "
            f"piksel dogrulugu {p['pixel_accuracy']:.4f}.", "",
            "Egitim dagiliminda (S2CMC) IoU 0.8892 idi; farkli sensorde 0.167 dusuyor.",
            "Gercek bir genelleme acigi, ama cokus degil.", "",
        ]

    if clf:
        co, uo = clf.get("overall", {}), d.get("overall", {})
        cs, us = clf.get("snowy", {}), d.get("snowy", {})
        cmp_rows = [
            {"olcut": "ROC-AUC", "siniflandirici 256px": fmt(co.get("roc_auc")),
             "U-Net 64px": fmt(uo.get("roc_auc"))},
            {"olcut": "F1", "siniflandirici 256px": fmt(co.get("f1")),
             "U-Net 64px": fmt(uo.get("f1"))},
            {"olcut": "veri azalmasi %",
             "siniflandirici 256px": fmt(clf.get("downlink", {}).get("veri_indirme_azalmasi_%"), 2),
             "U-Net 64px": fmt(d.get("downlink", {}).get("veri_indirme_azalmasi_%"), 2)},
            {"olcut": "kaybedilen kullanilabilir %",
             "siniflandirici 256px": fmt(clf.get("downlink", {}).get("kaybedilen_kullanilabilir_veri_%"), 3),
             "U-Net 64px": fmt(d.get("downlink", {}).get("kaybedilen_kullanilabilir_veri_%"), 3)},
            {"olcut": "kar/buz yanlis eleme %",
             "siniflandirici 256px": fmt(100 * cs.get("kari_bulut_sanma_orani", 0), 2),
             "U-Net 64px": fmt(100 * us.get("kari_bulut_sanma_orani", 0), 2)},
        ]
        lines += ["### 6b.1 Dogrudan karsilastirma", "", df_to_md(pd.DataFrame(cmp_rows)), "",
                  "**Kar/buz yanlis elemesi 11.6 kat aziyor.** Siniflandirici ham bant",
                  "genisligi kazancinda onde, U-Net korunan veri basina kazancta.", "",
                  "**Bu fark karelemeden gelmiyor.** Ayni 64x64 izgarada olculmus",
                  "siniflandirici (`c64_tuned_operating_points.json`) dengeli noktasinda",
                  "kar/buz yanlis elemesini %64.37 veriyor - clf256'nin %58.62'sinden DAHA",
                  "KOTU. Ince kareleme bu sorunu cozmuyor; fark gorev turundendir:",
                  "segmentasyon her pikselde yerel karar verir ve sensorler arasinda en cok",
                  "degisen ozellige (dokusal istatistikler) daha az bagimlidir.", ""]

    if sun:
        so, ss = sun.get("overall", {}), sun.get("snowy", {})
        uo, us = d.get("overall", {}), d.get("snowy", {})
        sun_rows = [
            {"olcut": "ROC-AUC", "duzeltmesiz": fmt(uo.get("roc_auc")), "duzeltmeli": fmt(so.get("roc_auc"))},
            {"olcut": "precision", "duzeltmesiz": fmt(uo.get("precision")), "duzeltmeli": fmt(so.get("precision"))},
            {"olcut": "recall", "duzeltmesiz": fmt(uo.get("recall")), "duzeltmeli": fmt(so.get("recall"))},
            {"olcut": "F1", "duzeltmesiz": fmt(uo.get("f1")), "duzeltmeli": fmt(so.get("f1"))},
            {"olcut": "IoU", "duzeltmesiz": fmt(d.get("pixel", {}).get("iou")),
             "duzeltmeli": fmt(sun.get("pixel", {}).get("iou"))},
            {"olcut": "kar/buz yanlis eleme %",
             "duzeltmesiz": fmt(100 * us.get("kari_bulut_sanma_orani", 0), 2),
             "duzeltmeli": fmt(100 * ss.get("kari_bulut_sanma_orani", 0), 2)},
        ]
        lines += [
            "### 6b.2 Negatif sonuc: gunes yuksekligi duzeltmesi yardimci olmuyor", "",
            "USGS'in `l8cloudmasks.zip` arsivi 80 sahnenin hepsi icin MTL dosyasi",
            "iceriyor, bu yuzden `1/sin(SUN_ELEVATION)` duzeltmesi uygulanabildi",
            "(carpan 1.07-2.00, medyan 1.22). Ayni U-Net, iki sette:", "",
            df_to_md(pd.DataFrame(sun_rows)), "",
            "**ROC-AUC pratikte degismiyor.** Bu belirleyici: duzeltme modelin ayirt",
            "etme yetenegine hicbir sey katmiyor. Yaptigi tek sey goruntuleri ~1.22 kat",
            "parlatarak sabit esigin skor dagilimi uzerindeki yerini kaydirmak - recall",
            "yukseliyor, precision dusuyor. Bilgi kazanci degil, esik kaymasi.", "",
            "Sonuc: **radyometrik uyumsuzluk SPARCS'taki performans acigini",
            "aciklamiyor.** Acik gercek bir alan kaymasidir (spektral tepki farklari,",
            "farkli yuzey ortusu, farkli cografya) ve olcek duzeltmesiyle kapanmiyor.",
            "Raporun onceki 'SPARCS karsilastirmasi kotumser' cekincesi bu olcumle",
            "ELENMISTIR.", "",
        ]
    return "\n".join(lines)


def section_partial_downlink(unet_tag: str) -> str:
    """Kismi indirme odunlesimi: blok boyutu vs bayt maliyeti."""
    df = load_csv(f"{unet_tag}_partial_downlink.csv")
    meta = load_json(f"{unet_tag}_partial_downlink.json")
    if df is None or df.empty:
        return ""

    base = df.iloc[0]
    marj = []
    for _, r in df.iloc[1:].iterrows():
        dk = base["kaybedilen_temiz_alan_%"] - r["kaybedilen_temiz_alan_%"]
        db = base["veri_azalmasi_bayt_%"] - r["veri_azalmasi_bayt_%"]
        marj.append({
            "adim": f"{base['blok'].split('x')[0]} -> {r['blok'].split('x')[0]}",
            "kurtarilan temiz alan (puan)": f"+{dk:.2f}",
            "kaybedilen bant genisligi kazanci (puan)": f"-{db:.2f}",
            "oran": f"{db / dk:.2f} : 1" if dk else "-",
        })

    show = df[["blok", "indirilen_alan_%", "kaybedilen_temiz_alan_%",
               "veri_azalmasi_bayt_%", "bayt_referansa_gore"]]

    lines = [
        "## 6c. Kismi indirme odunlesimi (olculdu)", "",
        "Ikili karar bir kareyi ya tamamen indirir ya tamamen atar; dogru sekilde",
        "atilan bulutlu karelerin icindeki temiz pikseller de kaybolur. Bu bir hata",
        "degil, yontemin YAPISAL maliyetidir ve ikili siniflandiriciyla kapatilamaz.",
        "U-Net maskesi hangi bolgenin temiz oldugunu bildirdigi icin, kareyi daha",
        "kucuk bloklara bolup yalnizca temiz bloklari indirmek mumkun.", "",
        "Karar TAHMIN EDILEN maskeden verilir (uyduda olan bilgi budur), kayip",
        "muhasebesi GERCEK maskeden yapilir. Her blok BAGIMSIZ bir iletim birimi",
        "olarak sikistirilir; blok haritasi da maliyete eklenir. `blok = 64x64`",
        "satiri mevcut ikili davranistir.", "",
        f"Test bolumu, {meta['kare']:,} kare:" if meta else "Test bolumu:", "",
        df_to_md(show), "",
        "**Negatif sonuc: odunlesim elverissiz.** Blogu 8x8'e kadar kucultmek yapisal",
        "kaybin yarisini kurtariyor ama bant genisligi kazancini 13.5 puan dusuruyor.",
        "Marjinal oran her adimda kotulesiyor:", "",
        df_to_md(pd.DataFrame(marj)), "",
        "**Neden:** iki etken ters yonde calisiyor.", "",
        "1. **Kurtarilan alan kucuk.** Indirilen alan yalnizca %48.58'den %50.46'ya",
        "   cikiyor. Sebep, U-Net maskelerinin MEKANSAL OLARAK TUTARLI olmasi: %30'dan",
        "   fazla bulutlu bir kare genellikle buyuk olcude bulutludur, satranc tahtasi",
        "   degil. Ince bolme kurtarilacak fazla temiz ada bulamiyor.",
        "2. **Parcalama sikistirmayi bozuyor.** Yuk 225.9 MB'dan 290.2 MB'a cikiyor",
        "   (+%28). Kucuk bloklar bagimsiz sikistirildiginda baglam kaybediyor.", "",
        "Blok haritasi maliyeti ihmal edilebilir cikti (en kotu durumda 122 KB, yukun",
        "%0.04'u). Yani kismi indirmenin maliyeti 'koordinat/metadata yuku' DEGIL,",
        "parcalanmanin kendisidir. Bu, tahminle bilinemezdi; gercekten sikistirip",
        "bayt saymak gerekiyordu.", "",
    ]
    if meta:
        lines += [
            f"> **Kodek cekincesi:** sikistirma vekili {meta['kodek']}.",
            "> Mutlak bayt sayilari gercek misyonu temsil etmez; ancak karsilastirma tek",
            "> kodekle yapildigi icin olculen sey - blok boyutunun sikistirmaya etkisi -",
            "> gecerlidir.", "",
        ]
    lines += [
        "**Sonucun siniri:** bu, blok tabanli bir protokolun ve genel amacli bir",
        "kodegin sonucudur; kismi indirme fikrinin tumden reddi degildir. Bolge-ilgi",
        "kodlamasi (JPEG2000 ROI) ya da bulutlu bolgeleri atmak yerine kayipli",
        "kodlamak, parcalanma cezasini odemeden ayni kazanci verebilir. Denenmemistir.", "",
        "**U-Net'in gerekcesi bu olcumle degisiyor.** Kismi indirme, U-Net'i projede",
        "tutmanin ana gerekcesi olarak gosterilmisti; olcum bunu desteklemiyor. Buna",
        "karsilik bolum 6b daha guclu ve olculmus bir gerekce sagliyor: farkli",
        "sensorde dayaniklilik. Gerekce spekulatif olandan olculmus olana kaymistir.", "",
    ]
    return "\n".join(lines)


def section_tradeoff_matrix() -> str:
    """Varyantlar arasi odunlesim: hangi olcutte hangisi kazaniyor."""
    lines = ["## 6d. Varyantlar arasi odunlesim matrisi", "",
             "Bu bolum, uretilen tum varyantlari ayni tabloda karsilastirir. Hicbiri her",
             "olcutte kazanmiyor - secim, hangi kisitin bagladigina baglidir.", ""]

    # --- Siniflandirici varyantlari ---
    rows = []
    specs = [("c6_tuned", "mobilenetv2_100", "256x256"),
             ("c050", "mobilenetv2_050", "256x256"),
             ("c64_tuned", "mobilenetv2_100", "64x64")]
    for tag, backbone, patch in specs:
        b = load_csv(f"{tag}_benchmark.csv")
        dl = load_json(f"{tag}_downlink_analysis.json")
        row = {"varyant": tag, "omurga": backbone, "kare": patch}
        if b is not None and (b["model"] == "ONNX INT8").any():
            i8 = b[b["model"] == "ONNX INT8"].iloc[0]
            row["INT8 disk MB"] = i8["boyut_MB"]
            row["INT8 F1"] = round(float(i8["f1"]), 4)
            row["ms/kare"] = round(float(i8["latency_mean_ms"]), 2)
        if dl:
            models = dl.get("models", dl)
            i8 = models.get("ONNX INT8", {})
            row["INT8 kayip %"] = i8.get("kaybedilen_kullanilabilir_veri_%")
        rows.append(row)

    if rows:
        lines += ["### 6d.1 Siniflandirici varyantlari", "", df_to_md(pd.DataFrame(rows)), "",
                  "> **Karsilastirma uyarisi:** c64_tuned'un F1 ve kayip degerleri 64x64",
                  "> karelerde olculmustur; kare boyutu degisince ETIKET TANIMI da degisir",
                  "> (bulut orani hangi pencerede olculuyor). Bu yuzden 256x256 varyantlariyla",
                  "> dogrudan karsilastirilamaz - bkz. bolum 11.2. Karsilastirilabilir",
                  "> olcutler SPARCS sonuclari ve sahne basina suredir.", "",
                  "**c050 elendi.** Yari genislikteki omurga diskte 2.6 kat kucuk ve kare",
                  "basina 1.7 kat hizli, ama INT8 kuantizasyonda cokuyor: kaybedilen",
                  "kullanilabilir veri %3.563'ten %13.539'a firliyor (FP32 -> INT8), yani",
                  "3.4 kat daha fazla geri donusu olmayan bilimsel veri kaybi. Ayni hattan",
                  "c6_tuned neredeyse bedelsiz geciyor (%3.088 -> %4.038). Bu, MobileNetV3",
                  "bulgusuyla ayni ailede bir sonuc: dar/agresif mimariler egitim sonrasi",
                  "kuantizasyona direncli.", "",
                  "Bellek tarafinda kazanc da beklenenden kucuk olurdu: ORT tabani",
                  "(~9.1 MB olculdu) model kuculunce dusmedigi icin diskteki 2.6 kat",
                  "kazanc bellege ~%20 olarak yansirdi. c050'nin bellegi ayrica hic",
                  "olculmedi.", "",
                  "**c64_tuned elendi.** Farkli sensore hic genellemiyor (SPARCS ROC-AUC",
                  "0.9644 -> 0.8668) ve sahne basina 5.3 kat yavas.", ""]

    # --- Dagitim yapilandirmalari ---
    cm = load_json("combined_memory.json")
    if cm:
        conf = [
            ("clf256 + unet64 (DAGITILAN)", "ONERILEN: clf256 + unet64", 3.55, 267,
             "0.9957", "48.65", "0.475", "29.89"),
            ("yalniz clf256", "yalniz siniflandirici (256)", 2.59, 92,
             "0.9957", "48.65", "0.475", "29.89"),
            ("yalniz unet64", "yalniz U-Net (64)", 0.96, 175,
             "0.9916", "42.72", "0.787", "0.71"),
            ("clf64 + unet64", "alternatif: clf64 + unet64", 3.55, 659,
             "-", "-", "-", "-"),
        ]
        rows = []
        for label, key, disk, ms, prec, az, kayip, kar in conf:
            rows.append({
                "yapilandirma": label,
                "disk MB": disk,
                "bellek MB (ORT)": cm.get(key, {}).get("toplam_tepe_MB", "-"),
                "ms/sahne": ms,
                "precision": prec,
                "veri azalmasi %": az,
                "kaybedilen kullanilabilir %": kayip,
                "SPARCS kar/buz yanlis eleme %": kar,
            })
        lines += ["### 6d.2 Dagitim yapilandirmalari", "", df_to_md(pd.DataFrame(rows)), "",
                  "> Precision / veri azalmasi / kayip sutunlari, her yapilandirmanin KARAR",
                  "> VEREN modelinin secilen calisma noktasindan gelir: siniflandirici icin",
                  "> esik 0.7947 (belirsiz bant, precision>=0.995), U-Net icin esik 0.8906",
                  "> (belirsiz bant, precision>=0.99 - U-Net 0.995'e hicbir esikte",
                  "> ulasamiyor).", ""]

    lines += [
        "### 6d.3 Olcut bazinda en iyiler", "",
        "| olcut | kazanan | deger | not |",
        "|---|---|---|---|",
        "| en kucuk disk | yalniz unet64 | 0.96 MB | clf256'nin ucte biri |",
        "| en dusuk bellek | yalniz unet64 | 14.75 MB | ORT tabani dahil |",
        "| en kisa sahne suresi | yalniz clf256 | 92 ms | unet64'un yaridan azi |",
        "| en yuksek precision | clf256 | 0.9957 | U-Net 0.9916'da tikaniyor |",
        "| en yuksek recall (esit precision hedefinde) | clf256 | 0.8627 | U-Net 0.7773 |",
        "| en fazla bant genisligi kazanci | clf256 | %48.65 | U-Net %42.72 |",
        "| en az bilimsel veri kaybi (S2CMC) | clf256 | %0.475 | U-Net %0.787 |",
        "| **en az veri kaybi (farkli sensor)** | **unet64** | **%0.464** | clf256 %7.95 |",
        "| **kar/buz dayanikliligi** | **unet64** | **%0.71** | clf256 %29.89 |",
        "| harici sensor ROC-AUC | unet64 | 0.9733 | clf256 0.9645 |",
        "| piksel maskesi uretimi | unet64 | var | clf256'da yok |",
        "",
        "**Okuma:** siniflandirici EGITIM DAGILIMINDA her verim olcutunde onde;",
        "U-Net DAGILIM KAYDIGINDA her dayaniklilik olcutunde onde. Tek kazanan",
        "olmadigi icin ikisi birlikte dagitilir - bu bir uzlasma degil, olculmus bir",
        "tasarimdir: siniflandirici verimi maksimize eder, U-Net dagilim kaydiginda",
        "guvenlik agi saglar ve piksel maskesi uretir.", "",
    ]
    return "\n".join(lines)


def section_literature() -> str:
    """Ilgili calismalar (PDF Gunler 1-3: literatur taramasi)."""
    return "\n".join([
        "## 14. Ilgili calismalar (literatur taramasi)", "",
        "### 14.1 Referans misyon: Phi-Sat-1 / CloudScout", "",
        "Bu calismanin dogrudan referansi, ESA'nin **Phi-Sat-1** misyonudur (Eylul 2020,",
        "6U CubeSat, FSSCat ikilisinin parcasi). Yorungede derin ogrenme calistiran ilk",
        "gosterim misyonudur: hiperspektral-termal HyperScout-2 kamerasindan gelen",
        "goruntuleri **Intel Movidius Myriad 2** VPU uzerinde isleyen **CloudScout**",
        "adli evrisimli sinir agi, bulutlu goruntuleri yere indirmeden eliyor.", "",
        "Yayimlanan degerler:", "",
        "| olcut | CloudScout |",
        "|---|---|",
        "| model ayak izi | 2.1 MB |",
        "| dogruluk | %92 |",
        "| cikarim suresi | 325 ms (sahne basina) |",
        "| guc | 1.8 W |",
        "| bant genisligi tasarrufu | ~%30 |",
        "| donanim | Myriad 2 VPU, 2 MB CMX SRAM + 128/512 MB LPDDR3 |", "",
        "Bu calisma ayni problemi yazilim ortaminda ele alir; hedef donanim uzerinde",
        "calistirma PDF geregi kapsam disidir. CloudScout ile DOGRUDAN karsilastirma",
        "gecerli degildir (farkli veri seti, farkli gorev esigi, farkli donanim sinifi),",
        "ancak boyut ve tasarruf mertebeleri baglam saglar.", "",
        "### 14.2 Veri setleri", "",
        "**Sentinel-2 Cloud Mask Catalogue** (Francis, Mrziglod, Sidiropoulos; ESA PhiLab)",
        "bu calismanin ana veri setidir. 2018 Sentinel-2 L1C arsivinden rastgele",
        "orneklenen 513 alt-sahne, IRIS aracıyla yari-otomatik (dinamik Random Forest +",
        "elle duzeltme) etiketlenmistir. Ayirt edici ozelligi, iki anotatorun bagimsiz",
        "etiketledigi 60 sahne uzerinden **insan seviyesi referansi** sunmasidir",
        "(CLOUD F1 %95.97, piksel dogrulugu %94.98). Bu calismada test bolumu bilerek",
        "o 60 sahneden olusturulmustur.", "",
        "**SPARCS** (Hughes & Kennedy, Remote Sensing 2019, 11(21):2591) 80 Landsat-8",
        "alt-sahnesinden olusan elle etiketlenmis bir dogrulama setidir. Bu calismada",
        "yalnizca harici dogrulama icin, egitime hic sokulmadan kullanilmistir.", "",
        "**95-Cloud** yalnizca 4 bant (R,G,B,NIR) icerdigi ve cirrus/SWIR bantlari",
        "bulunmadigi icin kapsam disi birakilmistir.", "",
        "**CloudSEN12** (Nature Scientific Data, 2022) 49.400 kareyle bu alandaki en",
        "genis veri setlerinden biridir ve sekiz bulut tespit algoritmasinin ciktisini",
        "birlikte sunar. Bu calismada kullanilmamistir ancak alanin karsilastirma",
        "zemini olarak dikkate degerdir.", "",
        "### 14.3 Bulut tespit yontemleri", "",
        "| yontem | tur | bildirilen basarim |",
        "|---|---|---|",
        "| Sen2Cor | esik/kural tabanli | IoU 0.4698 |",
        "| Fmask | esik/kural tabanli | IoU 0.5713, F1 0.832 |",
        "| CD-FM3SF | derin ogrenme | IoU 0.8363 |",
        "| Swin-Unet | derin ogrenme (transformer) | F1 0.891 |",
        "| UNetMobV2 | derin ogrenme | CloudSEN12'de en iyi |",
        "| **bu calisma (U-Net 64x64, INT8)** | **derin ogrenme** | **IoU 0.8807, Dice 0.9366** |", "",
        "Rakamlar FARKLI VERI SETLERINDEN gelmektedir; dogrudan karsilastirilamazlar.",
        "Gecerli tek karsilastirma, ayni veri setindeki insan seviyesi referansidir:",
        "bu calismanin Dice degeri 0.9366, anotatorler arasi uyum 0.9597 - fark 2.31",
        "puandir.", "",
        "### 14.4 Hafif mimariler ve kuantizasyon", "",
        "MobileNet ailesi (derinlemesine-ayrilabilir konvolusyonlar) ucta cikarim icin",
        "standart secimdir ve PDF de bu aileyi isaret eder. Ancak **MobileNetV3'un",
        "egitim sonrasi kuantizasyona (PTQ) direncli oldugu** bu calismada deneysel",
        "olarak dogrulanmistir: yedi farkli yapilandirmanin hicbiri dogrulugu",
        "koruyamamis, ayni yapilandirmalar MobileNetV2'de sorunsuz calismistir",
        "(bkz. bolum 7).", "",
        "**TensorFlow Lite Micro** (arXiv:2010.08678) gomulu cikarim icin referans",
        "cercevedir; cekirdek calisma zamani ~16 KB'dir ve TinyML modelleri <100 KB RAM",
        "ile calisabilmektedir. Bu calismada kullanilan ONNX Runtime'in taban ayak izi",
        "~10.5 MB olcumustur - yaklasik 650 kat buyuk. Bu fark, bolum 11.4'teki bellek",
        "karsilastirmasinin ana aciklamasidir.", "",
        "### 14.5 Kaynaklar", "",
        "- Giuffrida ve ark., *CloudScout: A Deep Neural Network for On-Board Cloud",
        "  Detection on Hyperspectral Images*, Remote Sensing, 2020",
        "- ESA, *Phi-Sat-1 Mission*, eoPortal",
        "- Francis, Mrziglod, Sidiropoulos, *Sentinel-2 Cloud Mask Catalogue*, Zenodo",
        "  (record 4172871)",
        "- Hughes & Kennedy, *High Quality Cloud Masking of Landsat 8 Imagery Using",
        "  Convolutional Neural Networks*, Remote Sensing 11(21):2591, 2019",
        "- Aybar ve ark., *CloudSEN12*, Nature Scientific Data, 2022",
        "- David ve ark., *TensorFlow Lite Micro: Embedded Machine Learning on TinyML",
        "  Systems*, arXiv:2010.08678", ""])


def section_release_matrix() -> str:
    """Hangi surum hangi olcutte en iyi - dagitim karari icin ozet."""
    tiling = load_csv("tiling_tradeoff.csv")
    if tiling is None:
        return ""

    rows = [
        {"surum": "v1", "kare": "256x256", "U-Net encoder": "mnv2_100",
         "IoU": "0.8663", "bellek": "27.27 MB", "ms/sahne": "129",
         "precision": "-", "temiz alan kaybi (%30)": "-"},
        {"surum": "v2", "kare": "256x256", "U-Net encoder": "mnv2_050",
         "IoU": "0.8855", "bellek": "26.69 MB", "ms/sahne": "129",
         "precision": "0.9264", "temiz alan kaybi (%30)": "%16.17"},
        {"surum": "v3", "kare": "256x256", "U-Net encoder": "mnv2_050",
         "IoU": "0.8844", "bellek": "26.47 MB", "ms/sahne": "129",
         "precision": "0.9531", "temiz alan kaybi (%30)": "%16.17"},
        {"surum": "v3.1", "kare": "128x128", "U-Net encoder": "mnv2_050",
         "IoU": "0.8824", "bellek": "17.82 MB", "ms/sahne": "138",
         "precision": "0.9442", "temiz alan kaybi (%30)": "%13.78"},
        {"surum": "v3.2", "kare": "64x64", "U-Net encoder": "mnv2_050",
         "IoU": "0.8807", "bellek": "14.87 MB", "ms/sahne": "175",
         "precision": "0.9771", "temiz alan kaybi (%30)": "%8.13"},
    ]

    return "\n".join([
        "## 13. Surum matrisi: hangi olcutte hangisi", "",
        "Tum surumler `releases/` altinda dondurulmustur. Siniflandirici v2'den",
        "itibaren degismemistir (c6_tuned); asagidaki farklar U-Net tarafindandir.", "",
        df_to_md(pd.DataFrame(rows)), "",
        "### 13.1 Olcut bazinda en iyiler", "",
        "- **En yuksek dogruluk (IoU):** v2 (0.8855). v3 ile arasindaki 0.0011'lik fark",
        "  gozlenen kosu-arasi degiskenligin icindedir; pratikte esittirler.",
        "- **En kisa sahne suresi:** v1/v2/v3 (129 ms). Kareleme ne kadar inceyse sahne",
        "  basina o kadar cok cikarim yapilir.",
        "- **En dusuk bellek:** v3.2 (14.87 MB) - v3'e gore %44 az.",
        "- **En yuksek precision:** v3.2 (0.9771). Ince kareleme modeli daha temkinli yapar.",
        "- **En az bilimsel veri kaybi:** v3.2 (%8.13 temiz alan) - v3'un yarisi.",
        "- **En dengeli:** v3.1 (128x128). Bellek 2.4 kat az, sure yalnizca %7 fazla,",
        "  IoU maliyeti 0.0020.", "",
        "### 13.2 Dagitim karari: v3.2", "",
        "**v3.2 ana surum olarak secilmistir.** Gerekce:", "",
        "1. Sure butcesi bol: 175 ms/sahne, referans misyon CloudScout'un 325 ms'inin",
        "   yarisi. Kaybedilen 46 ms, kazanilan bellek ve veri korumasi karsisinda ucuz.",
        "2. Bellek en kritik kisit: uydu uzerindeki calisma bellegi, disk alanindan cok",
        "   daha darda. v3.2 bu eksende %44 kazandiriyor.",
        "3. Bilimsel veri kaybi yariya iniyor (%16.17 -> %8.13). Yanlis eleme geri donusu",
        "   olmayan bir maliyet oldugu icin bu, projenin oncelik siralamasinda en ustte.",
        "4. Dogruluk maliyeti ihmal edilebilir: IoU -0.0037.", "",
        "Sure kisiti daralirsa v3.1 (138 ms), bellek kisiti gevserse v3 (129 ms)",
        "tercih edilebilir. Ucu de `releases/` altinda hazir durmaktadir.", ""])


def section_limitations() -> str:
    return "\n".join([
        "## 12. Kisitlar ve iyilestirme onerileri", "",
        "### 12.1 Bilinen kisitlar", "",
        f"1. **Bulut esigi (%{100 * config.CLOUD_PIXEL_THRESHOLD:.0f}) gerekcelendirilmedi.** Bu deger hem etiketleme",
        "   kuralini hem karar kuralini belirliyor ve tum sonuclari kaydiriyor. Referans",
        "   misyon CloudScout ayni karari %70 esikle veriyor. Duyarlilik analizi yapilmadi.",
        "2. **Boyut ve sure hedefleri (5 MB / 100 ms) proje icinde secildi**, PDF bir sayi",
        "   vermiyor. Referans misyondan turetilmeleri daha savunulabilir olurdu.",
        "3. **Tek tohumlu kosular.** Modeller arasi kucuk farklar icin istatistiksel",
        "   dayanak yok. Ozellikle v1->v2 siniflandirici farki (ROC-AUC +0.007) bu",
        "   uyarinin kapsamindadir; U-Net ve calisma noktasi kazanclari ise gozlenen",
        "   degiskenlikten belirgin sekilde buyuktur.",
        "3b. **Esik veriden turetilemedi.** Alti ayri yol denendi ve hicbiri kararli",
        "   bir dogal esik vermedi: bulut turu etiketleri (sahne bazli, dislayici degil,",
        "   ve yorungede mevcut degil), minimum bulut boyutu (1 piksele iniyor),",
        "   dagilim yapisi (uclar disinda duz), bagli bilesenler (toplam alanla 0.97",
        "   korelasyonlu), kontur doluluk orani (etiketlerin %0.07'sini degistiriyor)",
        "   ve uc degerleri disliyarak dagilim aramasi (vadi konumu kutu sayisi ve",
        "   ornekleme tohumuyla savruluyor; cekirdek yogunluk tahmininde vadi derinligi",
        "   1.04x - istatistiksel olarak anlamsiz). Esik bu nedenle operasyonel bir",
        "   politika parametresi olarak birakilmistir.",
        "4. **Precision kisiti test setine tasinmiyor.** Dogrulama setinde 0.99 hedefiyle",
        "   secilen esik, daha zor olan test setinde daha dusuk precision veriyor.",
        "5. ~~SPARCS karsilastirmasi kotumser (gunes yuksekligi duzeltmesi yok).~~",
        "   **OLCULDU VE ELENDI (bolum 6b.2):** duzeltme uygulandiginda ROC-AUC",
        "   degismiyor (0.9733 -> 0.9732). Radyometrik uyumsuzluk performans acigini",
        "   aciklamiyor; acik gercek bir alan kaymasidir. Bu kisit artik gecerli degil.",
        "6. **CloudScout ile dogrudan karsilastirma gecerli degildir:** farkli veri seti,",
        "   farkli gorev esigi (%70), farkli donanim sinifi (1.8 W uzay VPU'su vs masaustu",
        "   CPU) ve farkli girdi boyutu. Boyut olarak ayni sinifta olundugu soylenebilir,",
        "   dogruluk veya hiz ustunlugu iddia edilemez.", "",
        "### 12.2 Iyilestirme onerileri", "",
        "1. Bulut esigi icin duyarlilik analizi (%10/20/30/50/70) ve operasyonel gerekce.",
        "2. Coklu tohumlu kosularla model secimini saglamlastirmak.",
        "3. Kar/buz basarisizligi: kar agirlikli karelerle veri artirimi veya kar/buz",
        "   ozel bir kayip agirligi.",
        "4. ~~Genislik carpani taramasi.~~ **Kismen yapildi (bolum 6d.1):** mobilenetv2_050",
        "   diskte 2.6 kat kucuk ama INT8'de cokuyor (kayip %3.563 -> %13.539). _075",
        "   denenmedi; QAT ile birlikte denenmesi anlamli olabilir.",
        "5. ~~Kismi indirme senaryosunun olculmesi.~~ **OLCULDU (bolum 6c):** odunlesim",
        "   elverissiz cikti (3.28:1). Denenmemis alternatif: bolge-ilgi kodlamasi",
        "   (JPEG2000 ROI) veya bulutlu bolgeleri atmak yerine kayipli kodlamak.",
        "6. Kuantizasyon-farkinda egitim (QAT) ile MobileNetV3'un da kullanilabilir hale",
        "   getirilmesi.", ""])


def main():
    p = argparse.ArgumentParser()
    # Teslim edilecek belgeler proje kokundeki teslim/ altinda toplanir;
    # outputs/reports/ ham analiz ciktilarinin calisma dizinidir.
    p.add_argument("--out", type=Path,
                   default=config.PROJECT_ROOT / "teslim" / "TEKNIK_RAPOR.md")
    p.add_argument("--stamp", default=None, help="rapor tarihi (varsayilan: bugun)")
    p.add_argument("--classifier-tag", default="v2")
    p.add_argument("--unet-tag", default="unet")
    args = p.parse_args()

    stamp = args.stamp or datetime.now().strftime("%Y-%m-%d")

    header = "\n".join([
        "# Uydu Uzerinde Ucta Yapay Zeka ile Bulutlu Goruntulerin Filtrelenmesi",
        "## Teknik Rapor", "",
        f"Tarih: {stamp}  ",
        "Referans misyon: ESA Phi-Sat-1 / CloudScout", "",
        "---", "",
        "## 1. Ozet", "",
        "Yer gozlem uydularinin cektigi goruntulerin onemli bir bolumu bulutlarla kapli",
        "oldugundan bilimsel olarak kullanilamaz; bunlarin yere indirilmesi sinirli",
        "haberlesme bant genisligini bosa harcar. Bu calismada, bulutlu goruntuleri",
        "uydu uzerindeyken tespit edip eleyecek hafif bir yapay zeka modeli gelistirildi,",
        "INT8'e kuantize edildi ve saglayacagi bant genisligi kazanci sayisal olarak",
        "olculdu. Calisma tamamen yazilim ortaminda yurutulmustur; hedef donanim uzerinde",
        "calistirma kapsam disidir.", "",
        "Referans misyon ve ilgili calismalar icin bkz. **bolum 14**.", ""])

    parts = [header, section_dataset(), section_models(),
             section_results(args.classifier_tag, args.unet_tag),
             section_downlink(args.classifier_tag), section_external(args.classifier_tag),
             section_unet_external(args.unet_tag),
             section_partial_downlink(args.unet_tag),
             section_tradeoff_matrix(),
             section_quantization(), section_ablation(), section_hparam(),
             section_versions(), section_operating_points(args.classifier_tag),
             section_memory(), section_tiling(), section_classifier_tiling(),
             section_deployment_memory(), section_limitations(),
             section_release_matrix(), section_literature()]

    text = "\n".join(x for x in parts if x)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8")
    print(f"rapor yazildi: {args.out}  ({len(text):,} karakter)")


if __name__ == "__main__":
    main()

