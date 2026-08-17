# Teslim edilen belgeler

Bu klasor stajin teslim edilecek ciktilarini icerir.

| dosya | icerik |
|---|---|
| `TEKNIK_RAPOR.md` | Teknik rapor: veri, yontem, sonuclar, fayda analizi, harici dogrulama (siniflandirici + U-Net), kismi indirme odunlesimi, varyant karsilastirma matrisi, kuantizasyon calismasi, bellek profili, kisitlar, surum matrisi, literatur taramasi |
| `TEKNIK_RAPOR.docx` | Ayni raporun Word surumu |
| `PROJE_REHBERI.docx` | Proje rehberi |
| `SUNUM.pptx` | 15 slaytlik sunum |
| `demo_ornek.png` | Demo uygulamasinin ciktisi: RGB onizleme, Grad-CAM dikkat haritasi, U-Net bulut maskesi |

> **Not:** `TEKNIK_RAPOR.docx` artik `.md` dosyasindan betikle uretilmektedir:
> `python -m src.md_to_docx`. Rapor yeniden uretildiginde bu komutu da
> calistirin; aksi halde `.docx` bayatlar.

## Nerede ne var

- **Modeller ve surumler:** `releases/` — v1, v2, v3, v3.1, v3.2. Dagitilan surum
  v3.2'dir; her surum kendi checkpoint'leri, ONNX dosyalari, calisma noktalari ve
  olculmus sonuclariyla dondurulmustur.
- **Ham analiz ciktilari:** `outputs/reports/` — benchmark tablolari, kuantizasyon
  taramalari, bellek olcumleri, esik duyarlilik analizleri, etiket bazli basarim.
  Rapordaki her sayi bu dosyalardan uretilir.
- **Kaynak kod:** `src/`
- **Demo:** proje kokunde `python app.py`

## Rapor nasil yeniden uretilir

```
python -m src.make_report --classifier-tag c6_tuned --unet-tag unet_t64
```

Sayilar elle yazilmaz; `outputs/reports/` altindaki JSON/CSV dosyalarindan okunur.
Yeni bir deney kosuldugunda rapor tekrar uretilerek guncel tutulur.
