"""Reflektans normalizasyonu - veri hazirlama ve cikarim ayni fonksiyonu kullanir.

Tek yerde tutulmasi kritik: prepare_data.py ile inference.py farkli normalizasyon
uygularsa model egitimde gordugu dagilimdan farkli girdi alir ve dogruluk
sessizce duser.
"""

import numpy as np

from src import config


def normalize_reflectance(arr: np.ndarray) -> np.ndarray:
    """TOA reflektans degerlerini [0, 1] araligina getirir.

    Sentinel-2 Cloud Mask Catalogue README'sinden onemli not: reflektans
    degerlerinin bir kismi 1'i ASAR. Bu bir hata degil - yuzey goruş acisina
    gore normalden fazla isik aliyorsa gorunur reflektans 1'den buyuk olabilir.
    Bu yuzden [0, 1] arasina kirpmak gercek veriyi yok eder (ozellikle parlak
    bulut tepeleri ve kar/buz, yani tam olarak ayirt etmemiz gereken sinifi).

    Cozum: config.REFLECTANCE_CLIP'e kirp, sonra o degere bol. Boylece 1 ustu
    degerler korunur ama uc aykiri degerler egitimi bozmaz.
    """
    return np.clip(arr, 0.0, config.REFLECTANCE_CLIP) / config.REFLECTANCE_CLIP


def denormalize_reflectance(arr: np.ndarray) -> np.ndarray:
    """Gorselleştirme icin ters donusum."""
    return arr * config.REFLECTANCE_CLIP
