"""Tek goruntu icin cikarim: karar, olasilik, Grad-CAM isi haritasi, maske.

Hem demo arayuzu (app.py) hem komut satiri bu modulu kullanir.
"""

from pathlib import Path

import numpy as np
import torch

from src import config
from src.export import load_from_checkpoint
from src.preprocess import normalize_reflectance


def read_image(path: Path) -> np.ndarray:
    """Uydu goruntusunu (C, H, W) float32, normalize edilmis olarak okur.

    Uc girdi turunu ayirt eder. Bu ayrim kritik: hazir kare zaten normalize
    edilmistir, ikinci kez normalize edilirse model egitimde gordugunden
    farkli bir dagilim alir ve dogruluk sessizce duser.

      1. (C, H, W) float, C == IN_CHANNELS
         -> prepare_data.py ciktisi, ZATEN normalize, dokunulmaz
      2. (H, W, 13) float
         -> Cloud Mask Catalogue ham subscene; bant secilir + normalize edilir
            (reflektans bolmesi arsivde uygulanmis, tekrar bolunmez)
      3. .tif / .tiff
         -> ham L1C GeoTIFF; rasterio ile okunur, /10000 + normalize
    """
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".npy":
        arr = np.load(path).astype(np.float32)

        if arr.ndim != 3:
            raise ValueError(f"3 boyutlu dizi bekleniyordu, {arr.shape} bulundu")

        if arr.shape[0] == config.IN_CHANNELS:
            return arr  # hazir kare: normalize edilmis, oldugu gibi dondur

        if arr.shape[2] == len(config.S2_BANDS):
            arr = arr[..., config.BAND_INDICES].transpose(2, 0, 1)
            return normalize_reflectance(arr).astype(np.float32)

        raise ValueError(
            f"tanimlanamayan sekil {arr.shape}; ({config.IN_CHANNELS}, H, W) hazir kare "
            f"veya (H, W, {len(config.S2_BANDS)}) ham subscene bekleniyordu"
        )

    if suffix in (".tif", ".tiff"):
        import rasterio

        with rasterio.open(path) as src:
            arr = src.read().astype(np.float32)  # (bands, H, W)

        if arr.shape[0] == len(config.S2_BANDS):
            arr = arr[config.BAND_INDICES]
        elif arr.shape[0] != config.IN_CHANNELS:
            raise ValueError(
                f"{arr.shape[0]} bant bulundu; {len(config.S2_BANDS)} (ham Sentinel-2) "
                f"veya {config.IN_CHANNELS} (secilmis bantlar) bekleniyordu"
            )
        arr = arr / config.S2_REFLECTANCE_SCALE
        return normalize_reflectance(arr).astype(np.float32)

    raise ValueError(f"desteklenmeyen format: {suffix} (.npy, .tif bekleniyor)")


def to_rgb_preview(arr: np.ndarray, percentile: float = 2.0) -> np.ndarray:
    """Cok bantli goruntuden goruntulenebilir RGB (H, W, 3) uint8 uretir.

    Yuzdelik germe (percentile stretch) uygulanir: ham reflektans degerleri
    dogrudan gosterildiginde goruntu neredeyse siyah cikar.
    """
    try:
        rgb_idx = [config.BANDS.index(b) for b in ("B04", "B03", "B02")]
    except ValueError:
        # RGB bantlari secili degilse ilk uc kanali kullan
        rgb_idx = list(range(min(3, arr.shape[0])))

    rgb = arr[rgb_idx].transpose(1, 2, 0)
    lo = np.percentile(rgb, percentile, axis=(0, 1), keepdims=True)
    hi = np.percentile(rgb, 100 - percentile, axis=(0, 1), keepdims=True)
    rgb = (rgb - lo) / np.clip(hi - lo, 1e-6, None)
    return (np.clip(rgb, 0, 1) * 255).astype(np.uint8)


def overlay_heatmap(rgb: np.ndarray, heatmap: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Isi haritasini RGB onizleme uzerine bindirir (jet renk haritasi)."""
    import matplotlib.cm as cm

    heatmap = (heatmap - heatmap.min()) / np.clip(heatmap.max() - heatmap.min(), 1e-6, None)
    colored = (cm.jet(heatmap)[..., :3] * 255).astype(np.uint8)
    return (alpha * colored + (1 - alpha) * rgb).astype(np.uint8)


class CloudClassifier:
    """Siniflandirici + Grad-CAM."""

    def __init__(self, checkpoint: Path, threshold: float = config.DECISION_THRESHOLD):
        self.model, task = load_from_checkpoint(Path(checkpoint))
        if task != "classification":
            raise ValueError(f"bu checkpoint '{task}' gorevine ait, siniflandirici bekleniyordu")
        self.threshold = threshold
        self._cam = None

    def _get_cam(self):
        if self._cam is None:
            from pytorch_grad_cam import GradCAM

            # timm'in MobileNet ailesinde (V2/V3) son evrisim blogu: mekansal
            # bilgiyi tasiyan en derin katman. Daha derini (conv_head sonrasi)
            # global havuzlama nedeniyle mekansal cozunurlugu kaybeder.
            # Dagitilan model mobilenetv2_100'dur; .blocks her iki mimaride var.
            target_layers = [self.model.blocks[-1]]
            self._cam = GradCAM(model=self.model, target_layers=target_layers)
        return self._cam

    @torch.no_grad()
    def predict(self, arr: np.ndarray) -> dict:
        x = torch.from_numpy(arr).unsqueeze(0)
        prob = torch.sigmoid(self.model(x).squeeze()).item()
        return {
            "probability": prob,
            "decision": "BULUTLU (elenir)" if prob >= self.threshold else "KULLANILABILIR (indirilir)",
            "is_cloudy": prob >= self.threshold,
            "threshold": self.threshold,
        }

    def heatmap(self, arr: np.ndarray) -> np.ndarray:
        """Grad-CAM dikkat haritasi (H, W), [0,1]."""
        from pytorch_grad_cam.utils.model_targets import BinaryClassifierOutputTarget

        x = torch.from_numpy(arr).unsqueeze(0)
        cam = self._get_cam()
        # Tek logitli model: hedef sinif 1 ("bulutlu")
        grayscale = cam(input_tensor=x, targets=[BinaryClassifierOutputTarget(1)])
        return grayscale[0]


class CloudSegmenter:
    """U-Net segmentasyon (opsiyonel; checkpoint varsa yuklenir)."""

    def __init__(self, checkpoint: Path, pixel_threshold: float = 0.5):
        self.model, task = load_from_checkpoint(Path(checkpoint))
        if task != "segmentation":
            raise ValueError(f"bu checkpoint '{task}' gorevine ait, U-Net bekleniyordu")
        self.pixel_threshold = pixel_threshold

    @torch.no_grad()
    def predict(self, arr: np.ndarray) -> dict:
        x = torch.from_numpy(arr).unsqueeze(0)
        probs = torch.sigmoid(self.model(x)).squeeze().numpy()
        mask = (probs >= self.pixel_threshold).astype(np.float32)
        cloud_fraction = float(mask.mean())
        return {
            "mask": mask,
            "probability_map": probs,
            "cloud_fraction": cloud_fraction,
            "is_cloudy": cloud_fraction >= config.CLOUD_PIXEL_THRESHOLD,
        }
