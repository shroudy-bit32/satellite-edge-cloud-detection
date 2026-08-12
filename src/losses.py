"""Segmentasyon kayip fonksiyonlari ve metrikleri."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceBCELoss(nn.Module):
    """BCE + soft Dice.

    Neden ikisi birlikte: bulut maskeleri dengesizdir (bazi karelerde bulut
    pikseli %1'in altinda). Tek basina BCE bu karelerde "hepsi temiz" diyerek
    dusuk kayip alir; Dice ortusme oranini dogrudan optimize ettigi icin bu
    coku engeller. BCE ise Dice'in egitim basindaki kararsizligini dengeler.
    """

    def __init__(self, bce_weight: float = 0.5, pos_weight: torch.Tensor | None = None,
                 smooth: float = 1.0):
        super().__init__()
        self.bce_weight = bce_weight
        self.smooth = smooth
        self._pos_weight = pos_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(
            logits, targets,
            pos_weight=self._pos_weight.to(logits.device) if self._pos_weight is not None else None,
        )

        probs = torch.sigmoid(logits)
        dims = (1, 2, 3)
        intersection = (probs * targets).sum(dims)
        cardinality = probs.sum(dims) + targets.sum(dims)
        dice = (2 * intersection + self.smooth) / (cardinality + self.smooth)
        dice_loss = 1 - dice.mean()

        return self.bce_weight * bce + (1 - self.bce_weight) * dice_loss


@torch.no_grad()
def segmentation_metrics(logits: torch.Tensor, targets: torch.Tensor,
                         threshold: float = 0.5, eps: float = 1e-7) -> dict:
    """Piksel bazli IoU, Dice, precision, recall.

    Toplamlar batch genelinde toplanir (ortalama-of-oranlar degil): bos maskeli
    karelerin oranlari sonsuza/sifira gitmesin diye.
    """
    preds = (torch.sigmoid(logits) >= threshold).float()

    tp = (preds * targets).sum()
    fp = (preds * (1 - targets)).sum()
    fn = ((1 - preds) * targets).sum()
    tn = ((1 - preds) * (1 - targets)).sum()

    return {
        "iou": float((tp / (tp + fp + fn + eps)).item()),
        "dice": float((2 * tp / (2 * tp + fp + fn + eps)).item()),
        "precision": float((tp / (tp + fp + eps)).item()),
        "recall": float((tp / (tp + fn + eps)).item()),
        "pixel_accuracy": float(((tp + tn) / (tp + tn + fp + fn + eps)).item()),
        "_counts": (tp.item(), fp.item(), fn.item(), tn.item()),
    }


def accumulate_metrics(counts_list: list) -> dict:
    """Batch bazli sayimlari toplayip veri seti geneli metrik hesaplar."""
    eps = 1e-7
    tp = sum(c[0] for c in counts_list)
    fp = sum(c[1] for c in counts_list)
    fn = sum(c[2] for c in counts_list)
    tn = sum(c[3] for c in counts_list)
    return {
        "iou": tp / (tp + fp + fn + eps),
        "dice": 2 * tp / (2 * tp + fp + fn + eps),
        "precision": tp / (tp + fp + eps),
        "recall": tp / (tp + fn + eps),
        "pixel_accuracy": (tp + tn) / (tp + tn + fp + fn + eps),
    }
