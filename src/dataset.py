"""Patch veri seti.

Kaynak veri setinden bagimsizdir: prepare_data.py her veri seti icin
index.csv uretir (path,label,split), buradaki Dataset onu okur.
Boylece Sentinel-2 Cloud Mask Catalogue, 95-Cloud ve SPARCS ayni hatta girer.

Patch dosyalari .npy formatinda, sekil (C, H, W), dtype float32, [0,1] araliginda.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from src import config


class CloudPatchDataset(Dataset):
    def __init__(self, index_csv: Path, split: str, augment: bool = False):
        df = pd.read_csv(index_csv)
        self.df = df[df["split"] == split].reset_index(drop=True)
        if len(self.df) == 0:
            raise ValueError(f"'{split}' bolumunde ornek yok: {index_csv}")
        self.augment = augment

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        arr = np.load(row["path"]).astype(np.float32)

        if self.augment:
            # Uydu goruntusunde yon anlamsiz -> flip/rot90 guvenli artirimlar.
            # Renk/parlaklik artirimi YAPMIYORUZ: reflektans degerleri fiziksel,
            # bozmak bulut/kar ayrimini zorlastirir.
            k = np.random.randint(4)
            if k:
                arr = np.rot90(arr, k, axes=(1, 2))
            if np.random.rand() < 0.5:
                arr = np.flip(arr, axis=2)
            arr = np.ascontiguousarray(arr)

        x = torch.from_numpy(arr)
        y = torch.tensor(float(row["label"]), dtype=torch.float32)
        return x, y

    def pos_weight(self) -> torch.Tensor:
        """Dengesiz sinif icin BCEWithLogitsLoss agirligi."""
        pos = float((self.df["label"] == 1).sum())
        neg = float((self.df["label"] == 0).sum())
        return torch.tensor([neg / max(pos, 1.0)], dtype=torch.float32)


class CloudSegDataset(Dataset):
    """Piksel bazli segmentasyon icin (genisletilmis hedef).

    index.csv'de mask_path sutunu bulunmalidir. Maske .npy, sekil (H, W) veya
    (1, H, W), degerler {0, 1}.
    """

    def __init__(self, index_csv: Path, split: str, augment: bool = False):
        df = pd.read_csv(index_csv)
        if "mask_path" not in df.columns:
            raise ValueError(f"{index_csv} icinde mask_path sutunu yok - "
                             "prepare_data.py'yi --with-masks ile calistir")
        self.df = df[df["split"] == split].reset_index(drop=True)
        if len(self.df) == 0:
            raise ValueError(f"'{split}' bolumunde ornek yok: {index_csv}")
        self.augment = augment

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img = np.load(row["path"]).astype(np.float32)
        mask = np.load(row["mask_path"]).astype(np.float32)
        if mask.ndim == 2:
            mask = mask[None]

        if self.augment:
            # Goruntu ve maskeye AYNI donusum uygulanmali, yoksa etiketler kayar.
            k = np.random.randint(4)
            if k:
                img = np.rot90(img, k, axes=(1, 2))
                mask = np.rot90(mask, k, axes=(1, 2))
            if np.random.rand() < 0.5:
                img = np.flip(img, axis=2)
                mask = np.flip(mask, axis=2)
            img = np.ascontiguousarray(img)
            mask = np.ascontiguousarray(mask)

        return torch.from_numpy(img), torch.from_numpy(mask)

    def pos_weight(self) -> torch.Tensor:
        """Bulut pikseli orani uzerinden BCE agirligi.

        Ilk 200 maskeden ornekleyerek tahmin edilir; tum veri setini taramak
        buyuk setlerde gereksiz pahali.
        """
        sample = self.df["mask_path"].head(200)
        pos = neg = 0.0
        for p in sample:
            m = np.load(p)
            pos += float((m > 0.5).sum())
            neg += float((m <= 0.5).sum())
        return torch.tensor([neg / max(pos, 1.0)], dtype=torch.float32)


def build_seg_loaders(batch_size: int = config.BATCH_SIZE, num_workers: int = config.NUM_WORKERS):
    train_ds = CloudSegDataset(config.PATCH_INDEX, "train", augment=True)
    val_ds = CloudSegDataset(config.PATCH_INDEX, "val", augment=False)
    test_ds = CloudSegDataset(config.PATCH_INDEX, "test", augment=False)

    common = dict(num_workers=num_workers, pin_memory=True, persistent_workers=num_workers > 0)
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True, **common),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False, **common),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False, **common),
    )


def build_loaders(batch_size: int = config.BATCH_SIZE, num_workers: int = config.NUM_WORKERS,
                  balanced: bool = False):
    train_ds = CloudPatchDataset(config.PATCH_INDEX, "train", augment=True)
    val_ds = CloudPatchDataset(config.PATCH_INDEX, "val", augment=False)
    test_ds = CloudPatchDataset(config.PATCH_INDEX, "test", augment=False)

    common = dict(num_workers=num_workers, pin_memory=True, persistent_workers=num_workers > 0)

    if balanced:
        # Sinif dengesizligi buyukse her batch'te iki sinifi esit gorulme
        # olasiligiyla ornekle. pos_weight'e alternatif; ikisini birden
        # kullanmak azinlik sinifini asiri agirlastirir.
        labels = train_ds.df["label"].to_numpy()
        counts = np.bincount(labels, minlength=2).astype(np.float64)
        sample_weights = (1.0 / np.clip(counts, 1, None))[labels]
        sampler = WeightedRandomSampler(
            torch.as_tensor(sample_weights, dtype=torch.double),
            num_samples=len(train_ds), replacement=True,
        )
        train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                                  drop_last=True, **common)
    else:
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                  drop_last=True, **common)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, **common)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, **common)
    return train_loader, val_loader, test_loader


def dataset_summary(index_csv: Path = config.PATCH_INDEX) -> pd.DataFrame:
    """Split x sinif dagilimi - raporda tablo olarak kullanilir."""
    df = pd.read_csv(index_csv)
    return df.pivot_table(index="split", columns="label", values="path", aggfunc="count", fill_value=0)


if __name__ == "__main__":
    print(dataset_summary())
