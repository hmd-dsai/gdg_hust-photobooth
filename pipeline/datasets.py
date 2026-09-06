"""
datasets.py

PyTorch Dataset classes for the preprocessed landmark features written by
preprocess.py (features/face_{train,test}.npz and features/hand_{train,test}.npz).
Everything is small enough (at most a few thousand vectors of a few hundred
to ~1400 floats each) to load fully into memory.
"""

from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset


class LandmarkDataset(Dataset):
    """Generic in-memory dataset for a single features/*.npz file (arrays X, y)."""

    def __init__(self, npz_path: str):
        data = np.load(npz_path)
        self.X = torch.from_numpy(data["X"]).float()
        self.y = torch.from_numpy(data["y"]).long()
        if len(self.X) != len(self.y):
            raise ValueError(f"Feature/label count mismatch in {npz_path}: {len(self.X)} vs {len(self.y)}")

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]

    def class_counts(self, num_classes: Optional[int] = None) -> torch.Tensor:
        """
        Per-class sample counts, for building loss weights / a WeightedRandomSampler.
        Pass `num_classes` explicitly when a rare class might have zero samples in
        this particular split (otherwise it would silently be dropped from the count).
        """
        if len(self.y) == 0:
            return torch.zeros(num_classes or 0, dtype=torch.long)
        if num_classes is None:
            num_classes = int(self.y.max().item()) + 1
        return torch.bincount(self.y, minlength=num_classes)


class GestureDataset(LandmarkDataset):
    """Binary GDG-vs-background dataset. Labels: 0 = noise/background, 1 = gdg. 126-d hand vectors."""
    pass


class EmotionDataset(LandmarkDataset):
    """7-class FER-2013 facial emotion dataset. 1404-d face-mesh vectors."""
    pass
