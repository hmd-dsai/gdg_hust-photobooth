"""
models.py

Lightweight MLP classifiers over MediaPipe landmark coordinates.

Both models consume vectors that have already been made translation- and
scale-invariant by landmark_utils.normalize_points (see preprocess.py), so
every input dimension is a *geometric* quantity -- a coordinate relative to
a fixed anatomical reference point, not raw pixel/frame position. Stacked
Linear+BatchNorm layers over such a vector learn combinations of those
relative coordinates (differences, ratios, roughly angle-like combinations
between landmarks), which is what actually encodes hand/face shape.

(A graph-based encoder -- e.g. reshaping the flat vector back to (N, 3) and
running a shared per-landmark MLP with max/mean pooling, PointNet-style --
would be a reasonable drop-in alternative to `_MLPBlock` below if you want
an architecture that's explicitly permutation-aware. It's not used here to
keep the pipeline small and easy to train/debug on a modest dataset.)
"""

import torch
import torch.nn as nn


class _MLPBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GestureClassifier(nn.Module):
    """
    Binary classifier over two-hand landmark vectors
    (126-d = 21 landmarks x 3 coords x 2 hands, zero-padded per missing hand).

    Outputs a single raw logit per sample -- pair with BCEWithLogitsLoss for
    training and torch.sigmoid(...) for a probability at inference time.
    """

    def __init__(self, input_dim: int = 126, hidden_dims=(128, 64), dropout: float = 0.3):
        super().__init__()
        dims = [input_dim] + list(hidden_dims)
        self.blocks = nn.Sequential(*[
            _MLPBlock(dims[i], dims[i + 1], dropout) for i in range(len(dims) - 1)
        ])
        self.head = nn.Linear(dims[-1], 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.blocks(x)).squeeze(-1)  # (B,) raw logit


class EmotionClassifier(nn.Module):
    """
    7-class classifier over face-mesh landmark vectors
    (1404-d = 468 landmarks x 3 coords).

    Outputs `num_classes` raw logits per sample -- pair with CrossEntropyLoss
    for training and torch.softmax(...) for probabilities at inference time.
    """

    def __init__(self, input_dim: int = 1404, hidden_dims=(512, 128, 64), num_classes: int = 7, dropout: float = 0.4):
        super().__init__()
        dims = [input_dim] + list(hidden_dims)
        self.blocks = nn.Sequential(*[
            _MLPBlock(dims[i], dims[i + 1], dropout) for i in range(len(dims) - 1)
        ])
        self.head = nn.Linear(dims[-1], num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.blocks(x))
