"""
train.py

Trains the two models in the hierarchical gesture/emotion pipeline:
  - GestureClassifier: binary GDG-vs-background, from features/hand_{train,test}.npz
  - EmotionClassifier: 7-class FER-2013 emotion, from features/face_{train,test}.npz

Both classes are almost certainly imbalanced (e.g. ~217 "gdg" vs. ~1615
"noise" samples; FER-2013's "disgust" class is ~15-18x rarer than "happy").
This is handled two complementary ways, both explicit in the code below:

  1. WeightedRandomSampler on the *training* DataLoader only -- each sample's
     draw probability is 1 / (count of its class), so a mini-batch is, on
     average, class-balanced regardless of how skewed the raw dataset is.
  2. Inverse-frequency class weights passed directly into the loss function
     (`pos_weight=` for BCEWithLogitsLoss, `weight=` for CrossEntropyLoss),
     so misclassifying a rare class is penalized more heavily even within
     an already-resampled batch.

Validation always uses the raw (unweighted, non-resampled) distribution, so
reported metrics reflect real-world class frequencies.

Usage:
    python train.py --task both --features-dir ../features --out-dir ../checkpoints
    python train.py --task gesture --epochs 40
"""

import argparse
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from datasets import EmotionDataset, GestureDataset
from models import EmotionClassifier, GestureClassifier

NUM_EMOTION_CLASSES = 7
NUM_GESTURE_CLASSES = 2  # 0 = noise/background, 1 = gdg


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_weighted_sampler(dataset, num_classes: int) -> WeightedRandomSampler:
    """Per-sample weight = 1 / (count of that sample's class): rare classes get
    oversampled and common classes get undersampled, on average, each epoch."""
    counts = dataset.class_counts(num_classes).float()
    class_weights = 1.0 / counts.clamp(min=1)
    sample_weights = class_weights[dataset.y]
    return WeightedRandomSampler(sample_weights, num_samples=len(dataset), replacement=True)


def class_weights_for_ce(dataset, num_classes: int) -> torch.Tensor:
    """Inverse-frequency weights for CrossEntropyLoss: total / (n_classes * count[c])."""
    counts = dataset.class_counts(num_classes).float()
    return counts.sum() / (num_classes * counts.clamp(min=1))


def pos_weight_for_bce(dataset) -> torch.Tensor:
    """neg/pos ratio for BCEWithLogitsLoss's pos_weight (upweights the rarer positive class)."""
    counts = dataset.class_counts(NUM_GESTURE_CLASSES).float()
    neg, pos = counts[0], counts[1]
    return (neg / pos.clamp(min=1)).unsqueeze(0)


def run_epoch(model, loader, criterion, optimizer, device, is_binary: bool, train: bool):
    model.train(train)
    total_loss, total_correct, total_n = 0.0, 0, 0

    with torch.set_grad_enabled(train):
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            if train:
                optimizer.zero_grad()

            logits = model(X)
            if is_binary:
                loss = criterion(logits, y.float())
                preds = (torch.sigmoid(logits) > 0.5).long()
            else:
                loss = criterion(logits, y)
                preds = logits.argmax(dim=1)

            if train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * X.size(0)
            total_correct += (preds == y).sum().item()
            total_n += X.size(0)

    return total_loss / max(total_n, 1), total_correct / max(total_n, 1)


def train_task(task: str, features_dir: str, out_dir: str, epochs: int, batch_size: int,
               lr: float, patience: int) -> str:
    device = get_device()
    print(f"[{task}] device: {device}")

    if task == "gesture":
        train_ds = GestureDataset(os.path.join(features_dir, "hand_train.npz"))
        val_ds = GestureDataset(os.path.join(features_dir, "hand_test.npz"))
        model = GestureClassifier(input_dim=train_ds.X.shape[1]).to(device)
        pos_weight = pos_weight_for_bce(train_ds).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        is_binary = True
        sampler = make_weighted_sampler(train_ds, NUM_GESTURE_CLASSES)
        print(f"  train class counts (noise, gdg): {train_ds.class_counts(NUM_GESTURE_CLASSES).tolist()}")
        print(f"  BCE pos_weight: {pos_weight.item():.3f}")
    elif task == "emotion":
        train_ds = EmotionDataset(os.path.join(features_dir, "face_train.npz"))
        val_ds = EmotionDataset(os.path.join(features_dir, "face_test.npz"))
        model = EmotionClassifier(input_dim=train_ds.X.shape[1], num_classes=NUM_EMOTION_CLASSES).to(device)
        weights = class_weights_for_ce(train_ds, NUM_EMOTION_CLASSES).to(device)
        criterion = nn.CrossEntropyLoss(weight=weights)
        is_binary = False
        sampler = make_weighted_sampler(train_ds, NUM_EMOTION_CLASSES)
        print(f"  train class counts: {train_ds.class_counts(NUM_EMOTION_CLASSES).tolist()}")
        print(f"  CE class weights: {[round(w, 3) for w in weights.tolist()]}")
    else:
        raise ValueError(f"Unknown task: {task}")

    if len(train_ds) == 0 or len(val_ds) == 0:
        raise RuntimeError(
            f"[{task}] empty dataset (train={len(train_ds)}, val={len(val_ds)}). "
            f"Run preprocess.py first."
        )

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    os.makedirs(out_dir, exist_ok=True)
    ckpt_path = os.path.join(out_dir, f"{task}_model.pt")
    best_val_acc = 0.0
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, is_binary, train=True)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer, device, is_binary, train=False)
        print(f"[{task}] epoch {epoch:3d}/{epochs}  "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f}  "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_without_improvement = 0
            torch.save(
                {"model_state": model.state_dict(), "input_dim": train_ds.X.shape[1]},
                ckpt_path,
            )
            print(f"  -> new best (val_acc={val_acc:.4f}), saved to {ckpt_path}")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"  early stopping: no improvement in {patience} epochs")
                break

    print(f"[{task}] finished. best val_acc={best_val_acc:.4f}")
    return ckpt_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the gesture and/or emotion classifiers.")
    p.add_argument("--task", choices=["gesture", "emotion", "both"], default="both")
    p.add_argument("--features-dir", default="features")
    p.add_argument("--out-dir", default="checkpoints")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=8, help="Early-stopping patience, in epochs.")
    return p.parse_args()


def main():
    args = parse_args()
    tasks = ["gesture", "emotion"] if args.task == "both" else [args.task]
    for task in tasks:
        train_task(task, args.features_dir, args.out_dir, args.epochs, args.batch_size, args.lr, args.patience)


if __name__ == "__main__":
    main()
