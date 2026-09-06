"""
validate_pretrained_emotion.py

Independent validation of a candidate pretrained emotion model against this
project's OWN held-out fer-2013/test images -- not just trusting the model
card's self-reported numbers. This is what pretrained_emotion.py's accuracy
claim (70.13%) is based on.

Usage:
    python validate_pretrained_emotion.py [--fer-test-root ../fer-2013/test] [--n-per-class 150]
"""
import argparse
import os
import random
import sys
import time

import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

MODEL_ID = "trpakov/vit-face-expression"
OUR_CLASSES = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
SEED = 42


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fer-test-root", default="fer-2013/test")
    p.add_argument("--n-per-class", type=int, default=150, help="Stratified sample size per class.")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Loading {MODEL_ID} on {device} ...")
    t0 = time.time()
    processor = AutoImageProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForImageClassification.from_pretrained(MODEL_ID).to(device).eval()
    print(f"Loaded in {time.time() - t0:.1f}s")

    id2label = model.config.id2label
    print("Model id2label:", id2label)
    label_lower_to_id = {v.lower(): k for k, v in id2label.items()}
    missing = [c for c in OUR_CLASSES if c not in label_lower_to_id]
    if missing:
        print(f"WARNING: model is missing our classes: {missing}")
        sys.exit(1)

    rng = random.Random(SEED)
    samples = []  # (path, true_class)
    for cls in OUR_CLASSES:
        cls_dir = os.path.join(args.fer_test_root, cls)
        files = sorted(os.listdir(cls_dir))
        rng.shuffle(files)
        chosen = files[:args.n_per_class]
        samples.extend((os.path.join(cls_dir, f), cls) for f in chosen)
    rng.shuffle(samples)
    print(f"Evaluating on {len(samples)} images ({args.n_per_class}/class, stratified, seed={SEED})")

    correct = 0
    per_class_correct = {c: 0 for c in OUR_CLASSES}
    per_class_total = {c: 0 for c in OUR_CLASSES}
    confusion = {c: {c2: 0 for c2 in OUR_CLASSES} for c in OUR_CLASSES}

    batch_size = 32
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, len(samples), batch_size):
            batch = samples[i:i + batch_size]
            images = [Image.open(p).convert("RGB") for p, _ in batch]
            inputs = processor(images=images, return_tensors="pt").to(device)
            logits = model(**inputs).logits
            preds = logits.argmax(dim=-1).cpu().tolist()

            for (path, true_cls), pred_id in zip(batch, preds):
                pred_label = id2label[pred_id].lower()
                # map predicted label back to our closest class name (should match exactly)
                pred_cls = pred_label if pred_label in OUR_CLASSES else pred_label
                per_class_total[true_cls] += 1
                if pred_cls == true_cls:
                    correct += 1
                    per_class_correct[true_cls] += 1
                if pred_cls in confusion[true_cls]:
                    confusion[true_cls][pred_cls] += 1

            if (i // batch_size) % 5 == 0:
                print(f"  {i + len(batch)}/{len(samples)} ({time.time() - t0:.1f}s)...")

    acc = correct / len(samples)
    print(f"\n=== Overall accuracy on {len(samples)} real FER-2013 test images: {acc:.4f} ===\n")
    print(f"{'class':10s} {'acc':>6s} {'n':>5s}")
    for c in OUR_CLASSES:
        n = per_class_total[c]
        a = per_class_correct[c] / n if n else 0.0
        print(f"{c:10s} {a:6.3f} {n:5d}")

    print("\nConfusion (rows=true, cols=predicted):")
    header = "true\\pred".ljust(10) + "".join(f"{c[:4]:>6s}" for c in OUR_CLASSES)
    print(header)
    for c in OUR_CLASSES:
        row = c.ljust(10) + "".join(f"{confusion[c][c2]:6d}" for c2 in OUR_CLASSES)
        print(row)


if __name__ == "__main__":
    main()
