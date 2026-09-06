"""
preprocess.py

Feature extraction for the hierarchical gesture/emotion pipeline.

Runs MediaPipe Holistic ONCE per image and independently populates two
datasets from whichever landmarks Holistic actually finds:

  - FACE dataset (468 landmarks -> 1404-d vector), from
        fer-2013/<split>/<emotion>/*.jpg
    A face-less image is skipped (logged), since a supervised emotion
    classifier needs a real face signal -- there's nothing meaningful to
    zero-pad here.

  - HAND dataset (21 landmarks x 2 hands -> 126-d vector, zero-padded),
    from
        dataset/gdg/*.jpg    (label 1, "gdg")
        dataset/noise/*.jpg  (label 0, background)
    A missing hand is zero-padded, NOT discarded -- both classes legitimately
    contain frames with zero, one, or two hands visible, and the two
    per-image outputs (face vs. hand) are populated independently, so a
    missing hand on a gesture-dataset image never affects (and is never
    affected by) any face detection in that same frame.

Usage:
    python preprocess.py \
        --fer-root ../fer-2013 --gesture-root ../dataset --output-dir ../features

    # Re-extract only the (fast, ~1-2 min) hand/gesture dataset, e.g. after
    # collecting more dataset/gdg or dataset/noise images -- skips the slow
    # (~20 min) FER-2013 face pass entirely:
    python preprocess.py --only hand
"""

import argparse
import json
import os
import time

import cv2
import mediapipe as mp
import numpy as np

from landmark_utils import extract_face_vector, extract_two_hand_vector

# Fixed, human-readable label order -- reused by datasets.py / train.py / inference.py
# via the label_map.json this script writes out.
EMOTION_CLASS_ORDER = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
GESTURE_CLASS_ORDER = ["noise", "gdg"]  # index 0 = background/negative, 1 = gdg/positive


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract MediaPipe Holistic landmark features.")
    p.add_argument("--fer-root", default="fer-2013", help="Root of the FER-2013 image tree (train/, test/).")
    p.add_argument("--gesture-root", default="dataset", help="Root containing gdg/ and noise/ image folders.")
    p.add_argument("--output-dir", default="features", help="Where to write the .npz feature files.")
    p.add_argument(
        "--upscale", type=int, default=256,
        help="Resize the shorter side of each image to this size before running Holistic. "
             "FER-2013 faces are only 48x48px; upscaling substantially improves landmark-"
             "detection reliability (Holistic's face detector was not trained on such tiny inputs).",
    )
    p.add_argument("--min-detection-confidence", type=float, default=0.5)
    p.add_argument("--val-split", type=float, default=0.2, help="Fraction of the gesture dataset held out for validation.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--only", choices=["all", "face", "hand"], default="all",
        help="Which dataset(s) to (re-)extract. 'hand' skips the slow FER-2013 face pass "
             "entirely -- use it when you've only added/changed dataset/gdg or dataset/noise images.",
    )
    return p.parse_args()


def load_and_upscale(path: str, target: int):
    img = cv2.imread(path)
    if img is None:
        return None
    h, w = img.shape[:2]
    scale = target / min(h, w)
    if scale > 1.0:
        img = cv2.resize(img, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_CUBIC)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def iter_images(root_dir: str, classes):
    for label_idx, cls in enumerate(classes):
        cls_dir = os.path.join(root_dir, cls)
        if not os.path.isdir(cls_dir):
            print(f"  [warn] class dir not found, skipping: {cls_dir}")
            continue
        for fname in sorted(os.listdir(cls_dir)):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                yield os.path.join(cls_dir, fname), label_idx


def extract_split(holistic, root_dir: str, classes, split_name: str, args, mode: str):
    """mode: 'face' extracts face vectors only; 'hand' extracts (zero-padded) hand vectors only."""
    X, y = [], []
    n_total = 0
    n_skipped = 0
    t0 = time.time()

    for path, label_idx in iter_images(root_dir, classes):
        n_total += 1
        rgb = load_and_upscale(path, args.upscale)
        if rgb is None:
            n_skipped += 1
            continue

        rgb.flags.writeable = False
        results = holistic.process(rgb)

        if mode == "face":
            vec = extract_face_vector(results.face_landmarks)
            if vec is None:
                n_skipped += 1
                continue
            X.append(vec)
            y.append(label_idx)
        else:  # mode == "hand"
            vec = extract_two_hand_vector(results.left_hand_landmarks, results.right_hand_landmarks)
            # A positive ("gdg") sample with an all-zero vector means MediaPipe detected
            # NO hands at all in this frame -- that's a detection failure, not evidence
            # of the two-hand gesture, and keeping it mislabels "no hands visible" as a
            # confident positive. Drop it rather than propagate that label noise.
            # (A "noise"/background sample with an all-zero vector is legitimate --
            # "no hands visible" genuinely is a valid negative example -- so it's kept.)
            is_gdg = classes[label_idx] == "gdg"
            if is_gdg and not vec.any():
                n_skipped += 1
                continue
            X.append(vec)  # otherwise never discarded -- zero-padded when hand(s) absent
            y.append(label_idx)

        if n_total % 200 == 0:
            print(f"  [{split_name}] {n_total} processed ({time.time() - t0:.1f}s)...")

    reason = "no face detected" if mode == "face" else "unreadable image / no-hands-detected gdg sample"
    print(f"  [{split_name}] done: {len(X)} kept, {n_skipped}/{n_total} skipped ({reason}).")

    if not X:
        dim = 0
        return np.zeros((0, dim), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    return np.stack(X).astype(np.float32), np.array(y, dtype=np.int64)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    mp_holistic = mp.solutions.holistic

    with mp_holistic.Holistic(
        static_image_mode=True,
        model_complexity=1,
        refine_face_landmarks=False,  # keep the 468-point topology (no iris refinement -> 478)
        min_detection_confidence=args.min_detection_confidence,
    ) as holistic:

        # ---- Face / emotion dataset, from FER-2013's own train/test split ----
        if args.only in ("all", "face"):
            for split in ("train", "test"):
                split_root = os.path.join(args.fer_root, split)
                print(f"Extracting FACE features: {split_root}")
                X, y = extract_split(holistic, split_root, EMOTION_CLASS_ORDER, f"face/{split}", args, mode="face")
                out_path = os.path.join(args.output_dir, f"face_{split}.npz")
                np.savez_compressed(out_path, X=X, y=y)
                print(f"  saved -> {out_path}  X={X.shape} y={y.shape}")

        # ---- Hand / gesture dataset, from dataset/gdg + dataset/noise ----
        if args.only in ("all", "hand"):
            print(f"Extracting HAND features: {args.gesture_root}")
            X, y = extract_split(holistic, args.gesture_root, GESTURE_CLASS_ORDER, "hand", args, mode="hand")

            rng = np.random.default_rng(args.seed)
            idx = np.arange(len(y))
            rng.shuffle(idx)
            n_val = int(round(args.val_split * len(idx))) if len(idx) > 0 else 0
            val_idx, train_idx = idx[:n_val], idx[n_val:]

            np.savez_compressed(os.path.join(args.output_dir, "hand_train.npz"), X=X[train_idx], y=y[train_idx])
            np.savez_compressed(os.path.join(args.output_dir, "hand_test.npz"), X=X[val_idx], y=y[val_idx])
            print(f"  saved -> hand_train.npz ({len(train_idx)}) / hand_test.npz ({len(val_idx)})")

    label_map = {"emotion_classes": EMOTION_CLASS_ORDER, "gesture_classes": GESTURE_CLASS_ORDER}
    label_map_path = os.path.join(args.output_dir, "label_map.json")
    with open(label_map_path, "w") as f:
        json.dump(label_map, f, indent=2)
    print(f"Saved label map -> {label_map_path}")


if __name__ == "__main__":
    main()
