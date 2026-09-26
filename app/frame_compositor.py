"""
frame_compositor.py

Composites 4 player captures + 4 reference photos into the designed GDG
photobooth frame (app/static/branding/frame.png).

Layering order matters: photos are placed onto a blank canvas FIRST, and the
frame (with its alpha channel) is drawn on top of that SECOND. This is the
opposite of the plain grid build_photobooth_strip() in photobooth_challenge.py
uses, and it's deliberate -- the frame has decorative elements meant to
overlap the photo edges (border, shadow, corner accents), and those only look
right sitting above the photos, not hidden underneath them.

Slot coordinates were NOT taken from the original manual estimate as-is --
they were measured directly from the frame's own alpha channel (the
transparent "windows" are ground truth) via connected-component analysis.
That caught 3 typos in the manual estimate: the right edge of the label-image
column in rows 2-4 was given as x=992, which would have been visibly
non-square and misaligned with the frame art; the actual window edge is at
x=922 (matching row 1). See SLOTS below for the corrected, verified values --
re-run the `if __name__ == "__main__"` block's --detect-slots flag if the
frame is ever redesigned/replaced, rather than hand-editing coordinates again.

Usage:
    # Build a strip from a photobooth_challenge.py session directory:
    python app/frame_compositor.py --session-dir demo/output/session_<timestamp>

    # Re-detect slot coordinates from a new/updated frame.png (prints a dict
    # you can paste in to replace SLOTS below):
    python app/frame_compositor.py --detect-slots
"""

import argparse
import os
import sys

import cv2
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "demo"))

from split_screen_demo import LABEL_IMAGE_MAP  # noqa: E402

FRAME_PATH = os.path.join(os.path.dirname(__file__), "static", "branding", "frame.png")
GESTURE_SEQUENCE = ["happy", "angry", "surprise", "gdg"]  # row order, top to bottom

# (x1, y1, x2, y2) in frame.png's own pixel space, verified against the frame's
# alpha channel -- see module docstring.
SLOTS = {
    "happy":    {"capture": (158, 361, 555, 659),   "reference": (625, 361, 923, 658)},
    "angry":    {"capture": (158, 708, 555, 1005),  "reference": (625, 708, 922, 1005)},
    "surprise": {"capture": (158, 1042, 555, 1340), "reference": (625, 1042, 922, 1339)},
    "gdg":      {"capture": (158, 1375, 555, 1673), "reference": (625, 1375, 922, 1673)},
}


def crop_resize_to_box(img: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
    """Center-crop `img` to the box's aspect ratio (never stretch -- would distort
    faces), then resize to exactly fill (x2-x1) x (y2-y1)."""
    target_w, target_h = x2 - x1, y2 - y1
    target_ratio = target_w / target_h
    h, w = img.shape[:2]
    current_ratio = w / h
    if current_ratio > target_ratio:
        new_w = int(round(h * target_ratio))
        cx0 = (w - new_w) // 2
        cropped = img[:, cx0:cx0 + new_w]
    else:
        new_h = int(round(w / target_ratio))
        cy0 = (h - new_h) // 2
        cropped = img[cy0:cy0 + new_h, :]
    interp = cv2.INTER_AREA if target_w < cropped.shape[1] else cv2.INTER_CUBIC
    return cv2.resize(cropped, (target_w, target_h), interpolation=interp)


def build_framed_strip(captures: dict, references: dict, frame_path: str = FRAME_PATH) -> np.ndarray:
    """
    `captures`: {label: BGR image} -- the 4 player photos (from
        photobooth_challenge.py, already 4:3-cropped, but any shape works --
        this crops to fit regardless).
    `references`: {label: BGR image} -- the 4 raw reference photos (NOT
        pre-square-cropped; this function crops each to its own slot's shape).
    Returns the final BGR composite, at the frame's native pixel size.
    """
    frame_rgba = cv2.imread(frame_path, cv2.IMREAD_UNCHANGED)
    if frame_rgba is None:
        raise FileNotFoundError(frame_path)
    if frame_rgba.shape[2] != 4:
        raise ValueError(f"{frame_path} has no alpha channel -- re-export as PNG with transparency.")

    missing = [l for l in GESTURE_SEQUENCE if l not in captures or l not in references]
    if missing:
        raise ValueError(f"Missing captures/references for: {missing}")

    h, w = frame_rgba.shape[:2]
    canvas = np.zeros((h, w, 3), dtype=np.uint8)  # the photo layer, built first

    for label in GESTURE_SEQUENCE:
        x1, y1, x2, y2 = SLOTS[label]["capture"]
        canvas[y1:y2, x1:x2] = crop_resize_to_box(captures[label], x1, y1, x2, y2)

        x1, y1, x2, y2 = SLOTS[label]["reference"]
        canvas[y1:y2, x1:x2] = crop_resize_to_box(references[label], x1, y1, x2, y2)

    # Alpha-composite the frame over the assembled photo layer -- this is what
    # lets overlapping frame artwork sit correctly above the photos.
    alpha = frame_rgba[:, :, 3:4].astype(np.float32) / 255.0
    frame_rgb = frame_rgba[:, :, :3].astype(np.float32)
    composited = alpha * frame_rgb + (1.0 - alpha) * canvas.astype(np.float32)
    return composited.astype(np.uint8)


def detect_slots(frame_path: str = FRAME_PATH, alpha_threshold: int = 30, min_area: int = 500):
    """
    Detect the frame's transparent "window" rectangles directly from its alpha
    channel via connected-component analysis, sorted top-to-bottom then
    left-to-right. Use this to re-derive SLOTS whenever the frame is replaced,
    instead of re-measuring coordinates by hand.
    """
    frame_rgba = cv2.imread(frame_path, cv2.IMREAD_UNCHANGED)
    if frame_rgba is None or frame_rgba.shape[2] != 4:
        raise ValueError(f"{frame_path} must be a PNG with an alpha channel.")
    alpha = frame_rgba[:, :, 3]
    mask = (alpha < alpha_threshold).astype(np.uint8) * 255
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    boxes = []
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if area < min_area:
            continue
        boxes.append((x, y, x + bw, y + bh))
    boxes.sort(key=lambda b: (b[1], b[0]))
    return boxes


def load_session_captures(session_dir: str) -> dict:
    captures = {}
    for label in GESTURE_SEQUENCE:
        path = os.path.join(session_dir, f"{label}.jpg")
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"Missing capture: {path}")
        captures[label] = img
    return captures


def load_references(references_dir: str) -> dict:
    references = {}
    for label in GESTURE_SEQUENCE:
        path = os.path.join(references_dir, LABEL_IMAGE_MAP[label])
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"Missing reference: {path}")
        references[label] = img
    return references


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--session-dir", help="A photobooth_challenge.py demo/output/session_<ts>/ directory.")
    p.add_argument("--reference-dir", default=os.path.join(PROJECT_ROOT, "reference_photos"))
    p.add_argument("--frame", default=FRAME_PATH)
    p.add_argument("--output", help="Where to save the result (default: <session-dir>/framed_strip.jpg)")
    p.add_argument("--detect-slots", action="store_true", help="Print measured slot coordinates and exit.")
    return p.parse_args()


def main():
    args = parse_args()

    if args.detect_slots:
        for x1, y1, x2, y2 in detect_slots(args.frame):
            print(f"  ({x1},{y1}) -> ({x2},{y2})   size={x2-x1}x{y2-y1}")
        return

    if not args.session_dir:
        raise SystemExit("Pass --session-dir <path> or --detect-slots.")

    captures = load_session_captures(args.session_dir)
    references = load_references(args.reference_dir)
    result = build_framed_strip(captures, references, args.frame)

    output_path = args.output or os.path.join(args.session_dir, "framed_strip.jpg")
    cv2.imwrite(output_path, result)
    print(f"Saved -> {output_path}  ({result.shape[1]}x{result.shape[0]})")


if __name__ == "__main__":
    main()
