"""
split_screen_demo.py

LOCAL TESTING TOOL ONLY -- not part of the core pipeline.

Deliberately kept outside pipeline/ and imports GestureEmotionPipeline as a
consumer would (a plain library import, nothing modified) rather than adding
this visualization into the core inference code. The intent is for
pipeline/inference.py's GestureEmotionPipeline to ship as-is to a web backend
later; this script is just a throwaway way to eyeball predictions locally.

Shows a 1x2 split screen instead of a text label overlay:
  - left:  the live webcam feed, center-cropped to CAMERA_ASPECT_RATIO (4:3)
           right after capture -- the same cropped frame is what's shown,
           what's fed to the model, and (in photobooth_challenge.py) what's
           saved, so there's never a mismatch between what you see and what
           gets analyzed.
  - right: the reference photo for whatever the pipeline just predicted
           (from labeled/), square-cropped and size-normalized so all 8
           reference photos -- which arrive in wildly different sizes and
           aspect ratios -- display consistently.

Usage:
    python split_screen_demo.py
    python split_screen_demo.py --camera 1 --panel-size 400
"""

import argparse
import os
import sys

import cv2
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "pipeline"))

from inference import GestureEmotionPipeline, DEFAULT_GESTURE_THRESHOLD  # noqa: E402

# Reference photos don't share the emotion classes' exact naming (gdg's file
# is "gdgoc_hand_sign.jpeg", and extensions vary between .jpg/.jpeg) -- an
# explicit map is simpler and more robust than guessing.
LABEL_IMAGE_MAP = {
    "angry": "angry.jpg",
    "disgust": "disgust.jpg",
    "fear": "fear.jpg",
    "happy": "happy.jpg",
    "neutral": "neutral.jpg",
    "sad": "sad.jpg",
    "surprise": "surprise.jpeg",
    "gdg": "gdgoc_hand_sign.jpeg",
}

CAPTION_HEIGHT = 40
PLACEHOLDER_BG = (40, 40, 40)  # dark gray, for "no detection"

# The webcam's native frame is cropped to this aspect ratio (width / height) ONCE,
# immediately after capture -- before display, before the model, before anything
# else -- so the live view, the model's input, and the saved photo are always the
# exact same framing (WYSIWYG). This is a plain geometric center-crop with no face
# detection involved: it can never single out one person in a multi-person shot,
# it just consistently keeps the middle of the frame. 4:3 landscape, matching a
# classic photobooth look while trimming less than a full square crop would.
CAMERA_ASPECT_RATIO = 4 / 3


def crop_to_aspect(frame: np.ndarray, target_ratio: float = CAMERA_ASPECT_RATIO) -> np.ndarray:
    """
    Center-crop `frame` to `target_ratio` (width / height), trimming whichever
    dimension is in excess -- the *other* dimension is kept at its full native
    extent. No resizing here, so this only changes framing, not resolution/quality.
    """
    h, w = frame.shape[:2]
    current_ratio = w / h
    if current_ratio > target_ratio:
        # wider than target -> trim width, keep full height
        new_w = int(round(h * target_ratio))
        x0 = (w - new_w) // 2
        return frame[:, x0:x0 + new_w]
    else:
        # taller than target (or already narrower) -> trim height, keep full width
        new_h = int(round(w / target_ratio))
        y0 = (h - new_h) // 2
        return frame[y0:y0 + new_h, :]


def square_crop_resize(img: np.ndarray, size: int) -> np.ndarray:
    """Center-crop to a square (from the shorter side) then resize to size x size."""
    h, w = img.shape[:2]
    side = min(h, w)
    y0, x0 = (h - side) // 2, (w - side) // 2
    cropped = img[y0:y0 + side, x0:x0 + side]
    interp = cv2.INTER_AREA if side > size else cv2.INTER_CUBIC
    return cv2.resize(cropped, (size, size), interpolation=interp)


def load_reference_panels(labeled_dir: str, panel_size: int) -> dict:
    """Preprocess every reference photo once at startup -> {label: square BGR image}."""
    panels = {}
    for label, filename in LABEL_IMAGE_MAP.items():
        path = os.path.join(labeled_dir, filename)
        img = cv2.imread(path)
        if img is None:
            print(f"  [warn] could not load reference image for '{label}': {path}")
            continue
        panels[label] = square_crop_resize(img, panel_size)
    return panels


def make_placeholder_panel(panel_size: int, text: str) -> np.ndarray:
    panel = np.full((panel_size, panel_size, 3), PLACEHOLDER_BG, dtype=np.uint8)
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    x = max(10, (panel_size - tw) // 2)
    y = panel_size // 2
    cv2.putText(panel, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 2)
    return panel


def with_caption(panel: np.ndarray, caption: str, panel_size: int) -> np.ndarray:
    """Stack a dark caption bar under a square panel so left/right heights match."""
    out = np.zeros((panel_size + CAPTION_HEIGHT, panel_size, 3), dtype=np.uint8)
    out[:panel_size] = panel
    cv2.putText(out, caption, (10, panel_size + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return out


def build_canvas(webcam_frame: np.ndarray, right_panel: np.ndarray, right_caption: str, panel_size: int) -> np.ndarray:
    h = panel_size
    wh, ww = webcam_frame.shape[:2]
    scale = h / wh
    left = cv2.resize(webcam_frame, (int(round(ww * scale)), h))
    left_captioned = np.zeros((h + CAPTION_HEIGHT, left.shape[1], 3), dtype=np.uint8)
    left_captioned[:h] = left
    cv2.putText(left_captioned, "webcam  |  [q] quit", (10, h + 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

    right_captioned = with_caption(right_panel, right_caption, panel_size)
    return np.hstack([left_captioned, right_captioned])


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Local split-screen test view for the gesture/emotion cascade.")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--checkpoints-dir", default=os.path.join(PROJECT_ROOT, "checkpoints"))
    p.add_argument("--features-dir", default=os.path.join(PROJECT_ROOT, "features"))
    p.add_argument("--labeled-dir", default=os.path.join(PROJECT_ROOT, "labeled"))
    p.add_argument("--panel-size", type=int, default=480, help="Side length (px) of the square reference panel.")
    p.add_argument("--threshold", type=float, default=DEFAULT_GESTURE_THRESHOLD)
    return p.parse_args()


def main():
    args = parse_args()

    print(f"Loading reference panels from {args.labeled_dir} ...")
    panels = load_reference_panels(args.labeled_dir, args.panel_size)
    placeholder = make_placeholder_panel(args.panel_size, "no face / gesture")

    with GestureEmotionPipeline(args.checkpoints_dir, args.features_dir, static_image_mode=False) as pipeline:
        cap = cv2.VideoCapture(args.camera)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open webcam at index {args.camera}.")
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.flip(frame, 1)
                frame = crop_to_aspect(frame)  # WYSIWYG: display, model input, and any capture all use this

                result = pipeline.predict(frame, gesture_threshold=args.threshold)
                label = result["label"]
                right_panel = panels.get(label, placeholder)
                caption = f"{label}  ({result['confidence']:.2f})"

                canvas = build_canvas(frame, right_panel, caption, args.panel_size)
                cv2.imshow("Gesture/Emotion Cascade -- split screen (local test)", canvas)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
