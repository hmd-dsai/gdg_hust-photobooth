"""
photobooth_challenge.py

LOCAL TESTING TOOL ONLY -- built on top of split_screen_demo.py's helpers,
still outside pipeline/ and still importing GestureEmotionPipeline as a plain
consumer would (nothing in pipeline/ is modified).

A 4-step photobooth minigame:
  0. Every frame is center-cropped to CAMERA_ASPECT_RATIO (4:3, see
     split_screen_demo.crop_to_aspect) immediately after capture, before
     anything else touches it. This is a plain geometric crop -- no face
     detection -- so it can never single out one person in a multi-person
     shot; it also means the live view, the model's input, and the saved
     photo are always exactly the same framing (WYSIWYG).
  1. Prompts the player through a fixed sequence of gestures/emotions
     (happy -> angry -> surprise -> gdg), showing each target's reference
     photo as a thumbnail in the corner of the live webcam view.
  2. A gesture must be held continuously -- the predicted label must not
     change to anything else -- for `--hold-seconds` (default 1.0s) before
     it's accepted. A small progress bar under the thumbnail shows how close
     the current hold is; any change in predicted label resets it to zero.
  3. The instant a hold completes, the CLEAN webcam frame (no overlays) is
     captured for that step, a brief "captured!" flash is shown, and the
     game advances to the next target.
  4. After all 4 steps, composes a 4x2 photobooth strip -- one row per
     gesture, left column = the player's captured photo at its real 4:3 shape
     (resized, not cropped further -- the framing was already decided once,
     at capture time), right column = the (square) reference photo -- and
     saves both the strip and the 4 individual 4:3 captures to
     demo/output/session_<timestamp>/.

Usage:
    python photobooth_challenge.py
    python photobooth_challenge.py --hold-seconds 1.5 --panel-size 400
"""

import argparse
import os
import sys
import time

import cv2
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "pipeline"))
sys.path.insert(0, os.path.dirname(__file__))

from inference import GestureEmotionPipeline, DEFAULT_GESTURE_THRESHOLD  # noqa: E402
from split_screen_demo import CAMERA_ASPECT_RATIO, crop_to_aspect, load_reference_panels  # noqa: E402

GESTURE_SEQUENCE = ["happy", "angry", "surprise", "gdg"]
HOLD_SECONDS_DEFAULT = 1.0
THUMB_SIZE = 140
CAPTION_HEIGHT = 40
WINDOW_NAME = "Photobooth Challenge (local test)"


def draw_prompt_overlay(frame, target_label, thumb, hold_elapsed, hold_seconds,
                         step_idx, total_steps, live_label):
    """Corner thumbnail prompt + hold progress bar + status text, on a COPY of frame."""
    out = frame.copy()
    h, w = out.shape[:2]

    margin = 20
    tx0 = w - THUMB_SIZE - margin
    ty0 = margin
    out[ty0:ty0 + THUMB_SIZE, tx0:tx0 + THUMB_SIZE] = thumb
    cv2.rectangle(out, (tx0, ty0), (tx0 + THUMB_SIZE, ty0 + THUMB_SIZE), (255, 255, 255), 2)
    cv2.putText(out, f"Do: {target_label}", (tx0, ty0 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.putText(out, f"Step {step_idx + 1}/{total_steps}", (margin, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    cv2.putText(out, f"seeing: {live_label}", (margin, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

    bar_w, bar_h = THUMB_SIZE, 14
    bx0, by0 = tx0, ty0 + THUMB_SIZE + 10
    frac = max(0.0, min(1.0, hold_elapsed / hold_seconds))
    cv2.rectangle(out, (bx0, by0), (bx0 + bar_w, by0 + bar_h), (90, 90, 90), -1)
    cv2.rectangle(out, (bx0, by0), (bx0 + int(bar_w * frac), by0 + bar_h), (0, 220, 0), -1)
    cv2.rectangle(out, (bx0, by0), (bx0 + bar_w, by0 + bar_h), (255, 255, 255), 1)

    return out


def draw_captured_flash(frame, target_label):
    out = frame.copy()
    h, w = out.shape[:2]
    text = f"{target_label.upper()} CAPTURED!"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)
    x, y = (w - tw) // 2, h // 2
    cv2.rectangle(out, (x - 20, y - th - 20), (x + tw + 20, y + 20), (0, 0, 0), -1)
    cv2.putText(out, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
    return out


def build_photobooth_strip(captures: dict, panels: dict, panel_size: int) -> np.ndarray:
    """
    4x2 grid: one row per gesture (GESTURE_SEQUENCE order), left=player photo,
    right=reference. The player's photo is shown at its real CAMERA_ASPECT_RATIO
    (4:3) shape -- just resized to a fixed height, not cropped down to a square --
    since it was already framed once (crop_to_aspect, at capture time) and cropping
    it *again* here would just be re-losing more of what the player actually saw
    and posed within. Only the reference panel (a separate, pre-existing square
    asset) stays square; there's no requirement the two match shape until there's
    a real designed frame to composite into instead of this plain grid.
    """
    user_cell_width = int(round(panel_size * CAMERA_ASPECT_RATIO))
    rows = []
    for label in GESTURE_SEQUENCE:
        user_img = captures[label]
        interp = cv2.INTER_AREA if panel_size < user_img.shape[0] else cv2.INTER_CUBIC
        user_cell = cv2.resize(user_img, (user_cell_width, panel_size), interpolation=interp)
        ref_cell = panels[label]
        row = np.hstack([user_cell, ref_cell])
        cap = np.zeros((CAPTION_HEIGHT, row.shape[1], 3), dtype=np.uint8)
        cv2.putText(cap, label, (10, CAPTION_HEIGHT - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        rows.append(np.vstack([cap, row]))
    return np.vstack(rows)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="4-step gesture/emotion photobooth minigame (local test).")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--checkpoints-dir", default=os.path.join(PROJECT_ROOT, "checkpoints"))
    p.add_argument("--features-dir", default=os.path.join(PROJECT_ROOT, "features"))
    p.add_argument("--labeled-dir", default=os.path.join(PROJECT_ROOT, "labeled"))
    p.add_argument("--output-dir", default=os.path.join(os.path.dirname(__file__), "output"))
    p.add_argument("--panel-size", type=int, default=480)
    p.add_argument("--hold-seconds", type=float, default=HOLD_SECONDS_DEFAULT)
    p.add_argument("--threshold", type=float, default=DEFAULT_GESTURE_THRESHOLD)
    return p.parse_args()


def main():
    args = parse_args()

    print(f"Loading reference panels from {args.labeled_dir} ...")
    panels = load_reference_panels(args.labeled_dir, args.panel_size)
    missing = [g for g in GESTURE_SEQUENCE if g not in panels]
    if missing:
        raise RuntimeError(f"Missing reference images for: {missing}")
    thumbs = {label: cv2.resize(img, (THUMB_SIZE, THUMB_SIZE)) for label, img in panels.items()}

    captures = {}
    aborted = False

    with GestureEmotionPipeline(args.checkpoints_dir, args.features_dir, static_image_mode=False) as pipeline:
        cap = cv2.VideoCapture(args.camera)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open webcam at index {args.camera}.")

        try:
            step_idx = 0
            hold_start = None

            while step_idx < len(GESTURE_SEQUENCE):
                target_label = GESTURE_SEQUENCE[step_idx]
                ret, frame = cap.read()
                if not ret:
                    print("Failed to read frame from webcam.")
                    aborted = True
                    break
                frame = cv2.flip(frame, 1)
                frame = crop_to_aspect(frame)  # WYSIWYG: same crop for display, model input, and capture
                clean_frame = frame.copy()  # captured, if this frame wins the hold -- no overlays baked in

                result = pipeline.predict(frame, gesture_threshold=args.threshold)
                live_label = result["label"]

                now = time.monotonic()
                if live_label == target_label:
                    if hold_start is None:
                        hold_start = now
                    hold_elapsed = now - hold_start
                else:
                    hold_start = None
                    hold_elapsed = 0.0

                display = draw_prompt_overlay(
                    frame, target_label, thumbs[target_label], hold_elapsed, args.hold_seconds,
                    step_idx, len(GESTURE_SEQUENCE), live_label,
                )
                cv2.imshow(WINDOW_NAME, display)

                if hold_elapsed >= args.hold_seconds:
                    captures[target_label] = clean_frame
                    print(f"Captured '{target_label}' ({step_idx + 1}/{len(GESTURE_SEQUENCE)})")
                    cv2.imshow(WINDOW_NAME, draw_captured_flash(clean_frame, target_label))
                    cv2.waitKey(800)
                    step_idx += 1
                    hold_start = None
                    continue

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    print("Aborted by user.")
                    aborted = True
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()

    if aborted or len(captures) < len(GESTURE_SEQUENCE):
        print("Session incomplete -- no photobooth strip saved.")
        return

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    session_dir = os.path.join(args.output_dir, f"session_{timestamp}")
    os.makedirs(session_dir, exist_ok=True)
    for label, img in captures.items():
        cv2.imwrite(os.path.join(session_dir, f"{label}.jpg"), img)

    strip = build_photobooth_strip(captures, panels, args.panel_size)
    strip_path = os.path.join(session_dir, "photobooth_strip.jpg")
    cv2.imwrite(strip_path, strip)
    print(f"Saved photobooth strip -> {strip_path}")

    cv2.imshow("Your Photobooth Strip -- press any key to close", strip)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
