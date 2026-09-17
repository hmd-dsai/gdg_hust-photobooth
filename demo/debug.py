"""
debug.py

LOCAL TESTING TOOL ONLY -- not part of the core pipeline. Deliberately built
by reaching into pipeline/'s lower-level pieces directly (GestureClassifier,
PretrainedEmotionClassifier, extract_two_hand_vector) rather than wrapping
GestureEmotionPipeline -- a debug tool needs to show every intermediate value
the production cascade normally discards (the full 7-class emotion
distribution even when gesture wins, the raw distance1/distance2 that feed
into P(gdg)), which GestureEmotionPipeline.predict() doesn't expose by design
(it only returns the cascade's final decision).

Three modes:

  live     Continuous live analysis -- every frame gets the full treatment
           (hand mesh, gesture readout, emotion distribution) in real time.
  capture  Live raw preview (cheap -- no model calls) until you press 'c',
           which freezes that frame and runs the full analysis on it. Press
           'c' again to go back to live preview, 'q' to quit. Useful for
           methodically checking specific poses without the numbers
           constantly changing while you read them.
  image    Same detailed analysis as 'capture', but on a single provided
           image file instead of the webcam -- no crop-to-4:3 applied (unlike
           the webcam modes), matching how pipeline/inference.py --image
           itself treats a provided file: analyze exactly what was given.

All three show, on a dark side panel next to the (mesh-annotated) frame:
  - which hand(s) Holistic detected
  - P(gdg) from the raw GestureClassifier, plus the raw distance1/distance2
    inter-hand features (informational only -- there's no hard distance gate
    on the actual cascade decision; an earlier version had one, removed after
    it produced false rejections on genuine gdg attempts)
  - the face crop used for emotion (drawn as a box on the frame) and the full
    7-class probability distribution, not just the argmax
  - the FINAL label the real cascade would report, for direct comparison
    against the raw numbers above it

Usage:
    python debug.py                                    # mode: live
    python debug.py --mode capture
    python debug.py --mode image --image path/to/photo.jpg
"""

import argparse
import os
import sys

import cv2
import mediapipe as mp
import numpy as np
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "pipeline"))

from inference import DEFAULT_GESTURE_THRESHOLD, get_device  # noqa: E402
from landmark_utils import extract_two_hand_vector  # noqa: E402
from models import GestureClassifier  # noqa: E402
from pretrained_emotion import PretrainedEmotionClassifier, face_bbox_from_landmarks  # noqa: E402

from split_screen_demo import CAMERA_ASPECT_RATIO, crop_to_aspect  # noqa: E402

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

PANEL_WIDTH = 460
DISPLAY_HEIGHT = 560
BG = (24, 23, 22)
FG = (255, 255, 255)
DIM = (150, 150, 150)
GREEN = (100, 220, 100)
RED = (90, 90, 255)
YELLOW = (60, 220, 230)
WINDOW_NAME = "debug -- gesture/emotion cascade internals"


# --------------------------------------------------------------------------- #
# Analysis: reach into pipeline/'s pieces directly, keep every intermediate
# value instead of collapsing to the cascade's final decision.
# --------------------------------------------------------------------------- #

def analyze_frame(frame_bgr: np.ndarray, holistic, gesture_model, emotion_model, device,
                   gesture_threshold: float) -> dict:
    frame_h, frame_w = frame_bgr.shape[:2]
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False
    results = holistic.process(rgb)

    left_present = results.left_hand_landmarks is not None
    right_present = results.right_hand_landmarks is not None

    gesture = {"left": left_present, "right": right_present, "prob": 0.0,
               "distance1": None, "distance2": None, "reason": "no hands detected"}

    if left_present or right_present:
        hand_vec = extract_two_hand_vector(
            results.left_hand_landmarks, results.right_hand_landmarks, frame_w, frame_h
        )
        with torch.no_grad():
            hand_t = torch.from_numpy(hand_vec).float().unsqueeze(0).to(device)
            prob = torch.sigmoid(gesture_model(hand_t)).item()

        gesture["prob"] = prob
        # Purely informational -- the cascade doesn't hard-gate on these (see
        # module docstring). Still shown since they're part of the model's
        # input and worth being able to eyeball.
        gesture["distance1"] = float(hand_vec[126])
        gesture["distance2"] = float(hand_vec[127])
        gesture["reason"] = "passes" if prob > gesture_threshold else f"below threshold ({gesture_threshold:.2f})"

    gesture_wins = gesture["prob"] > gesture_threshold

    emotion = {"face_detected": False, "bbox": None, "probs": None, "label": None, "confidence": None}
    if results.face_landmarks is not None:
        x1, y1, x2, y2 = face_bbox_from_landmarks(results.face_landmarks, frame_bgr.shape)
        if x2 > x1 and y2 > y1:
            emotion["face_detected"] = True
            emotion["bbox"] = (x1, y1, x2, y2)
            face_crop_rgb = cv2.cvtColor(frame_bgr[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
            result = emotion_model.predict(face_crop_rgb)
            emotion["probs"] = result["probs"]
            emotion["label"] = result["label"]
            emotion["confidence"] = result["confidence"]

    if gesture_wins:
        cascade_label = "gdg"
    elif emotion["face_detected"]:
        cascade_label = emotion["label"]
    else:
        cascade_label = "no_face_or_gesture_detected"

    return {"results": results, "gesture": gesture, "emotion": emotion, "cascade_label": cascade_label}


# --------------------------------------------------------------------------- #
# Drawing
# --------------------------------------------------------------------------- #

def draw_on_frame(frame_bgr: np.ndarray, analysis: dict) -> np.ndarray:
    out = frame_bgr.copy()
    results = analysis["results"]
    for hand_landmarks in (results.left_hand_landmarks, results.right_hand_landmarks):
        if hand_landmarks is not None:
            mp_drawing.draw_landmarks(
                out, hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style(),
            )
    if analysis["emotion"]["bbox"] is not None:
        x1, y1, x2, y2 = analysis["emotion"]["bbox"]
        cv2.rectangle(out, (x1, y1), (x2, y2), YELLOW, 1)
    return out


def _text(panel, s, x, y, scale=0.55, color=FG, thickness=1):
    cv2.putText(panel, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def _bar(panel, x, y, w, h, frac, color):
    frac = max(0.0, min(1.0, frac))
    cv2.rectangle(panel, (x, y), (x + w, y + h), (70, 70, 70), -1)
    cv2.rectangle(panel, (x, y), (x + int(w * frac), y + h), color, -1)
    cv2.rectangle(panel, (x, y), (x + w, y + h), (100, 100, 100), 1)


def build_panel(height: int, analysis: dict, gesture_threshold: float) -> np.ndarray:
    panel = np.full((height, PANEL_WIDTH, 3), BG, dtype=np.uint8)
    x = 16
    y = 28

    _text(panel, "GESTURE", x, y, 0.65, YELLOW, 2)
    y += 26
    g = analysis["gesture"]
    _text(panel, f"left hand:  {'yes' if g['left'] else 'no'}", x, y, color=FG if g["left"] else DIM)
    y += 22
    _text(panel, f"right hand: {'yes' if g['right'] else 'no'}", x, y, color=FG if g["right"] else DIM)
    y += 26

    if g["distance1"] is None:
        _text(panel, "(no hands -- gesture model never called)", x, y, color=DIM)
        y += 24
    else:
        prob_color = GREEN if g["prob"] > gesture_threshold else RED
        _text(panel, f"P(gdg) = {g['prob']:.3f}", x, y, color=prob_color)
        y += 20
        _bar(panel, x, y, PANEL_WIDTH - 2 * x, 10, g["prob"], prob_color)
        y += 24

        # Informational only -- no hard gate on these (see module docstring).
        # Bar fill is just a fixed visual scale, not a pass/fail line.
        DISTANCE_VISUAL_SCALE = 3.0
        for name, val in (("distance1", g["distance1"]), ("distance2", g["distance2"])):
            _text(panel, f"{name} = {val:.2f}", x, y, color=FG)
            y += 20
            _bar(panel, x, y, PANEL_WIDTH - 2 * x, 8, val / DISTANCE_VISUAL_SCALE, DIM)
            y += 22

        verdict_color = GREEN if g["reason"] == "passes" else RED
        _text(panel, f"-> {g['reason']}", x, y, color=verdict_color)
        y += 28

    cv2.line(panel, (x, y), (PANEL_WIDTH - x, y), (70, 70, 70), 1)
    y += 30

    _text(panel, "EMOTION", x, y, 0.65, YELLOW, 2)
    y += 26
    e = analysis["emotion"]
    if not e["face_detected"]:
        _text(panel, "no face detected", x, y, color=DIM)
        y += 24
    else:
        # Fixed order = the model's own id2label order (already alphabetical),
        # so bars don't jump around frame to frame in live mode. Label and
        # percentage each get a fixed-width column (sized for the longest
        # label, "surprise", and "100.0%") so the bar starting position never
        # collides with text of varying width.
        label_col_w, percent_col_w, bar_gap = 80, 70, 10
        bar_x = x + label_col_w + percent_col_w + bar_gap
        for label, p in e["probs"].items():
            is_top = label == e["label"]
            color = GREEN if is_top else FG
            _text(panel, label, x, y, color=color)
            pct_str = f"{p*100:.1f}%"
            (pct_w, _), _ = cv2.getTextSize(pct_str, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            _text(panel, pct_str, x + label_col_w + percent_col_w - pct_w, y, color=color)
            _bar(panel, bar_x, y - 10, PANEL_WIDTH - x - bar_x, 12, p, color if is_top else DIM)
            y += 24
        y += 6

    cv2.line(panel, (x, y), (PANEL_WIDTH - x, y), (70, 70, 70), 1)
    y += 30

    _text(panel, "FINAL (cascade decision)", x, y, 0.55, DIM)
    y += 26
    label = analysis["cascade_label"]
    label_display = "GDG (< >)" if label == "gdg" else label.upper()
    # Shrink to fit -- most labels are short emotion words, but
    # "no_face_or_gesture_detected" is long enough to overflow the panel
    # at a fixed scale.
    max_w = PANEL_WIDTH - 2 * x
    scale = 0.85
    while scale > 0.35:
        (tw, _), _ = cv2.getTextSize(label_display, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
        if tw <= max_w:
            break
        scale -= 0.05
    _text(panel, label_display, x, y, scale, YELLOW, 2)

    return panel


def compose_display(frame_bgr: np.ndarray, analysis: dict, gesture_threshold: float) -> np.ndarray:
    annotated = draw_on_frame(frame_bgr, analysis)
    h, w = annotated.shape[:2]
    scale = DISPLAY_HEIGHT / h
    left = cv2.resize(annotated, (int(round(w * scale)), DISPLAY_HEIGHT))
    panel = build_panel(DISPLAY_HEIGHT, analysis, gesture_threshold)
    return np.hstack([left, panel])


# --------------------------------------------------------------------------- #
# Modes
# --------------------------------------------------------------------------- #

def load_models(checkpoints_dir: str, device):
    gesture_ckpt = torch.load(os.path.join(checkpoints_dir, "gesture_model.pt"), map_location=device)
    gesture_model = GestureClassifier(input_dim=gesture_ckpt["input_dim"]).to(device)
    gesture_model.load_state_dict(gesture_ckpt["model_state"])
    gesture_model.eval()
    emotion_model = PretrainedEmotionClassifier(device=device)
    return gesture_model, emotion_model


def run_live(args, gesture_model, emotion_model, device):
    with mp_holistic.Holistic(static_image_mode=False, model_complexity=1, refine_face_landmarks=False,
                               min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        cap = cv2.VideoCapture(args.camera)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open webcam at index {args.camera}.")
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = crop_to_aspect(cv2.flip(frame, 1))
                analysis = analyze_frame(frame, holistic, gesture_model, emotion_model, device, args.threshold)
                cv2.imshow(WINDOW_NAME, compose_display(frame, analysis, args.threshold))
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()


def run_capture(args, gesture_model, emotion_model, device):
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open webcam at index {args.camera}.")
    try:
        while True:
            # Cheap live preview -- no model calls -- until 'c' is pressed.
            while True:
                ret, frame = cap.read()
                if not ret:
                    return
                frame = crop_to_aspect(cv2.flip(frame, 1))
                preview = frame.copy()
                cv2.putText(preview, "[c] capture & analyze   [q] quit", (10, preview.shape[0] - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
                cv2.imshow(WINDOW_NAME, preview)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("c"):
                    break
                if key == ord("q"):
                    return

            # static_image_mode=True: one isolated frame, no tracking assumptions,
            # matching how every other "single still image" path in this project works.
            with mp_holistic.Holistic(static_image_mode=True, model_complexity=1, refine_face_landmarks=False,
                                       min_detection_confidence=0.5) as holistic:
                analysis = analyze_frame(frame, holistic, gesture_model, emotion_model, device, args.threshold)
            cv2.imshow(WINDOW_NAME, compose_display(frame, analysis, args.threshold))
            print(analysis["cascade_label"], analysis["gesture"], analysis["emotion"]["label"])

            # Hold the result on screen until 'c' (recapture) or 'q' (quit).
            while True:
                key = cv2.waitKey(0) & 0xFF
                if key == ord("c"):
                    break
                if key == ord("q"):
                    return
    finally:
        cap.release()
        cv2.destroyAllWindows()


def run_image(args, gesture_model, emotion_model, device):
    frame = cv2.imread(args.image)
    if frame is None:
        raise FileNotFoundError(args.image)
    # No crop_to_aspect here on purpose -- analyze exactly what was given,
    # matching pipeline/inference.py --image's own behavior on a provided file.
    with mp_holistic.Holistic(static_image_mode=True, model_complexity=1, refine_face_landmarks=False,
                               min_detection_confidence=0.5) as holistic:
        analysis = analyze_frame(frame, holistic, gesture_model, emotion_model, device, args.threshold)
    print(analysis["cascade_label"], analysis["gesture"], analysis["emotion"]["label"])
    cv2.imshow(WINDOW_NAME, compose_display(frame, analysis, args.threshold))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Inspect every intermediate value in the gesture/emotion cascade.")
    p.add_argument("--mode", choices=["live", "capture", "image"], default="live")
    p.add_argument("--image", help="Path to an image file (required for --mode image).")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--checkpoints-dir", default=os.path.join(PROJECT_ROOT, "checkpoints"))
    p.add_argument("--threshold", type=float, default=DEFAULT_GESTURE_THRESHOLD)
    return p.parse_args()


def main():
    args = parse_args()
    if args.mode == "image" and not args.image:
        raise SystemExit("--mode image requires --image <path>.")

    device = get_device()
    print(f"Loading models on {device} ...")
    gesture_model, emotion_model = load_models(args.checkpoints_dir, device)

    if args.mode == "live":
        run_live(args, gesture_model, emotion_model, device)
    elif args.mode == "capture":
        run_capture(args, gesture_model, emotion_model, device)
    else:
        run_image(args, gesture_model, emotion_model, device)


if __name__ == "__main__":
    main()
