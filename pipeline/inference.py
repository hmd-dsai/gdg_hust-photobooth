"""
inference.py

Cascade inference for the hierarchical gesture/emotion pipeline:

  1. Run MediaPipe Holistic once on the frame.
  2. Run the (landmark-based) GestureClassifier on the two-hand vector, but
     ONLY if Holistic found at least one hand -- a frame with zero hands is
     categorically not "gdg", and we never even ask the model in that case.
     Otherwise, trust the model's own P(gdg) directly: P(gdg) > gesture_threshold
     (default 0.85) -> report "gdg" immediately, skipping the emotion model
     entirely. (An earlier version also hard-rejected based on the two
     inter-hand distance features -- landmark_utils.extract_two_hand_vector's
     distance1/distance2 -- when they exceeded a fixed threshold. Removed: in
     practice it produced false rejections on genuine gdg attempts. Those two
     features are still part of the model's input and still influence P(gdg)
     through its learned weights, just no longer as a separate hard override.)
  3. Otherwise, fall back to the pretrained ViT emotion classifier
     (pretrained_emotion.PretrainedEmotionClassifier) on a face crop derived
     from Holistic's face landmarks, and report the predicted emotion -- or
     "no_face_or_gesture_detected" if Holistic found neither a confident
     gesture nor a face in the frame.

Usage:
    python inference.py --image path/to/photo.jpg
    python inference.py --webcam
"""

import argparse
import json
import os

import cv2
import mediapipe as mp
import numpy as np
import torch

from landmark_utils import extract_two_hand_vector
from models import GestureClassifier
from pretrained_emotion import PretrainedEmotionClassifier, face_bbox_from_landmarks

DEFAULT_GESTURE_THRESHOLD = 0.85


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class GestureEmotionPipeline:
    """Loads both trained models once and exposes a single-frame cascade `predict`."""

    def __init__(self, checkpoints_dir: str = "checkpoints", features_dir: str = "features", device=None,
                 static_image_mode: bool = False):
        """
        `static_image_mode`: pass True for single, unrelated still images (each frame
        is detected fresh, no cross-frame tracking assumptions -- matches preprocess.py,
        and is what makes cold-start detection reliable on a single photo). Pass False
        (the default) for a live webcam stream, where Holistic's frame-to-frame tracking
        makes detection faster and smoother once a hand/face has been picked up.
        """
        self.device = device or get_device()

        with open(os.path.join(features_dir, "label_map.json")) as f:
            label_map = json.load(f)
        self.gesture_classes = label_map["gesture_classes"]  # ["noise", "gdg"]

        gesture_ckpt = torch.load(os.path.join(checkpoints_dir, "gesture_model.pt"), map_location=self.device)
        self.gesture_model = GestureClassifier(input_dim=gesture_ckpt["input_dim"]).to(self.device)
        self.gesture_model.load_state_dict(gesture_ckpt["model_state"])
        self.gesture_model.eval()

        # Pretrained ViT (trpakov/vit-face-expression), not the from-scratch
        # landmark MLP -- see pretrained_emotion.py for why, and for the
        # independently-measured accuracy (70.1% vs. 33.5%).
        self.emotion_model = PretrainedEmotionClassifier(device=self.device)

        self.holistic = mp.solutions.holistic.Holistic(
            static_image_mode=static_image_mode,
            model_complexity=1,
            refine_face_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def close(self):
        self.holistic.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    @torch.no_grad()
    def predict(self, bgr_frame: np.ndarray, gesture_threshold: float = DEFAULT_GESTURE_THRESHOLD) -> dict:
        """
        Run the full cascade on one BGR frame (as read by cv2.imread / cv2.VideoCapture).
        Returns a dict with at least {"label", "stage", "confidence"}.
        """
        frame_h, frame_w = bgr_frame.shape[:2]
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self.holistic.process(rgb)

        # --- Stage 1: gesture ---
        # A frame with NO hands detected at all is categorically not "gdg" -- gate on
        # that *before* trusting the classifier, so this stays correct even if the model
        # is later retrained on data that again contains a no-hands-detected mislabeled
        # sample (see preprocess.py's cleanup of exactly that). We deliberately do NOT
        # require *both* hands here: MediaPipe's hand detector inside Holistic is known
        # to miss a hand fairly often even when the gesture is genuinely being formed
        # (weaker than the standalone Hands solution, since it derives hand ROIs from
        # pose landmarks) -- being that strict would trade real recall for a case the
        # zero-padded model already handles fine as a genuine partial-evidence input.
        no_hands_detected = results.left_hand_landmarks is None and results.right_hand_landmarks is None
        if no_hands_detected:
            gesture_prob = 0.0
        else:
            hand_vec = extract_two_hand_vector(
                results.left_hand_landmarks, results.right_hand_landmarks, frame_w, frame_h
            )
            hand_t = torch.from_numpy(hand_vec).float().unsqueeze(0).to(self.device)
            gesture_prob = torch.sigmoid(self.gesture_model(hand_t)).item()

        if gesture_prob > gesture_threshold:
            return {"label": "gdg", "stage": "gesture", "confidence": gesture_prob}

        # --- Stage 2: fall back to emotion ---
        if results.face_landmarks is None:
            return {
                "label": "no_face_or_gesture_detected",
                "stage": "none",
                "confidence": 1.0 - gesture_prob,
                "gesture_confidence": gesture_prob,
            }

        x1, y1, x2, y2 = face_bbox_from_landmarks(results.face_landmarks, bgr_frame.shape)
        if x2 <= x1 or y2 <= y1:
            return {
                "label": "no_face_or_gesture_detected",
                "stage": "none",
                "confidence": 1.0 - gesture_prob,
                "gesture_confidence": gesture_prob,
            }
        face_crop_rgb = cv2.cvtColor(bgr_frame[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)

        emotion_result = self.emotion_model.predict(face_crop_rgb)
        return {
            "label": emotion_result["label"],
            "stage": "emotion",
            "confidence": emotion_result["confidence"],
            "gesture_confidence": gesture_prob,
        }


def demo_image(path: str, checkpoints_dir: str, features_dir: str, threshold: float):
    frame = cv2.imread(path)
    if frame is None:
        raise FileNotFoundError(path)
    # static_image_mode=True: this is a single, standalone photo, not a video frame.
    with GestureEmotionPipeline(checkpoints_dir, features_dir, static_image_mode=True) as pipeline:
        result = pipeline.predict(frame, gesture_threshold=threshold)
    print(result)


def demo_webcam(checkpoints_dir: str, features_dir: str, threshold: float, camera: int = 0):
    with GestureEmotionPipeline(checkpoints_dir, features_dir, static_image_mode=False) as pipeline:
        cap = cv2.VideoCapture(camera)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open webcam at index {camera}.")
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.flip(frame, 1)
                result = pipeline.predict(frame, gesture_threshold=threshold)
                text = f"{result['label']} ({result['confidence']:.2f})"
                cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                cv2.putText(frame, "[q] quit", (10, frame.shape[0] - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
                cv2.imshow("Gesture/Emotion Cascade", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the gesture->emotion cascade.")
    p.add_argument("--image", help="Path to a single image to classify.")
    p.add_argument("--webcam", action="store_true", help="Run a live webcam demo instead.")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--checkpoints-dir", default="checkpoints")
    p.add_argument("--features-dir", default="features")
    p.add_argument("--threshold", type=float, default=DEFAULT_GESTURE_THRESHOLD)
    return p.parse_args()


def main():
    args = parse_args()
    if args.webcam:
        demo_webcam(args.checkpoints_dir, args.features_dir, args.threshold, args.camera)
    elif args.image:
        demo_image(args.image, args.checkpoints_dir, args.features_dir, args.threshold)
    else:
        raise SystemExit("Pass --image <path> or --webcam.")


if __name__ == "__main__":
    main()
