"""
collect_gesture_data.py

Data collection tool for building a custom hand-gesture dataset.

Opens the primary webcam, runs MediaPipe Hands on each frame to detect and
draw hand landmarks for visual confirmation, and lets the user save clean
(landmark-free) snapshots to a target directory by pressing 's'. Press 'c'
to toggle hands-free continuous (burst) capture, which keeps auto-saving
frames at a fixed interval while at least one hand is visible. Press 'q'
to quit.

Usage:
    python collect_gesture_data.py --label thumbs_up --output-dir data/gesture
    python collect_gesture_data.py -l peace -o data/gesture --camera 0
    python collect_gesture_data.py -l wave -o data/gesture --interval 0.1
"""

import argparse
import os
import time
from datetime import datetime

import cv2
import mediapipe as mp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect webcam images for a custom hand-gesture dataset."
    )
    parser.add_argument(
        "-l", "--label",
        type=str,
        default="gesture",
        help="Name of the gesture being recorded (used as a subfolder name).",
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=str,
        default="data/gesture",
        help="Root directory where captured images are saved.",
    )
    parser.add_argument(
        "-c", "--camera",
        type=int,
        default=0,
        help="Index of the webcam to open (default: 0, the primary webcam).",
    )
    parser.add_argument(
        "--max-hands",
        type=int,
        default=2,
        help="Maximum number of hands for MediaPipe Hands to track.",
    )
    parser.add_argument(
        "--detection-confidence",
        type=float,
        default=0.7,
        help="Minimum confidence for MediaPipe hand detection.",
    )
    parser.add_argument(
        "--tracking-confidence",
        type=float,
        default=0.5,
        help="Minimum confidence for MediaPipe hand tracking.",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=0.2,
        help="Seconds between auto-saved frames while continuous capture ('c') is on.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Images for this gesture go into <output_dir>/<label>/
    save_dir = os.path.join(args.output_dir, args.label)
    os.makedirs(save_dir, exist_ok=True)

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open webcam at index {args.camera}.")

    saved_count = 0
    continuous_mode = False
    last_save_time = 0.0

    def save_frame(image) -> str:
        # Timestamp-based filename (microsecond precision) avoids overwrites.
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{args.label}_{timestamp}.jpg"
        filepath = os.path.join(save_dir, filename)
        cv2.imwrite(filepath, image)
        return filepath

    print("Controls:")
    print("  's' - save a clean snapshot of the current frame")
    print("  'c' - toggle hands-free continuous capture (auto-saves while a hand is visible)")
    print("  'q' - quit and release the webcam")
    print(f"Saving images to: {os.path.abspath(save_dir)}")

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=args.max_hands,
        min_detection_confidence=args.detection_confidence,
        min_tracking_confidence=args.tracking_confidence,
    ) as hands:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to read frame from webcam. Exiting.")
                break

            # Mirror for a more natural, "selfie-style" view.
            frame = cv2.flip(frame, 1)

            # Keep an untouched copy so saved images have no drawn landmarks.
            clean_frame = frame.copy()

            # MediaPipe expects RGB input; OpenCV frames are BGR.
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_frame.flags.writeable = False
            results = hands.process(rgb_frame)

            display_frame = frame
            hand_detected = bool(results.multi_hand_landmarks)

            if hand_detected:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        display_frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style(),
                    )

            # Auto-save while continuous capture is on and a hand is in frame.
            now = time.time()
            if continuous_mode and hand_detected and (now - last_save_time) >= args.interval:
                filepath = save_frame(clean_frame)
                saved_count += 1
                last_save_time = now
                print(f"Saved: {filepath}")

            # On-screen status/help text.
            status_text = "Hand detected" if hand_detected else "No hand detected"
            status_color = (0, 255, 0) if hand_detected else (0, 0, 255)
            cv2.putText(display_frame, status_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
            cv2.putText(display_frame, f"Label: {args.label}  Saved: {saved_count}",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            if continuous_mode:
                cv2.putText(display_frame, "REC (continuous)", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.putText(display_frame, "[s] save   [c] toggle continuous   [q] quit",
                        (10, display_frame.shape[0] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 2)

            cv2.imshow("Hand Gesture Data Collection", display_frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("s"):
                filepath = save_frame(clean_frame)
                saved_count += 1
                print(f"Saved: {filepath}")
            elif key == ord("c"):
                continuous_mode = not continuous_mode
                last_save_time = 0.0  # allow an immediate save on toggle-on
                print(f"Continuous capture {'ON' if continuous_mode else 'OFF'}")
            elif key == ord("q"):
                print("Quitting...")
                break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Done. Total images saved: {saved_count}")


if __name__ == "__main__":
    main()
