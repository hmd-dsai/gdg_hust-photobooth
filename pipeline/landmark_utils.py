"""
landmark_utils.py

Shared helpers for turning MediaPipe Holistic output into fixed-size,
translation- and scale-invariant feature vectors. Used by both preprocess.py
(offline dataset building) and inference.py (live prediction), so the exact
same math is applied at train time and at inference time.
"""

from typing import Optional

import numpy as np

NUM_FACE_LANDMARKS = 468
NUM_HAND_LANDMARKS = 21

# --- Face reference points -------------------------------------------------
# Widely-used community reference indices in MediaPipe's 468-point Face Mesh
# topology (Google does not publish official "nose"/"eye" names for indices,
# but these are the de-facto standard points used across face-mesh tutorials
# and verified visually to sit at the nose tip / outer eye corners).
FACE_CENTER_IDX = 1     # nose tip -> translation reference
FACE_SCALE_IDX_A = 33   # right eye, outer corner
FACE_SCALE_IDX_B = 263  # left eye, outer corner  -> inter-ocular distance = scale reference

# --- Hand reference points --------------------------------------------------
# Canonical MediaPipe HandLandmark indices (officially documented).
HAND_WRIST_IDX = 0            # WRIST -> translation reference
HAND_SCALE_IDX = 9            # MIDDLE_FINGER_MCP -> stable "palm size" scale reference
                               # (using palm size rather than max finger extent keeps
                               # finger-spread/shape information intact post-normalization,
                               # which is exactly what a gesture classifier needs to see).


def landmarks_to_array(landmark_list) -> Optional[np.ndarray]:
    """Convert a MediaPipe NormalizedLandmarkList into an (N, 3) float32 array, or None."""
    if landmark_list is None:
        return None
    return np.array(
        [[lm.x, lm.y, lm.z] for lm in landmark_list.landmark], dtype=np.float32
    )


def normalize_points(points: np.ndarray, center_idx: int, scale_a: int, scale_b: Optional[int] = None) -> np.ndarray:
    """
    Make a landmark set translation- and scale-invariant:
      1. Subtract the `center_idx` point from every point, so it no longer
         matters where in the frame the face/hand is.
      2. Divide by a reference distance, so it no longer matters how close
         the subject is to the camera / how large the face or hand appears.

    If `scale_b` is given, the reference distance is ||points[scale_a] - points[scale_b]||
    (e.g. inter-ocular distance for a face). Otherwise it's ||points[scale_a] - points[center_idx]||
    (e.g. wrist-to-palm distance for a hand).
    """
    centered = points - points[center_idx]
    if scale_b is not None:
        scale = float(np.linalg.norm(points[scale_a] - points[scale_b]))
    else:
        scale = float(np.linalg.norm(points[scale_a] - points[center_idx]))
    scale = max(scale, 1e-6)  # guard against degenerate/collapsed detections
    return centered / scale


def extract_face_vector(face_landmarks) -> Optional[np.ndarray]:
    """Flattened, normalized (NUM_FACE_LANDMARKS * 3,) face feature vector, or None if no face."""
    pts = landmarks_to_array(face_landmarks)
    if pts is None:
        return None
    pts = normalize_points(pts, FACE_CENTER_IDX, FACE_SCALE_IDX_A, FACE_SCALE_IDX_B)
    return pts.flatten()


def extract_hand_vector(hand_landmarks) -> np.ndarray:
    """
    Flattened, normalized (NUM_HAND_LANDMARKS * 3,) hand feature vector.
    Zero-padded (all zeros) if this hand was not detected in the frame --
    the robust "missing hand" strategy: a fixed-size vector either way.
    """
    pts = landmarks_to_array(hand_landmarks)
    if pts is None:
        return np.zeros(NUM_HAND_LANDMARKS * 3, dtype=np.float32)
    pts = normalize_points(pts, HAND_WRIST_IDX, HAND_SCALE_IDX)
    return pts.flatten()


def extract_two_hand_vector(left_hand_landmarks, right_hand_landmarks) -> np.ndarray:
    """
    Concatenate left + right hand vectors -> (NUM_HAND_LANDMARKS * 3 * 2,).
    Whichever hand is absent contributes a zero-padded slice rather than
    causing the sample to be discarded -- required because the "gdg" gesture
    is defined by two hands together, but plenty of valid frames (both
    positive mid-gesture and negative/background) only have zero or one
    hand visible.
    """
    left = extract_hand_vector(left_hand_landmarks)
    right = extract_hand_vector(right_hand_landmarks)
    return np.concatenate([left, right])
