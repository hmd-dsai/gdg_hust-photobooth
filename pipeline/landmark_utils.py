"""
landmark_utils.py

Shared helpers for turning MediaPipe Holistic output into fixed-size,
translation- and scale-invariant feature vectors. Used by both preprocess.py
(offline dataset building) and inference.py (live prediction), so the exact
same math is applied at train time and at inference time.

The 126-d two-hand vector (extract_two_hand_vector) normalizes each hand in
its own independent local frame (wrist-centered, palm-scaled), which means it
cannot see how far apart the two hands are from each other -- two correctly-
shaped V-signs held far apart are nearly indistinguishable from two held
close together forming "gdg", since neither hand's local vector encodes the
other hand's position. compute_inter_hand_distances() / the two appended
distance features address exactly that gap, making the full vector 128-d.
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

# --- Inter-hand distance features -------------------------------------------
# Used only by compute_inter_hand_distances() / extract_two_hand_vector() below --
# NOT part of the per-hand local normalization above. These measure how far apart
# the two hands' fingertips are (something the wrist-centered, palm-scaled 126-d
# vector structurally cannot see, since each hand lives in its own independent
# local coordinate frame -- see the module docstring below).
INDEX_TIP_IDX = 8
MIDDLE_TIP_IDX = 12
INDEX_BONE_CHAIN = (5, 6, 7, 8)      # INDEX_MCP -> PIP -> DIP -> TIP
MIDDLE_BONE_CHAIN = (9, 10, 11, 12)  # MIDDLE_MCP -> PIP -> DIP -> TIP

# Sentinel for "can't be computed" (a hand is missing) -- deliberately NOT 0.0,
# since a distance of 0 would read as "the two fingertips are touching", the
# opposite of the truth. A large distance instead correctly reads as "these
# fingertips are nowhere near each other".
MISSING_HAND_DISTANCE_PENALTY = 99.0


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


def _aspect_corrected_points(hand_landmarks, aspect_ratio: float) -> Optional[np.ndarray]:
    """
    Raw (NUM_HAND_LANDMARKS, 3) landmark array with x scaled by the image's aspect
    ratio (width / height). MediaPipe's x, y are each normalized to [0, 1] against
    their own axis (width, height respectively) -- on a non-square frame those two
    axes are not the same physical unit, so a plain Euclidean distance across x and
    y would be warped. Scaling x by width/height re-expresses both axes in a common
    unit (heights) before any distance is computed. z is left as-is (MediaPipe does
    not document an equivalent correction for it). None if the hand wasn't detected.
    """
    pts = landmarks_to_array(hand_landmarks)
    if pts is None:
        return None
    corrected = pts.copy()
    corrected[:, 0] *= aspect_ratio
    return corrected


def _bone_chain_length(points: np.ndarray, chain) -> float:
    """
    Sum of consecutive-joint distances along `chain` (e.g. MCP->PIP->DIP->TIP).
    Robust to a slightly bent finger, unlike a single straight-line MCP-to-tip
    measurement, which shortens as soon as the finger curls even a little.
    """
    return float(sum(
        np.linalg.norm(points[chain[i]] - points[chain[i + 1]])
        for i in range(len(chain) - 1)
    ))


def compute_inter_hand_distances(left_hand_landmarks, right_hand_landmarks,
                                  image_width: int, image_height: int) -> tuple:
    """
    (distance1, distance2): fingertip-to-fingertip distance between the two hands,
    each normalized by that finger's own (bone-sum) length so the result reflects
    "how many finger-lengths apart" rather than an absolute, frame-scale-dependent
    number:
      distance1 = ||left index tip (8) - right index tip (8)|| / avg(left, right index length)
      distance2 = ||left middle tip (12) - right middle tip (12)|| / avg(left, right middle length)

    Both distances (tip-to-tip and finger bone lengths) are computed in aspect-ratio-
    corrected, full (x, y, z) space -- see _aspect_corrected_points.

    Returns (MISSING_HAND_DISTANCE_PENALTY, MISSING_HAND_DISTANCE_PENALTY) if either
    hand is missing: with only one (or zero) hands present there is no second
    fingertip to measure to, and this is deliberately not 0.0 (see the constant's
    docstring above).
    """
    aspect_ratio = image_width / image_height
    left = _aspect_corrected_points(left_hand_landmarks, aspect_ratio)
    right = _aspect_corrected_points(right_hand_landmarks, aspect_ratio)

    if left is None or right is None:
        return MISSING_HAND_DISTANCE_PENALTY, MISSING_HAND_DISTANCE_PENALTY

    def normalized_tip_distance(tip_idx: int, chain) -> float:
        tip_distance = float(np.linalg.norm(left[tip_idx] - right[tip_idx]))
        avg_finger_length = (_bone_chain_length(left, chain) + _bone_chain_length(right, chain)) / 2.0
        avg_finger_length = max(avg_finger_length, 1e-6)  # guard against a degenerate detection
        return tip_distance / avg_finger_length

    distance1 = normalized_tip_distance(INDEX_TIP_IDX, INDEX_BONE_CHAIN)
    distance2 = normalized_tip_distance(MIDDLE_TIP_IDX, MIDDLE_BONE_CHAIN)
    return distance1, distance2


def extract_two_hand_vector(left_hand_landmarks, right_hand_landmarks,
                             image_width: int, image_height: int) -> np.ndarray:
    """
    Concatenate left + right locally-normalized hand vectors, plus two inter-hand
    distance features -> (NUM_HAND_LANDMARKS * 3 * 2 + 2,) = 128-d:
      [0:63]   left hand  (wrist-centered, palm-scaled -- see extract_hand_vector)
      [63:126] right hand (same)
      [126]    distance1 -- normalized left/right index-fingertip distance
      [127]    distance2 -- normalized left/right middle-fingertip distance

    Whichever hand is absent contributes a zero-padded slice in [0:126] rather than
    causing the sample to be discarded -- required because the "gdg" gesture is
    defined by two hands together, but plenty of valid frames (both positive
    mid-gesture and negative/background) only have zero or one hand visible. The
    two distance features get MISSING_HAND_DISTANCE_PENALTY instead of 0.0 in that
    same situation, for the reason documented on that constant.

    `image_width` / `image_height` (the source frame's pixel dimensions) are
    required to aspect-ratio-correct the distance features -- see
    compute_inter_hand_distances.
    """
    left = extract_hand_vector(left_hand_landmarks)
    right = extract_hand_vector(right_hand_landmarks)
    distance1, distance2 = compute_inter_hand_distances(
        left_hand_landmarks, right_hand_landmarks, image_width, image_height
    )
    return np.concatenate([left, right, np.array([distance1, distance2], dtype=np.float32)])
