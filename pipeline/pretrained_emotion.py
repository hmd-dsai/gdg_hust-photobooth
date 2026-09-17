"""
pretrained_emotion.py

Swaps the from-scratch landmark-MLP emotion classifier (models.EmotionClassifier,
33.45% val accuracy) for a much stronger pretrained model:

    trpakov/vit-face-expression -- a ViT (vit-base-patch16-224-in21k) fine-tuned
    directly on FER-2013's 7 classes.

This was NOT taken on faith from the model card. It was independently verified
against 1,011 real, held-out images from this project's own fer-2013/test/
(150/class, stratified, seed=42) -- see validate_pretrained_emotion.py:

    Overall accuracy: 70.13%   (model card claims 71.16% -- consistent)
    angry 65.3%  disgust 66.7%  fear 54.0%  happy 94.0%
    sad 67.3%  surprise 79.3%  neutral 63.3%

That's roughly 2x the landmark MLP's accuracy, so this is what the inference
cascade uses. models.EmotionClassifier / train.py's "emotion" task are still
here and still work (useful as a from-scratch baseline / for the write-up),
they're just no longer wired into GestureEmotionPipeline.

IMPORTANT -- input format: unlike the landmark MLP, this model consumes a
*cropped face image*, not landmark coordinates. It was trained on FER-2013's
already face-cropped images, so feeding it a raw, uncropped webcam frame
would badly mismatch its training distribution and tank accuracy -- this is
the most common reason a "pretrained emotion model" performs poorly in a live
demo despite a strong benchmark number, and is the likely explanation for the
earlier OpenCV pretrained model also underperforming. `face_bbox_from_landmarks`
derives a proper crop from MediaPipe's own face-landmark bounding box.
"""

from typing import Optional

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

MODEL_ID = "trpakov/vit-face-expression"


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class PretrainedEmotionClassifier:
    """Thin wrapper around the HF ViT model, downloaded/cached on first use."""

    def __init__(self, device: Optional[torch.device] = None):
        self.device = device or get_device()
        self.processor = AutoImageProcessor.from_pretrained(MODEL_ID)
        self.model = AutoModelForImageClassification.from_pretrained(MODEL_ID).to(self.device).eval()
        # Lowercased so labels line up with the rest of the pipeline
        # ("Happy" from the model card -> "happy" everywhere else here).
        self.id2label = {idx: name.lower() for idx, name in self.model.config.id2label.items()}

    @torch.no_grad()
    def predict(self, face_rgb: np.ndarray) -> dict:
        """
        `face_rgb`: an (H, W, 3) RGB uint8 crop containing just the face.
        `probs` is the full 7-class distribution (already computed for the
        argmax anyway, so returning it costs nothing extra) -- useful for
        anything that wants to show/inspect the full call, not just the winner.
        """
        image = Image.fromarray(face_rgb)
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        logits = self.model(**inputs).logits
        probs = torch.softmax(logits, dim=-1).squeeze(0)
        pred_id = int(probs.argmax().item())
        return {
            "label": self.id2label[pred_id],
            "confidence": probs[pred_id].item(),
            "probs": {self.id2label[i]: probs[i].item() for i in range(len(probs))},
        }


def face_bbox_from_landmarks(face_landmarks, frame_shape, margin: float = 0.25):
    """
    Pixel-space bounding box (x1, y1, x2, y2) around a MediaPipe face-landmark
    set, expanded by `margin` (as a fraction of box size) on each side so the
    crop includes the whole face -- forehead, chin, ears -- similar to
    FER-2013's own curated crops, rather than just the tight landmark hull
    (which hugs the eyes/nose/mouth and cuts off too much).
    """
    h, w = frame_shape[:2]
    xs = [lm.x for lm in face_landmarks.landmark]
    ys = [lm.y for lm in face_landmarks.landmark]
    x1, x2 = min(xs) * w, max(xs) * w
    y1, y2 = min(ys) * h, max(ys) * h
    bw, bh = x2 - x1, y2 - y1
    x1 -= bw * margin
    x2 += bw * margin
    y1 -= bh * margin
    y2 += bh * margin
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(w, int(x2)), min(h, int(y2))
    return x1, y1, x2, y2
