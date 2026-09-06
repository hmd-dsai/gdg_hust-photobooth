# Integration guide

How to use the trained gesture/emotion cascade in another project -- as a
bare Python model, behind a web backend, or as a base for new features.
For how the model was built/trained, see [README.md](README.md).

## What you need

- `pipeline/*.py` (the inference code) -- `landmark_utils.py`, `models.py`,
  `pretrained_emotion.py`, `inference.py` are the ones actually loaded at
  runtime; the rest (`preprocess.py`, `train.py`, `datasets.py`,
  `validate_pretrained_emotion.py`) are training-only.
- `checkpoints/gesture_model.pt` -- the trained gesture MLP.
- `features/label_map.json` -- tiny, but required at runtime (`GestureEmotionPipeline`
  reads it directly for `gesture_classes`).
- `requirements.txt` installed (`pip install -r requirements.txt`).
- Internet access on first run, OR a pre-copied
  `~/.cache/huggingface/hub/models--trpakov--vit-face-expression` (~336MB) --
  the emotion model auto-downloads from Hugging Face the first time
  `GestureEmotionPipeline` is constructed. There's no local checkpoint file
  for it; `checkpoints/emotion_model.pt` if present is an unused, unrelated
  from-scratch baseline -- don't load it.

## Quick start

```python
import cv2
import sys
sys.path.insert(0, "pipeline")  # or install pipeline/ as a package
from inference import GestureEmotionPipeline

pipeline = GestureEmotionPipeline(
    checkpoints_dir="checkpoints",
    features_dir="features",
    static_image_mode=True,  # True = single photos, False = live video stream
)

frame = cv2.imread("photo.jpg")          # BGR, like any cv2.imread/VideoCapture frame
result = pipeline.predict(frame)
print(result)
# {'label': 'happy', 'stage': 'emotion', 'confidence': 0.94, 'gesture_confidence': 0.02}

pipeline.close()
```

`predict()` returns:
- `label`: `"gdg"`, one of the 7 FER-2013 emotions (`angry`, `disgust`,
  `fear`, `happy`, `sad`, `surprise`, `neutral`), or
  `"no_face_or_gesture_detected"`.
- `stage`: which stage produced the label -- `"gesture"`, `"emotion"`, or `"none"`.
- `confidence`: that stage's own confidence (sigmoid prob for gesture, softmax
  prob for emotion).
- `gesture_confidence`: always present when `stage != "gesture"` -- useful for
  debugging why the cascade fell through.

Use `pipeline.predict(frame, gesture_threshold=0.9)` to require a higher bar
before reporting `"gdg"` (default `0.85`).

## Adding to a web app

`GestureEmotionPipeline` is plain Python/PyTorch, so the usual shape is a
small backend endpoint (Flask/FastAPI) that the browser posts webcam frames
to: `getUserMedia`/canvas in the browser -> JPEG bytes -> `cv2.imdecode` ->
`pipeline.predict()` -> JSON back. See `demo/photobooth_challenge.py` for the
capture-and-hold UX this project already prototyped locally (not shippable
as-is -- it's an OpenCV desktop window -- but the state machine translates
directly: run `predict()` per frame, track how long the label has matched a
target continuously, act once it holds).

**Construct one `GestureEmotionPipeline` per worker/process, not per request**
-- model loading takes a few seconds. Reuse it across requests.

## Building new features on top

- **New gesture classes**: retrain `GestureClassifier` (README's "Retraining
  the hand-gesture model") after adding a new `dataset/<label>/` folder --
  `GESTURE_CLASS_ORDER` in `preprocess.py` is the only place class identity
  is defined.
- **Different emotion set / better accuracy**: swap `pretrained_emotion.py`'s
  `MODEL_ID` for another Hugging Face image-classification checkpoint with a
  compatible `id2label`; nothing else in the cascade needs to change as long
  as it still takes a face-crop image and outputs class probabilities.
- **Per-user/session state** (e.g. a multi-step challenge like
  `photobooth_challenge.py`): keep that state outside `GestureEmotionPipeline`
  -- the pipeline itself is stateless per call (Holistic's internal video
  tracking state is the only exception, see caveats below).

## Caveats

- **`static_image_mode` matters more than it looks.** `True` = detect fresh
  every call (needed for unrelated single photos -- a live-mode instance can
  fail to detect a hand/face reliably on a single cold frame). `False` = video
  tracking (smoother and faster across a *continuous* stream from the same
  session, but each `GestureEmotionPipeline` instance's Holistic then carries
  state between calls -- don't share one instance's video-mode Holistic
  across unrelated users/streams).
- **Frames are BGR**, matching `cv2.imread`/`cv2.VideoCapture` directly --
  don't pre-convert to RGB yourself, `predict()` does that internally.
- **Tiny images hurt detection.** MediaPipe's detectors weren't trained on
  very small crops; if your source images are much smaller than a typical
  webcam frame (under ~256px on the short side), upscale before calling
  `predict()` -- see `preprocess.py`'s `load_and_upscale` for the exact
  approach used to build the training data.
- **The `gdg` far-apart threshold is empirical, not universal.**
  `MAX_PLAUSIBLE_GDG_DISTANCE = 1.5` in `inference.py` was calibrated against
  this project's own training captures (a specific camera, distance, and
  framing). If your deployment's camera setup is very different, sanity-check
  it against a few real captures before trusting it as-is.
- **Emotion accuracy ceiling is ~70%, not near-100%.** The pretrained ViT was
  independently measured at 70.1% on FER-2013's own test set (see
  `pipeline/validate_pretrained_emotion.py`) -- solid for a 7-class problem,
  but don't build a UX that assumes near-perfect emotion recognition, especially
  for the harder classes (`fear`, `disgust`).
- **First run downloads ~336MB.** Budget for that (or pre-cache it) before a
  live/offline demo -- see "What you need" above.
- **Not GPU-required, but the emotion stage is the heavy part.** ~30ms/frame
  (Holistic) + ~60ms/frame (ViT, CPU-only) on a modern laptop CPU -- fine for
  "predict once per capture," not necessarily for classifying every live
  frame at 30fps on weaker hardware. Throttle emotion-stage calls if you need
  a continuously-updating live label.
