# GDG-HUST Photobooth

A hierarchical gesture/emotion recognition photobooth. A live webcam feed is run through a two-stage cascade:

1. **Gesture stage**: a tiny MLP looks at MediaPipe hand landmarks. If it's confident the player is forming the "gdg" sign (two V-signs, `< >`), that wins immediately.
2. **Emotion stage** (fallback): otherwise, a pretrained ViT classifies the player's cropped face into one of FER-2013's 7 emotions (angry, disgust, fear, happy, sad, surprise, neutral).

## 1. Setup

Requires **Python 3.10**.

### 1.1. Without Docker

**macOS / Linux:**
```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

**Windows (PowerShell / Command Prompt):**
```powershell
py -3.10 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000`. First run downloads the `trpakov/vit-face-expression` ViT weights (~336MB) from Hugging Face (needs internet once, then it's cached under `~/.cache/huggingface/hub/`).

### 1.2. With Docker

```bash
docker compose up --build -d
```
Open `http://localhost:8000`. The image bakes the ViT weights in at build time, so the container never needs internet at runtime. The first build is slow (installs PyTorch/MediaPipe/Transformers) -- that's expected, do it once rather than per machine.

### 1.3. Access from other devices (phones, tablets) over Wi-Fi

With the server running on the organizer's laptop, find that machine's LAN IP (e.g. `192.168.1.100`), then on another device connected to the **same Wi-Fi network**, open `http://<that-ip>:8000`.

Browsers block camera access on a plain HTTP origin that isn't `localhost`, so on Chrome you may need to visit `chrome://flags/#unsafely-treat-insecure-origin-as-secure`, add `http://<that-ip>:8000`, and relaunch the browser before the camera permission prompt works.

## 2. Retraining the hand-gesture model

The gesture model is the one you'll retrain most often (e.g. after collecting more `data/gesture/gdg` or `data/gesture/noise` images to fix a misclassification you saw live). Full pipeline:

1. **Collect more data** for whichever class needs it:
   ```bash
   python scripts/collect_gesture_data.py --label gdg --output-dir data/gesture --max-hands 2
   python scripts/collect_gesture_data.py --label noise --output-dir data/gesture
   ```
   Use `c` (continuous capture) to gather many frames quickly while holding a pose; vary angle lighting/distance for the `gdg` class especially, since that's the smaller, harder-to-generalize class.

2. **Re-extract hand features only** -- `--only hand` skips the slow (~20 min)
   FER-2013 face pass entirely, so this takes ~1-2 minutes:
   ```bash
   python pipeline/preprocess.py --only hand
   ```
   This regenerates `features/hand_train.npz`, `features/hand_test.npz`, and `features/label_map.json` (harmless to rewrite; content is unchanged unless you also changed the class folder names). `features/face_*.npz` is left untouched.

   Note: any `data/gesture/gdg/*.jpg` frame where Holistic fails to detect *any* hand is automatically dropped rather than kept as a false-positive "gdg with no visible hands" label -- see the comment in `preprocess.py` if you're wondering why the kept count is slightly less than the file count.

3. **Retrain the gesture model**:
   ```bash
   python pipeline/train.py --task gesture --features-dir features --out-dir checkpoints
   ```
   Watch `train_acc`/`val_acc` in the printed per-epoch log. It early-stops automatically (default patience: 10 epochs) and only overwrites `checkpoints/gesture_model.pt` when validation accuracy improves, so a bad run can't clobber a working checkpoint with something worse.

4. **Sanity-check it**, ideally on real frames rather than trusting the validation number alone:
   ```bash
   python pipeline/inference.py --image path/to/a/test/photo.jpg
   # or, for a live feel:
   python demo/split_screen_demo.py
   ```

5. **If you change the gesture classes themselves** (rename `data/gesture/gdg` -> something else, or add a third gesture folder), update `GESTURE_CLASS_ORDER` in `pipeline/preprocess.py` first, then repeat from step 2. The 7 emotion classes (`EMOTION_CLASS_ORDER`) are independent of this and don't need touching.

**Retraining the emotion model** is not part of this workflow by design -- the live pipeline uses the pretrained ViT (`pretrained_emotion.py`), not `EmotionClassifier`. `pipeline/train.py --task emotion` still works if you want to reproduce or extend the from-scratch baseline for comparison, but it won't change what `inference.py` actually predicts.

**Moving to another machine without re-downloading the ViT weights**: copy `~/.cache/huggingface/hub/models--trpakov--vit-face-expression` to the same path on the target machine.

## 3. Architecture

```
                     ┌──────────────────────┐
   webcam frame ---> │  MediaPipe Holistic  │   (one pass: hands + face)
                     └──────────┬───────────┘
                                │
               no hands at all?  --yes-->  skip gesture model (P=0)
                                │no
                                v
                     ┌──────────────────────┐
                     │  GestureClassifier   │   126-d hand vector -> 1 logit
                     │  (from-scratch MLP)  │
                     └──────────┬───────────┘
                       [ P(gdg) > 0.85 ? ]
                        │yes           │no
                        v              v
                     "gdg"  ┌──────────────┐
                            │face detected?│--no-->"no_face_or_gesture_detected"
                            └──────┬───────┘
                                   │yes
                                   v
                        ┌──────────────────────┐
                        │    pretrained ViT    │  face crop -> 7-class softmax
                        │ (vit-face-expression)│
                        └──────────┬───────────┘
                                   v
                          predicted emotion
```

## 4. Directory layout

```
data/fer-2013/train/<emotion>/*.jpg  FER-2013 dataset (7 emotion classes)
data/fer-2013/test/<emotion>/*.jpg
data/gesture/gdg/*.jpg               Webcam captures of the "gdg" gesture (positive)
data/gesture/noise/*.jpg             Webcam captures of everything else (negative)
reference_photos/                    One reference photo per class, for on-screen prompts
features/                            Extracted landmark vectors (.npz) -- generated
checkpoints/                         Trained model weights (.pt) -- generated
pipeline/                            Core library: preprocessing, models, training, inference
demo/                                Local-only test/demo tools built on top of pipeline/
scripts/                             Standalone utility scripts (data collection, model download)
app/                                 Web app: FastAPI backend (main.py) + browser frontend (static/)
app/frame_compositor.py              Composites captures into the designed strip -- production code, used by app/main.py
app/static/branding/frame.png        Designed photobooth frame, composited by app/frame_compositor.py
output/                              Web app's saved sessions/strips (gitignored) -- separate from demo/output/
```

## 5. Scripts

### 5.1. `scripts/` (standalone utilities)

**`collect_gesture_data.py`**
Webcam data-collection tool for building `data/gesture/<label>/`. Runs `mediapipe.solutions.hands` live, draws landmarks for visual confirmation, and saves a *clean* (landmark-free) snapshot on `s`, or continuously at a fixed interval while a hand is visible on `c` (hands-free burst capture -- toggle again to stop). Files are timestamped so nothing overwrites. `q` quits.

```bash
python scripts/collect_gesture_data.py --label gdg --output-dir data/gesture --max-hands 2
python scripts/collect_gesture_data.py --label noise --output-dir data/gesture
```

### 5.2. `pipeline/` (core library)

This is the part meant to ship as-is to a backend later; nothing here should be edited for local-testing purposes (see `demo/` instead).

| File | Role |
|---|---|
| `landmark_utils.py` | Shared math: turns raw MediaPipe landmarks into fixed-size, translation-/scale-invariant vectors. Faces are centered on the nose tip and scaled by inter-ocular distance; each hand is centered on the wrist and scaled by palm size. A missing hand zero-pads to a 63-d zero vector. Used identically by `preprocess.py` (offline) and `inference.py` (live), so training and inference math never drift apart. |
| `preprocess.py` | Batch feature extraction. Runs Holistic once per image and writes `features/face_{train,test}.npz` (1404-d, from `data/fer-2013/`) and `features/hand_{train,test}.npz` (126-d, from `data/gesture/gdg` + `data/gesture/noise`), plus `features/label_map.json`. See [Retraining the hand-gesture model](#retraining-the-hand-gesture-model) for the `--only hand` fast path. |
| `datasets.py` | `GestureDataset` (binary) and `EmotionDataset` (7-class) -- thin in-memory `torch.utils.data.Dataset` wrappers over the `.npz` files. |
| `models.py` | `GestureClassifier` (126-d -> 1 logit) and `EmotionClassifier` (1404-d -> 7 logits): small BatchNorm+Dropout MLPs. `EmotionClassifier` is a from-scratch baseline kept for reference/comparison -- it's **not** what the live pipeline actually uses for emotion (see below). |
| `train.py` | Trains `GestureClassifier` and/or `EmotionClassifier` with Adam + `BCEWithLogitsLoss` / `CrossEntropyLoss`. Explicitly handles class imbalance two ways: a `WeightedRandomSampler` on the training loader, and inverse-frequency `pos_weight`/`weight` in the loss itself. Prints per-epoch train/val loss+accuracy, saves the best checkpoint, early-stops on a validation-accuracy plateau. |
| `pretrained_emotion.py` | The emotion classifier the live pipeline actually uses: `trpakov/vit-face-expression`, a ViT fine-tuned directly on FER-2013's 7 classes, wrapped to accept a face-crop image. Also has `face_bbox_from_landmarks()`, which derives that crop from MediaPipe's face-landmark bounding box (feeding this model a raw, uncropped frame badly mismatches its training distribution). |
| `validate_pretrained_emotion.py` | Independent accuracy check for the pretrained ViT against this project's own held-out `data/fer-2013/test/` images (not just the model card's claim). Last run: **70.13%** on 1,011 stratified test images, vs. **33.45%** for the from-scratch `EmotionClassifier` MLP. |
| `inference.py` | The cascade itself: `GestureEmotionPipeline`. Runs Holistic once, gates the gesture model on "was at least one hand detected at all" (a true zero-hands frame is never even asked), and falls back to the pretrained emotion model on a Holistic-derived face crop. Has an `--image` mode (single photo) and a `--webcam` demo. |

**Why a pretrained model for emotion but not for gesture?** The from-scratch landmark MLP tops out around 33% on FER-2013 -- landmark geometry alone discards texture cues (wrinkles, skin shading) that a lot of FER-2013's classes actually depend on, and that's an architectural ceiling, not a training issue. There's no such problem for the 2-class gesture task: "gdg" is a well-separated hand shape, and the from-scratch `GestureClassifier` MLP reaches ~99% validation accuracy on it, so there was no reason to swap it out.

### 5.3. `demo/` (local-testing-only tools)

Not part of the shippable pipeline -- everything here imports `GestureEmotionPipeline` the way an external consumer would (plain library import, nothing in `pipeline/` is touched), kept separate on purpose since the plan is to hand `pipeline/` off to a web backend as-is later. (`frame_compositor.py` used to live here too, but it moved to `app/` -- see below -- since the web app actually depends on it at runtime.)

**`split_screen_demo.py`**
A 1x2 live view: left is the raw webcam feed, right is the `reference_photos/` reference photo matching whatever the cascade just predicted (square-cropped and size-normalized, since the 8 source photos range from 217x220 to 4050x3240). Replaces a plain text-overlay label with a visual comparison.

```bash
python demo/split_screen_demo.py
```

**`photobooth_challenge.py`**
The actual minigame, built on `split_screen_demo.py`'s helpers. Prompts the player through `happy -> angry -> surprise -> gdg` in order, showing each target's reference photo as a corner thumbnail with a hold-progress bar. A gesture must match continuously for `--hold-seconds` (default 1.0s) -- any change resets the bar to zero -- before it's captured (a clean frame, no overlays). After all 4 steps, saves a 4x2 strip (row = gesture, columns = player photo | reference photo) plus the 4 individual captures to `demo/output/session_<timestamp>/`.

```bash
python demo/photobooth_challenge.py --hold-seconds 1.0 --panel-size 480
```

### 5.4. `app/` -- web application

A FastAPI backend (`main.py`) wrapping the same `GestureEmotionPipeline` used everywhere else in this repo, serving a browser frontend (`static/`) that implements the same challenge flow as `demo/photobooth_challenge.py` -- camera capture happens in the browser (`getUserMedia`), the backend never touches a camera directly. See [Setup](#setup) above for how to launch it, and [INTEGRATION.md](INTEGRATION.md) for the API surface if you're building against it rather than running it as-is.

**`frame_compositor.py`**
Lives in `app/` rather than `demo/` because `main.py`'s `/api/challenge/compose` endpoint depends on it directly at runtime, not just for local testing. Composites 4 player captures + 4 reference photos into the designed `app/static/branding/frame.png` strip -- photos are placed on a blank canvas first, then the frame (with its alpha channel) is drawn on top, so the frame's decorative elements correctly overlap the photo edges instead of being hidden underneath them. `build_framed_strip(captures, references)` is the reusable function. Slot coordinates for a replacement frame can be re-measured directly from its alpha channel with `--detect-slots`, rather than hand-measured.

```bash
python app/frame_compositor.py --session-dir demo/output/session_<timestamp>
python app/frame_compositor.py --detect-slots   # after replacing frame.png
```

## 6. Future features

- [ ] Add START button before starting webcam and inference
- [ ] Use better emotion detector ViT (currently it's easy to do happy and surprised, hard to do angry, and nearly impossible to do any other emotions)
- [ ] Use more meme reference images, and randomize them
- [ ] Train more hand gestures, maybe hand + face as well (eg. mewing, so I need to reintroduce face mesh)
- [ ] Collect hand gestures data from more people
- [ ] Deal with >= 2 people taking photos (current behavior is unknown to users)
- [ ] Improve output image resolution
- [ ] Add boomerang/GIF output like in real photobooth
- [ ] Use a better web to upload temp output photots (tmpfiles.org has a fake download button)
- [ ] The gigantic PyTorch is installed only for MLP, so migrate to NumPy