# gdg_hust-photobooth

A hierarchical gesture/emotion recognition photobooth. A live webcam feed is run
through a two-stage cascade:

1. **Gesture stage** -- a tiny MLP looks at MediaPipe hand landmarks. If it's
   confident the player is forming the "gdg" sign (two V-signs, `< >`), that
   wins immediately.
2. **Emotion stage** (fallback) -- otherwise, a pretrained ViT classifies the
   player's cropped face into one of FER-2013's 7 emotions (angry, disgust,
   fear, happy, sad, surprise, neutral).

Both stages run on **MediaPipe Holistic landmarks / face crops**, not raw-pixel
CNNs trained from scratch -- see [Architecture](#architecture) for why the
emotion stage in particular ended up as a pretrained model rather than a
from-scratch landmark classifier.

## 1. Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

First run of anything emotion-related (`inference.py`, `pretrained_emotion.py`,
`validate_pretrained_emotion.py`) downloads the `trpakov/vit-face-expression`
ViT weights (~336MB) from Hugging Face and caches them under
`~/.cache/huggingface/hub/`. Do this once, ahead of time, if you plan to run
on a machine with unreliable internet -- see the note in
[Retraining the hand-gesture model](#retraining-the-hand-gesture-model) for
copying that cache to another machine instead of re-downloading.

## 2. Directory layout

```
fer-2013/train/<emotion>/*.jpg    FER-2013 dataset (7 emotion classes)
fer-2013/test/<emotion>/*.jpg
dataset/gdg/*.jpg                 Webcam captures of the "gdg" gesture (positive)
dataset/noise/*.jpg               Webcam captures of everything else (negative)
labeled/                          One reference photo per class, for on-screen prompts
features/                         Extracted landmark vectors (.npz) -- generated
checkpoints/                      Trained model weights (.pt) -- generated
pipeline/                         Core library: preprocessing, models, training, inference
demo/                             Local-only test/demo tools built on top of pipeline/
demo/assets/frame.png             Designed photobooth frame, composited by demo/frame_compositor.py
app/                              Web app: FastAPI backend (main.py) + browser frontend (static/)
output/                           Web app's saved sessions/strips (gitignored) -- separate from demo/output/
```

## 3. Scripts

### 3.1. Root

**`collect_gesture_data.py`**
Webcam data-collection tool for building `dataset/<label>/`. Runs
`mediapipe.solutions.hands` live, draws landmarks for visual confirmation, and
saves a *clean* (landmark-free) snapshot on `s`, or continuously at a fixed
interval while a hand is visible on `c` (hands-free burst capture -- toggle
again to stop). Files are timestamped so nothing overwrites. `q` quits.

```bash
python collect_gesture_data.py --label gdg --output-dir dataset --max-hands 2
python collect_gesture_data.py --label noise --output-dir dataset
```

### 3.2. `pipeline/` -- core library

This is the part meant to ship as-is to a backend later; nothing here should
be edited for local-testing purposes (see `demo/` instead).

| File | Role |
|---|---|
| `landmark_utils.py` | Shared math: turns raw MediaPipe landmarks into fixed-size, translation-/scale-invariant vectors. Faces are centered on the nose tip and scaled by inter-ocular distance; each hand is centered on the wrist and scaled by palm size. A missing hand zero-pads to a 63-d zero vector. Used identically by `preprocess.py` (offline) and `inference.py` (live), so training and inference math never drift apart. |
| `preprocess.py` | Batch feature extraction. Runs Holistic once per image and writes `features/face_{train,test}.npz` (1404-d, from `fer-2013/`) and `features/hand_{train,test}.npz` (126-d, from `dataset/gdg` + `dataset/noise`), plus `features/label_map.json`. See [Retraining the hand-gesture model](#retraining-the-hand-gesture-model) for the `--only hand` fast path. |
| `datasets.py` | `GestureDataset` (binary) and `EmotionDataset` (7-class) -- thin in-memory `torch.utils.data.Dataset` wrappers over the `.npz` files. |
| `models.py` | `GestureClassifier` (126-d -> 1 logit) and `EmotionClassifier` (1404-d -> 7 logits): small BatchNorm+Dropout MLPs. `EmotionClassifier` is a from-scratch baseline kept for reference/comparison -- it's **not** what the live pipeline actually uses for emotion (see below). |
| `train.py` | Trains `GestureClassifier` and/or `EmotionClassifier` with Adam + `BCEWithLogitsLoss` / `CrossEntropyLoss`. Explicitly handles class imbalance two ways: a `WeightedRandomSampler` on the training loader, and inverse-frequency `pos_weight`/`weight` in the loss itself. Prints per-epoch train/val loss+accuracy, saves the best checkpoint, early-stops on a validation-accuracy plateau. |
| `pretrained_emotion.py` | The emotion classifier the live pipeline actually uses: `trpakov/vit-face-expression`, a ViT fine-tuned directly on FER-2013's 7 classes, wrapped to accept a face-crop image. Also has `face_bbox_from_landmarks()`, which derives that crop from MediaPipe's face-landmark bounding box (feeding this model a raw, uncropped frame badly mismatches its training distribution). |
| `validate_pretrained_emotion.py` | Independent accuracy check for the pretrained ViT against this project's own held-out `fer-2013/test/` images (not just the model card's claim). Last run: **70.13%** on 1,011 stratified test images, vs. **33.45%** for the from-scratch `EmotionClassifier` MLP. |
| `inference.py` | The cascade itself: `GestureEmotionPipeline`. Runs Holistic once, gates the gesture model on "was at least one hand detected at all" (a true zero-hands frame is never even asked), and falls back to the pretrained emotion model on a Holistic-derived face crop. Has an `--image` mode (single photo) and a `--webcam` demo. |

**Why a pretrained model for emotion but not for gesture?** The from-scratch
landmark MLP tops out around 33% on FER-2013 -- landmark geometry alone
discards texture cues (wrinkles, skin shading) that a lot of FER-2013's classes
actually depend on, and that's an architectural ceiling, not a training issue.
There's no such problem for the 2-class gesture task: "gdg" is a
well-separated hand shape, and the from-scratch `GestureClassifier` MLP
reaches ~99% validation accuracy on it, so there was no reason to swap it out.

### 3.3. `demo/` -- local-testing-only tools

Not part of the shippable pipeline. Both scripts import `GestureEmotionPipeline`
the way an external consumer would (plain library import, nothing in
`pipeline/` is touched) -- kept separate on purpose, since the plan is to hand
`pipeline/` off to a web backend as-is later.

**`split_screen_demo.py`**
A 1x2 live view: left is the raw webcam feed, right is the `labeled/`
reference photo matching whatever the cascade just predicted (square-cropped
and size-normalized, since the 8 source photos range from 217x220 to
4050x3240). Replaces a plain text-overlay label with a visual comparison.

```bash
python demo/split_screen_demo.py
```

**`photobooth_challenge.py`**
The actual minigame, built on `split_screen_demo.py`'s helpers. Prompts the
player through `happy -> angry -> surprise -> gdg` in order, showing each
target's reference photo as a corner thumbnail with a hold-progress bar. A
gesture must match continuously for `--hold-seconds` (default 1.0s) -- any
change resets the bar to zero -- before it's captured (a clean frame, no
overlays). After all 4 steps, saves a 4x2 strip (row = gesture, columns =
player photo | reference photo) plus the 4 individual captures to
`demo/output/session_<timestamp>/`.

```bash
python demo/photobooth_challenge.py --hold-seconds 1.0 --panel-size 480
```

**`frame_compositor.py`**
Composites 4 player captures + 4 reference photos into the designed
`demo/assets/frame.png` strip -- photos are placed on a blank canvas first,
then the frame (with its alpha channel) is drawn on top, so the frame's
decorative elements correctly overlap the photo edges instead of being hidden
underneath them. `build_framed_strip(captures, references)` is the reusable
function; it's what `app/main.py`'s `/api/challenge/compose` endpoint calls.
Slot coordinates for a replacement frame can be re-measured directly from its
alpha channel with `--detect-slots`, rather than hand-measured.

```bash
python demo/frame_compositor.py --session-dir demo/output/session_<timestamp>
python demo/frame_compositor.py --detect-slots   # after replacing frame.png
```

### 3.4. `app/` -- web application

A FastAPI backend (`main.py`) wrapping the same `GestureEmotionPipeline` used
everywhere else in this repo, serving a browser frontend (`static/`) that
implements the same challenge flow as `demo/photobooth_challenge.py` -- camera
capture happens in the browser (`getUserMedia`), the backend never touches a
camera directly. See [Running the web app](#running-the-web-app) below for how
to launch it, and [INTEGRATION.md](INTEGRATION.md) for the API surface if
you're building against it rather than running it as-is.

## 4. Architecture

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

## 5. Running the web app

Two ways to set up `app/`, depending on how many machines you're deploying to
and whether you want Docker's isolation or a lighter native install.

### 5.1. Docker

```bash
docker compose up --build -d
```
Open `http://localhost:8000`. The image bakes the ViT weights in at build
time (`download_models.py` runs during the build), so a running container
never needs internet access or a host's Hugging Face cache.

**The first build is slow and network-heavy -- budget real time for it.**
In testing, the `pip install -r requirements-docker.txt` step alone (mediapipe
+ torch + transformers + opencv) took 20-36 minutes depending on connection
quality, and the final image is **~8GB**. This is inherent to those
dependencies, not something `--build` can speed up. **Do this once, not per
machine** -- see "Distributing to multiple laptops" below.

```bash
docker compose logs -f       # tail logs
docker compose restart       # picks up changes under app/ (bind-mounted) -- no rebuild needed
docker compose build         # rebuild -- required for changes outside app/ (pipeline/, demo/,
                              # requirements-docker.txt, Dockerfile itself)
docker compose down          # stop and remove the container
```
Optional: set `IMGUR_CLIENT_ID` in your shell (or a `.env` file next to
`docker-compose.yml`) before `up` to enable Imgur uploads -- omit it and the
app falls back to an anonymous CDN, then a local LAN URL.

### 5.2. Manual (no Docker)

```bash
.venv/bin/pip install "fastapi>=0.110.0" "uvicorn[standard]>=0.28.0" \
  "python-multipart>=0.0.9" "pydantic>=2.0.0"
.venv/bin/python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```
This assumes the ML dependencies are already installed via the root
[Setup](#setup) (`requirements.txt`) -- use that file, not
`requirements-docker.txt`, for a native install. `requirements-docker.txt`
deliberately pins `mediapipe==0.10.18` (not `0.10.21`, which doesn't have a
`linux/arm64` wheel and would fail the Docker build outright) and skips
`torch`/`torchvision` (installed as a separate step from PyTorch's own CPU
index) -- neither restriction applies outside Docker.

### 5.3. Accessing from another device on the same network

Both paths serve on `0.0.0.0:8000`, so a phone/tablet/other laptop on the same
wifi can open `http://<host-laptop-LAN-IP>:8000`. If the browser refuses
camera access over plain HTTP on a non-`localhost` address, allow it via
`chrome://flags/#unsafely-treat-insecure-origin-as-secure` (add
`http://<that-IP>:8000`) -- camera access requires a secure context, and
`localhost` gets one for free but a LAN IP doesn't.

### 5.4. Distributing to multiple laptops (e.g. a club fest with several PICs)

Whichever path you pick, **never repeat the slow dependency install live, on
each laptop, at the venue.** Build or download once, distribute the result,
and let each laptop's setup be offline and fast:

- **Docker**: build once, `docker save gdg-photobooth:latest -o gdg-photobooth.tar`
  (~8GB), copy it to each laptop (USB / shared drive), `docker load -i
  gdg-photobooth.tar` there -- fully offline from that point on. For a larger
  or recurring fleet, push to a registry instead (e.g. GitHub Container
  Registry, `ghcr.io/<org>/<repo>`) so each laptop just does `docker pull`
  ahead of time, on good wifi, days before the event. If the fleet mixes
  Windows (`amd64`) and Apple Silicon Mac (`arm64`) laptops, build with
  `docker buildx build --platform linux/amd64,linux/arm64 --push` so one pull
  automatically gets the right variant per machine -- a plain `docker build`
  only targets the machine you build on (this repo's images have so far only
  been built for `arm64`; a Windows laptop needs an `amd64` build first).

- **Manual (recommended for a small, known set of laptops)**: build a "wheelhouse"
  once per distinct OS/CPU in the fleet -- no installation happens yet, just
  downloading the files:
  ```bash
  pip download --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple \
    --platform win_amd64 --python-version 310 --only-binary=:all: \
    -d wheelhouse-win torch torchvision -r requirements.txt
  ```
  (swap `--platform` for `macosx_11_0_arm64` / `macosx_10_9_x86_64` on Mac).
  Copy the resulting folder + this repo to each matching laptop, then there,
  with no network involved at all:
  ```bash
  python -m venv .venv
  .venv/bin/pip install --no-index --find-links=wheelhouse-win -r requirements.txt
  ```
  Each laptop builds its own venv locally (avoiding the broken-symlink risk of
  copying a `.venv` folder wholesale across machines), from a payload a few GB
  smaller than the Docker image, with no Docker Desktop install required on
  any target machine.

## 6. Retraining the hand-gesture model

The gesture model is the one you'll retrain most often -- e.g. after
collecting more `dataset/gdg` or `dataset/noise` images to fix a
misclassification you saw live. Full pipeline:

1. **Collect more data** for whichever class needs it:
   ```bash
   python collect_gesture_data.py --label gdg --output-dir dataset --max-hands 2
   python collect_gesture_data.py --label noise --output-dir dataset
   ```
   Use `c` (continuous capture) to gather many frames quickly while holding a
   pose; vary angle/lighting/distance for the `gdg` class especially, since
   that's the smaller, harder-to-generalize class.

2. **Re-extract hand features only** -- `--only hand` skips the slow (~20 min)
   FER-2013 face pass entirely, so this takes ~1-2 minutes:
   ```bash
   python pipeline/preprocess.py --only hand
   ```
   This regenerates `features/hand_train.npz`, `features/hand_test.npz`, and
   `features/label_map.json` (harmless to rewrite; content is unchanged unless
   you also changed the class folder names). `features/face_*.npz` is left
   untouched.

   Note: any `dataset/gdg/*.jpg` frame where Holistic fails to detect *any*
   hand is automatically dropped rather than kept as a false-positive "gdg
   with no visible hands" label -- see the comment in `preprocess.py` if
   you're wondering why the kept count is slightly less than the file count.

3. **Retrain the gesture model**:
   ```bash
   python pipeline/train.py --task gesture --features-dir features --out-dir checkpoints
   ```
   Watch `train_acc`/`val_acc` in the printed per-epoch log. It early-stops
   automatically (default patience: 10 epochs) and only overwrites
   `checkpoints/gesture_model.pt` when validation accuracy improves, so a bad
   run can't clobber a working checkpoint with something worse.

4. **Sanity-check it**, ideally on real frames rather than trusting the
   validation number alone:
   ```bash
   python pipeline/inference.py --image path/to/a/test/photo.jpg
   # or, for a live feel:
   python demo/split_screen_demo.py
   ```

5. **If you change the gesture classes themselves** (rename `dataset/gdg` ->
   something else, or add a third gesture folder), update
   `GESTURE_CLASS_ORDER` in `pipeline/preprocess.py` first, then repeat from
   step 2. The 7 emotion classes (`EMOTION_CLASS_ORDER`) are independent of
   this and don't need touching.

**Retraining the emotion model** is not part of this workflow by design --
the live pipeline uses the pretrained ViT (`pretrained_emotion.py`), not
`EmotionClassifier`. `pipeline/train.py --task emotion` still works if you want
to reproduce or extend the from-scratch baseline for comparison, but it won't
change what `inference.py` actually predicts.

**Moving to another machine without re-downloading the ViT weights**: copy
`~/.cache/huggingface/hub/models--trpakov--vit-face-expression` to the same
path on the target machine.