"""
FastAPI Server for GDG HUST Photobooth.
Exposes real-time cascade prediction and photobooth strip composition endpoints with Imgur & Cloud QR sharing.
"""
import base64
from datetime import datetime
import io
import json
import os
import sys
import time
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "pipeline"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "demo"))

from inference import DEFAULT_GESTURE_THRESHOLD, GestureEmotionPipeline
from split_screen_demo import LABEL_IMAGE_MAP, crop_to_aspect, load_reference_panels
from frame_compositor import build_framed_strip, load_references

CHECKPOINTS_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
FEATURES_DIR = os.path.join(PROJECT_ROOT, "features")
REFERENCE_PHOTOS_DIR = os.path.join(PROJECT_ROOT, "reference_photos")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

app = FastAPI(title="GDG Photobooth API", version="1.1.0")

# Enable CORS for cross-device access on local network
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global pipeline instance (loaded once at startup)
pipeline: Optional[GestureEmotionPipeline] = None
reference_panels: Dict[str, np.ndarray] = {}

GESTURE_SEQUENCE = ["happy", "angry", "surprise", "gdg"]


@app.on_event("startup")
def startup_event():
    global pipeline, reference_panels
    print("Loading GestureEmotionPipeline...")
    pipeline = GestureEmotionPipeline(
        checkpoints_dir=CHECKPOINTS_DIR,
        features_dir=FEATURES_DIR,
        static_image_mode=True,  # True for independent request frames
    )
    print(f"Pipeline loaded on device: {pipeline.device}")

    print("Loading reference panels...")
    reference_panels = load_reference_panels(REFERENCE_PHOTOS_DIR, panel_size=480)
    print(f"Loaded {len(reference_panels)} reference panels.")


@app.on_event("shutdown")
def shutdown_event():
    global pipeline
    if pipeline:
        pipeline.close()


class PredictRequest(BaseModel):
    image: str  # Base64 data URL or raw base64 string
    threshold: Optional[float] = DEFAULT_GESTURE_THRESHOLD
    crop_aspect: Optional[bool] = True


class ComposeRequest(BaseModel):
    captures: Dict[str, str]  # Map of label -> base64 image
    panel_size: Optional[int] = 420  # unused now that the strip uses the fixed designed frame
    imgur_client_id: Optional[str] = None


class ImgurVerifyRequest(BaseModel):
    client_id: str


def decode_base64_image(data_str: str) -> np.ndarray:
    """Decodes a base64 image string (with or without data URI prefix) into a BGR numpy array."""
    if "," in data_str:
        data_str = data_str.split(",", 1)[1]
    image_bytes = base64.b64decode(data_str)
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Invalid image data.")
    return img


def encode_image_to_base64(img_bgr: np.ndarray, ext: str = ".jpg") -> str:
    """Encodes a BGR image to base64 JPEG data URI."""
    success, buffer = cv2.imencode(ext, img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    if not success:
        raise ValueError("Failed to encode image.")
    encoded = base64.b64encode(buffer).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"


@app.get("/api/health")
def health_check():
    return {
        "status": "ok",
        "device": str(pipeline.device) if pipeline else "unknown",
        "gesture_sequence": GESTURE_SEQUENCE,
        "has_env_imgur": bool(os.getenv("IMGUR_CLIENT_ID")),
    }


@app.get("/api/reference/{label}")
def get_reference_image(label: str):
    if label not in LABEL_IMAGE_MAP:
        raise HTTPException(status_code=404, detail=f"Label '{label}' not found.")
    filename = LABEL_IMAGE_MAP[label]
    filepath = os.path.join(REFERENCE_PHOTOS_DIR, filename)
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="File missing.")
    media_type = "image/jpeg" if filename.endswith((".jpg", ".jpeg")) else "image/png"
    return FileResponse(filepath, media_type=media_type)


@app.post("/api/predict")
def predict(req: PredictRequest):
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized.")

    try:
        frame = decode_base64_image(req.image)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Cannot decode image: {e}")

    if req.crop_aspect:
        frame = crop_to_aspect(frame)

    result = pipeline.predict(frame, gesture_threshold=req.threshold)
    label = result["label"]

    return {
        "label": label,
        "stage": result["stage"],
        "confidence": round(float(result["confidence"]), 4),
        "gesture_confidence": round(float(result.get("gesture_confidence", 0.0)), 4),
        "reference_url": f"/api/reference/{label}" if label in LABEL_IMAGE_MAP else None,
    }


def get_lan_ip() -> str:
    """Gets the machine's local LAN IP address for cross-device mobile access."""
    import socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "localhost"


def upload_to_imgur(image_path: str, client_id: str) -> Optional[str]:
    """Uploads strip image to Imgur using Client-ID header authentication."""
    try:
        import urllib.request
        boundary = "----WebKitFormBoundaryImgurUploadGDGHust"
        with open(image_path, "rb") as f:
            file_bytes = f.read()

        filename = os.path.basename(image_path)
        body = (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="image"; filename="{filename}"\r\n'
            f"Content-Type: image/jpeg\r\n\r\n"
        ).encode("utf-8") + file_bytes + f"\r\n--{boundary}--\r\n".encode("utf-8")

        req = urllib.request.Request(
            "https://api.imgur.com/3/image",
            data=body,
            headers={
                "Authorization": f"Client-ID {client_id.strip()}",
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko)",
                "Content-Type": f"multipart/form-data; boundary={boundary}",
            }
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            if data.get("success") and "data" in data and "link" in data["data"]:
                link = data["data"]["link"]
                print(f"[Imgur Upload Success] Link: {link}")
                return link
    except Exception as e:
        print(f"[Imgur Upload Notice]: {e}")
    return None


def upload_to_cloud(image_path: str, custom_imgur_id: Optional[str] = None) -> Tuple[Optional[str], str]:
    """
    Multi-tier image upload:
    Tier 1: Imgur API (if client_id is passed or in IMGUR_CLIENT_ID env)
    Tier 2: Anonymous temporary CDN (temp.sh)
    Tier 3: Local LAN network URL
    Returns (url, provider_name: 'imgur' | 'cdn' | 'local')
    """
    # 1. Imgur API
    cid = (custom_imgur_id or os.getenv("IMGUR_CLIENT_ID", "")).strip()
    if cid:
        imgur_link = upload_to_imgur(image_path, cid)
        if imgur_link:
            return imgur_link, "imgur"

    # 2. Anonymous CDN fallback (temp.sh). NOTE: unlike tmpfile.link, the URL
    # temp.sh returns is NOT a direct-download link on a plain GET -- it's an
    # HTML landing page with a "Click here to download" button (only a POST
    # to that same URL returns the raw file). Being tried anyway at the
    # user's explicit request, to compare upload latency firsthand before
    # deciding on a provider -- see the module's git history / conversation
    # for the measured tradeoffs (tmpfile.link: ~5-24s, temp.sh: ~12s in
    # testing, neither reliably fast on venue wifi).
    try:
        import urllib.request
        boundary = "----WebKitFormBoundaryPhotoboothUpload7MA4"
        with open(image_path, "rb") as f:
            file_bytes = f.read()

        filename = os.path.basename(image_path)
        body = (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
            f"Content-Type: image/jpeg\r\n\r\n"
        ).encode("utf-8") + file_bytes + f"\r\n--{boundary}--\r\n".encode("utf-8")

        req = urllib.request.Request(
            "https://temp.sh/upload",
            data=body,
            headers={
                "Content-Type": f"multipart/form-data; boundary={boundary}",
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }
        )
        # Same reasoning as before: a real photostrip took well over 8s in
        # testing, so give it real headroom rather than silently falling back.
        with urllib.request.urlopen(req, timeout=30) as resp:
            # temp.sh's response body is plain text containing just the URL
            # (not JSON, unlike tmpfile.link/tmpfiles.org).
            direct_url = resp.read().decode("utf-8").strip()
            if direct_url.startswith("http"):
                print(f"[Cloud Upload Success] Fallback CDN: {direct_url}")
                return direct_url, "cdn"
    except Exception as e:
        print(f"[CDN Upload Fallback Notice]: {e}")

    return None, "local"


@app.post("/api/imgur/verify")
def verify_imgur(req: ImgurVerifyRequest):
    """Checks if an Imgur Client ID is valid by attempting a test request."""
    cid = req.client_id.strip()
    if not cid:
        raise HTTPException(status_code=400, detail="Client ID cannot be empty.")

    # Try an upload with the sample image
    test_path = os.path.join(REFERENCE_PHOTOS_DIR, "happy.jpg")
    url = upload_to_imgur(test_path, cid)
    if url:
        return {"valid": True, "url": url, "message": "Imgur Client ID hoạt động xuất sắc!"}
    else:
        return {"valid": False, "message": "Không thể kết nối đến Imgur API với Client ID này. Vui lòng kiểm tra lại."}


@app.post("/api/challenge/compose")
def compose_strip(req: ComposeRequest):
    """Composes the GDG-designed Photobooth Strip (demo/assets/frame.png) from 4 challenge captures."""
    missing = [k for k in GESTURE_SEQUENCE if k not in req.captures]
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing captures for: {missing}")

    session_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = os.path.join(OUTPUT_DIR, f"session_{session_ts}")
    os.makedirs(session_dir, exist_ok=True)

    captures = {}
    for label in GESTURE_SEQUENCE:
        try:
            user_img = decode_base64_image(req.captures[label])
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Bad image for {label}: {e}")
        cv2.imwrite(os.path.join(session_dir, f"{label}.jpg"), user_img)  # save raw capture
        captures[label] = user_img

    references = load_references(REFERENCE_PHOTOS_DIR)
    final_strip = build_framed_strip(captures, references)

    strip_path = os.path.join(session_dir, "photobooth_strip.jpg")
    cv2.imwrite(strip_path, final_strip)

    strip_base64 = encode_image_to_base64(final_strip)

    # Cloud upload
    cloud_url, provider = upload_to_cloud(strip_path, req.imgur_client_id)
    lan_ip = get_lan_ip()
    local_url = f"http://{lan_ip}:8000/output/session_{session_ts}/photobooth_strip.jpg"
    final_download_url = cloud_url if cloud_url else local_url

    return {
        "session_id": f"session_{session_ts}",
        "strip_image": strip_base64,
        "saved_path": strip_path,
        "cloud_url": cloud_url,
        "provider": provider,
        "download_url": final_download_url,
    }


# Mount output directory for image downloads
app.mount("/output", StaticFiles(directory=OUTPUT_DIR), name="output")

# Mount web static directory at root
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(STATIC_DIR):
    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
