/**
 * GDG-HUST - Photobooth Studio
 * Professional interactive client logic with Web Audio shutter sounds,
 * Imgur Cloud upload, tactile hold mechanics, and QR distribution.
 */

const GESTURE_SEQUENCE = ["happy", "angry", "surprise", "gdg"];
const PREDICT_INTERVAL_MS = 180; // ~5.5 FPS

// Persistent Settings State
const SETTINGS = {
  imgurClientId: localStorage.getItem("gdg_imgur_client_id") || "",
  holdDurationMs: parseInt(localStorage.getItem("gdg_hold_duration") || "1000", 10),
  soundEnabled: localStorage.getItem("gdg_sound_enabled") !== "false",
};

let currentMode = "challenge"; // 'challenge' | 'mirror'
let currentStepIdx = 0;
let holdElapsed = 0;
let lastTimestamp = performance.now();
let captures = {};
let isPredicting = false;
let stream = null;
let backendOnline = null; // null = not checked yet
let hasLastResult = false; // true once a strip has been composed at least once this session

// DOM Elements — Viewports & Feeds
const videoFeed = document.getElementById("videoFeed");
const mirrorVideoFeed = document.getElementById("mirrorVideoFeed");
const captureCanvas = document.getElementById("captureCanvas");
const flashOverlay = document.getElementById("flashOverlay");

const btnChallengeMode = document.getElementById("btnChallengeMode");
const btnMirrorMode = document.getElementById("btnMirrorMode");
const challengeSection = document.getElementById("challengeSection");
const mirrorSection = document.getElementById("mirrorSection");
const cameraSelect = document.getElementById("cameraSelect");

// DOM Elements — Target & Progress HUD
const targetLabelText = document.getElementById("targetLabelText");
const targetThumbImg = document.getElementById("targetThumbImg");
const holdProgressBar = document.getElementById("holdProgressBar");
const holdHint = document.getElementById("holdHint");
const stepProgressItems = document.querySelectorAll(".step-progress-item");

const hudLiveLabel = document.getElementById("hudLiveLabel");
const hudConfidence = document.getElementById("hudConfidence");

const mirrorRefImg = document.getElementById("mirrorRefImg");
const mirrorCaption = document.getElementById("mirrorCaption");

// DOM Elements — Backend connection status + persistent header controls
const backendStatusDot = document.getElementById("backendStatusDot");
const backendStatusText = document.getElementById("backendStatusText");
const backendStatusSpec = document.getElementById("backendStatusSpec");
const btnHeaderRestart = document.getElementById("btnHeaderRestart");
const btnViewLastResult = document.getElementById("btnViewLastResult");

// DOM Elements — Result Dialog
const resultModal = document.getElementById("resultModal");
const stripResultImg = document.getElementById("stripResultImg");
const btnDownloadStrip = document.getElementById("btnDownloadStrip");
const btnCloseModal = document.getElementById("btnCloseModal");
const btnRestartChallenge = document.getElementById("btnRestartChallenge");
const qrcodeContainer = document.getElementById("qrcodeContainer");
const cloudDirectLink = document.getElementById("cloudDirectLink");
const cloudDirectUrlInput = document.getElementById("cloudDirectUrlInput");
const btnCopyLink = document.getElementById("btnCopyLink");
const cloudBadgeText = document.getElementById("cloudBadgeText");

// DOM Elements — Studio Settings Dialog
const btnOpenSettings = document.getElementById("btnOpenSettings");
const settingsModal = document.getElementById("settingsModal");
const btnCloseSettingsModal = document.getElementById("btnCloseSettingsModal");
const settingImgurClientId = document.getElementById("settingImgurClientId");
const btnVerifyImgur = document.getElementById("btnVerifyImgur");
const imgurVerifyStatus = document.getElementById("imgurVerifyStatus");
const settingHoldDuration = document.getElementById("settingHoldDuration");
const settingSoundEnabled = document.getElementById("settingSoundEnabled");
const btnSaveSettings = document.getElementById("btnSaveSettings");

// Audio Engine (Acoustic Web Audio Dual-stage Camera Shutter)
let audioCtx = null;
function getAudioContext() {
  if (!audioCtx) {
    audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  }
  if (audioCtx.state === "suspended") {
    audioCtx.resume();
  }
  return audioCtx;
}

function playShutterSound() {
  if (!SETTINGS.soundEnabled) return;
  try {
    const ctx = getAudioContext();
    const now = ctx.currentTime;

    // First mechanical click: blade release
    const osc1 = ctx.createOscillator();
    const gain1 = ctx.createGain();
    osc1.type = "sine";
    osc1.frequency.setValueAtTime(1400, now);
    osc1.frequency.exponentialRampToValueAtTime(180, now + 0.035);
    gain1.gain.setValueAtTime(0.35, now);
    gain1.gain.exponentialRampToValueAtTime(0.01, now + 0.035);
    osc1.connect(gain1);
    gain1.connect(ctx.destination);
    osc1.start(now);
    osc1.stop(now + 0.04);

    // Second mechanical click: curtain close
    const osc2 = ctx.createOscillator();
    const gain2 = ctx.createGain();
    osc2.type = "triangle";
    osc2.frequency.setValueAtTime(800, now + 0.05);
    osc2.frequency.exponentialRampToValueAtTime(90, now + 0.11);
    gain2.gain.setValueAtTime(0.4, now + 0.05);
    gain2.gain.exponentialRampToValueAtTime(0.01, now + 0.11);
    osc2.connect(gain2);
    gain2.connect(ctx.destination);
    osc2.start(now + 0.05);
    osc2.stop(now + 0.12);
  } catch (e) {
    console.warn("Audio error:", e);
  }
}

function playSuccessFanfare() {
  if (!SETTINGS.soundEnabled) return;
  try {
    const ctx = getAudioContext();
    const notes = [523.25, 659.25, 783.99, 1046.50]; // C5, E5, G5, C6
    notes.forEach((freq, idx) => {
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();
      const t = ctx.currentTime + idx * 0.1;
      osc.type = "sine";
      osc.frequency.setValueAtTime(freq, t);
      gain.gain.setValueAtTime(0.2, t);
      gain.gain.exponentialRampToValueAtTime(0.001, t + 0.35);
      osc.connect(gain);
      gain.connect(ctx.destination);
      osc.start(t);
      osc.stop(t + 0.36);
    });
  } catch (e) {
    console.warn("Fanfare error:", e);
  }
}

function launchCelebration() {
  if (typeof confetti === "function") {
    confetti({
      particleCount: 80,
      spread: 65,
      origin: { y: 0.65 },
      colors: ["#4285F4", "#EA4335", "#FBBC04", "#34A853"]
    });
  }
}

function showToast(message) {
  const toast = document.getElementById("toast");
  if (!toast) return;
  toast.textContent = message;
  toast.classList.add("show");
  setTimeout(() => toast.classList.remove("show"), 2400);
}

// Backend Connection Status
// `online`: true only when we got ANY HTTP response (even a 4xx/5xx still proves the
// server is reachable) -- false only when the request itself failed to complete
// (fetch threw: connection refused, timeout, DNS/CORS failure, etc).
function setBackendStatus(online, deviceLabel) {
  const changed = online !== backendOnline;
  backendOnline = online;

  if (online) {
    backendStatusDot.style.backgroundColor = "var(--google-green)";
    backendStatusDot.style.boxShadow = "0 0 8px rgba(52, 168, 83, 0.7)";
    backendStatusText.textContent = "BACKEND ONLINE";
    if (deviceLabel) backendStatusSpec.textContent = deviceLabel.toUpperCase();
  } else {
    backendStatusDot.style.backgroundColor = "var(--google-red)";
    backendStatusDot.style.boxShadow = "0 0 8px rgba(234, 67, 53, 0.7)";
    backendStatusText.textContent = "BACKEND OFFLINE";
    backendStatusSpec.textContent = "RETRYING...";
    if (changed) showToast("⚠️ Mất kết nối với máy chủ. Đang thử kết nối lại...");
  }
}

async function checkBackendHealth() {
  try {
    const resp = await fetch("/api/health", { cache: "no-store" });
    if (resp.ok) {
      const data = await resp.json();
      setBackendStatus(true, data.device);
    } else {
      setBackendStatus(true); // reachable, just an unexpected status -- not "offline"
    }
  } catch (err) {
    setBackendStatus(false);
  }
}

// Camera Management
async function initCameras() {
  try {
    const devices = await navigator.mediaDevices.enumerateDevices();
    const videoDevices = devices.filter(d => d.kind === "videoinput");
    cameraSelect.innerHTML = "";

    videoDevices.forEach((dev, idx) => {
      const opt = document.createElement("option");
      opt.value = dev.deviceId;
      opt.textContent = dev.label || `Camera ${idx + 1}`;
      cameraSelect.appendChild(opt);
    });

    const chosenId = videoDevices.length > 0 ? videoDevices[0].deviceId : undefined;
    await startCameraStream(chosenId);
  } catch (err) {
    console.error("Camera access error:", err);
    hudLiveLabel.textContent = "Lỗi Camera";
  }
}

async function startCameraStream(deviceId) {
  if (stream) {
    stream.getTracks().forEach(t => t.stop());
  }

  const constraints = {
    video: {
      deviceId: deviceId ? { exact: deviceId } : undefined,
      width: { ideal: 1280 },
      height: { ideal: 960 },
      facingMode: "user"
    },
    audio: false
  };

  try {
    stream = await navigator.mediaDevices.getUserMedia(constraints);
    videoFeed.srcObject = stream;
    mirrorVideoFeed.srcObject = stream;
  } catch (e) {
    console.error("getUserMedia error:", e);
  }
}

cameraSelect.addEventListener("change", (e) => {
  startCameraStream(e.target.value);
});

// Mode Switcher
btnChallengeMode.addEventListener("click", () => {
  currentMode = "challenge";
  btnChallengeMode.classList.add("active");
  btnMirrorMode.classList.remove("active");
  challengeSection.classList.add("active");
  mirrorSection.classList.remove("active");
});

btnMirrorMode.addEventListener("click", () => {
  currentMode = "mirror";
  btnMirrorMode.classList.add("active");
  btnChallengeMode.classList.remove("active");
  mirrorSection.classList.add("active");
  challengeSection.classList.remove("active");
});

// UI Target Display
const TARGET_TITLES = {
  "happy": "HAPPY (CƯỜI)",
  "angry": "ANGRY (TỨC GIẬN)",
  "surprise": "SURPRISE (BẤT NGỜ)",
  "gdg": "GDG SIGN (< >)"
};

function updateTargetUI() {
  if (currentStepIdx >= GESTURE_SEQUENCE.length) return;
  const target = GESTURE_SEQUENCE[currentStepIdx];

  targetLabelText.textContent = TARGET_TITLES[target] || target.toUpperCase();
  targetThumbImg.src = `/api/reference/${target}`;

  holdProgressBar.style.width = "0%";
  holdHint.textContent = `Giữ tư thế ${(SETTINGS.holdDurationMs / 1000).toFixed(1)}s`;

  stepProgressItems.forEach((step, idx) => {
    step.classList.toggle("active", idx === currentStepIdx);
    step.classList.toggle("completed", idx < currentStepIdx);
  });
}

function triggerFlash() {
  playShutterSound();
  flashOverlay.classList.remove("flashing");
  void flashOverlay.offsetWidth;
  flashOverlay.classList.add("flashing");
  setTimeout(() => {
    flashOverlay.classList.remove("flashing");
  }, 350);
}

// Frame Grabber
function grabCurrentFrame(flipHorizontal = true) {
  const v = (currentMode === "challenge") ? videoFeed : mirrorVideoFeed;
  if (!v.videoWidth || !v.videoHeight) return null;

  captureCanvas.width = v.videoWidth;
  captureCanvas.height = v.videoHeight;
  const ctx = captureCanvas.getContext("2d");

  if (flipHorizontal) {
    ctx.translate(captureCanvas.width, 0);
    ctx.scale(-1, 1);
  }
  ctx.drawImage(v, 0, 0, captureCanvas.width, captureCanvas.height);
  return captureCanvas.toDataURL("image/jpeg", 0.92);
}

// Prediction Loop
async function predictionLoop() {
  const now = performance.now();
  const dt = now - lastTimestamp;
  lastTimestamp = now;

  if (!isPredicting && (videoFeed.videoWidth || mirrorVideoFeed.videoWidth)) {
    const frameData = grabCurrentFrame(true);

    if (frameData) {
      isPredicting = true;
      try {
        const resp = await fetch("/api/predict", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ image: frameData, threshold: 0.85, crop_aspect: true })
        });
        setBackendStatus(true); // got a response at all -- server is reachable

        if (resp.ok) {
          const res = await resp.json();
          handlePredictionResult(res, dt, frameData);
        }
      } catch (err) {
        console.error("Predict fetch error:", err);
        setBackendStatus(false);
      } finally {
        isPredicting = false;
      }
    }
  }

  setTimeout(predictionLoop, PREDICT_INTERVAL_MS);
}

function handlePredictionResult(res, dt, currentFrameData) {
  const liveLabel = res.label;
  const confidence = (res.confidence * 100).toFixed(0);

  // Update HUD Telemetry
  hudLiveLabel.textContent = liveLabel === "gdg" ? "GDG (< >)" : liveLabel.toUpperCase();
  hudConfidence.textContent = `${confidence}%`;

  // Mirror mode updates
  if (currentMode === "mirror") {
    mirrorCaption.textContent = `${liveLabel.toUpperCase()} (${confidence}%)`;
    if (res.reference_url) {
      mirrorRefImg.src = res.reference_url;
    }
    return;
  }

  // Challenge mode logic
  if (currentStepIdx >= GESTURE_SEQUENCE.length) return;

  const currentTarget = GESTURE_SEQUENCE[currentStepIdx];
  const targetDuration = SETTINGS.holdDurationMs;

  if (liveLabel === currentTarget) {
    holdElapsed += dt;
    const pct = Math.min(100, (holdElapsed / targetDuration) * 100);
    holdProgressBar.style.width = `${pct}%`;
    holdProgressBar.style.backgroundColor = "var(--google-green)";
    const remaining = Math.max(0, ((targetDuration - holdElapsed) / 1000)).toFixed(1);
    holdHint.textContent = `Giữ thêm ${remaining}s...`;

    if (holdElapsed >= targetDuration) {
      completeStep(currentFrameData);
    }
  } else {
    holdElapsed = 0;
    holdProgressBar.style.width = "0%";
    holdProgressBar.style.backgroundColor = "var(--google-blue)";
    holdHint.textContent = `Tạo biểu cảm: ${currentTarget.toUpperCase()}`;
  }
}

// Marks the current challenge step as captured with `frameData` and advances
// to the next one (or finishes the challenge). Shared by the normal
// hold-for-N-seconds path above and the hidden manual-capture shortcut below,
// so both end up going through the exact same completion logic.
function completeStep(frameData) {
  const currentTarget = GESTURE_SEQUENCE[currentStepIdx];
  triggerFlash();
  captures[currentTarget] = frameData;
  holdElapsed = 0;
  currentStepIdx++;

  if (currentStepIdx < GESTURE_SEQUENCE.length) {
    updateTargetUI();
  } else {
    onChallengeComplete();
  }
}

// When challenge finishes: compose strip & generate QR
async function onChallengeComplete() {
  hudLiveLabel.textContent = "COMPOSE...";
  try {
    const payload = {
      captures: captures,
      panel_size: 420,
      imgur_client_id: SETTINGS.imgurClientId || undefined,
    };

    const resp = await fetch("/api/challenge/compose", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });
    setBackendStatus(true); // got a response at all -- server is reachable

    if (resp.ok) {
      const data = await resp.json();
      stripResultImg.src = data.strip_image;
      btnDownloadStrip.href = data.strip_image;
      btnDownloadStrip.download = `gdg_photobooth_${data.session_id}.jpg`;

      const downloadUrl = data.download_url || data.cloud_url || window.location.href;
      
      // Update link inputs
      cloudDirectUrlInput.value = downloadUrl;
      cloudDirectLink.href = downloadUrl;

      // Update provider tag
      if (data.provider === "imgur") {
        cloudBadgeText.textContent = "Imgur Cloud";
        cloudBadgeText.style.color = "#1bb666";
        cloudBadgeText.style.background = "rgba(27, 182, 102, 0.12)";
      } else if (data.provider === "cdn") {
        cloudBadgeText.textContent = "Cloud CDN";
        cloudBadgeText.style.color = "var(--google-blue)";
        cloudBadgeText.style.background = "rgba(66, 133, 244, 0.12)";
      } else {
        cloudBadgeText.textContent = "LAN Server";
        cloudBadgeText.style.color = "var(--text-secondary)";
        cloudBadgeText.style.background = "rgba(255, 255, 255, 0.08)";
      }

      // Generate Crisp QR Code
      qrcodeContainer.innerHTML = "";
      new QRCode(qrcodeContainer, {
        text: downloadUrl,
        width: 150,
        height: 150,
        colorDark: "#0c0d12",
        colorLight: "#ffffff",
        correctLevel: QRCode.CorrectLevel.M
      });

      hasLastResult = true;
      btnViewLastResult.hidden = false;

      resultModal.classList.add("show");
      playSuccessFanfare();
      launchCelebration();
    } else {
      alert("Không thể tạo strip ảnh. Vui lòng thử lại!");
    }
  } catch (err) {
    console.error("Compose error:", err);
    setBackendStatus(false);
    alert("Mất kết nối với máy chủ khi tạo ảnh. Vui lòng kiểm tra kết nối mạng và bấm nút \"Bắt đầu lại\" ở góc trên khi đã sẵn sàng thử lại.");
  }
}

function restartChallenge() {
  currentStepIdx = 0;
  holdElapsed = 0;
  captures = {};
  resultModal.classList.remove("show");
  updateTargetUI();
}

btnCloseModal.addEventListener("click", () => resultModal.classList.remove("show"));
btnRestartChallenge.addEventListener("click", restartChallenge);

// Persistent header controls -- reachable regardless of modal/challenge state, so
// closing the result modal (or getting stuck after the challenge completes) never
// requires an F5 to recover from.
btnHeaderRestart.addEventListener("click", restartChallenge);
btnViewLastResult.addEventListener("click", () => {
  if (hasLastResult) resultModal.classList.add("show");
});

// Hidden manual-capture override -- intentionally not surfaced anywhere in the
// UI (no button, no hint, no settings entry). Some target expressions (e.g.
// "angry") are hard to get the model to recognize confidently on demand;
// pressing 'c' captures the current frame for whatever step is active right
// now, skipping the hold-for-N-seconds requirement entirely. Goes through the
// exact same completeStep() as a normal capture, so the rest of the flow
// (flash, sound, strip composition) behaves identically either way.
document.addEventListener("keydown", (e) => {
  if (e.repeat) return; // ignore OS key-repeat -- each press should complete at most one step
  if (e.key.toLowerCase() !== "c") return;

  // Diagnostic logging (console only, not UI) -- if 'c' does nothing visible,
  // open DevTools console and press it again: this prints exactly which
  // guard blocked it, if any.
  if (currentMode !== "challenge") { console.log("[c-capture] blocked: not in challenge mode"); return; }
  if (currentStepIdx >= GESTURE_SEQUENCE.length) { console.log("[c-capture] blocked: challenge already complete"); return; }
  if (settingsModal.classList.contains("show") || resultModal.classList.contains("show")) {
    console.log("[c-capture] blocked: a modal is open"); return;
  }
  const activeTag = (document.activeElement && document.activeElement.tagName) || "";
  if (["INPUT", "TEXTAREA", "SELECT"].includes(activeTag)) {
    console.log("[c-capture] blocked: focus is in a", activeTag); return;
  }

  const frameData = grabCurrentFrame(true);
  if (!frameData) { console.log("[c-capture] blocked: grabCurrentFrame() returned nothing (video not ready?)"); return; }
  console.log("[c-capture] firing for step", GESTURE_SEQUENCE[currentStepIdx]);
  completeStep(frameData);
});

// Copy link action
btnCopyLink.addEventListener("click", async () => {
  const url = cloudDirectUrlInput.value;
  if (!url) return;
  try {
    await navigator.clipboard.writeText(url);
    showToast("Đã sao chép link ảnh! 📋");
  } catch (e) {
    cloudDirectUrlInput.select();
    document.execCommand("copy");
    showToast("Đã sao chép link ảnh! 📋");
  }
});

// Settings Modal Management
btnOpenSettings.addEventListener("click", () => {
  settingImgurClientId.value = SETTINGS.imgurClientId;
  settingHoldDuration.value = SETTINGS.holdDurationMs.toString();
  settingSoundEnabled.checked = SETTINGS.soundEnabled;
  imgurVerifyStatus.innerHTML = "";

  settingsModal.classList.add("show");
});

btnCloseSettingsModal.addEventListener("click", () => {
  settingsModal.classList.remove("show");
});

btnVerifyImgur.addEventListener("click", async () => {
  const cid = settingImgurClientId.value.trim();
  if (!cid) {
    imgurVerifyStatus.className = "form-status error";
    imgurVerifyStatus.innerHTML = '<span class="material-symbols-rounded">error</span><span>Vui lòng nhập Client ID trước.</span>';
    return;
  }

  imgurVerifyStatus.className = "form-status";
  imgurVerifyStatus.innerHTML = '<span class="material-symbols-rounded">sync</span><span>Đang kiểm tra với Imgur API...</span>';

  try {
    const resp = await fetch("/api/imgur/verify", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ client_id: cid })
    });
    const res = await resp.json();
    if (res.valid) {
      imgurVerifyStatus.className = "form-status success";
      imgurVerifyStatus.innerHTML = `<span class="material-symbols-rounded">check_circle</span><span>${res.message}</span>`;
    } else {
      imgurVerifyStatus.className = "form-status error";
      imgurVerifyStatus.innerHTML = `<span class="material-symbols-rounded">cancel</span><span>${res.message}</span>`;
    }
  } catch (e) {
    imgurVerifyStatus.className = "form-status error";
    imgurVerifyStatus.innerHTML = '<span class="material-symbols-rounded">cancel</span><span>Lỗi kết nối máy chủ khi kiểm tra.</span>';
  }
});

btnSaveSettings.addEventListener("click", () => {
  SETTINGS.imgurClientId = settingImgurClientId.value.trim();
  SETTINGS.holdDurationMs = parseInt(settingHoldDuration.value, 10);
  SETTINGS.soundEnabled = settingSoundEnabled.checked;

  localStorage.setItem("gdg_imgur_client_id", SETTINGS.imgurClientId);
  localStorage.setItem("gdg_hold_duration", SETTINGS.holdDurationMs.toString());
  localStorage.setItem("gdg_sound_enabled", SETTINGS.soundEnabled ? "true" : "false");

  settingsModal.classList.remove("show");
  showToast("Cài đặt đã được lưu thành công.");
  updateTargetUI();
});

// Close modals when clicking on backdrop
document.querySelectorAll(".dialog-backdrop").forEach(bd => {
  bd.addEventListener("click", () => {
    resultModal.classList.remove("show");
    settingsModal.classList.remove("show");
  });
});

// Start
window.addEventListener("DOMContentLoaded", () => {
  updateTargetUI();
  initCameras();
  setTimeout(predictionLoop, 400);

  // Backend connection status: check immediately, then keep polling as a heartbeat
  // independent of the predict loop (so it still reports correctly even if the
  // user is idle, mid-settings, or on the result modal -- not actively predicting).
  checkBackendHealth();
  setInterval(checkBackendHealth, 5000);
});
