FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_DEFAULT_TIMEOUT=120 \
    HF_HOME=/root/.cache/huggingface

# Install system libraries needed by OpenCV and MediaPipe
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install lightweight PyTorch CPU wheels
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple

# Install application dependencies
COPY requirements-docker.txt .
RUN pip install --no-cache-dir -r requirements-docker.txt

# Bake the pretrained ViT emotion model into the image at build time, so the
# container is genuinely self-contained -- no download, no internet dependency,
# no reliance on a host machine's ~/.cache/huggingface, at first run on any
# laptop. This step only depends on requirements-docker.txt (already installed
# above), so it's cached independently of source code changes below.
COPY scripts/download_models.py .
RUN python download_models.py

# Copy source code and model checkpoints
COPY . .

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
