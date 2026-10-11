# ─── Production Dockerfile for Smart Vision Monitor API ───
# Base image: Official lightweight Python 3.13 Debian-slim
FROM python:3.13-slim AS base

# Prevent Python from writing .pyc files and buffer stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    YOLO_CONFIG_DIR=/tmp/yolo_config

WORKDIR /app

# Install minimal OS dependencies required by OpenCV headless
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch CPU build first to keep image lean (no CUDA blobs)
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Copy project definition and install dependencies
COPY pyproject.toml README.md /app/
RUN pip install --no-cache-dir ".[dev]"

# Pre-download YOLO26n weights so container starts instantly without external network calls
RUN python -c "from ultralytics import YOLO; YOLO('yolo26n.pt')"

# Copy application source code
COPY src/ /app/src/

# Install the application package
RUN pip install --no-deps -e .

# Security: Run as dedicated non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Healthcheck hitting the FastAPI liveness endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

CMD ["uvicorn", "vision_monitor.api:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
