# syntax=docker/dockerfile:1.7
# ByteTrack multi-object-tracking adapter — contract-compliant per AI Adapter Contract v1.
#
# Build:
#   docker build -f adapters/bytetrack/Dockerfile -t opennvr/bytetrack-adapter:1.0.0 .
#
# Run (CPU-only, ByteTrack doesn't use GPU):
#   docker run --rm -p 9007:9007 \
#     -e OPENNVR_ADAPTER_TOKEN=<token> \
#     opennvr/bytetrack-adapter:1.0.0
#
# This adapter doesn't take pixels — it takes the JSON detections an
# upstream detector (YOLOv8, fast-plate-ocr, etc.) produced and adds
# persistent track_ids. No model weights to mount.

FROM python:3.11-slim AS runtime

# supervision pulls opencv (for some annotators, not for tracking
# itself) which needs libGL at runtime. opencv-python-headless avoids
# the X11 deps but still needs libgomp1 for its numpy-OpenMP layers.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# supervision >=0.21 ships ByteTrack with the keyword-argument
# constructor service.py uses (track_activation_threshold,
# lost_track_buffer, etc.). Upper bound is <0.30 because supervision
# 0.28 deprecated ByteTrack and 0.30 removes it entirely — when
# upstream lands the replacement tracker (the deprecation notice
# hints at moving tracking to a separate package), port service.py
# and bump the pin in lock-step. Until then we stay on the API that
# actually has ByteTrack.
RUN pip install --no-cache-dir \
        fastapi==0.127.0 \
        uvicorn==0.40.0 \
        pydantic==2.12.5 \
        httpx==0.28.1 \
        python-multipart==0.0.20 \
        numpy==1.26.4 \
        "supervision>=0.21,<0.30"

# Install the OpenNVR adapter SDK so ``adapters.bytetrack.main`` can
# import AdapterApp et al. Mirrors yolov8 / piper / whisper.
COPY opennvr_adapter_sdk/                       ./opennvr_adapter_sdk/
RUN pip install --no-cache-dir ./opennvr_adapter_sdk

COPY app/__init__.py                            ./app/__init__.py
COPY app/interfaces/                            ./app/interfaces/
COPY app/adapters/__init__.py                   ./app/adapters/__init__.py
COPY app/adapters/base.py                       ./app/adapters/base.py
COPY adapters/bytetrack/                        ./adapters/bytetrack/

ENV PYTHONPATH=/app \
    OPENNVR_ADAPTER_TOKEN="" \
    BYTETRACK_IDLE_TTL_SECONDS=300

EXPOSE 9007

HEALTHCHECK --interval=30s --timeout=3s --retries=3 \
    CMD python -c "import httpx, sys; sys.exit(0 if httpx.get('http://127.0.0.1:9007/health', timeout=2).status_code == 200 else 1)"

CMD ["uvicorn", "adapters.bytetrack.main:app", "--host", "0.0.0.0", "--port", "9007"]
