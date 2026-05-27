# syntax=docker/dockerfile:1.7
# __SLUG__ adapter — contract-compliant per AI Adapter Contract v1.
#
# Build:
#   docker build -f adapters/__DIR_NAME__/Dockerfile -t opennvr/__IMAGE_NAME__:1.0.0 .
#
# Run:
#   docker run --rm -p __PORT__:__PORT__ \
#     -e OPENNVR_ADAPTER_TOKEN=<token> \
#     opennvr/__IMAGE_NAME__:1.0.0

FROM python:3.11-slim AS runtime

# TODO: add any system packages your ML library needs at runtime.
# Common ones:
#   * libgl1 / libglib2.0-0 — for OpenCV (vision adapters)
#   * libgomp1 — for numpy / OpenMP-backed libraries
#   * libsndfile1 + ffmpeg — for audio decode (Whisper, audio adapters)
#   * ca-certificates — almost always needed for HTTPS model downloads
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# TODO: add your model's Python dependencies to this pip install.
# Pin exact versions for reproducibility (yolov8 / piper / whisper
# Dockerfiles are reference examples). Keep the contract layer
# versions in sync with the SDK's pyproject.toml.
RUN pip install --no-cache-dir \
        fastapi==0.127.0 \
        uvicorn==0.40.0 \
        pydantic==2.12.5 \
        httpx==0.28.1 \
        python-multipart==0.0.20

# Install the OpenNVR adapter SDK so the adapter package can import
# AdapterApp et al. Same shape as every other per-adapter Dockerfile.
COPY opennvr_adapter_sdk/                       ./opennvr_adapter_sdk/
RUN pip install --no-cache-dir ./opennvr_adapter_sdk

# Copy just what this adapter needs from the broader repo. Mirror
# yolov8/whisper/etc. — explicit COPY lines keep the image small and
# the dependency surface obvious.
COPY app/__init__.py                            ./app/__init__.py
COPY app/interfaces/                            ./app/interfaces/
COPY app/adapters/__init__.py                   ./app/adapters/__init__.py
COPY app/adapters/base.py                       ./app/adapters/base.py
COPY adapters/__DIR_NAME__/                     ./adapters/__DIR_NAME__/

# TODO: if your adapter needs to mount weights at runtime, add a
# WEIGHTS_DIR env var here (yolov8 / whisper use this pattern). If
# weights are baked into the image at build time, COPY them in
# above instead.

ENV PYTHONPATH=/app \
    OPENNVR_ADAPTER_TOKEN=""

EXPOSE __PORT__

HEALTHCHECK --interval=30s --timeout=3s --retries=3 \
    CMD python -c "import httpx, sys; sys.exit(0 if httpx.get('http://127.0.0.1:__PORT__/health', timeout=2).status_code == 200 else 1)"

CMD ["uvicorn", "adapters.__DIR_NAME__.main:app", "--host", "0.0.0.0", "--port", "__PORT__"]
