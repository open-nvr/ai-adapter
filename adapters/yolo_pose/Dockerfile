# syntax=docker/dockerfile:1.7
# YOLO-pose human-keypoint adapter — contract-compliant per AI Adapter Contract v1.
#
# Build:
#   docker build -f adapters/yolo_pose/Dockerfile -t opennvr/yolo-pose-adapter:1.0.0 .
#
# Run (typical, CPU — this adapter's design target):
#   docker run --rm -p 9009:9009 \
#     -e OPENNVR_ADAPTER_TOKEN=<token> \
#     -v $(pwd)/model_weights:/weights:ro \
#     opennvr/yolo-pose-adapter:1.0.0
#
# Run with GPU passthrough (NVIDIA Container Toolkit required; only
# useful if the image was rebuilt against onnxruntime-gpu):
#   docker run --rm -p 9009:9009 --gpus all \
#     -e OPENNVR_ADAPTER_TOKEN=<token> \
#     -v $(pwd)/model_weights:/weights:ro \
#     opennvr/yolo-pose-adapter:1.0.0
#
# The weights are NOT baked in: /weights must contain
# yolo11n-pose.onnx (see the README's export recipe, or run
# `python download_models.py`). Set YOLO_POSE_MODEL_URL to have the
# adapter fetch it once on first boot instead.

FROM python:3.11-slim AS runtime

# System deps: ca-certificates for the optional first-boot weights
# fetch. No libGL needed because we install opencv-python-headless.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Versions track pyproject.toml exactly — same pin set as
# adapters/yolov8/Dockerfile, since this adapter runs the same
# onnxruntime + OpenCV stack.
#
# websockets is NOT optional here even though no code imports it:
# bare uvicorn ships no WebSocket implementation, and without one it
# answers the /infer/stream upgrade with a bare 404 plus a log warning
# ("No supported WebSocket library detected") while /health stays
# green the whole time. An image smoke test that curls /health
# therefore never catches it, and the adapter simply has no streaming
# -- which for this adapter is the primary path. (The SDK's own
# `serve` extra pulls uvicorn[standard] for exactly this reason.)
RUN pip install --no-cache-dir \
        fastapi==0.127.0 \
        uvicorn==0.40.0 \
        pydantic==2.12.5 \
        httpx==0.28.1 \
        python-multipart==0.0.20 \
        websockets==17.1 \
        numpy==1.26.4 \
        opencv-python-headless==4.10.0.84 \
        onnxruntime==1.23.2

# Install the OpenNVR adapter SDK so ``adapters.yolo_pose.main`` can
# import ``AdapterApp``, ``BodyShape``, etc. Sister Dockerfiles
# (adapters/yolov8/Dockerfile, adapters/insightface/Dockerfile) do the
# same. Without this the container crashes at uvicorn boot with
# ModuleNotFoundError: opennvr_adapter_sdk.
COPY opennvr_adapter_sdk/                       ./opennvr_adapter_sdk/
RUN pip install --no-cache-dir ./opennvr_adapter_sdk

# Unlike adapters/yolov8/, nothing from the legacy ``app/`` package is
# copied: this adapter owns its ONNX session and post-processing, so
# the image contains the SDK and this one package.
COPY adapters/yolo_pose/                        ./adapters/yolo_pose/

ENV PYTHONPATH=/app \
    OPENNVR_ADAPTER_TOKEN="" \
    YOLO_POSE_WEIGHTS_DIR=/weights \
    YOLO_POSE_MODEL_URL=""

EXPOSE 9009

HEALTHCHECK --interval=30s --timeout=3s --retries=3 \
    CMD python -c "import httpx, sys; sys.exit(0 if httpx.get('http://127.0.0.1:9009/health', timeout=2).status_code == 200 else 1)"

CMD ["uvicorn", "adapters.yolo_pose.main:app", "--host", "0.0.0.0", "--port", "9009"]
