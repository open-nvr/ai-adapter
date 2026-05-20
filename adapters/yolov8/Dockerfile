# syntax=docker/dockerfile:1.7
# YOLOv8 object-detection adapter — contract-compliant per AI Adapter Contract v1.
#
# Build:
#   docker build -f adapters/yolov8/Dockerfile -t opennvr/yolov8-adapter:1.0.0 .
#
# Run (typical, CPU):
#   docker run --rm -p 9002:9002 \
#     -e OPENNVR_ADAPTER_TOKEN=<token> \
#     -v $(pwd)/model_weights:/weights:ro \
#     opennvr/yolov8-adapter:1.0.0
#
# Run with GPU passthrough (NVIDIA Container Toolkit required):
#   docker run --rm -p 9002:9002 --gpus all \
#     -e OPENNVR_ADAPTER_TOKEN=<token> \
#     -v $(pwd)/model_weights:/weights:ro \
#     opennvr/yolov8-adapter:1.0.0

FROM python:3.11-slim AS runtime

# System deps: libGL for OpenCV at runtime; build-essential not needed because
# we install opencv-python-headless (no GL deps).
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Versions track pyproject.toml exactly.
RUN pip install --no-cache-dir \
        fastapi==0.127.0 \
        uvicorn==0.40.0 \
        pydantic==2.12.5 \
        httpx==0.28.1 \
        python-multipart==0.0.20 \
        numpy==1.26.4 \
        opencv-python-headless==4.10.0.84 \
        onnxruntime==1.23.2

COPY app/__init__.py                            ./app/__init__.py
COPY app/interfaces/                            ./app/interfaces/
COPY app/adapters/__init__.py                   ./app/adapters/__init__.py
COPY app/adapters/base.py                       ./app/adapters/base.py
COPY app/adapters/vision/__init__.py            ./app/adapters/vision/__init__.py
COPY app/adapters/vision/yolov8_adapter.py      ./app/adapters/vision/yolov8_adapter.py
COPY app/config/                                ./app/config/
COPY app/utils/__init__.py                      ./app/utils/__init__.py
COPY app/utils/image_utils.py                   ./app/utils/image_utils.py
COPY adapters/yolov8/                           ./adapters/yolov8/

ENV PYTHONPATH=/app \
    OPENNVR_ADAPTER_TOKEN="" \
    YOLOV8_WEIGHTS_DIR=/weights

EXPOSE 9002

HEALTHCHECK --interval=30s --timeout=3s --retries=3 \
    CMD python -c "import httpx, sys; sys.exit(0 if httpx.get('http://127.0.0.1:9002/health', timeout=2).status_code == 200 else 1)"

CMD ["uvicorn", "adapters.yolov8.main:app", "--host", "0.0.0.0", "--port", "9002"]
