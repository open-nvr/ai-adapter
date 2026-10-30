# syntax=docker/dockerfile:1.7
# Package-detection adapter — contract-compliant per AI Adapter Contract v1.
#
# Build:
#   docker build -f adapters/package_detection/Dockerfile -t opennvr/package-detection-adapter:1.0.0 .
#
# Run (typical, CPU — this adapter's design target):
#   docker run --rm -p 9010:9010 \
#     -e OPENNVR_ADAPTER_TOKEN=<token> \
#     -v $(pwd)/model_weights:/weights:ro \
#     opennvr/package-detection-adapter:1.0.0
#
# Or let it fetch the published weights once on first boot:
#   docker run --rm -p 9010:9010 \
#     -e OPENNVR_ADAPTER_TOKEN=<token> \
#     -e PACKAGE_DETECTION_MODEL_URL=https://github.com/open-nvr/ai-adapter/releases/download/package-detection-v1.0.0/yolov8n-package.onnx \
#     -v opennvr_package_detection_weights:/weights \
#     opennvr/package-detection-adapter:1.0.0
#
# The weights are NOT baked in: /weights must contain
# yolov8n-package.onnx (the README's training recipe, or the release
# asset above), or PACKAGE_DETECTION_MODEL_URL must point at it.

#
# OpenNVR Accelerate flavours — one Dockerfile, ACCEL picks the runtime:
#   cpu   (default)  plain onnxruntime; amd64 + arm64
#   intel            onnxruntime-openvino + Intel OpenCL/NPU user-space
#   qualcomm         + onnxruntime-qnn, the Qualcomm AI Runtime PLUGIN provider
#                    (+ libatomic1: the HTP backend dlopens it and python:slim
#                    lacks it — "Unable to load backend" otherwise)
#                    (Hexagon NPU; next to plain onnxruntime, which it needs
#                    >= 1.24.2 — the plugin API); arm64 only
#                    (scripts/install-intel-runtime.sh); amd64 only
#   docker build --build-arg ACCEL=intel -f adapters/<dir>/Dockerfile .
# Select the device at run time with ADAPTER_ONNX_PROVIDERS
# (e.g. OpenVINOExecutionProvider:NPU); see opennvr_adapter_sdk/providers.py.

FROM python:3.11-slim AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Versions track pyproject.toml exactly — same pin set as
# adapters/yolov8/Dockerfile, since this adapter runs the same
# onnxruntime + OpenCV stack. No websockets: this adapter serves HTTP
# /infer only (supports_stream=False), on purpose — see service.py.
RUN pip install --no-cache-dir \
        fastapi==0.127.0 \
        uvicorn==0.40.0 \
        pydantic==2.12.5 \
        httpx==0.28.1 \
        python-multipart==0.0.20 \
        numpy==1.26.4 \
        opencv-python-headless==4.10.0.84 \
        onnxruntime==1.23.2

ARG ACCEL=cpu
COPY scripts/install-intel-runtime.sh /tmp/install-intel-runtime.sh
RUN case "${ACCEL}" in \
      cpu) ;; \
      intel) sh /tmp/install-intel-runtime.sh \
             && pip uninstall -y onnxruntime \
             && pip install --no-cache-dir onnxruntime-openvino==1.24.1 ;; \
      qualcomm) apt-get update && apt-get install -y --no-install-recommends libatomic1 \
             && rm -rf /var/lib/apt/lists/* \
             && pip install --no-cache-dir "onnxruntime>=1.24.2,<2" onnxruntime-qnn==2.7.0 "onnx>=1.16" ;; \
      *) echo "unknown ACCEL=${ACCEL} (cpu|intel|qualcomm)" >&2; exit 1 ;; \
    esac && rm -f /tmp/install-intel-runtime.sh
ENV OPENNVR_ACCEL_FLAVOUR=${ACCEL}

COPY opennvr_adapter_sdk/                       ./opennvr_adapter_sdk/
RUN pip install --no-cache-dir ./opennvr_adapter_sdk

# Flavour step (ACCEL). The accelerator wheels CONFLICT with plain onnxruntime
# (same import name), so swap rather than add; the Intel user-space comes from
# Intel's pinned GitHub releases because Debian carries none of it.

# Self-contained: the SDK and this one package, nothing from app/.
COPY adapters/package_detection/                ./adapters/package_detection/

ENV PYTHONPATH=/app \
    OPENNVR_ADAPTER_TOKEN="" \
    PACKAGE_DETECTION_WEIGHTS_DIR=/weights \
    PACKAGE_DETECTION_MODEL_URL="" \
    PACKAGE_DETECTION_LABELS=""

EXPOSE 9010

HEALTHCHECK --interval=30s --timeout=3s --retries=3 \
    CMD python -c "import httpx, sys; sys.exit(0 if httpx.get('http://127.0.0.1:9010/health', timeout=2).status_code == 200 else 1)"

CMD ["uvicorn", "adapters.package_detection.main:app", "--host", "0.0.0.0", "--port", "9010"]
