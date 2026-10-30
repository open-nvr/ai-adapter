# syntax=docker/dockerfile:1.7
# fast-plate-ocr license-plate adapter — contract-compliant per AI Adapter Contract v1.
#
# Build (from the ai-adapter repo root):
#   docker build -f adapters/fast_plate_ocr/Dockerfile -t opennvr/fast-plate-ocr-adapter:1.0.0 .
#
# Run (CPU only — fast-plate-ocr does not benefit from GPU):
#   docker run --rm -p 9004:9004 \
#     -e OPENNVR_ADAPTER_TOKEN=<token> \
#     opennvr/fast-plate-ocr-adapter:1.0.0
#
# The model weights (~5 MB) are downloaded by fast-plate-ocr on first
# load() and cached inside the container. To avoid the first-run
# download in air-gapped deployments, pre-warm and mount the cache:
#
#   docker run --rm -v $(pwd)/fp-cache:/root/.cache \
#     opennvr/fast-plate-ocr-adapter:1.0.0 \
#     python -c "from fast_plate_ocr import LicensePlateRecognizer; LicensePlateRecognizer('cct-s-v2-global-model')"
#
#   # subsequent runs:
#   docker run --rm -p 9004:9004 -v $(pwd)/fp-cache:/root/.cache \
#     -e OPENNVR_ADAPTER_TOKEN=<token> \
#     opennvr/fast-plate-ocr-adapter:1.0.0

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

# Versions track pyproject.toml exactly. NOTE: onnxruntime is an
# OPTIONAL extra of fast-plate-ocr 1.x ("[onnx]") — NOT a hard dep as
# an earlier comment here claimed. Installing without the extra
# produced an image whose model could never load: every /infer
# answered 503 while /health stayed green enough for Docker, so a
# whole deployment ran plate-blind with no build-time signal. The
# import check below makes that a build FAILURE instead.
#
# fast-plate-ocr is pinned to >=1.1.0 because 2.x doesn't exist on
# PyPI — latest published is 1.1.0 (verified via the PyPI JSON
# index at https://pypi.org/pypi/fast-plate-ocr/json). An earlier
# pin against >=2.0 broke the smoke job with "No matching
# distribution found" and won't unblock until upstream publishes a
# 2.x release. Matches the pyproject.toml [lpr] extra exactly.
RUN pip install --no-cache-dir \
        fastapi==0.127.0 \
        uvicorn==0.40.0 \
        pydantic==2.12.5 \
        httpx==0.28.1 \
        python-multipart==0.0.20 \
        "fast-plate-ocr[onnx]>=1.1.0" \
        "open-image-models[onnx]>=0.6.0"

# Build-time guard: importing LicensePlateRecognizer pulls onnxruntime
# (and cv2/numpy) — if any runtime dependency is missing, the BUILD
# fails here rather than the first plate read failing in production.
# Import only; the ~5 MB model still downloads on first load() so the
# image stays weights-free.
RUN python -c "from fast_plate_ocr import LicensePlateRecognizer; from open_image_models import create_detector; import onnxruntime, cv2, numpy"

COPY adapters/fast_plate_ocr/  ./adapters/fast_plate_ocr/

# Baked custom models (selected with OPENNVR_LPR_MODEL=<name>; the hub
# model stays the default). Fetched from this repo's releases at build
# time and checksum-verified, so the image is reproducible and the
# weights never go through git. See adapters/fast_plate_ocr/models/README.md.
ARG INDIAN_PLATES_RELEASE=indian-plates-v1
ARG INDIAN_PLATES_ONNX_SHA256=4c5cc1b784728312ddb01dac88cc81e8c330a1cd6e8de5f73df7c0713c5812bc
ARG INDIAN_PLATES_CONFIG_SHA256=236af234cfd64b85a3993f37504c6c2d3c0de066aaaa61d3e37993508ea8b243
RUN set -eux; d=./adapters/fast_plate_ocr/models/indian-plates-v1; mkdir -p "$d"; \
    base="https://github.com/open-nvr/ai-adapter/releases/download/${INDIAN_PLATES_RELEASE}"; \
    python -c "import urllib.request,sys; urllib.request.urlretrieve(sys.argv[1], sys.argv[2])" "$base/indian-plates-v1.onnx" "$d/model.onnx"; \
    python -c "import urllib.request,sys; urllib.request.urlretrieve(sys.argv[1], sys.argv[2])" "$base/plate_config.yaml" "$d/plate_config.yaml"; \
    echo "${INDIAN_PLATES_ONNX_SHA256}  $d/model.onnx" | sha256sum -c -; \
    echo "${INDIAN_PLATES_CONFIG_SHA256}  $d/plate_config.yaml" | sha256sum -c -

# The SDK lives in this same repo; copy the sdist-style source so the
# container is self-contained. (Once the SDK is published on PyPI we
# can replace this with `pip install opennvr-adapter-sdk` for parity
# with external adapter authors.)
COPY opennvr_adapter_sdk/  ./opennvr_adapter_sdk/
RUN pip install --no-cache-dir ./opennvr_adapter_sdk

# Flavour step (ACCEL). The accelerator wheels CONFLICT with plain onnxruntime
# (same import name), so swap rather than add; the Intel user-space comes from
# Intel's pinned GitHub releases because Debian carries none of it.
ARG ACCEL=cpu
COPY scripts/install-intel-runtime.sh /tmp/install-intel-runtime.sh
RUN case "${ACCEL}" in \
      cpu) ;; \
      intel) sh /tmp/install-intel-runtime.sh \
             && pip uninstall -y onnxruntime \
             && pip install --no-cache-dir onnxruntime-openvino==1.24.1 ;; \
      qualcomm) apt-get update && apt-get install -y --no-install-recommends libatomic1 \
             && rm -rf /var/lib/apt/lists/* \
             && pip install --no-cache-dir "onnxruntime>=1.24.2,<2" onnxruntime-qnn==2.7.0 ;; \
      *) echo "unknown ACCEL=${ACCEL} (cpu|intel|qualcomm)" >&2; exit 1 ;; \
    esac && rm -f /tmp/install-intel-runtime.sh
ENV OPENNVR_ACCEL_FLAVOUR=${ACCEL}

ENV PYTHONPATH=/app \
    OPENNVR_ADAPTER_TOKEN="" \
    OPENNVR_LPR_MODEL=cct-s-v2-global-model

EXPOSE 9004

HEALTHCHECK --interval=30s --timeout=3s --retries=3 \
    CMD python -c "import httpx, sys; sys.exit(0 if httpx.get('http://127.0.0.1:9004/health', timeout=2).status_code == 200 else 1)"

CMD ["uvicorn", "adapters.fast_plate_ocr.main:app", "--host", "0.0.0.0", "--port", "9004"]
