# syntax=docker/dockerfile:1.7
# SigLIP2 (base/16, 224) image + text embeddings on the Qualcomm Hexagon NPU
# — the `embed` task, contract-compliant per AI Adapter Contract v1.
# arm64, Qualcomm hosts only.
#
# Build (from the ai-adapter repo root):
#   docker build -f adapters/siglip_qnn/Dockerfile -t ghcr.io/open-nvr/siglip-qnn-adapter:latest .
#
# Run (the compose overlay docker-compose.accel-qualcomm.yml does this):
#   docker run --rm -p 9011:9011 --device /dev/fastrpc-cdsp \
#     -e OPENNVR_ADAPTER_TOKEN=<token> ghcr.io/open-nvr/siglip-qnn-adapter:latest
#
# The model is Qualcomm AI Hub's export of google/siglip2-base-patch16-224
# (Apache-2.0): two quantized ONNX graphs (w8a16) that ONNX Runtime's QNN
# provider compiles for the HTP at the first start; the compiled context is
# cached under ADAPTER_ACCEL_CACHE_DIR, so mount that path to keep it across
# container recreations. The 1.4 GB of graphs and the Gemma tokenizer are
# BAKED IN: no volume, no network, boots on a sealed box.

FROM python:3.11-slim AS runtime

# libatomic1: the QNN HTP backend dlopens it and python:slim lacks it.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        libatomic1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --no-cache-dir \
        fastapi==0.127.0 \
        uvicorn==0.40.0 \
        pydantic==2.12.5 \
        httpx==0.28.1 \
        python-multipart==0.0.20 \
        numpy==1.26.4 \
        Pillow==11.1.0 \
        "onnxruntime>=1.24.2,<2" \
        onnxruntime-qnn==2.7.0 \
        tokenizers==0.22.1 \
        huggingface_hub
ENV OPENNVR_ACCEL_FLAVOUR=qualcomm

# The assets: Qualcomm's public release bucket (no login), pinned by
# version; the tokenizer from the upstream model on HuggingFace.
ARG SIGLIP_QNN_VERSION=v0.66.0
ARG SIGLIP_QNN_ASSET_BASE=https://qaihub-public-assets.s3.us-west-2.amazonaws.com/qai-hub-models/models/siglip2/releases
ARG SIGLIP_HF_REPO=google/siglip2-base-patch16-224
ENV SIGLIP_QNN_ASSETS_DIR=/app/model_weights/siglip2-qnn \
    ADAPTER_ACCEL_CACHE_DIR=/app/model_weights/siglip2-qnn/.opennvr-accel
RUN set -eux; \
    mkdir -p /tmp/asset "${SIGLIP_QNN_ASSETS_DIR}"; \
    curl -fsSL -o /tmp/asset/a.zip "${SIGLIP_QNN_ASSET_BASE}/${SIGLIP_QNN_VERSION}/siglip2-onnx-w8a16.zip"; \
    python -c "import zipfile; zipfile.ZipFile('/tmp/asset/a.zip').extractall('/tmp/asset')"; \
    mv /tmp/asset/siglip2-onnx-w8a16/* "${SIGLIP_QNN_ASSETS_DIR}/"; \
    rm -rf /tmp/asset; \
    python -c "from huggingface_hub import hf_hub_download as d; import shutil, os; \
      shutil.copy(d('${SIGLIP_HF_REPO}', 'tokenizer.json'), os.path.join('${SIGLIP_QNN_ASSETS_DIR}', 'tokenizer.json'))"; \
    mkdir -p "${ADAPTER_ACCEL_CACHE_DIR}"; \
    ls -la "${SIGLIP_QNN_ASSETS_DIR}"

COPY opennvr_adapter_sdk/                       ./opennvr_adapter_sdk/
RUN pip install --no-cache-dir ./opennvr_adapter_sdk

COPY app/__init__.py                            ./app/__init__.py
COPY app/config/                                ./app/config/
COPY adapters/siglip_qnn/                       ./adapters/siglip_qnn/

# The image tower on the HTP; the text tower too unless
# SIGLIP_QNN_TEXT_PROVIDERS=CPUExecutionProvider keeps it on the CPU.
ENV ADAPTER_ONNX_PROVIDERS=QNNExecutionProvider:htp \
    OPENNVR_ADAPTER_PORT=9011 \
    PYTHONUNBUFFERED=1

EXPOSE 9011
# The first start compiles two graphs for the HTP: a long start period.
HEALTHCHECK --interval=30s --timeout=5s --start-period=600s --retries=5 \
    CMD curl -fsS http://localhost:9011/health || exit 1

CMD ["python", "-m", "uvicorn", "adapters.siglip_qnn.main:app", "--host", "0.0.0.0", "--port", "9011"]
