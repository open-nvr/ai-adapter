# syntax=docker/dockerfile:1.7
# Whisper-Small (w8a16) on the Qualcomm Hexagon NPU — ASR adapter,
# contract-compliant per AI Adapter Contract v1. arm64, Qualcomm hosts only.
#
# Build (from the ai-adapter repo root):
#   docker build -f adapters/whisper_qnn/Dockerfile -t ghcr.io/open-nvr/whisper-qnn-adapter:latest .
#
# Run (the compose overlay docker-compose.accel-qualcomm.yml does this):
#   docker run --rm -p 9020:9020 --device /dev/fastrpc-cdsp \
#     -e OPENNVR_ADAPTER_TOKEN=<token> ghcr.io/open-nvr/whisper-qnn-adapter:latest
#
# The model is Qualcomm's AI Hub export of openai/whisper-small: QNN context
# binaries compiled for ONE chip (default qcs8275 — Dragonwing IQ-8275,
# Arduino VENTUNO Q, HTP v75). Another Qualcomm chip is a rebuild:
#   docker build --build-arg WHISPER_QNN_CHIPSET=qcs6490 ...
# The ~340 MB of binaries, the tokenizer and the mel filter bank are BAKED
# IN (parakeet precedent): no volume, no network, boots on a sealed box.

FROM python:3.11-slim AS runtime

# ffmpeg: anything that is not already 16 kHz mono PCM WAV is converted.
# libatomic1: the QNN HTP backend dlopens it and python:slim lacks it.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
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
        "onnxruntime>=1.24.2,<2" \
        onnxruntime-qnn==2.7.0 \
        tokenizers==0.22.1 \
        huggingface_hub
ENV OPENNVR_ACCEL_FLAVOUR=qualcomm

# The assets: Qualcomm's public release bucket (no login), one zip per chip,
# pinned by version; plus the tokenizer and the feature extractor's mel
# filter bank from the upstream Whisper repository on HuggingFace.
ARG WHISPER_QNN_CHIPSET=qcs8275
ARG WHISPER_QNN_VERSION=v0.66.0
ARG WHISPER_QNN_ASSET_BASE=https://qaihub-public-assets.s3.us-west-2.amazonaws.com/qai-hub-models/models/whisper_small_quantized/releases
ARG WHISPER_HF_REPO=openai/whisper-small
ENV WHISPER_QNN_ASSETS_DIR=/app/model_weights/whisper-qnn
RUN set -eux; \
    name="whisper_small_quantized-precompiled_qnn_onnx-w8a16-qualcomm_${WHISPER_QNN_CHIPSET}"; \
    mkdir -p /tmp/asset "${WHISPER_QNN_ASSETS_DIR}"; \
    curl -fsSL -o /tmp/asset/a.zip "${WHISPER_QNN_ASSET_BASE}/${WHISPER_QNN_VERSION}/${name}.zip"; \
    python -c "import zipfile; zipfile.ZipFile('/tmp/asset/a.zip').extractall('/tmp/asset')"; \
    mv /tmp/asset/${name}/* "${WHISPER_QNN_ASSETS_DIR}/"; \
    rm -rf /tmp/asset; \
    python -c "from huggingface_hub import hf_hub_download as d; import shutil, os; \
      [shutil.copy(d('${WHISPER_HF_REPO}', f), os.path.join('${WHISPER_QNN_ASSETS_DIR}', f)) \
       for f in ('tokenizer.json', 'preprocessor_config.json')]"; \
    ls -la "${WHISPER_QNN_ASSETS_DIR}"

COPY opennvr_adapter_sdk/                       ./opennvr_adapter_sdk/
RUN pip install --no-cache-dir ./opennvr_adapter_sdk

COPY app/__init__.py                            ./app/__init__.py
COPY app/config/                                ./app/config/
COPY adapters/whisper_qnn/                      ./adapters/whisper_qnn/

# The only provider these binaries run on; the operator may override.
ENV ADAPTER_ONNX_PROVIDERS=QNNExecutionProvider:htp \
    OPENNVR_ADAPTER_PORT=9020 \
    PYTHONUNBUFFERED=1

EXPOSE 9020
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD curl -fsS http://localhost:9020/health || exit 1

CMD ["python", "-m", "uvicorn", "adapters.whisper_qnn.main:app", "--host", "0.0.0.0", "--port", "9020"]
