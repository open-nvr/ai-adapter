# syntax=docker/dockerfile:1.7
# text-embed adapter — sentence embeddings for search's caption arm, the
# `text_embed` task of the AI Adapter Contract v1. BAAI/bge-small-en-v1.5
# (MIT) as its own ONNX export, run by ONNX Runtime on the CPU: a caption
# costs a few milliseconds a core. The model and tokenizer are BAKED IN
# (issue #79): no volume, no egress, an air-gapped site loads the image
# and is done. linux/amd64 + linux/arm64.
#
# Build:
#   docker build -f adapters/text_embed/Dockerfile -t opennvr/text-embed-adapter:local .
# Run:
#   docker run --rm -p 9013:9013 -e OPENNVR_ADAPTER_TOKEN=<token> opennvr/text-embed-adapter:local

FROM python:3.11-slim AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --no-cache-dir \
        fastapi==0.127.0 \
        uvicorn==0.40.0 \
        pydantic==2.12.5 \
        httpx==0.28.1 \
        python-multipart==0.0.20 \
        numpy==1.26.4 \
        onnxruntime==1.23.2 \
        tokenizers==0.22.1 \
        huggingface_hub

ARG TEXT_EMBED_HF_REPO=BAAI/bge-small-en-v1.5
ARG TEXT_EMBED_HF_REVISION=main
ENV TEXT_EMBED_MODEL_DIR=/app/model_weights/text-embed
RUN set -eux; \
    mkdir -p "${TEXT_EMBED_MODEL_DIR}"; \
    python -c "from huggingface_hub import hf_hub_download as d; import shutil, os, json; \
      dst='${TEXT_EMBED_MODEL_DIR}'; \
      shutil.copy(d('${TEXT_EMBED_HF_REPO}', 'onnx/model.onnx', revision='${TEXT_EMBED_HF_REVISION}'), os.path.join(dst, 'model.onnx')); \
      shutil.copy(d('${TEXT_EMBED_HF_REPO}', 'tokenizer.json', revision='${TEXT_EMBED_HF_REVISION}'), os.path.join(dst, 'tokenizer.json')); \
      json.dump({'model_id': '${TEXT_EMBED_HF_REPO}', 'pooling': 'cls', 'max_tokens': 128, \
                 'query_prefix': '', 'licence': 'MIT', 'dim': 384}, open(os.path.join(dst, 'model.json'), 'w'))"; \
    rm -rf /root/.cache/huggingface; \
    ls -la "${TEXT_EMBED_MODEL_DIR}"

COPY opennvr_adapter_sdk/                       ./opennvr_adapter_sdk/
RUN pip install --no-cache-dir ./opennvr_adapter_sdk

COPY app/__init__.py                            ./app/__init__.py
COPY app/config/                                ./app/config/
COPY adapters/text_embed/                       ./adapters/text_embed/

ENV ADAPTER_ONNX_PROVIDERS=CPUExecutionProvider \
    TEXT_EMBED_THREADS=2 \
    OPENNVR_ADAPTER_PORT=9013 \
    PYTHONUNBUFFERED=1

EXPOSE 9013
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=5 \
    CMD curl -fsS http://localhost:9013/health || exit 1

CMD ["python", "-m", "uvicorn", "adapters.text_embed.main:app", "--host", "0.0.0.0", "--port", "9013"]
