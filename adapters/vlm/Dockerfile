# syntax=docker/dockerfile:1.7
# Open-vocabulary detection adapter (OWL-ViT v2) — contract-compliant per
# AI Adapter Contract v1.
#
# Build (from the ai-adapter repo root):
#   docker build -f adapters/vlm/Dockerfile -t ghcr.io/open-nvr/open-vocab-adapter:latest .
#
# Run (typical, CPU — slow; GPU strongly recommended for video rates):
#   docker run --rm -p 9012:9012 \
#     -e OPENNVR_ADAPTER_TOKEN=<token> \
#     ghcr.io/open-nvr/open-vocab-adapter:latest
#
# The ~600 MB OWL-ViT weights are BAKED IN (the blip/clip precedent,
# issue #79): the container needs no volume and no network, which is
# what lets the listing declare network_egress: [] and what makes it
# boot on a sealed box. Another OWL-ViT variant is a rebuild:
#   docker build --build-arg VLM_MODEL_ID=google/owlv2-large-patch14-ensemble ...

FROM python:3.11-slim AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --no-cache-dir \
        fastapi==0.127.0 \
        uvicorn==0.40.0 \
        pydantic==2.12.5 \
        httpx==0.28.1 \
        python-multipart==0.0.20 \
        pillow==12.0.0 \
        transformers==5.0.0

# The torch pin is not arch-portable (the long version is in
# adapters/blip/Dockerfile): download.pytorch.org/whl/cpu has no
# ``+cpu`` aarch64 wheel, and PyPI's plain aarch64 wheel IS the CPU
# build. Same upstream version either way; bump both arms together.
ARG TARGETARCH
RUN set -eux; \
    if [ "$TARGETARCH" = "arm64" ]; then \
        pip install --no-cache-dir torch==2.9.1 torchvision==0.24.1; \
    else \
        pip install --no-cache-dir \
            --index-url https://download.pytorch.org/whl/cpu \
            torch==2.9.1+cpu \
            torchvision==0.24.1+cpu; \
    fi

# Pre-bake the OWL-ViT weights so the adapter runs with NO network
# egress (issue #79, as blip and clip). Without this the first boot
# downloads ~600 MB from huggingface.co, which on an air-gapped site
# never arrives and leaves the adapter unhealthy forever.
ARG VLM_MODEL_ID=google/owlv2-base-patch16-ensemble
ENV HF_HUB_DISABLE_TELEMETRY=1
RUN python -c "from transformers import Owlv2ForObjectDetection, Owlv2Processor; \
mid='${VLM_MODEL_ID}'; Owlv2Processor.from_pretrained(mid); Owlv2ForObjectDetection.from_pretrained(mid)"

# Pin the RUNTIME fully offline so from_pretrained() loads only from the
# baked cache and never makes the metadata check to huggingface.co. Set
# AFTER the bake (the build needs the network; the container does not).
# OPENNVR_VLM_MODEL defaults to the baked id; pointing it at a model
# that is not baked fails to load, by design.
ENV TRANSFORMERS_OFFLINE=1 \
    HF_HUB_OFFLINE=1 \
    OPENNVR_VLM_MODEL=${VLM_MODEL_ID}

COPY opennvr_adapter_sdk/                       ./opennvr_adapter_sdk/
RUN pip install --no-cache-dir ./opennvr_adapter_sdk

COPY adapters/vlm/                              ./adapters/vlm/

ENV PYTHONPATH=/app

EXPOSE 9012

HEALTHCHECK --interval=30s --timeout=3s --retries=3 \
    CMD python -c "import httpx, sys; sys.exit(0 if httpx.get('http://127.0.0.1:9012/health', timeout=2).status_code == 200 else 1)"

CMD ["uvicorn", "adapters.vlm.main:app", "--host", "0.0.0.0", "--port", "9012"]
