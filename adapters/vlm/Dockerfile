# syntax=docker/dockerfile:1.7
# Open-vocabulary detection adapter (OWL-ViT v2) — contract-compliant per
# AI Adapter Contract v1.
#
# Build (from the ai-adapter repo root):
#   docker build -f adapters/vlm/Dockerfile -t opennvr/vlm-adapter:1.0.0 .
#
# Run (typical, CPU — slow; GPU strongly recommended for video rates):
#   docker run --rm -p 9012:9012 \
#     -e OPENNVR_ADAPTER_TOKEN=<token> \
#     -v $(pwd)/hf_cache:/root/.cache/huggingface \
#     opennvr/vlm-adapter:1.0.0
#
# The HuggingFace cache mount lets the ~600 MB OWL-ViT weights persist
# across container restarts.

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
        transformers==5.0.0 \
    && pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cpu \
        torch==2.9.1+cpu \
        torchvision==0.24.1+cpu

COPY opennvr_adapter_sdk/                       ./opennvr_adapter_sdk/
RUN pip install --no-cache-dir ./opennvr_adapter_sdk

COPY adapters/vlm/                              ./adapters/vlm/

ENV PYTHONPATH=/app

EXPOSE 9012

HEALTHCHECK --interval=30s --timeout=3s --retries=3 \
    CMD python -c "import httpx, sys; sys.exit(0 if httpx.get('http://127.0.0.1:9012/health', timeout=2).status_code == 200 else 1)"

CMD ["uvicorn", "adapters.vlm.main:app", "--host", "0.0.0.0", "--port", "9012"]
