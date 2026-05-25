# syntax=docker/dockerfile:1.7
# BLIP scene-captioning adapter — contract-compliant per AI Adapter Contract v1.
#
# Build (from the ai-adapter repo root):
#   docker build -f adapters/blip/Dockerfile -t opennvr/blip-adapter:1.0.0 .
#
# Run (typical, CPU):
#   docker run --rm -p 9006:9006 \
#     -e OPENNVR_ADAPTER_TOKEN=<token> \
#     -v $(pwd)/hf_cache:/root/.cache/huggingface \
#     opennvr/blip-adapter:1.0.0
#
# The HuggingFace cache mount lets the ~990 MB BLIP weights persist
# across container restarts — without it, the first /infer triggers a
# fresh download every time the container is recreated.

FROM python:3.11-slim AS runtime

# System deps for Pillow + torch wheels.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Pin everything to match pyproject.toml exactly. torch is the CPU
# wheel — operators with NVIDIA GPUs swap to the CUDA wheel by
# rebuilding with --build-arg USE_GPU=true and the matching index.
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

# Install the OpenNVR adapter SDK so ``adapters.blip.main`` can import
# AdapterApp, BodyShape, etc.
COPY opennvr_adapter_sdk/                       ./opennvr_adapter_sdk/
RUN pip install --no-cache-dir ./opennvr_adapter_sdk

COPY adapters/blip/                             ./adapters/blip/

ENV PYTHONPATH=/app

EXPOSE 9006

HEALTHCHECK --interval=30s --timeout=3s --retries=3 \
    CMD python -c "import httpx, sys; sys.exit(0 if httpx.get('http://127.0.0.1:9006/health', timeout=2).status_code == 200 else 1)"

CMD ["uvicorn", "adapters.blip.main:app", "--host", "0.0.0.0", "--port", "9006"]
