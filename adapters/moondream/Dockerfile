# syntax=docker/dockerfile:1.7
# Moondream2 vision-language adapter — VQA + captioning, Apache-2.0.
#
# Uses the official `moondream` package with a QUANTIZED model file (int8 .mf.gz)
# — built for fast VQA on CPU/edge. No torch/transformers, so the image is small
# and there's no remote-code / CPU-encoding fragility.
#
# Build (from the ai-adapter repo root). You MUST supply the model-file URL —
# get the int8 .mf.gz from the Moondream model page (https://moondream.ai/p/models
# or the HuggingFace mirror). 0.5B is the fast edge default; 2B is more capable:
#   docker build -f adapters/moondream/Dockerfile \
#     --build-arg MOONDREAM_MODEL_URL=<url-to-moondream-0_5b-int8.mf.gz> \
#     -t opennvr/moondream-adapter:1.0.0 .
#
# Run (CPU):
#   docker run --rm -p 9008:9008 -e OPENNVR_ADAPTER_TOKEN=<token> \
#     opennvr/moondream-adapter:1.0.0

FROM python:3.11-slim AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Lightweight deps — the moondream package bundles its own CPU runtime.
RUN pip install --no-cache-dir \
        fastapi==0.127.0 \
        uvicorn==0.40.0 \
        pydantic==2.12.5 \
        httpx==0.28.1 \
        python-multipart==0.0.20 \
        "pillow>=10.4,<11" \
        "moondream>=0.1,<1.0"

# Bake the quantized model file into the image so the adapter runs offline under
# AI_SOVEREIGNTY=local_only (no runtime egress). 0.5B int8 ~593 MiB by default.
ARG MOONDREAM_MODEL_URL
ARG MOONDREAM_MODEL_FILE=moondream-0_5b-int8.mf.gz
# Bake the model when a URL is supplied (production / offline / local_only).
# When it's not (e.g. the CI smoke build that builds every adapter without
# args), build WITHOUT the model so the image still builds — the adapter then
# reports unhealthy at runtime until a model is provided. Pass the URL for a
# complete, offline-capable image.
RUN mkdir -p /models \
    && if [ -n "$MOONDREAM_MODEL_URL" ]; then \
         echo "Baking model ${MOONDREAM_MODEL_FILE}…"; \
         curl -fSL "$MOONDREAM_MODEL_URL" -o "/models/${MOONDREAM_MODEL_FILE}"; \
       else \
         echo "WARN: no MOONDREAM_MODEL_URL — image built WITHOUT a model; pass --build-arg MOONDREAM_MODEL_URL=<url to ${MOONDREAM_MODEL_FILE}> for an offline/local_only image." >&2; \
       fi
ENV OPENNVR_MOONDREAM_MODEL_PATH=/models/${MOONDREAM_MODEL_FILE}

# OpenNVR adapter SDK + the adapter package.
COPY opennvr_adapter_sdk/                       ./opennvr_adapter_sdk/
RUN pip install --no-cache-dir ./opennvr_adapter_sdk
COPY adapters/moondream/                        ./adapters/moondream/

ENV PYTHONPATH=/app \
    OPENNVR_ADAPTER_TOKEN=""

EXPOSE 9008

HEALTHCHECK --interval=30s --timeout=3s --retries=3 \
    CMD python -c "import httpx, sys; sys.exit(0 if httpx.get('http://127.0.0.1:9008/health', timeout=2).status_code == 200 else 1)"

CMD ["uvicorn", "adapters.moondream.main:app", "--host", "0.0.0.0", "--port", "9008"]
