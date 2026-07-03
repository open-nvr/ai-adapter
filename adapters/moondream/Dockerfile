# syntax=docker/dockerfile:1.7
# Moondream2 vision-language adapter — VQA + captioning, Apache-2.0.
#
# Uses the official `moondream` package with a QUANTIZED model file (int8 .mf.gz)
# — built for fast VQA on CPU/edge. No torch/transformers, so the image is small
# and there's no remote-code / CPU-encoding fragility.
#
# Build (from the ai-adapter repo root). The quantized 0.5B int8 model is
# BAKED INTO THE IMAGE BY DEFAULT (issue #79 treatment, same as BLIP) from a
# revision-pinned, checksum-verified URL — so the published image runs with
# zero runtime egress under AI_SOVEREIGNTY=local_only. The download happens
# once at BUILD time (CI), never on the operator's box:
#   docker build -f adapters/moondream/Dockerfile -t opennvr/moondream-adapter:1.0.0 .
# Override the model (e.g. the more capable 2B) with
#   --build-arg MOONDREAM_MODEL_URL=<url> --build-arg MOONDREAM_MODEL_SHA256=<sha256> \
#   --build-arg MOONDREAM_MODEL_FILE=<file>
# or build a weightless dev image with --build-arg MOONDREAM_MODEL_URL="".
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
        moondream==0.0.6

# Bake the quantized model file into the image so the adapter runs offline
# under AI_SOVEREIGNTY=local_only — no runtime egress, ever (issue #79, same
# posture as the BLIP image). The default is the official Moondream 0.5B int8
# file (~593 MiB), pinned to an immutable HuggingFace revision and verified
# against a sha256 so a moved/tampered file fails the BUILD, not the operator.
# The download happens at build time only; the running container never
# touches the network. Set MOONDREAM_MODEL_URL="" for a weightless dev image
# (the adapter then reports unhealthy until a model is mounted).
ARG MOONDREAM_MODEL_URL="https://huggingface.co/vikhyatk/moondream2/resolve/9dddae84d54db4ac56fe37817aeaeb502ed083e2/moondream-0_5b-int8.mf.gz"
ARG MOONDREAM_MODEL_SHA256="355c534d5f71fff8524dfeb41ba65f7d07948f910f754de4df242d0037d8be4a"
ARG MOONDREAM_MODEL_FILE=moondream-0_5b-int8.mf.gz
RUN mkdir -p /models \
    && if [ -n "$MOONDREAM_MODEL_URL" ]; then \
         echo "Baking model ${MOONDREAM_MODEL_FILE}…"; \
         curl -fSL --retry 3 "$MOONDREAM_MODEL_URL" -o "/models/${MOONDREAM_MODEL_FILE}"; \
         if [ -n "$MOONDREAM_MODEL_SHA256" ]; then \
           echo "${MOONDREAM_MODEL_SHA256}  /models/${MOONDREAM_MODEL_FILE}" | sha256sum -c - \
             || { echo "ERROR: model checksum mismatch — refusing to bake" >&2; exit 1; }; \
         fi; \
       else \
         echo "WARN: MOONDREAM_MODEL_URL is empty — image built WITHOUT a model (dev only); the adapter reports unhealthy until a model is provided at ${MOONDREAM_MODEL_FILE}." >&2; \
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
