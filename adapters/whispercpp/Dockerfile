# syntax=docker/dockerfile:1.7
# whisper.cpp STT adapter — ggml Whisper via pywhispercpp. Torch-free.
#
# Build (context = ai-adapter repo root):
#   docker build -f adapters/whispercpp/Dockerfile -t opennvr/whispercpp-adapter:1.0.0 .
# Run (mount the ggml model):
#   docker run --rm -p 9013:9013 -e OPENNVR_ADAPTER_TOKEN=<token> \
#     -v $(pwd)/model_weights/whispercpp:/app/models:ro \
#     opennvr/whispercpp-adapter:1.0.0

# ── build stage: compile the pywhispercpp extension ────────────────────
FROM python:3.11-slim AS build
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake git ca-certificates \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir --prefix=/install \
        numpy pywhispercpp

# ── runtime stage: slim, no compiler ───────────────────────────────────
FROM python:3.11-slim AS runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 ca-certificates \
    && rm -rf /var/lib/apt/lists/*
COPY --from=build /install /usr/local

WORKDIR /app
RUN pip install --no-cache-dir \
        fastapi==0.127.0 uvicorn==0.40.0 pydantic==2.12.5 \
        httpx==0.28.1 python-multipart==0.0.20

COPY opennvr_adapter_sdk/  ./opennvr_adapter_sdk/
RUN pip install --no-cache-dir ./opennvr_adapter_sdk
COPY adapters/whispercpp/  ./adapters/whispercpp/

ENV PYTHONPATH=/app \
    OPENNVR_ADAPTER_TOKEN="" \
    OPENNVR_STT_MODEL_PATH=/app/models/ggml-base.en.bin

EXPOSE 9013
HEALTHCHECK --interval=30s --timeout=3s --retries=3 --start-period=60s \
    CMD python -c "import httpx,sys; sys.exit(0 if httpx.get('http://127.0.0.1:9013/health', timeout=2).status_code==200 else 1)"
CMD ["uvicorn", "adapters.whispercpp.main:app", "--host", "0.0.0.0", "--port", "9013"]
