# syntax=docker/dockerfile:1.7
# Voice-bundle adapter — Whisper + Piper + BLIP in ONE image/container, each on
# its existing port with the SAME per-adapter /infer contract (issue #82).
#
# Why: the full camera-agent edition otherwise runs three separate adapter
# containers. This collapses them to one (fewer images to pull, fewer containers
# to start/health-check). RAM is dominated by the models (LLM + BLIP), so this is
# an operational-simplicity win, not a halving of memory — the editions and model
# choices are the real RAM lever.
#
# Build (from the ai-adapter repo root):
#   docker build -f adapters/voice-bundle/Dockerfile -t opennvr/voice-adapter:1.0.0 .
#
# Run (CPU): exposes 9001 (piper), 9003 (whisper), 9006 (blip)
#   docker run --rm -p 9001:9001 -p 9003:9003 -p 9006:9006 \
#     -e OPENNVR_ADAPTER_TOKEN=<token> -e WHISPER_MODEL_SIZE=base.en \
#     opennvr/voice-adapter:1.0.0

FROM python:3.11-slim AS runtime

# System deps: libsndfile1 + ffmpeg for faster-whisper audio decode; ca-certs
# for the build-time weight fetch.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libsndfile1 \
        ffmpeg \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Union of the three adapters' runtime deps, pinned to match the per-adapter
# Dockerfiles. faster-whisper (whisper) and piper-tts/onnxruntime (piper) are
# torch-free; only BLIP pulls torch — installed once here for all three.
RUN pip install --no-cache-dir \
        fastapi==0.127.0 \
        uvicorn==0.40.0 \
        pydantic==2.12.5 \
        httpx==0.28.1 \
        python-multipart==0.0.20 \
        pillow==12.0.0 \
        transformers==5.0.0 \
        "faster-whisper>=1.1.0" \
        piper-tts==1.2.0 \
        onnxruntime==1.23.2 \
    && pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cpu \
        torch==2.9.1+cpu \
        torchvision==0.24.1+cpu

# Pre-bake BLIP weights so the bundle captions offline under local_only (#79).
ARG BLIP_MODEL_ID=Salesforce/blip-image-captioning-base
ENV HF_HUB_DISABLE_TELEMETRY=1
RUN python -c "from transformers import BlipProcessor, BlipForConditionalGeneration; \
mid='${BLIP_MODEL_ID}'; BlipProcessor.from_pretrained(mid); BlipForConditionalGeneration.from_pretrained(mid)"
# Pin the runtime fully offline AFTER the bake (build needs network; runtime must not).
ENV TRANSFORMERS_OFFLINE=1 \
    HF_HUB_OFFLINE=1

# OpenNVR adapter SDK (imported by all three adapter apps).
COPY opennvr_adapter_sdk/                       ./opennvr_adapter_sdk/
RUN pip install --no-cache-dir ./opennvr_adapter_sdk

# App modules + the three adapter packages + the launcher. Copy the whole app/
# tree so each adapter's transitive imports (e.g. app.adapters.audio for whisper)
# resolve without enumerating them file-by-file.
COPY app/                                       ./app/
COPY adapters/whisper/                          ./adapters/whisper/
COPY adapters/piper/                            ./adapters/piper/
COPY adapters/blip/                             ./adapters/blip/
COPY adapters/voice-bundle/launch.py            ./adapters/voice-bundle/launch.py

ENV PYTHONPATH=/app \
    OPENNVR_ADAPTER_TOKEN="" \
    WHISPER_MODEL_SIZE=base.en

EXPOSE 9001 9003 9006

# Healthy when all three apps answer. (Any disabled via *_PORT=0 are skipped by
# the launcher; adjust this check accordingly if you disable one.)
HEALTHCHECK --interval=30s --timeout=4s --retries=5 \
    CMD python -c "import httpx,sys; \
ports=[9001,9003,9006]; \
sys.exit(0 if all(httpx.get(f'http://127.0.0.1:{p}/health',timeout=2).status_code==200 for p in ports) else 1)"

CMD ["python", "adapters/voice-bundle/launch.py"]
