# syntax=docker/dockerfile:1.7
# Piper TTS adapter — piper-tts (onnxruntime) in-process. Torch-free.
#
#   docker build -f adapters/pipertts/Dockerfile -t opennvr/pipertts-adapter:1.0.0 .
#   docker run --rm -p 9012:9012 -e OPENNVR_ADAPTER_TOKEN=<token> \
#     -v $(pwd)/model_weights/pipertts:/app/models:ro opennvr/pipertts-adapter:1.0.0

FROM python:3.11-slim AS runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
        libsndfile1 ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
RUN pip install --no-cache-dir \
        fastapi==0.127.0 uvicorn==0.40.0 pydantic==2.12.5 \
        httpx==0.28.1 python-multipart==0.0.20 \
        piper-tts==1.2.0 onnxruntime==1.23.2

COPY opennvr_adapter_sdk/  ./opennvr_adapter_sdk/
RUN pip install --no-cache-dir ./opennvr_adapter_sdk
COPY adapters/pipertts/    ./adapters/pipertts/

ENV PYTHONPATH=/app \
    OPENNVR_ADAPTER_TOKEN="" \
    OPENNVR_TTS_VOICE_PATH=/app/models/en_US-amy-medium.onnx

EXPOSE 9012
HEALTHCHECK --interval=30s --timeout=3s --retries=3 --start-period=30s \
    CMD python -c "import httpx,sys; sys.exit(0 if httpx.get('http://127.0.0.1:9012/health', timeout=2).status_code==200 else 1)"
CMD ["uvicorn", "adapters.pipertts.main:app", "--host", "0.0.0.0", "--port", "9012"]
