# syntax=docker/dockerfile:1.7
# Ollama-backed VLM adapter — contract-compliant per AI Adapter Contract v1.
#
# No weights, no torch, no onnx: this adapter PROXIES visual_qa /
# scene_caption to an Ollama endpoint (OPENNVR_OLLAMA_VLM_URL — the host
# machine by default, where inference gets the real GPU on macOS/Windows).
# The image is therefore tiny (~80 MB) and arch-independent.
#
# Build (from the ai-adapter repo root):
#   docker build -f adapters/ollamavlm/Dockerfile -t opennvr/ollamavlm-adapter:local .
#
# Run:
#   docker run --rm -p 9018:9018 \
#     -e OPENNVR_ADAPTER_TOKEN=<token> \
#     -e OPENNVR_OLLAMA_VLM_URL=http://host.docker.internal:11434 \
#     -e OPENNVR_OLLAMA_VLM_MODEL=moondream \
#     --add-host host.docker.internal:host-gateway \
#     opennvr/ollamavlm-adapter:local

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
        python-multipart==0.0.20

COPY opennvr_adapter_sdk/   ./opennvr_adapter_sdk/
RUN pip install --no-cache-dir ./opennvr_adapter_sdk
COPY adapters/ollamavlm/    ./adapters/ollamavlm/

ENV PYTHONPATH=/app \
    OPENNVR_ADAPTER_TOKEN="" \
    OPENNVR_OLLAMA_VLM_URL=http://host.docker.internal:11434 \
    OPENNVR_OLLAMA_VLM_MODEL=moondream

EXPOSE 9018
HEALTHCHECK --interval=30s --timeout=3s --retries=3 \
    CMD python -c "import httpx, sys; sys.exit(0 if httpx.get('http://127.0.0.1:9018/health', timeout=2).status_code == 200 else 1)"
CMD ["uvicorn", "adapters.ollamavlm.main:app", "--host", "0.0.0.0", "--port", "9018"]
