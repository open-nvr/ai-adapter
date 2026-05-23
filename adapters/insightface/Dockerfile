# syntax=docker/dockerfile:1.7
# InsightFace face-recognition adapter — contract-compliant per AI Adapter Contract v1.
#
# Build (from the ai-adapter repo root):
#   docker build -f adapters/insightface/Dockerfile -t opennvr/insightface-adapter:1.0.0 .
#
# Run (CPU; InsightFace runs well on CPU for typical homelab fps):
#   docker run --rm -p 9005:9005 \
#     -e OPENNVR_ADAPTER_TOKEN=<token> \
#     -v $(pwd)/model_weights:/weights:ro \
#     -v $(pwd)/face-db:/data \
#     -e OPENNVR_INSIGHTFACE_FACE_DB=/data/faces.json \
#     opennvr/insightface-adapter:1.0.0
#
# The face DB JSON survives container restarts as long as the
# /data mount survives. Embeddings live there; raw faces never do.

FROM python:3.11-slim AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        libglib2.0-0 \
        libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Versions track pyproject.toml exactly.
RUN pip install --no-cache-dir \
        fastapi==0.127.0 \
        uvicorn==0.40.0 \
        pydantic==2.12.5 \
        httpx==0.28.1 \
        python-multipart==0.0.20 \
        numpy==1.26.4 \
        opencv-python-headless==4.10.0.84 \
        "insightface>=0.7.3" \
        "onnxruntime==1.23.2" \
        "scipy==1.16.3"

COPY adapters/insightface/  ./adapters/insightface/

# Vendored SDK install — when the SDK is up on PyPI this becomes
# `pip install opennvr-adapter-sdk`.
COPY opennvr_adapter_sdk/  ./opennvr_adapter_sdk/
RUN pip install --no-cache-dir ./opennvr_adapter_sdk

ENV PYTHONPATH=/app \
    OPENNVR_ADAPTER_TOKEN="" \
    OPENNVR_INSIGHTFACE_MODEL=buffalo_l \
    OPENNVR_INSIGHTFACE_FACE_DB=/data/faces.json

EXPOSE 9005

HEALTHCHECK --interval=30s --timeout=3s --retries=3 \
    CMD python -c "import httpx, sys; sys.exit(0 if httpx.get('http://127.0.0.1:9005/health', timeout=2).status_code == 200 else 1)"

CMD ["uvicorn", "adapters.insightface.main:app", "--host", "0.0.0.0", "--port", "9005"]
