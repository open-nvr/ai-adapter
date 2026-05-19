# syntax=docker/dockerfile:1.7
# Piper TTS adapter — contract-compliant per AI Adapter Contract v1.
#
# Build:
#   docker build -f adapters/piper/Dockerfile -t opennvr/piper-adapter:1.0.0 .
#
# Run (typical):
#   docker run --rm -p 9001:9001 \
#     -e OPENNVR_ADAPTER_TOKEN=<token> \
#     -v $(pwd)/model_weights/piper:/voices:ro \
#     -v $(pwd)/audio:/audio \
#     opennvr/piper-adapter:1.0.0

FROM python:3.11-slim AS runtime

# System deps for piper-tts (ONNX runtime + libsndfile for wave probing).
RUN apt-get update && apt-get install -y --no-install-recommends \
        libsndfile1 \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install runtime deps. Versions track ``pyproject.toml`` exactly — any
# drift between this list and the pyproject would make the
# containerised Piper diverge from what conformance tests run against.
RUN pip install --no-cache-dir \
        fastapi==0.127.0 \
        uvicorn==0.40.0 \
        pydantic==2.12.5 \
        httpx==0.28.1 \
        python-multipart==0.0.20 \
        piper-tts==1.2.0 \
        onnxruntime==1.23.2

# Copy the layers needed at runtime:
#   - app.interfaces.contract (the Pydantic types)
#   - app.adapters.audio.piper_adapter (the legacy inference class)
#   - app.adapters.base + app.interfaces.adapter (transitive base classes)
#   - app.utils.audio_utils + app.config (supporting helpers)
#   - adapters.piper.* (this service)
COPY app/__init__.py                            ./app/__init__.py
COPY app/interfaces/                            ./app/interfaces/
COPY app/adapters/__init__.py                   ./app/adapters/__init__.py
COPY app/adapters/base.py                       ./app/adapters/base.py
COPY app/adapters/audio/                        ./app/adapters/audio/
COPY app/config/                                ./app/config/
COPY app/utils/__init__.py                      ./app/utils/__init__.py
COPY app/utils/audio_utils.py                   ./app/utils/audio_utils.py
COPY adapters/piper/                            ./adapters/piper/

# Default voice directory + audio output. These match the legacy
# PiperAdapter defaults. Operators override with bind mounts.
ENV PYTHONPATH=/app \
    OPENNVR_ADAPTER_TOKEN="" \
    PIPER_VOICE_DIR=/voices \
    BASE_AUDIO_DIR=/audio

EXPOSE 9001

# /health is exempt from auth (§3.8) so the Docker healthcheck works
# even when OPENNVR_ADAPTER_TOKEN is set.
HEALTHCHECK --interval=30s --timeout=3s --retries=3 \
    CMD python -c "import httpx, sys; sys.exit(0 if httpx.get('http://127.0.0.1:9001/health', timeout=2).status_code == 200 else 1)"

CMD ["uvicorn", "adapters.piper.main:app", "--host", "0.0.0.0", "--port", "9001"]
