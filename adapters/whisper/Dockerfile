# syntax=docker/dockerfile:1.7
# Whisper ASR adapter — contract-compliant per AI Adapter Contract v1.
#
# Build:
#   docker build -f adapters/whisper/Dockerfile -t opennvr/whisper-adapter:1.0.0 .
#
# Run (CPU; whisper-tiny ~75 MB, base ~150 MB, small ~500 MB):
#   docker run --rm -p 9003:9003 \
#     -e OPENNVR_ADAPTER_TOKEN=<token> \
#     -e WHISPER_MODEL_SIZE=base \
#     -v $(pwd)/model_weights/whisper:/weights:ro \
#     opennvr/whisper-adapter:1.0.0
#
# Run with GPU passthrough:
#   docker run --rm --gpus all -p 9003:9003 \
#     -e OPENNVR_ADAPTER_TOKEN=<token> \
#     -e WHISPER_MODEL_SIZE=base \
#     -v $(pwd)/model_weights/whisper:/weights:ro \
#     opennvr/whisper-adapter:1.0.0

FROM python:3.11-slim AS runtime

# libsndfile1 + ffmpeg let faster-whisper decode any audio format.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libsndfile1 \
        ffmpeg \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# faster-whisper>=1.1.0 (matching pyproject.toml [stt] extra) so the
# transitive ``av`` dep is unpinned and pip can pick a recent PyAV
# (currently 17.x) that has cp311-linux-x86_64 manylinux wheels.
# faster-whisper 1.0.0 / 1.0.1 pinned ``av==11.*`` and av 11.0.0 ships
# only a source tarball (no binary wheels), so pip falls back to a
# from-source build that needs pkg-config + libav*-dev headers we
# don't install here. The smoke job failed with "pkg-config is
# required for building PyAV" before this bump.
RUN pip install --no-cache-dir \
        fastapi==0.127.0 \
        uvicorn==0.40.0 \
        pydantic==2.12.5 \
        httpx==0.28.1 \
        python-multipart==0.0.20 \
        "faster-whisper>=1.1.0"

# Install the OpenNVR adapter SDK so ``adapters.whisper.main`` can
# import AdapterApp et al. Mirrors insightface/fast_plate_ocr.
COPY opennvr_adapter_sdk/                       ./opennvr_adapter_sdk/
RUN pip install --no-cache-dir ./opennvr_adapter_sdk

COPY app/__init__.py                            ./app/__init__.py
COPY app/interfaces/                            ./app/interfaces/
COPY app/adapters/__init__.py                   ./app/adapters/__init__.py
COPY app/adapters/base.py                       ./app/adapters/base.py
COPY app/adapters/audio/__init__.py             ./app/adapters/audio/__init__.py
COPY app/adapters/audio/whisper_adapter.py      ./app/adapters/audio/whisper_adapter.py
COPY app/config/                                ./app/config/
COPY app/utils/__init__.py                      ./app/utils/__init__.py
COPY app/utils/audio_utils.py                   ./app/utils/audio_utils.py
COPY adapters/whisper/                          ./adapters/whisper/

ENV PYTHONPATH=/app \
    OPENNVR_ADAPTER_TOKEN="" \
    WHISPER_MODEL_SIZE=base

EXPOSE 9003

HEALTHCHECK --interval=30s --timeout=3s --retries=3 \
    CMD python -c "import httpx, sys; sys.exit(0 if httpx.get('http://127.0.0.1:9003/health', timeout=2).status_code == 200 else 1)"

CMD ["uvicorn", "adapters.whisper.main:app", "--host", "0.0.0.0", "--port", "9003"]
