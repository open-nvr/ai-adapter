# syntax=docker/dockerfile:1.7
# Parakeet ASR adapter (NVIDIA Parakeet TDT 0.6B v3 via onnx-asr) —
# contract-compliant per AI Adapter Contract v1.
#
# Build (from the ai-adapter repo root):
#   docker build -f adapters/parakeet/Dockerfile -t ghcr.io/open-nvr/parakeet-adapter:latest .
#
# Run:
#   docker run --rm -p 9019:9019 -e OPENNVR_ADAPTER_TOKEN=<token> \
#     ghcr.io/open-nvr/parakeet-adapter:latest
#
# The ~670 MB int8 weights are BAKED IN (blip/clip/open-vocab precedent,
# issue #79): no volume, no network, boots on a sealed box. Another
# onnx-asr model is a rebuild:
#   docker build --build-arg PARAKEET_MODEL=nemo-parakeet-tdt-0.6b-v2 ...

FROM python:3.11-slim AS runtime

# ffmpeg: anything that is not already 16 kHz mono PCM WAV is converted.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --no-cache-dir \
        fastapi==0.127.0 \
        uvicorn==0.40.0 \
        pydantic==2.12.5 \
        httpx==0.28.1 \
        python-multipart==0.0.20 \
        "onnx-asr[cpu,hub]==0.12.0" \
        huggingface_hub

# Pre-bake the weights: onnx-asr's own loader resolves the model name
# to its HuggingFace repository and the int8 files, into
# PARAKEET_WEIGHTS_DIR, which the service loads from by path.
ARG PARAKEET_MODEL=nemo-parakeet-tdt-0.6b-v3
ENV PARAKEET_MODEL=${PARAKEET_MODEL} \
    PARAKEET_WEIGHTS_DIR=/app/model_weights/parakeet \
    HF_HUB_DISABLE_TELEMETRY=1
RUN python - <<'PY'
import os, shutil
import onnx_asr
m = onnx_asr.load_model(os.environ["PARAKEET_MODEL"], quantization="int8", providers=["CPUExecutionProvider"])
# The loader cached the files under the HF hub cache; copy the snapshot
# into the weights dir so the runtime never touches the cache or the net.
import huggingface_hub as hh
cache = hh.constants.HF_HUB_CACHE
snaps = [os.path.join(r, d) for r, ds, _ in os.walk(cache) for d in ds if r.endswith("snapshots")]
assert snaps, "no snapshot downloaded"
src = max(snaps, key=os.path.getmtime)
dst = os.environ["PARAKEET_WEIGHTS_DIR"]
os.makedirs(dst, exist_ok=True)
for n in os.listdir(src):
    shutil.copy(os.path.join(src, n), os.path.join(dst, n))
print("baked", sorted(os.listdir(dst)))
shutil.rmtree(cache, ignore_errors=True)
PY

# Offline from here: the service loads by path and never resolves a name.
ENV HF_HUB_OFFLINE=1

COPY opennvr_adapter_sdk/                       ./opennvr_adapter_sdk/
RUN pip install --no-cache-dir ./opennvr_adapter_sdk

# OpenNVR Accelerate flavours — one Dockerfile, ACCEL picks the runtime:
#   cpu   (default)  plain onnxruntime (onnx-asr[cpu])
#   qualcomm         + onnxruntime-qnn, the Qualcomm AI Runtime PLUGIN provider
#                    (next to plain onnxruntime >= 1.24.2); arm64 only. The
#                    encoder/decoder graphs are dynamic-shape, so what the
#                    HTP takes is measured per release (docs/HARDWARE.md).
ARG ACCEL=cpu
RUN case "${ACCEL}" in \
      cpu) ;; \
      qualcomm) pip install --no-cache-dir "onnxruntime>=1.24.2,<2" onnxruntime-qnn==2.7.0 ;; \
      *) echo "unknown ACCEL=${ACCEL} (cpu|qualcomm)" >&2; exit 1 ;; \
    esac
ENV OPENNVR_ACCEL_FLAVOUR=${ACCEL}

COPY app/__init__.py                            ./app/__init__.py
COPY app/config/                                ./app/config/
COPY adapters/parakeet/                         ./adapters/parakeet/

ENV PYTHONPATH=/app \
    OPENNVR_ADAPTER_TOKEN=""

EXPOSE 9019

HEALTHCHECK --interval=30s --timeout=3s --retries=3 \
    CMD python -c "import httpx, sys; sys.exit(0 if httpx.get('http://127.0.0.1:9019/health', timeout=2).status_code == 200 else 1)"

CMD ["uvicorn", "adapters.parakeet.main:app", "--host", "0.0.0.0", "--port", "9019"]
