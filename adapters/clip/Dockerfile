# syntax=docker/dockerfile:1.7
# CLIP embedding adapter — contract-compliant per AI Adapter Contract v1.
#
# Build:
#   docker build -f adapters/clip/Dockerfile -t opennvr/clip-adapter:1.0.0 .
#
# Run (CPU — the design target):
#   docker run --rm -p 9011:9011 \
#     -e OPENNVR_ADAPTER_TOKEN=<token> \
#     opennvr/clip-adapter:1.0.0
#
# The weights are BAKED IN, so the container runs with no network egress
# at all — the same call adapters/blip makes (issue #79). No volume to
# mount, nothing to pre-populate, and an air-gapped site can load the
# image and be done.

FROM python:3.11-slim AS runtime

# System deps for Pillow + torch wheels.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Pinned to match pyproject.toml exactly.
RUN pip install --no-cache-dir \
        fastapi==0.127.0 \
        uvicorn==0.40.0 \
        pydantic==2.12.5 \
        httpx==0.28.1 \
        python-multipart==0.0.20 \
        pillow==12.0.0

# TARGETARCH is set automatically by BuildKit to the arch being built
# ("amd64" / "arm64"). It is needed because the torch pin is not
# arch-portable — the same reason, and the same fix, as
# adapters/blip/Dockerfile:
#
#   * linux/amd64 — download.pytorch.org/whl/cpu carries a CPU-only
#     build tagged with the ``+cpu`` local version, which is what keeps
#     the ~2 GB CUDA wheel out of the image.
#   * linux/arm64 — that index has no ``+cpu`` aarch64 wheel
#     (pytorch/pytorch#136275), so pinning ``torch==2.9.1+cpu`` there
#     fails to resolve and the arm64 build dies. PyPI's plain
#     ``torch==2.9.1`` manylinux_2_28_aarch64 wheel IS the CPU build —
#     there is no CUDA variant for generic aarch64 — so it is the
#     correct equivalent, not a fallback.
#
# Same upstream version either way; only the index and the local version
# suffix differ. Bump both arms together, and keep them in step with the
# cpu/gpu extras in pyproject.toml.
ARG TARGETARCH
RUN set -eux; \
    if [ "$TARGETARCH" = "arm64" ]; then \
        pip install --no-cache-dir torch==2.9.1 torchvision==0.24.1; \
    else \
        pip install --no-cache-dir \
            --index-url https://download.pytorch.org/whl/cpu \
            torch==2.9.1+cpu \
            torchvision==0.24.1+cpu; \
    fi

RUN pip install --no-cache-dir open-clip-torch==3.3.0

# Pre-bake the weights so the adapter runs with NO network egress under
# AI_SOVEREIGNTY=local_only (issue #79, the same call adapters/blip
# makes). Without this, the first request would pull ~600 MB from
# huggingface.co at runtime — which means relaxing the sovereignty gate
# on a box whose whole reason for existing is not having to. Baking
# keeps the secure-by-default posture, and it is why main.py's
# _weights_fetch_egress() reports an empty list on this image.
ARG CLIP_MODEL=ViT-B-32
ARG CLIP_PRETRAINED=laion2b_s34b_b79k
ENV HF_HUB_DISABLE_TELEMETRY=1
RUN python -c "import open_clip; \
open_clip.create_model_and_transforms('${CLIP_MODEL}', pretrained='${CLIP_PRETRAINED}', cache_dir='/models/clip'); \
open_clip.get_tokenizer('${CLIP_MODEL}')"

# Pin the RUNTIME fully offline so a load never makes the default
# metadata check to huggingface.co — which would contradict
# network_egress=[] and stall in a network-isolated deployment. Set
# AFTER the bake: the build needs the network, the container does not.
ENV HF_HUB_OFFLINE=1 \
    CLIP_OFFLINE=1

COPY opennvr_adapter_sdk/                       ./opennvr_adapter_sdk/
RUN pip install --no-cache-dir ./opennvr_adapter_sdk

# Self-contained: the SDK and this one package, nothing from app/.
COPY adapters/clip/                             ./adapters/clip/

ENV PYTHONPATH=/app \
    OPENNVR_ADAPTER_TOKEN="" \
    CLIP_MODEL=ViT-B-32 \
    CLIP_PRETRAINED=laion2b_s34b_b79k \
    CLIP_CACHE_DIR=/models/clip \
    CLIP_THREADS=2

EXPOSE 9011

HEALTHCHECK --interval=30s --timeout=3s --retries=3 \
    CMD python -c "import httpx, sys; sys.exit(0 if httpx.get('http://127.0.0.1:9011/health', timeout=2).status_code == 200 else 1)"

CMD ["uvicorn", "adapters.clip.main:app", "--host", "0.0.0.0", "--port", "9011"]
