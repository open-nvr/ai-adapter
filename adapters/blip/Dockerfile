# syntax=docker/dockerfile:1.7
# BLIP scene-captioning adapter — contract-compliant per AI Adapter Contract v1.
#
# Build (from the ai-adapter repo root):
#   docker build -f adapters/blip/Dockerfile -t opennvr/blip-adapter:1.0.0 .
#
# Run (typical, CPU):
#   docker run --rm -p 9006:9006 \
#     -e OPENNVR_ADAPTER_TOKEN=<token> \
#     -v $(pwd)/hf_cache:/root/.cache/huggingface \
#     opennvr/blip-adapter:1.0.0
#
# The HuggingFace cache mount lets the ~990 MB BLIP weights persist
# across container restarts — without it, the first /infer triggers a
# fresh download every time the container is recreated.

FROM python:3.11-slim AS runtime

# System deps for Pillow + torch wheels.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Pin everything to match pyproject.toml exactly. torch is the CPU
# wheel — operators with NVIDIA GPUs swap to the CUDA wheel by
# rebuilding with --build-arg USE_GPU=true and the matching index.
RUN pip install --no-cache-dir \
        fastapi==0.127.0 \
        uvicorn==0.40.0 \
        pydantic==2.12.5 \
        httpx==0.28.1 \
        python-multipart==0.0.20 \
        pillow==12.0.0 \
        transformers==5.0.0

# TARGETARCH is set automatically by BuildKit to the arch being built
# ("amd64" / "arm64"). It is needed because the torch pin is not
# arch-portable:
#
#   * linux/amd64 — download.pytorch.org/whl/cpu carries a CPU-only
#     build tagged with the ``+cpu`` local version, which is what keeps
#     the ~2 GB CUDA wheel out of the image.
#   * linux/arm64 — that index has no ``+cpu`` aarch64 wheel
#     (pytorch/pytorch#136275 is the open request for one), so pinning
#     ``torch==2.9.1+cpu`` there fails to resolve and the arm64 build
#     dies. PyPI's plain ``torch==2.9.1`` manylinux_2_28_aarch64 wheel
#     IS the CPU build — there is no CUDA variant for generic aarch64
#     — so it is the correct equivalent, not a fallback.
#
# Same upstream version either way; only the index and the local
# version suffix differ. Bump both arms together.
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

# Pre-bake the BLIP weights into the image so the adapter runs with NO
# network egress under AI_SOVEREIGNTY=local_only (issue #79). Without
# this, the first /infer would download ~990 MB from huggingface.co at
# runtime — which requires relaxing the sovereignty gate. Baking at
# build time keeps the secure-by-default posture intact.
ARG BLIP_MODEL_ID=Salesforce/blip-image-captioning-base
ENV HF_HUB_DISABLE_TELEMETRY=1
RUN python -c "from transformers import BlipProcessor, BlipForConditionalGeneration; \
mid='${BLIP_MODEL_ID}'; BlipProcessor.from_pretrained(mid); BlipForConditionalGeneration.from_pretrained(mid)"

# Pin the RUNTIME fully offline so from_pretrained() loads only from the
# baked cache and never makes the default metadata/update check to
# huggingface.co — which would otherwise contradict network_egress=[]
# and stall in a network-isolated local_only deployment. Set AFTER the
# bake above (the build needs the network; the running container does not).
ENV TRANSFORMERS_OFFLINE=1 \
    HF_HUB_OFFLINE=1

# Install the OpenNVR adapter SDK so ``adapters.blip.main`` can import
# AdapterApp, BodyShape, etc.
COPY opennvr_adapter_sdk/                       ./opennvr_adapter_sdk/
RUN pip install --no-cache-dir ./opennvr_adapter_sdk

COPY adapters/blip/                             ./adapters/blip/

ENV PYTHONPATH=/app

EXPOSE 9006

HEALTHCHECK --interval=30s --timeout=3s --retries=3 \
    CMD python -c "import httpx, sys; sys.exit(0 if httpx.get('http://127.0.0.1:9006/health', timeout=2).status_code == 200 else 1)"

CMD ["uvicorn", "adapters.blip.main:app", "--host", "0.0.0.0", "--port", "9006"]
