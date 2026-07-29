# syntax=docker/dockerfile:1.7
# llama.cpp LLM adapter — Qwen2.5-3B GGUF, contract-compliant. Torch-free.
#
# Build (context = ai-adapter repo root, so the SDK is copyable):
#   docker build -f adapters/llamacpp/Dockerfile -t opennvr/llamacpp-adapter:1.0.0 .
#
# Run (CPU; mount the GGUF):
#   docker run --rm -p 9014:9014 \
#     -e OPENNVR_ADAPTER_TOKEN=<token> \
#     -v $(pwd)/model_weights/llamacpp:/app/models:ro \
#     opennvr/llamacpp-adapter:1.0.0
#
# GPU: add --gpus all and -e LLAMACPP_GPU_LAYERS=999 (needs a CUDA llama.cpp base).

# ── Stage 1: the native llama-server binary + its shared libs ──────────
FROM ghcr.io/ggml-org/llama.cpp:server AS llamacpp

# ── Stage 2: slim python runtime + the adapter ─────────────────────────
FROM python:3.11-slim AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Bring in the llama.cpp server binary + shared libraries from the official
# image. Paths follow the ggml-org server image layout (binary + *.so in /app);
# adjust if the upstream layout changes.
COPY --from=llamacpp /app/llama-server /usr/local/bin/llama-server
COPY --from=llamacpp /app/*.so* /usr/local/lib/
# ggml discovers its backend libs by scanning NEXT TO the llama-server binary,
# not the linker path — without these symlinks it reports "no backends are
# loaded" and llama-server exits 1 (previously worked around per-deploy in
# compose ``command:`` overrides).
RUN ldconfig && ln -sf /usr/local/lib/libggml*.so* /usr/local/bin/

WORKDIR /app

RUN pip install --no-cache-dir \
        fastapi==0.127.0 \
        uvicorn==0.40.0 \
        pydantic==2.12.5 \
        httpx==0.28.1 \
        python-multipart==0.0.20

COPY opennvr_adapter_sdk/  ./opennvr_adapter_sdk/
RUN pip install --no-cache-dir ./opennvr_adapter_sdk
COPY adapters/llamacpp/    ./adapters/llamacpp/

ENV PYTHONPATH=/app \
    LD_LIBRARY_PATH=/usr/local/lib \
    OPENNVR_ADAPTER_TOKEN="" \
    OPENNVR_LLM_MODEL_PATH=/app/models/Qwen2.5-3B-Instruct-Q4_K_M.gguf \
    LLAMACPP_SERVER_BIN=/usr/local/bin/llama-server \
    LLAMACPP_GPU_LAYERS=0

EXPOSE 9014

HEALTHCHECK --interval=30s --timeout=3s --retries=3 --start-period=180s \
    CMD python -c "import httpx,sys; sys.exit(0 if httpx.get('http://127.0.0.1:9014/health', timeout=2).status_code==200 else 1)"

CMD ["uvicorn", "adapters.llamacpp.main:app", "--host", "0.0.0.0", "--port", "9014"]
