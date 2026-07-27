# syntax=docker/dockerfile:1.7
# SmolVLM2 VLM adapter — GGUF + mmproj via llama.cpp multimodal. Torch-free.
#
#   docker build -f adapters/smolvlm/Dockerfile -t opennvr/smolvlm-adapter:1.0.0 .
#   docker run --rm -p 9016:9016 -e OPENNVR_ADAPTER_TOKEN=<token> \
#     -v $(pwd)/model_weights/smolvlm:/app/models:ro opennvr/smolvlm-adapter:1.0.0

# ── native llama-server (multimodal) ───────────────────────────────────
FROM ghcr.io/ggml-org/llama.cpp:server AS llamacpp

# ── slim python runtime + adapter ──────────────────────────────────────
FROM python:3.11-slim AS runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=llamacpp /app/llama-server /usr/local/bin/llama-server
COPY --from=llamacpp /app/*.so* /usr/local/lib/
# ggml discovers its backend libs by scanning NEXT TO the llama-server binary
# (see the llamacpp adapter Dockerfile for the full story).
RUN ldconfig && ln -sf /usr/local/lib/libggml*.so* /usr/local/bin/

WORKDIR /app
RUN pip install --no-cache-dir \
        fastapi==0.127.0 uvicorn==0.40.0 pydantic==2.12.5 \
        httpx==0.28.1 python-multipart==0.0.20 Pillow==11.0.0

COPY opennvr_adapter_sdk/  ./opennvr_adapter_sdk/
RUN pip install --no-cache-dir ./opennvr_adapter_sdk
COPY adapters/smolvlm/     ./adapters/smolvlm/

ENV PYTHONPATH=/app \
    LD_LIBRARY_PATH=/usr/local/lib \
    OPENNVR_ADAPTER_TOKEN="" \
    OPENNVR_VLM_MODEL_PATH=/app/models/SmolVLM2-2.2B-Instruct-Q4_K_M.gguf \
    OPENNVR_VLM_MMPROJ_PATH=/app/models/mmproj-SmolVLM2-2.2B-Instruct-f16.gguf \
    LLAMACPP_SERVER_BIN=/usr/local/bin/llama-server \
    SMOLVLM_GPU_LAYERS=0

EXPOSE 9016
HEALTHCHECK --interval=30s --timeout=3s --retries=3 --start-period=180s \
    CMD python -c "import httpx,sys; sys.exit(0 if httpx.get('http://127.0.0.1:9016/health', timeout=2).status_code==200 else 1)"
CMD ["uvicorn", "adapters.smolvlm.main:app", "--host", "0.0.0.0", "--port", "9016"]
