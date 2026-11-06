# Gemma on the Qualcomm Hexagon NPU through LiteRT-LM — the caption/VQA adapter
# for Qualcomm Dragonwing boxes (Ventuno Q / IQ-8275, QCS8275, Hexagon v75).
#
# arm64 only. The NPU-only model (2.7 GB) is NOT baked: it is downloaded once
# into the model volume at first start when OPENNVR_LITERTLM_MODEL_URL is set.
# Qualcomm's AI Runtime libraries are fetched once at first start too (their
# terms do not allow redistribution). What IS baked: the Python package and the
# Linux arm64 NPU build of the LiteRT-LM native library with its Qualcomm
# dispatch library (Apache-2.0), pinned by SHA-256.
#
#   docker build --platform linux/arm64 -f adapters/litertlm/Dockerfile .
#
# The build-and-push-qualcomm job passes ACCEL=qualcomm like the ONNX adapters;
# this image has one flavour, so the argument is accepted and ignored.
FROM python:3.12-slim AS runtime
ARG ACCEL=qualcomm

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl libatomic1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
RUN pip install --no-cache-dir \
        fastapi==0.127.0 \
        uvicorn==0.40.0 \
        pydantic==2.12.5 \
        httpx==0.28.1 \
        python-multipart==0.0.20 \
        "pillow>=10.4,<11" \
        litert-lm-api==0.18.0

# The NPU build of the LiteRT-LM runtime + LiteRT's Qualcomm dispatch library.
ARG LITERTLM_BUNDLE_URL="https://github.com/DenisovAV/flutter_edge_ai/releases/download/native-v0.18.0-e/litertlm-linux_arm64.tar.gz"
ARG LITERTLM_BUNDLE_SHA256="29c2306f8d42d88b500f033acf878109a9de8e58d34eb0bc75c464c536be3214"
RUN set -eu; mkdir -p /opt/npu/lib /opt/npu/litertlm /opt/npu/dispatch /models /tmp/bundle; \
    curl -fSL --retry 3 "$LITERTLM_BUNDLE_URL" -o /tmp/bundle.tar.gz; \
    echo "${LITERTLM_BUNDLE_SHA256}  /tmp/bundle.tar.gz" | sha256sum -c -; \
    tar -xzf /tmp/bundle.tar.gz -C /tmp/bundle; \
    pick() { cp "$(find /tmp/bundle -name "$1" -type f | head -n1)" "$2/$1"; echo "$3  $2/$1" | sha256sum -c -; }; \
    pick libLiteRtLm.so                      /opt/npu/litertlm 5a64a057bbdf8baeea5c77e78e3c470680d9a4b5629ae98db629341f8113de39; \
    pick libLiteRt.so                        /opt/npu/litertlm eb5120d0887d8b532518a7f025900ba9b4c4ac6f43e71ca3570a7ba0bb726a24; \
    pick libGemmaModelConstraintProvider.so  /opt/npu/litertlm 016f5d615a6009e8c5f3a1c3700291be3f94a8638f423460953e1c290b451f43; \
    pick libLiteRtDispatch_Qualcomm.so       /opt/npu/dispatch 06a9be66f6e8bc58de6ec020844b381dfe3df10f891dbec0f4aaeeb68c7d269e; \
    rm -rf /tmp/bundle /tmp/bundle.tar.gz

# Linux reads LD_LIBRARY_PATH at process start: the Qualcomm libraries land in
# /opt/npu/lib (a volume, fetched at first start) before the engine dlopens
# them, which is enough. /opt/npu/dispatch holds the baked dispatch library and
# libatomic so an empty volume cannot hide them; the service copies the
# dispatch library next to the Qualcomm ones (LiteRT wants one directory).
# libcdsprpc.so comes from the host (compose mounts it into /usr/lib/aarch64-linux-gnu).
RUN ln -sf /usr/lib/aarch64-linux-gnu/libatomic.so.1 /opt/npu/dispatch/libatomic.so.1
ENV LD_LIBRARY_PATH=/opt/npu/lib:/opt/npu/dispatch:/opt/npu/litertlm \
    ADSP_LIBRARY_PATH=/opt/npu/lib \
    OPENNVR_LITERTLM_NPU_LIB_DIR=/opt/npu/lib \
    OPENNVR_LITERTLM_DISPATCH_DIR=/opt/npu/dispatch \
    OPENNVR_LITERTLM_RUNTIME_DIR=/opt/npu/litertlm \
    OPENNVR_LITERTLM_MODEL_PATH=/models/gemma-4-E2B-it_Qualcomm_QCS8275.litertlm

COPY opennvr_adapter_sdk/                       ./opennvr_adapter_sdk/
RUN pip install --no-cache-dir ./opennvr_adapter_sdk
COPY adapters/litertlm/                         ./adapters/litertlm/

ENV PYTHONPATH=/app \
    OPENNVR_ADAPTER_TOKEN=""
VOLUME ["/models", "/opt/npu/lib"]
EXPOSE 9020
HEALTHCHECK --interval=30s --timeout=3s --start-period=120s --retries=5 \
    CMD python -c "import httpx, sys; sys.exit(0 if httpx.get('http://127.0.0.1:9020/health', timeout=2).status_code == 200 else 1)"
CMD ["uvicorn", "adapters.litertlm.main:app", "--host", "0.0.0.0", "--port", "9020"]
