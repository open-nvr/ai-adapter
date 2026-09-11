# Copyright (c) 2026 __ADAPTER_NAME__ authors
# SPDX-License-Identifier: __LICENSE__
#
# __ADAPTER_NAME__ — an OpenNVR AI adapter.
#
#   docker build -t __ADAPTER_ID__ .
#   docker run -p __PORT__:__PORT__ -e OPENNVR_ADAPTER_TOKEN=dev-token __ADAPTER_ID__

FROM python:3.12-slim

# TODO: add the system libraries your runtime needs. Common ones:
#   libgl1 libglib2.0-0   (OpenCV)
#   ffmpeg                (audio)
# RUN apt-get update && apt-get install -y --no-install-recommends \
#         libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies first, so a code change does not re-download the world.
COPY pyproject.toml ./
RUN pip install --no-cache-dir "__SDK_REQUIREMENT__" "uvicorn[standard]>=0.30,<1.0" \
    # TODO: your model runtime goes here.
    && rm -rf /root/.cache/pip

COPY __ADAPTER_MODULE__.py ./
# TODO: COPY your weights, or fetch them at build time. Baking them in
# makes the image reproducible and the fingerprint stable.
# COPY models/ ./models/

# The contract's own liveness endpoint, so the orchestrator and KAI-C
# agree about when this adapter is ready.
HEALTHCHECK --interval=30s --timeout=3s --start-period=60s \
    CMD python -c "import urllib.request,sys; \
sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:__PORT__/health').status==200 else 1)"

EXPOSE __PORT__
CMD ["uvicorn", "__ADAPTER_MODULE__:app", "--host", "0.0.0.0", "--port", "__PORT__"]
