# Copyright (c) 2026 OpenNVR
# SPDX-License-Identifier: Apache-2.0

"""
SmolVLM2 VLM adapter — GGUF + mmproj via llama.cpp multimodal, contract-compliant.

On-demand VQA/captioning of a single camera frame. Mount the GGUF + projector at
OPENNVR_VLM_MODEL_PATH / OPENNVR_VLM_MMPROJ_PATH; bundle the ``llama-server`` binary.
Torch-free, CPU-first, GPU-optional via SMOLVLM_GPU_LAYERS.

Run:
    OPENNVR_VLM_MODEL_PATH=/models/SmolVLM2-2.2B-Instruct-Q4_K_M.gguf \\
    OPENNVR_VLM_MMPROJ_PATH=/models/mmproj-SmolVLM2-2.2B-Instruct-f16.gguf \\
    LLAMACPP_SERVER_BIN=/bin/llama-server \\
    python -m uvicorn adapters.smolvlm.main:app --host 0.0.0.0 --port 9016
Conformance:
    python -m conformance http://localhost:9016 --token $OPENNVR_ADAPTER_TOKEN
"""
from __future__ import annotations

from adapters.smolvlm.service import MAX_IMAGE_BYTES, SmolVlmService
from opennvr_adapter_sdk import (
    AdapterApp,
    BodyShape,
    Cost,
    FairQueuing,
    Permissions,
    Scheduling,
)

_adapter_app = AdapterApp(
    service_factory=SmolVlmService,
    name="smolvlm-vlm",
    version="1.0.0",
    vendor="camera-agent-lite",
    license="Apache-2.0",
    model_card_url="https://huggingface.co/ggml-org/SmolVLM2-2.2B-Instruct-GGUF",
    tasks_advertised=["scene_caption", "visual_question_answering"],
    body_shape=BodyShape.IMAGE,
    max_body_bytes=MAX_IMAGE_BYTES,
    permissions=Permissions(
        gpu=False,
        network_egress=[],
        # GGUF + mmproj download from HuggingFace ONCE on first boot into the
        # weights volume (whisper-adapter precedent). Pre-populate /app/models
        # (or set OPENNVR_VLM_MODEL_URL="" / _MMPROJ_URL="") for offline
        # installs.
        host_filesystem=["/app/models"],
        shared_memory_paths=[],
        host_metadata=False,
    ),
    scheduling=Scheduling(
        max_inflight=1,
        preferred_batch_size=1,
        fair_queuing=FairQueuing.PER_CAMERA,
    ),
    cost=Cost(currency="USD"),
    supports_stream=False,
)

app = _adapter_app.fastapi_app


def __getattr__(name: str):
    if name == "_service":
        return _adapter_app.service
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
