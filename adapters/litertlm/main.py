# Copyright (c) 2026 OpenNVR
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Gemma on the Qualcomm Hexagon NPU (LiteRT-LM) — contract-compliant FastAPI
service for captions and visual question answering. A drop-in for the
``caption-adapter`` slot on Qualcomm Dragonwing boxes (Ventuno Q, IQ-8275).

Run locally (on the board):
    OPENNVR_ADAPTER_TOKEN=secret \\
    python -m uvicorn adapters.litertlm.main:app --host 0.0.0.0 --port 9020
"""
from __future__ import annotations

import logging

from adapters.litertlm.service import MAX_IMAGE_BYTES, LitertLmService
from opennvr_adapter_sdk import (
    AdapterApp,
    BodyShape,
    Cost,
    FairQueuing,
    Permissions,
    Scheduling,
)

logger = logging.getLogger(__name__)


def _build_service() -> LitertLmService:
    return LitertLmService()


_adapter_app = AdapterApp(
    service_factory=_build_service,
    name="litertlm-gemma-npu",
    version="1.0.0",
    vendor="open-nvr",
    license="Apache-2.0",
    model_card_url="https://huggingface.co/litert-community/gemma-4-E2B-it-litert-lm",
    tasks_advertised=["visual_qa", "scene_caption"],
    body_shape=BodyShape.IMAGE,
    max_body_bytes=MAX_IMAGE_BYTES,
    permissions=Permissions(
        gpu=False,                       # the Hexagon NPU, not a GPU
        # First-boot fetches only: the model from Hugging Face and six
        # Qualcomm AI Runtime libraries from Qualcomm's download server.
        # Nothing leaves the box at inference time.
        network_egress=["huggingface.co", "softwarecenter.qualcomm.com"],
        host_filesystem=["/dev/fastrpc-cdsp", "/dev/dma_heap"],
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

# The chat endpoint rides the same app (and the SDK's bearer-token
# middleware) and the same engine as the captions: adapters/litertlm/chat.py.
from adapters.litertlm.chat import router as _chat_router  # noqa: E402

app.include_router(_chat_router)
# Resolved per request, not here: the SDK builds the service in the
# lifespan, and reading it at import time trips its assertion before
# uvicorn has even bound the port (the CI smoke test caught exactly that).
app.state.litertlm_service_getter = lambda: _adapter_app.service


def __getattr__(name: str):
    if name == "_service":
        return _adapter_app.service
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
