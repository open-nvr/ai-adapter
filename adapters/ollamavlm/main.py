# Copyright (c) 2026 OpenNVR
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Ollama-backed VLM adapter — contract-compliant FastAPI service.

Drop-in for the camera-agent's ``CAPTION_ADAPTER`` slot (same tasks and
result keys as moondream/blip), but inference happens on whatever Ollama
endpoint OPENNVR_OLLAMA_VLM_URL points at — the host machine's GPU when
that endpoint is a host-side Ollama (Metal on Apple Silicon), which is
the whole point on macOS/Windows where in-container inference is
CPU-only. See adapters/ollamavlm/service.py for the design decisions.

Run locally:
    OPENNVR_ADAPTER_TOKEN=secret \\
    OPENNVR_OLLAMA_VLM_URL=http://localhost:11434 \\
    python -m uvicorn adapters.ollamavlm.main:app --host 0.0.0.0 --port 9018

Conformance check:
    python -m conformance http://localhost:9018 --token $OPENNVR_ADAPTER_TOKEN
"""
from __future__ import annotations

import logging
import os

from adapters.ollamavlm.service import MAX_IMAGE_BYTES, OllamaVlmService
from opennvr_adapter_sdk import (
    AdapterApp,
    BodyShape,
    Cost,
    FairQueuing,
    Permissions,
    Scheduling,
)

logger = logging.getLogger(__name__)


def _build_service() -> OllamaVlmService:
    return OllamaVlmService()


_adapter_app = AdapterApp(
    service_factory=_build_service,
    name="ollamavlm",
    version="1.0.0",
    vendor="open-nvr",
    license="AGPL-3.0",
    model_card_url="https://ollama.com/library",
    tasks_advertised=["visual_qa", "scene_caption"],
    body_shape=BodyShape.IMAGE,
    max_body_bytes=MAX_IMAGE_BYTES,
    permissions=Permissions(
        gpu=False,
        # DECLARED egress: every inference goes to the configured Ollama
        # endpoint — host.docker.internal (the host machine) by default.
        # Under a strict local_only posture the sovereignty layer judges
        # this URL like any other adapter endpoint; frames go nowhere
        # else, and nothing here talks to the internet at runtime except
        # the optional first-boot model pull, which Ollama itself
        # performs from ITS host-side store.
        network_egress=[os.getenv("OPENNVR_OLLAMA_VLM_URL",
                                  "http://host.docker.internal:11434")],
        host_filesystem=[],
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
