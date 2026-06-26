# Copyright (c) 2026 OpenNVR
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Moondream2 vision-language adapter — contract-compliant FastAPI service.

Visual question answering + captioning. Drop-in for the BLIP ``caption_adapter``
slot in the camera-agent, but it can actually *answer questions* about the scene
("what is the person wearing?") — the model conversations-about-video need.

Run locally:
    OPENNVR_ADAPTER_TOKEN=secret \\
    python -m uvicorn adapters.moondream.main:app --host 0.0.0.0 --port 9008

Conformance check:
    python -m conformance http://localhost:9008 --token $OPENNVR_ADAPTER_TOKEN
"""
from __future__ import annotations

import logging

from adapters.moondream.service import MAX_IMAGE_BYTES, MoondreamService
from opennvr_adapter_sdk import (
    AdapterApp,
    BodyShape,
    Cost,
    FairQueuing,
    Permissions,
    Scheduling,
)

logger = logging.getLogger(__name__)


def _build_service() -> MoondreamService:
    return MoondreamService()


_adapter_app = AdapterApp(
    service_factory=_build_service,
    name="moondream-vlm",
    version="1.0.0",
    vendor="open-nvr",
    license="Apache-2.0",
    model_card_url="https://huggingface.co/vikhyatk/moondream2",
    tasks_advertised=["visual_qa", "scene_caption"],
    body_shape=BodyShape.IMAGE,
    max_body_bytes=MAX_IMAGE_BYTES,
    permissions=Permissions(
        # CPU works; GPU auto-used when torch.cuda.is_available() at load.
        gpu=False,
        # No runtime egress: weights are baked into the image at build time
        # (see Dockerfile) so it runs under AI_SOVEREIGNTY=local_only.
        network_egress=[],
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
