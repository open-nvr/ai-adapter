# Copyright (c) 2026 OpenNVR
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
BLIP scene-captioning adapter — contract-compliant FastAPI service.

The captioning logic lives in ``adapters/blip/service.py``; the SDK
provides everything else — auth, metrics, correlation_id, the six
mandatory contract endpoints, body parsing, error envelope
translation, lifespan.

Run locally:
    OPENNVR_ADAPTER_TOKEN=secret \\
    python -m uvicorn adapters.blip.main:app --host 0.0.0.0 --port 9006

Conformance check:
    python -m conformance http://localhost:9006 --token $OPENNVR_ADAPTER_TOKEN
"""
from __future__ import annotations

import logging

from adapters.blip.service import MAX_IMAGE_BYTES, BlipService
from opennvr_adapter_sdk import (
    AdapterApp,
    BodyShape,
    Cost,
    FairQueuing,
    Permissions,
    Scheduling,
)

logger = logging.getLogger(__name__)


def _build_service() -> BlipService:
    """Factory used by AdapterApp's lazy lifespan."""
    return BlipService()


_adapter_app = AdapterApp(
    service_factory=_build_service,
    name="blip-scene-caption",
    version="1.0.0",
    vendor="open-nvr",
    license="AGPL-3.0",
    model_card_url="https://huggingface.co/Salesforce/blip-image-captioning-base",
    tasks_advertised=["scene_caption"],
    body_shape=BodyShape.IMAGE,
    max_body_bytes=MAX_IMAGE_BYTES,
    permissions=Permissions(
        # CPU works. GPU is automatically used when
        # ``torch.cuda.is_available()`` returns True at load time;
        # operators with NVIDIA passthrough get GPU inference without
        # extra config. Override via the standard torch device-
        # selection env vars (CUDA_VISIBLE_DEVICES, etc.) — the
        # service doesn't define adapter-specific device knobs.
        gpu=False,
        # First-run model download from huggingface.co. Operators
        # with strict sovereignty either pre-bake the model or
        # operate a local HF mirror.
        network_egress=["huggingface.co", "*.huggingface.co"],
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


# PEP 562 module-level __getattr__ — exposes ``_service`` as a
# synthetic attribute so test fixtures that reach into
# ``main._service`` keep working. Same idiom as the InsightFace /
# YOLOv8 / fast-plate-ocr adapters.
def __getattr__(name: str):
    if name == "_service":
        return _adapter_app.service
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
