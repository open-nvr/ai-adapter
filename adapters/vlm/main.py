# Copyright (c) 2026 OpenNVR
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Open-vocabulary detection adapter — contract-compliant FastAPI service.

The detection logic lives in ``adapters/vlm/service.py``; the SDK
provides auth, metrics, correlation_id, the six mandatory contract
endpoints, body parsing, error-envelope translation, and lifespan.

Run locally:
    OPENNVR_ADAPTER_TOKEN=secret \\
    python -m uvicorn adapters.vlm.main:app --host 0.0.0.0 --port 9012

Conformance check:
    python -m conformance http://localhost:9012 --token $OPENNVR_ADAPTER_TOKEN

Example call (multipart):
    curl -H "Authorization: Bearer $OPENNVR_ADAPTER_TOKEN" \\
      -F frame=@street.jpg -F 'queries=red truck, person on a bicycle' \\
      http://localhost:9012/infer
"""
from __future__ import annotations

import logging

from adapters.vlm.service import MAX_IMAGE_BYTES, VlmService
from opennvr_adapter_sdk import (
    AdapterApp,
    BodyShape,
    Cost,
    FairQueuing,
    Permissions,
    Scheduling,
)

logger = logging.getLogger(__name__)


def _build_service() -> VlmService:
    return VlmService()


_adapter_app = AdapterApp(
    service_factory=_build_service,
    name="vlm-open-vocab-detection",
    version="1.0.0",
    vendor="open-nvr",
    license="AGPL-3.0",
    model_card_url="https://huggingface.co/google/owlv2-base-patch16-ensemble",
    tasks_advertised=["open_vocab_detection"],
    body_shape=BodyShape.IMAGE,
    max_body_bytes=MAX_IMAGE_BYTES,
    permissions=Permissions(
        # CPU works but is slow; GPU strongly recommended for video
        # rates. Auto-uses CUDA when torch.cuda.is_available() at load.
        gpu=False,
        # First-run model download from huggingface.co. Strict-
        # sovereignty operators pre-bake the model or run a local
        # HF mirror; under AI_SOVEREIGNTY=local_only KAI-C refuses to
        # register an adapter that declares egress, so bake the weights.
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


def __getattr__(name: str):
    if name == "_service":
        return _adapter_app.service
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
