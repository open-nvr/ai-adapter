# Copyright (c) 2026 OpenNVR
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
ByteTrack multi-object tracking adapter — contract-compliant FastAPI
service.

The tracking logic lives in ``adapters/bytetrack/service.py``; the SDK
provides everything else — auth, metrics, correlation_id, the six
mandatory contract endpoints, body parsing, error envelope translation,
lifespan.

Run locally:
    OPENNVR_ADAPTER_TOKEN=secret \\
    python -m uvicorn adapters.bytetrack.main:app --host 0.0.0.0 --port 9007

Conformance check:
    python -m conformance http://localhost:9007 --token $OPENNVR_ADAPTER_TOKEN
"""
from __future__ import annotations

import logging

from adapters.bytetrack.service import MAX_PAYLOAD_BYTES, ByteTrackService
from opennvr_adapter_sdk import (
    AdapterApp,
    BodyShape,
    Cost,
    FairQueuing,
    Permissions,
    Scheduling,
)

logger = logging.getLogger(__name__)


def _build_service() -> ByteTrackService:
    """Factory used by AdapterApp's lazy lifespan."""
    return ByteTrackService()


_adapter_app = AdapterApp(
    service_factory=_build_service,
    name="bytetrack-multi-object-tracker",
    version="1.0.0",
    vendor="open-nvr",
    license="AGPL-3.0",
    model_card_url="https://github.com/roboflow/supervision",
    tasks_advertised=["multi_object_tracking"],
    body_shape=BodyShape.TEXT,
    max_body_bytes=MAX_PAYLOAD_BYTES,
    permissions=Permissions(
        # CPU-only, no GPU acceleration available in supervision's
        # ByteTrack. Operators don't need to grant any escalated
        # permissions.
        gpu=False,
        # No outbound network — tracking is purely local.
        network_egress=[],
        host_filesystem=[],
        shared_memory_paths=[],
        host_metadata=False,
    ),
    scheduling=Scheduling(
        # ByteTrack updates are stateful PER CAMERA. Two concurrent
        # /infer calls for DIFFERENT cameras are safe to interleave;
        # two for the SAME camera would race the tracker's internal
        # state. KAI-C's fair_queuing=per_camera serialises calls per
        # camera_id, which is exactly what we need — set max_inflight
        # to a reasonable concurrency for cross-camera parallelism.
        max_inflight=4,
        preferred_batch_size=1,
        fair_queuing=FairQueuing.PER_CAMERA,
    ),
    cost=Cost(currency="USD"),
    # No streaming protocol — tracking is per-frame request/response.
    supports_stream=False,
)

app = _adapter_app.fastapi_app


# PEP 562 module-level __getattr__ — exposes ``_service`` as a synthetic
# attribute so test fixtures that reach into ``main._service`` keep
# working. Same idiom as the other contract adapters.
def __getattr__(name: str):
    if name == "_service":
        return _adapter_app.service
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
