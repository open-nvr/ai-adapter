# Copyright (c) 2026 OpenNVR
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
YOLOv8 object-detection adapter — contract-compliant FastAPI service.

Migrated to ``opennvr-adapter-sdk`` in A2.3b. This file is now the
minimum viable adapter §3.7 promised: ~30 lines of FastAPI app
construction. The YOLOv8-specific logic lives in
``adapters/yolov8/service.py`` (including the full §6 WS protocol
loop); the SDK provides everything else (auth, metrics,
correlation_id, all six contract endpoints, body parsing, error
envelope translation, lifespan).

Run locally:
    OPENNVR_ADAPTER_TOKEN=secret \\
    python -m uvicorn adapters.yolov8.main:app --host 0.0.0.0 --port 9002

Conformance check:
    python -m conformance http://localhost:9002 --token $OPENNVR_ADAPTER_TOKEN
"""
from __future__ import annotations

from adapters.yolov8.service import MAX_IMAGE_BYTES, YoloV8Service
from app.config import MODEL_WEIGHTS_DIR
from opennvr_adapter_sdk import (
    AdapterApp,
    BodyShape,
    Cost,
    FairQueuing,
    Permissions,
    Scheduling,
)

_adapter_app = AdapterApp(
    # ``service_factory`` (lazy build at lifespan startup) instead of
    # eager ``service=`` so test fixtures that monkey-patch
    # YoloV8Service.__init__ between module load and TestClient
    # __enter__ take effect. Production doesn't care which path is used.
    service_factory=YoloV8Service,
    name="yolov8-object-detection",
    version="1.0.0",
    vendor="open-nvr",
    license="AGPL-3.0",
    model_card_url="https://github.com/ultralytics/ultralytics",
    tasks_advertised=["object_detection"],
    body_shape=BodyShape.IMAGE,
    max_body_bytes=MAX_IMAGE_BYTES,
    permissions=Permissions(
        # §8 — GPU permission requires operator approval at KAI-C
        # registration time.
        gpu=True,
        network_egress=[],
        # Match pre-A2.3b: advertise the weights *directory* (no
        # trailing slash) — KAI-C / operator policy comparison is
        # string-equality on this value.
        host_filesystem=[MODEL_WEIGHTS_DIR],
        shared_memory_paths=[],
        host_metadata=False,
    ),
    scheduling=Scheduling(
        # max_inflight=1 is the honest value for v1: the underlying
        # onnxruntime session is a shared singleton and we don't
        # serialize inference calls across WS streams. KAI-C uses this
        # as its global cap per §9.
        max_inflight=1,
        preferred_batch_size=1,
        # §9 — opt in to KAI-C's per-camera fair queuing so one
        # chatty camera can't starve the rest.
        fair_queuing=FairQueuing.PER_CAMERA,
    ),
    cost=Cost(currency="USD"),
    supports_stream=True,
    stream_max_concurrent=16,
    # Shared-memory fast path is documented in §6.2 but not yet
    # implemented. Advertise false so KAI-C never sends frame_ref.
    # A2.2b will land shm support; bump
    # ``stream_supports_shared_memory=True`` then.
    stream_supports_shared_memory=False,
)

app = _adapter_app.fastapi_app


# PEP 562 module-level __getattr__ — exposes ``_service`` as a synthetic
# attribute so tests (and other introspection code) that reach into
# ``main._service`` keep working post-SDK-refactor. Reads the live
# service from the lazily-built AdapterApp.
def __getattr__(name: str):
    if name == "_service":
        return _adapter_app.service
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
