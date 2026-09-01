# Copyright (c) 2026 OpenNVR
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
fast-plate-ocr license-plate adapter — contract-compliant FastAPI service.

The OCR-specific logic lives in ``adapters/fast_plate_ocr/service.py``;
the SDK (``opennvr-adapter-sdk``) provides everything else — auth,
metrics, correlation_id, the six mandatory contract endpoints, body
parsing, error envelope translation, lifespan.

Run locally:
    OPENNVR_ADAPTER_TOKEN=secret \\
    python -m uvicorn adapters.fast_plate_ocr.main:app --host 0.0.0.0 --port 9004

Conformance check:
    python -m conformance http://localhost:9004 --token $OPENNVR_ADAPTER_TOKEN
"""
from __future__ import annotations

from adapters.fast_plate_ocr.service import (
    MAX_IMAGE_BYTES,
    FastPlateOcrService,
)
from opennvr_adapter_sdk import (
    AdapterApp,
    BodyShape,
    Cost,
    FairQueuing,
    Permissions,
    Scheduling,
)

_adapter_app = AdapterApp(
    # Lazy build at lifespan startup mirrors the YOLOv8 / Piper
    # adapters — lets test fixtures monkey-patch
    # FastPlateOcrService.__init__ between module load and TestClient
    # __enter__ take effect.
    service_factory=FastPlateOcrService,
    name="fast-plate-ocr",
    version="1.1.0",
    vendor="open-nvr",
    license="AGPL-3.0",
    model_card_url="https://github.com/ankandrew/fast-plate-ocr",
    tasks_advertised=["license_plate_recognition"],
    body_shape=BodyShape.IMAGE,
    max_body_bytes=MAX_IMAGE_BYTES,
    permissions=Permissions(
        # No GPU — fast-plate-ocr runs on CPU via onnxruntime, which
        # is the whole point of picking it for the LPR adapter.
        gpu=False,
        # No outbound calls at inference time. The model file is
        # downloaded on first ``load()`` by fast-plate-ocr itself
        # (~3 MB) and cached locally; subsequent runs are offline.
        # That one-time download is acceptable under operator-
        # approved permissions but not declared as ongoing egress.
        network_egress=[],
        host_filesystem=[],
        shared_memory_paths=[],
        host_metadata=False,
    ),
    scheduling=Scheduling(
        # Single ONNX session, no internal queuing. KAI-C enforces
        # the global cap from this value.
        max_inflight=1,
        preferred_batch_size=1,
        # LPR is event-driven (one inference per detected vehicle),
        # not frame-rate, but per-camera fair queuing still buys us
        # something in the busy-parking-lot case where two cameras
        # both see a wave of vehicles at the same time.
        fair_queuing=FairQueuing.PER_CAMERA,
    ),
    cost=Cost(currency="USD"),
    # Streaming WS is intentionally NOT supported. LPR is an
    # event-triggered call from the orchestrator (the
    # license-plate-recognition example app), not a per-frame
    # stream. Setting this False lets the SDK refuse /infer/stream
    # with HTTP 501 and keeps the per-call latency honest.
    supports_stream=False,
)

app = _adapter_app.fastapi_app


# PEP 562 module-level __getattr__ — exposes ``_service`` as a synthetic
# attribute so test fixtures that reach into ``main._service`` keep
# working post-SDK-refactor. Reads the live service from the lazily-
# built AdapterApp.
def __getattr__(name: str):
    if name == "_service":
        return _adapter_app.service
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
