# Copyright (c) 2026 OpenNVR
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
__SLUG__ adapter — contract-compliant FastAPI service.

The inference logic lives in ``adapters/__DIR_NAME__/service.py``;
the SDK provides everything else — auth, metrics, correlation_id, the
six mandatory contract endpoints, body parsing, error envelope
translation, lifespan.

Run locally:
    OPENNVR_ADAPTER_TOKEN=secret \\
    python -m uvicorn adapters.__DIR_NAME__.main:app --host 0.0.0.0 --port __PORT__

Conformance check:
    python -m conformance http://localhost:__PORT__ --token $OPENNVR_ADAPTER_TOKEN
"""
from __future__ import annotations

import logging

from adapters.__DIR_NAME__.service import __SERVICE_CLASS__
from opennvr_adapter_sdk import (
    AdapterApp,
    BodyShape,
    Cost,
    FairQueuing,
    Permissions,
    Scheduling,
)

logger = logging.getLogger(__name__)


def _build_service() -> __SERVICE_CLASS__:
    """Factory used by AdapterApp's lazy lifespan."""
    return __SERVICE_CLASS__()


_adapter_app = AdapterApp(
    service_factory=_build_service,
    # TODO: pick a stable name. Becomes the value in /capabilities.adapter.name.
    name="__SLUG__",
    version="1.0.0",
    # TODO: set vendor + license to match your distribution policy.
    vendor="open-nvr",
    license="AGPL-3.0",
    # TODO: link to your model card / weights provenance. Operators
    # use this to audit what's running.
    model_card_url=None,
    # TODO: list the §5.x task convention(s) this adapter advertises.
    # Examples: "object_detection", "asr", "scene_caption",
    # "multi_object_tracking". Match an existing convention if you can.
    # Empty list is the right placeholder — KAI-C refuses to route
    # work to an adapter with no advertised tasks, which is exactly
    # the failure mode you want if you forget to fill this in.
    tasks_advertised=[],
    body_shape=BodyShape.__BODY_SHAPE__,
    # TODO: tune to your input. 8 MiB is reasonable for image adapters;
    # text-only adapters typically want ~1 MiB; audio depends on
    # max-clip duration.
    max_body_bytes=8 * 1024 * 1024,
    permissions=Permissions(
        # TODO: declare every permission your adapter needs. KAI-C
        # refuses to register an adapter requesting more than the
        # operator has granted, so over-declaring blocks deployment.
        gpu=False,
        network_egress=[],
        host_filesystem=[],
        shared_memory_paths=[],
        host_metadata=False,
    ),
    scheduling=Scheduling(
        # TODO: max_inflight=1 is the honest default for most adapters;
        # bump only if your model is genuinely re-entrant. fair_queuing
        # =per_camera asks KAI-C to round-robin frames across cameras
        # under contention — almost always what you want.
        max_inflight=1,
        preferred_batch_size=1,
        fair_queuing=FairQueuing.PER_CAMERA,
    ),
    cost=Cost(currency="USD"),
    # TODO: flip to True if your adapter implements the §6 WebSocket
    # streaming protocol in service.py's handle_stream(). For HTTP-only
    # adapters, leave False — the /infer/stream endpoint returns 501.
    supports_stream=False,
)

app = _adapter_app.fastapi_app


# PEP 562 module-level __getattr__ — exposes ``_service`` as a synthetic
# attribute so test fixtures that reach into ``main._service`` keep
# working. Standard idiom across all contract adapters.
def __getattr__(name: str):
    if name == "_service":
        return _adapter_app.service
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
