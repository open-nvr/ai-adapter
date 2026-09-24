# Copyright (c) 2026 OpenNVR
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
CLIP embedding adapter — contract-compliant FastAPI service.

The embedding logic lives in ``adapters/clip/service.py``; the SDK
provides everything else (auth, metrics, correlation_id, all six
contract endpoints, body parsing, error envelope translation,
lifespan).

Run locally:
    OPENNVR_ADAPTER_TOKEN=secret \\
    python -m uvicorn adapters.clip.main:app --host 0.0.0.0 --port 9011

Conformance check:
    python -m conformance http://localhost:9011 --token $OPENNVR_ADAPTER_TOKEN
"""
from __future__ import annotations

import os

from adapters.clip.service import (
    CACHE_DIR_ENV,
    DEFAULT_CACHE_DIR,
    MAX_IMAGE_BYTES,
    ClipEmbeddingService,
)
from opennvr_adapter_sdk import (
    AdapterApp,
    BodyShape,
    Cost,
    FairQueuing,
    Permissions,
    Scheduling,
)

#: Where the weights come from on a cold start.
_WEIGHTS_HOST = "huggingface.co"


def _weights_fetch_egress() -> list[str]:
    """The host this adapter may contact, or nothing at all.

    Declared only when a fetch could actually happen. A populated cache
    directory means the adapter never dials anything, so it registers
    with an empty egress list and stays ``local_only``-clean — which is
    the posture an air-gapped site needs, and the one they will check.

    Pre-populating that cache is the supported air-gap path: mount it,
    and this list is empty on every boot.
    """
    cache = os.getenv(CACHE_DIR_ENV, "").strip() or DEFAULT_CACHE_DIR
    try:
        populated = os.path.isdir(cache) and any(os.scandir(cache))
    except OSError:
        populated = False
    if populated:
        return []
    # An operator can also assert "never download" explicitly, which is
    # stronger than a cache that happens to be full right now.
    if os.getenv("CLIP_OFFLINE", "").strip().lower() in {"1", "true", "yes"}:
        return []
    return [_WEIGHTS_HOST]


def _cuda_available() -> bool:
    """True only when torch actually sees a CUDA device. The default
    image is the CPU wheel, so it declares gpu=False and registers
    without a GPU-grant prompt."""
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


_adapter_app = AdapterApp(
    service_factory=ClipEmbeddingService,
    name="clip",
    version="1.0.0",
    vendor="open-nvr",
    license="AGPL-3.0",
    model_card_url="https://github.com/open-nvr/ai-adapter/blob/main/adapters/clip/README.md",
    # ONLY embed. It is tempting to also advertise image_captioning —
    # CLIP can rank captions — but an app asking for a caption expects a
    # sentence, and this returns 512 floats. A task must mean what it
    # says, or an app's `requires_tasks` becomes a lie.
    tasks_advertised=["embed"],
    # TEXT, not IMAGE, and this is the one genuinely unusual line in the
    # file. IMAGE makes the binary mandatory, and half of this adapter's
    # job is answering a request that carries no image at all — the
    # operator's query. service.py's module docstring has the full
    # reasoning, including why the size guard moves into the service.
    body_shape=BodyShape.TEXT,
    max_body_bytes=MAX_IMAGE_BYTES,
    permissions=Permissions(
        gpu=_cuda_available(),
        network_egress=_weights_fetch_egress(),
        # Weights live in a container-owned volume, not a host
        # bind-mount, so no host_filesystem scope is claimed (§8
        # "declare minimally").
        host_filesystem=[],
        shared_memory_paths=[],
        host_metadata=False,
    ),
    scheduling=Scheduling(
        # One model instance, called serially. Honest rather than
        # optimistic: torch will already be using several threads per
        # call, so admitting more concurrency here would oversubscribe
        # the same cores and make every caller slower while looking
        # like more throughput.
        max_inflight=1,
        preferred_batch_size=1,
        # Enrichment arrives in bursts when several cameras finish
        # tracks together; per-camera queuing stops one busy camera
        # starving the rest.
        fair_queuing=FairQueuing.PER_CAMERA,
    ),
    cost=Cost(currency="USD"),
    # One vector per finished visit, plus one per search that uses
    # words. Both are request/response; there is nothing to stream.
    supports_stream=False,
)

app = _adapter_app.fastapi_app


def __getattr__(name: str):
    if name == "_service":
        return _adapter_app.service
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
