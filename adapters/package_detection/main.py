# Copyright (c) 2026 OpenNVR
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Package-detection adapter — contract-compliant FastAPI service.

The parcel-specific logic lives in ``adapters/package_detection/
service.py``; the SDK provides everything else (auth, metrics,
correlation_id, all six contract endpoints, body parsing, error
envelope translation, lifespan).

Run locally:
    OPENNVR_ADAPTER_TOKEN=secret \\
    python -m uvicorn adapters.package_detection.main:app --host 0.0.0.0 --port 9010

Conformance check:
    python -m conformance http://localhost:9010 --token $OPENNVR_ADAPTER_TOKEN
"""
from __future__ import annotations

import os
from urllib.parse import urlparse

from adapters.package_detection.service import (
    MAX_IMAGE_BYTES,
    MODEL_URL_ENV,
    PackageDetectionService,
)
from opennvr_adapter_sdk import (
    AdapterApp,
    BodyShape,
    Cost,
    FairQueuing,
    Permissions,
    Scheduling,
)


def _model_fetch_egress() -> list[str]:
    """The host the adapter may contact on first boot, or nothing.

    ``PACKAGE_DETECTION_MODEL_URL`` is empty by default, so the stock
    deployment declares no egress at all and stays sovereignty-clean
    for ``local_only``. An operator who points it at a weights URL gets
    that ONE host declared, because ``ensure_model_file`` will
    genuinely dial it (§8: an undeclared egress host in an audit log is
    what gets an adapter removed). A present weights file still
    short-circuits the fetch; the declaration covers the capability.
    """
    url = os.getenv(MODEL_URL_ENV, "").strip()
    if not url:
        return []
    host = urlparse(url).hostname
    return [host] if host else []


def _cuda_provider_available() -> bool:
    """True only when the installed onnxruntime build ships the CUDA
    execution provider. The default image pins the CPU-only wheel, so
    it declares gpu=False and registers without a GPU-grant prompt."""
    try:
        import onnxruntime as ort

        return "CUDAExecutionProvider" in ort.get_available_providers()
    except Exception:
        return False


_adapter_app = AdapterApp(
    service_factory=PackageDetectionService,
    name="package-detection",
    version="1.0.0",
    vendor="open-nvr",
    license="AGPL-3.0",
    model_card_url="https://github.com/open-nvr/ai-adapter/blob/main/adapters/package_detection/README.md",
    # ONLY package_detection. This model knows one class; advertising
    # object_detection as well would let it satisfy an app whose
    # ``requires_tasks`` means "people and vehicles" (intrusion, loitering)
    # and it would see nothing. An app asks for a task, never for an
    # adapter by name — so the task must mean what it says.
    tasks_advertised=["package_detection"],
    body_shape=BodyShape.IMAGE,
    max_body_bytes=MAX_IMAGE_BYTES,
    permissions=Permissions(
        gpu=_cuda_provider_available(),
        network_egress=_model_fetch_egress(),
        # The deployed weights path is a container-owned named volume
        # (see the Dockerfile and open-nvr's compose), not a host
        # bind-mount — so no host_filesystem scope (§8 "declare
        # minimally"). An operator who really bind-mounts a host
        # directory adds it here.
        host_filesystem=[],
        shared_memory_paths=[],
        host_metadata=False,
    ),
    scheduling=Scheduling(
        # One shared onnxruntime session, serial. The consumer calls a
        # few times a day per door, so this is not a throughput limit
        # in practice; it is the honest cap per §9.
        max_inflight=1,
        preferred_batch_size=1,
        fair_queuing=FairQueuing.PER_CAMERA,
    ),
    cost=Cost(currency="USD"),
    # On demand by design — the package-delivery app fetches one
    # snapshot when Tier-0 says somebody left the doorstep and on a slow
    # recheck cadence. No per-frame stream; see service.py.
    supports_stream=False,
)

app = _adapter_app.fastapi_app


def __getattr__(name: str):
    if name == "_service":
        return _adapter_app.service
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
