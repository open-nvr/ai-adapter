# Copyright (c) 2026 OpenNVR
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
YOLOv8 object-detection adapter — contract-compliant FastAPI service.

Migrated to ``opennvr-adapter-sdk`` in the SDK migration. This file is now the
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
from opennvr_adapter_sdk import (
    AdapterApp,
    BodyShape,
    Cost,
    FairQueuing,
    Permissions,
    Scheduling,
)


def _cuda_provider_available() -> bool:
    """True only when the installed onnxruntime build ships the CUDA
    execution provider — i.e. a GPU image built with ``onnxruntime-gpu``.

    Same signal family as ``YoloV8Service._detect_gpu_in_use()`` /
    ``hardware_evaluation()`` (onnxruntime provider inspection), but
    checked against ``get_available_providers()`` at declaration time
    because the inference session doesn't exist yet. The default image
    (adapters/yolov8/Dockerfile) pins the CPU-only ``onnxruntime``
    wheel, which never lists CUDAExecutionProvider — so the CPU image
    declares gpu=False and only a GPU build declares gpu=True.
    """
    try:
        import onnxruntime as ort

        return "CUDAExecutionProvider" in ort.get_available_providers()
    except Exception:
        return False


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
        # §8 — declare build-accurately: gpu=True (an operator-approval
        # gate at KAI-C registration) only when this build can actually
        # use CUDA. The stock CPU image therefore declares gpu=False
        # and registers without a GPU-grant prompt.
        gpu=_cuda_provider_available(),
        network_egress=[],
        # No host_filesystem entry: the weights are not a host
        # bind-mount. In the OpenNVR stack (open-nvr/docker-compose.yml)
        # the ``yolov8-adapter`` service mounts the container-owned
        # named volume ``opennvr_yolov8_weights`` at /app/model_weights,
        # populated by the ``yolov8-weights-init`` one-shot from the
        # pre-baked ghcr.io/open-nvr/yolov8-weights image — no host
        # path is ever exposed, so declaring one would only add a
        # needless operator-approval scope (§8 "declare minimally").
        host_filesystem=[],
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
    # a planned follow-up will land shm support; bump
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
