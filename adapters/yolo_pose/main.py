# Copyright (c) 2026 OpenNVR
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
YOLO-pose human-keypoint adapter — contract-compliant FastAPI service.

This file is the ~30-line app construction §3.7 promises: everything
pose-specific lives in ``adapters/yolo_pose/service.py`` (including
the full §6 WS protocol loop), and the SDK provides the rest — auth,
metrics, correlation_id, all six contract endpoints, body parsing,
error envelope translation, lifespan.

Run locally:
    OPENNVR_ADAPTER_TOKEN=secret \\
    python -m uvicorn adapters.yolo_pose.main:app --host 0.0.0.0 --port 9009

Conformance check:
    python -m conformance http://localhost:9009 --token $OPENNVR_ADAPTER_TOKEN
"""
from __future__ import annotations

from adapters.yolo_pose.service import MAX_IMAGE_BYTES, YoloPoseService
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

    Same signal family as ``YoloPoseService._detect_gpu_in_use()`` /
    ``hardware_evaluation()`` (onnxruntime provider inspection), but
    checked against ``get_available_providers()`` at declaration time
    because the inference session doesn't exist yet. The default image
    (adapters/yolo_pose/Dockerfile) pins the CPU-only ``onnxruntime``
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
    # YoloPoseService.__init__ between module load and TestClient
    # __enter__ take effect. Production doesn't care which path is used.
    service_factory=YoloPoseService,
    name="yolo-pose-estimation",
    version="1.0.0",
    vendor="open-nvr",
    license="AGPL-3.0",
    model_card_url="https://docs.ultralytics.com/tasks/pose/",
    tasks_advertised=["pose_estimation"],
    body_shape=BodyShape.IMAGE,
    max_body_bytes=MAX_IMAGE_BYTES,
    permissions=Permissions(
        # §8 — declare build-accurately: gpu=True (an operator-approval
        # gate at KAI-C registration) only when this build can actually
        # use CUDA. The stock CPU image therefore declares gpu=False
        # and registers without a GPU-grant prompt — which is the
        # normal deployment for this adapter, since it is CPU-first by
        # design.
        gpu=_cuda_provider_available(),
        # No egress: the ONNX file is mounted (or fetched once, by the
        # operator's own explicitly-configured YOLO_POSE_MODEL_URL,
        # which is empty by default). Nothing this adapter does on the
        # steady-state path touches the network.
        network_egress=[],
        # No host_filesystem entry: as with the yolov8 adapter, the
        # weights belong in a container-owned named volume mounted at
        # /weights, not a host bind-mount, so declaring a host path
        # would only add a needless operator-approval scope
        # (§8 "declare minimally").
        host_filesystem=[],
        shared_memory_paths=[],
        host_metadata=False,
    ),
    scheduling=Scheduling(
        # max_inflight=1 is the honest value: the onnxruntime session
        # is a shared singleton and inference calls are not serialized
        # across WS streams. KAI-C uses this as its global cap per §9.
        max_inflight=1,
        preferred_batch_size=1,
        # §9 — opt in to KAI-C's per-camera fair queuing. It matters
        # more here than for an event-driven adapter: every camera
        # streaming pose at 10 fps is a steady load, and without fair
        # queuing the busiest entrance starves the others.
        fair_queuing=FairQueuing.PER_CAMERA,
    ),
    cost=Cost(currency="USD"),
    supports_stream=True,
    stream_max_concurrent=16,
    # Shared-memory fast path is documented in §6.2 but not
    # implemented. Advertise false so KAI-C never sends frame_ref.
    stream_supports_shared_memory=False,
)

app = _adapter_app.fastapi_app


# PEP 562 module-level __getattr__ — exposes ``_service`` as a synthetic
# attribute so tests (and other introspection code) that reach into
# ``main._service`` keep working. Reads the live service from the
# lazily-built AdapterApp.
def __getattr__(name: str):
    if name == "_service":
        return _adapter_app.service
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
