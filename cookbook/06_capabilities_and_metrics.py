# Copyright (c) 2026 OpenNVR
# SPDX-License-Identifier: Apache-2.0
"""What an operator sees — `/capabilities` and `/metrics`.

Demonstrates: `CapabilitiesResponse`, `AdapterInfo`, `ModelInfo`,
`Accelerator`, `Permissions`, `Scheduling`, `Cost`, `EndpointsInfo`,
`HealthResponse`, `Metrics`, `register_counter`, `inc_counter`,
`register_histogram`, `observe`, `set_queue_depth`.

Two audiences. `/capabilities` is read by KAI-C, to decide whether to
route work here and to notice the model changing underneath a running
deployment. `/metrics` is read by the operator's Prometheus, and is the
only view they have of whether the model is keeping up.
"""
from opennvr_adapter_sdk import Adapter

adapter = Adapter(
    "demo-metrics",
    version="1.0.0",
    tasks=["object_detection"],
    framework="onnxruntime",
    # Everything here lands in /capabilities. Declare honestly: KAI-C
    # refuses to register an adapter asking for more than the operator
    # granted, so over-declaring blocks the deployment, and
    # under-declaring means the permission you need is not there.
    gpu=True,
    network_egress=["huggingface.co"],
    max_inflight=2,
)


@adapter.load()
def load():
    """The SDK already exports the §3.4 baseline metrics — in-flight
    calls, inference latency and outcome per task, a model-loaded gauge,
    and `adapter_model_info` identity labels carrying the fingerprint.

    Register your own here for what only this model knows."""
    metrics = adapter.service.metrics

    metrics.register_counter(
        "adapter_frames_skipped_total",
        "Frames dropped before inference because they were too dark.")
    metrics.register_histogram(
        "adapter_preprocess_seconds",
        "Time spent resizing and normalising, excluding the model.",
        buckets=(0.001, 0.005, 0.01, 0.05, 0.1),
    )
    return object()


@adapter.on_image()
def detect(call):
    metrics = adapter.service.metrics

    if _too_dark(call.image):
        # A counter an operator can alert on beats a log line nobody
        # reads: "this camera has been sending unusable frames since 3am".
        metrics.inc_counter("adapter_frames_skipped_total")
        return []

    metrics.observe("adapter_preprocess_seconds", 0.004)
    # Depth the SDK cannot know, because only you own the queue.
    metrics.set_queue_depth(0)
    return [call.detection("person", 0.9, 0.1, 0.1, 0.2, 0.3)]


def what_kaic_reads() -> dict:
    """`GET /capabilities`, in full — identity, model, accelerator,
    permissions, scheduling, cost and endpoints.

    The field that matters most is `model.fingerprint`: KAI-C records it
    at registration and on every poll, and a change is a tamper signal
    that raises an audit event."""
    with _client() as client:
        return client.get("/capabilities").json()


def _too_dark(frame: bytes) -> bool:
    return False


def _client():
    from fastapi.testclient import TestClient

    return TestClient(adapter.app)
