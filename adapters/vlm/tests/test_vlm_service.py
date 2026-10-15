# Copyright (c) 2026 OpenNVR
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Unit tests for VlmService at the AdapterService level.

These exercise the contract surface (query/threshold validation, the
normalized-bbox conversion, the InferResponse shape) WITHOUT loading
real transformers/torch — ``_run_detect`` is stubbed and the service is
forced ready. CPU/GPU model behaviour is out of scope for a unit test.
"""
from __future__ import annotations

import pytest

from opennvr_adapter_sdk import ErrorCategory, HealthStatus, ServiceError

from adapters.vlm.service import VlmService, _to_normalized


def _ready_service(detections=None):
    """A VlmService forced ready, with _run_detect stubbed."""
    svc = VlmService(model_id="stub/owlv2")
    svc._load_state = HealthStatus.OK
    svc._device = "cpu"
    captured = {}

    def _fake_detect(image, *, queries, threshold):
        captured["queries"] = queries
        captured["threshold"] = threshold
        return detections if detections is not None else [
            {"label": queries[0], "confidence": 0.9,
             "bbox": {"x": 0.1, "y": 0.1, "w": 0.2, "h": 0.3}},
        ]

    svc._run_detect = _fake_detect  # type: ignore[assignment]

    # Patch image decode so we don't need a real JPEG.
    import adapters.vlm.service as mod

    class _FakeImage:
        size = (640, 480)

    mod._decode_image = lambda b: _FakeImage()  # type: ignore[assignment]
    return svc, captured


# ── bbox normalization ─────────────────────────────────────────────


def test_to_normalized_basic():
    bb = _to_normalized(64, 48, 320, 240, 640, 480)
    assert bb == {"x": 0.1, "y": 0.1, "w": 0.4, "h": 0.4}


def test_to_normalized_clamps_out_of_frame():
    bb = _to_normalized(-10, -10, 700, 500, 640, 480)
    assert bb["x"] == 0.0 and bb["y"] == 0.0
    assert bb["w"] == 1.0 and bb["h"] == 1.0


# ── infer happy path ───────────────────────────────────────────────


def test_infer_returns_detections():
    svc, captured = _ready_service()
    resp = svc.infer({"__file__": b"\xff\xd8stub", "queries": ["red truck", "person"]})
    assert resp.result["task"] == "open_vocab_detection"
    assert captured["queries"] == ["red truck", "person"]
    assert resp.result["detections"][0]["label"] == "red truck"


def test_queries_accepts_comma_separated_string():
    # Multipart form fields arrive as strings; "a, b" → ["a", "b"].
    svc, captured = _ready_service()
    svc.infer({"__file__": b"x", "queries": "red truck, person on a bicycle"})
    assert captured["queries"] == ["red truck", "person on a bicycle"]


# ── validation errors ──────────────────────────────────────────────


def test_missing_image_raises_transport_error():
    svc, _ = _ready_service()
    with pytest.raises(ServiceError) as ei:
        svc.infer({"queries": ["person"]})
    assert ei.value.category == ErrorCategory.TRANSPORT_ERROR


def test_missing_queries_raises():
    svc, _ = _ready_service()
    with pytest.raises(ServiceError) as ei:
        svc.infer({"__file__": b"x"})
    assert ei.value.code == "missing_queries"


def test_too_many_queries_raises():
    svc, _ = _ready_service()
    with pytest.raises(ServiceError) as ei:
        svc.infer({"__file__": b"x", "queries": [f"q{i}" for i in range(64)]})
    assert ei.value.code == "too_many_queries"


def test_threshold_out_of_range_raises():
    svc, _ = _ready_service()
    with pytest.raises(ServiceError) as ei:
        svc.infer({"__file__": b"x", "queries": ["person"], "threshold": 2.0})
    assert ei.value.code == "threshold_out_of_range"


def test_not_ready_raises_model_error():
    svc = VlmService(model_id="stub/owlv2")  # still LOADING
    with pytest.raises(ServiceError) as ei:
        svc.infer({"__file__": b"x", "queries": ["person"]})
    assert ei.value.category == ErrorCategory.MODEL_ERROR


def test_run_detect_uses_the_post_processor_transformers_5_ships(monkeypatch):
    """transformers 5 dropped ``post_process_object_detection`` from the
    OWL-ViT processors (``post_process_grounded_object_detection`` since
    4.5x). The first published image 500'd on every /infer with
    ``'Owlv2Processor' object has no attribute
    'post_process_object_detection'``; the stubbed tests never saw it.
    This processor has only the new name, as the pinned image's does."""
    import types

    import adapters.vlm.service as mod

    class _Box:
        def __init__(self, v):
            self._v = v

        def tolist(self):
            return self._v

    class _Processor:
        def __call__(self, *, text, images, return_tensors):
            return types.SimpleNamespace(to=lambda device: {"pixel_values": None})

        def post_process_grounded_object_detection(self, *, outputs, target_sizes, threshold):
            return [{"scores": [0.7], "labels": [1], "boxes": [_Box([10.0, 20.0, 60.0, 80.0])]}]

    class _Model:
        def __call__(self, **inputs):
            return object()

    class _Tensor:
        def to(self, device):
            return self

    fake_torch = types.SimpleNamespace(
        no_grad=lambda: __import__("contextlib").nullcontext(),
        tensor=lambda v: _Tensor(),
    )
    monkeypatch.setitem(__import__("sys").modules, "torch", fake_torch)

    svc = VlmService()
    svc._processor = _Processor()
    svc._model = _Model()
    svc._device = "cpu"
    image = types.SimpleNamespace(size=(100, 200))

    out = VlmService._run_detect(svc, image, queries=["car", "person"], threshold=0.1)

    assert out == [{
        "label": "person",
        "confidence": 0.7,
        "bbox": mod._to_normalized(10.0, 20.0, 60.0, 80.0, 100, 200),
    }]
