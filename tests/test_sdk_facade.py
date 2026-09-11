# Copyright (c) 2026 OpenNVR
# SPDX-License-Identifier: Apache-2.0
"""The `Adapter` facade — publishing a model without learning the contract.

The scaffolded adapter used to be 218 lines of TODOs: a health state
machine, a fingerprint convention, a hardware-evaluation payload and an
error taxonomy, all before wrapping a single model. These tests pin what
the facade derives instead, and that what it compiles to is an ordinary
`AdapterService`.

They deliberately use the JSON/base64 route into `/infer` rather than
multipart, so they run in any environment that has the SDK's own
dependencies — `python-multipart` is a FastAPI extra, not an SDK one.
"""
from __future__ import annotations

import base64
import hashlib
from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient

from opennvr_adapter_sdk import AdapterService, InferResponse
from opennvr_adapter_sdk.adapter_app import BodyShape
from opennvr_adapter_sdk.contract import (
    ErrorCategory, HardwareEvaluationResponse, HardwareVerdict,
)
from opennvr_adapter_sdk.facade import Adapter, InferCall, Overloaded
from opennvr_adapter_sdk.service import ServiceError

FRAME = b"\xff\xd8\xff\xe0 not really a jpeg"


def detector(**kwargs) -> Adapter:
    defaults = dict(version="1.0.0", vendor="ACME", license="Apache-2.0",
                    tasks=["object_detection"], framework="onnxruntime")
    defaults.update(kwargs)
    return Adapter(defaults.pop("adapter_id", "fall-detection"), **defaults)


def post_image(client: TestClient, **params) -> "object":
    body = {"frame_b64": base64.b64encode(FRAME).decode(), **params}
    return client.post("/infer", json=body)


# ── The whole adapter, in a handful of lines ────────────────────────


def test_a_model_wrapper_is_an_adapter():
    adapter = detector()

    @adapter.load()
    def load():
        return {"session": "loaded"}

    @adapter.on_image()
    def detect(call):
        assert call.model == {"session": "loaded"}
        assert call.image == FRAME
        return [call.detection("fallen", 0.91, 0.1, 0.2, 0.3, 0.4)]

    with TestClient(adapter.app) as client:
        assert client.get("/health").json()["status"] == "ok"
        body = post_image(client).json()
        assert body["model_name"] == "fall-detection"
        assert body["result"]["detections"] == [{
            "label": "fallen", "confidence": 0.91,
            "bbox": {"x": 0.1, "y": 0.2, "w": 0.3, "h": 0.4}}]


def test_it_compiles_to_an_ordinary_adapter_service():
    """Anything that accepts an AdapterService accepts this — the
    conformance runner, a custom server, the existing tests."""
    adapter = detector()
    adapter.on_image()(lambda call: [])
    assert isinstance(adapter.service, AdapterService)
    assert adapter.service is adapter.service          # built once


def test_the_body_shape_follows_the_handler():
    for decorate, shape in (("on_image", BodyShape.IMAGE),
                            ("on_audio", BodyShape.AUDIO),
                            ("on_text", BodyShape.TEXT),
                            ("on_data", BodyShape.GENERIC)):
        adapter = detector()
        getattr(adapter, decorate)()(lambda call: {})
        document = adapter.app.openapi()
        content = document["paths"]["/infer"]["post"]["requestBody"]["content"]
        if shape is BodyShape.TEXT:
            assert set(content) == {"application/json"}
        else:
            assert "multipart/form-data" in content


def test_one_handler_only_and_the_error_says_what_to_do_instead():
    adapter = detector()
    adapter.on_image()(lambda call: [])
    with pytest.raises(RuntimeError, match="branch on call.task"):
        adapter.on_image()(lambda call: [])


def test_an_adapter_with_no_handler_refuses_to_build():
    with pytest.raises(RuntimeError, match="@adapter.on_image"):
        _ = detector().service


def test_the_id_must_be_usable_as_an_identity():
    for bad in ("Fall Detection", "fall_detection", "FallDetection", "fall--x"):
        with pytest.raises(ValueError, match="kebab-case"):
            Adapter(bad)


# ── What the facade derives ─────────────────────────────────────────


def test_the_fingerprint_is_the_weights_hash(tmp_path):
    weights = tmp_path / "model.onnx"
    weights.write_bytes(b"weights bytes")
    adapter = detector(weights=weights)
    adapter.on_image()(lambda call: [])
    expected = hashlib.sha256(b"weights bytes").hexdigest()
    assert adapter.fingerprint == f"sha256:{expected}"
    assert adapter.model_info().size_mb == 0.0


def test_the_fingerprint_is_never_null():
    """KAI-C's drift detection skips a null fingerprint, so an adapter
    with one is silently exempt from the tamper check — the template's
    old default was `return None`."""
    adapter = detector()
    adapter.on_image()(lambda call: [])
    assert adapter.fingerprint.startswith("sha256:")
    assert adapter.model_info().fingerprint == adapter.fingerprint
    # Deterministic: the same adapter always reports the same value.
    assert adapter.fingerprint == detector().fingerprint


def test_modalities_follow_the_handler_and_can_be_overridden():
    image = detector()
    image.on_image()(lambda call: [])
    assert image.model_info().modalities_in == ["image"]
    assert image.model_info().modalities_out == ["bbox_classes"]

    asr = detector(modalities_out=["text"])
    asr.on_audio()(lambda call: {})
    assert asr.model_info().modalities_in == ["audio"]
    assert asr.model_info().modalities_out == ["text"]


def test_health_follows_the_loader():
    adapter = detector()
    adapter.on_image()(lambda call: [])

    @adapter.load()
    def load():
        raise RuntimeError("weights are missing")

    service = adapter.service
    assert service.is_ready() is False           # LOADING, before load()
    service.load()
    assert service.is_ready() is False           # ERROR, after it raised
    evaluation = service.hardware_evaluation()
    assert evaluation.verdict == HardwareVerdict.BLOCKED
    assert "weights are missing" in evaluation.reasoning


def test_a_failed_load_answers_infer_with_a_retryable_error():
    adapter = detector()
    adapter.on_image()(lambda call: [])

    @adapter.load()
    def load():
        raise RuntimeError("no CUDA device")

    with TestClient(adapter.app) as client:
        response = post_image(client)
        assert response.status_code == 503
        error = response.json()["error"]
        assert error["category"] == ErrorCategory.MODEL_ERROR.value
        assert "no CUDA device" in error["message"]


def test_hardware_evaluation_is_derived_and_overridable():
    plain = detector()
    plain.on_image()(lambda call: [])
    plain.service.load()
    assert plain.service.hardware_evaluation().verdict == HardwareVerdict.OK

    for returned, expected in (
        (True, HardwareVerdict.OK),
        (False, HardwareVerdict.BLOCKED),
        (HardwareVerdict.WARN, HardwareVerdict.WARN),
        (("warn", "only 2 GB of VRAM"), HardwareVerdict.WARN),
        (HardwareEvaluationResponse(verdict="blocked", reasoning="no NPU",
                                    checked_at=datetime.now(timezone.utc)),
         HardwareVerdict.BLOCKED),
    ):
        adapter = detector()
        adapter.on_image()(lambda call: [])
        adapter.check_hardware()(lambda model, r=returned: r)
        adapter.service.load()
        assert adapter.service.hardware_evaluation().verdict == expected


def test_a_raising_hardware_check_warns_rather_than_crashing_health():
    adapter = detector()
    adapter.on_image()(lambda call: [])
    adapter.check_hardware()(lambda model: 1 / 0)
    adapter.service.load()
    evaluation = adapter.service.hardware_evaluation()
    assert evaluation.verdict == HardwareVerdict.WARN
    assert "ZeroDivisionError" in evaluation.reasoning


def test_gpu_and_egress_reach_the_capabilities_payload():
    adapter = detector(gpu=True, network_egress=["api.example.com"],
                       max_inflight=4)
    adapter.on_image()(lambda call: [])
    with TestClient(adapter.app) as client:
        caps = client.get("/capabilities").json()
    assert caps["permissions"]["gpu"] is True
    assert caps["permissions"]["network_egress"] == ["api.example.com"]
    assert caps["scheduling"]["max_inflight"] == 4


# ── The error taxonomy, without the table ───────────────────────────


@pytest.mark.parametrize("raised,status,category,transient", [
    (ValueError("frame is not a JPEG"), 400, "transport_error", False),
    (KeyError("threshold"), 400, "transport_error", False),
    (RuntimeError("CUDA out of memory"), 500, "model_error", False),
    (Overloaded("queue is full", retry_after_ms=250), 503, "overloaded", True),
])
def test_ordinary_exceptions_become_the_right_contract_error(
        raised, status, category, transient):
    adapter = detector()

    @adapter.on_image()
    def detect(call):
        raise raised

    with TestClient(adapter.app) as client:
        response = post_image(client)
    assert response.status_code == status
    error = response.json()["error"]
    assert error["category"] == category
    assert error["transient"] is transient


def test_backpressure_tells_the_caller_when_to_come_back():
    adapter = detector()

    @adapter.on_image()
    def detect(call):
        raise Overloaded(retry_after_ms=250)

    with TestClient(adapter.app) as client:
        body = post_image(client).json()
    assert body["error"]["retry_after_ms"] == 250


def test_a_precise_handler_keeps_its_own_service_error():
    """The facade classifies for you; it never overrides a handler that
    was explicit."""
    adapter = detector()

    @adapter.on_image()
    def detect(call):
        raise ServiceError(ErrorCategory.PERMISSION_DENIED, code="policy",
                           message="Site policy refuses face matching.",
                           transient=False, http_status=403)

    with TestClient(adapter.app) as client:
        response = post_image(client)
    assert response.status_code == 403
    assert response.json()["error"]["code"] == "policy"


# ── Shaping the answer ──────────────────────────────────────────────


def test_a_list_is_the_detection_convention_and_a_dict_is_verbatim():
    adapter = detector()

    @adapter.on_image()
    def detect(call):
        if call.param("raw"):
            return {"caption": "a person falling"}
        return [call.detection("fallen", 0.5, 0, 0, 1, 1)]

    with TestClient(adapter.app) as client:
        assert "detections" in post_image(client).json()["result"]
        assert post_image(client, raw=True).json()["result"] == {
            "caption": "a person falling"}


def test_a_handler_may_return_a_full_infer_response():
    adapter = detector()

    @adapter.on_image()
    def detect(call):
        return InferResponse(model_name="other", model_version="9",
                             inference_ms=7, result={"ok": True})

    with TestClient(adapter.app) as client:
        body = post_image(client).json()
    assert (body["model_name"], body["model_version"], body["inference_ms"]) == \
        ("other", "9", 7)


def test_a_handler_returning_nonsense_says_what_to_return():
    adapter = detector()
    adapter.on_image()(lambda call: "a string")
    with TestClient(adapter.app) as client:
        message = post_image(client).json()["error"]["message"]
    assert "return a list of detections" in message


def test_detections_are_clamped_to_the_normalized_frame():
    """Pixel coordinates are the most common thing to get wrong, and the
    contract requires 0–1. Clamping keeps a bad box from failing
    validation downstream — the operator sees a wrong box, not a 500."""
    item = InferCall.detection("person", 1.4, -0.2, 0.5, 3.0, 0.1)
    assert item["confidence"] == 1.0
    assert item["bbox"] == {"x": 0.0, "y": 0.5, "w": 1.0, "h": 0.1}


def test_the_call_exposes_what_arrived():
    adapter = detector()
    seen = {}

    @adapter.on_image()
    def detect(call):
        seen.update(task=call.task, camera=call.camera_id,
                    params=call.params, threshold=call.param("threshold", 0.5),
                    repr=repr(call))
        return []

    with TestClient(adapter.app) as client:
        post_image(client, task="object_detection", camera_id="cam-3",
                   threshold=0.8)
    assert seen["task"] == "object_detection"
    assert seen["camera"] == "cam-3"
    assert seen["threshold"] == 0.8
    assert "frame_b64" not in seen["params"]      # the body, not a param
    assert "task='object_detection'" in seen["repr"]


def test_a_text_adapter_reads_the_conventional_keys():
    adapter = detector(tasks=["speech_synthesis"])
    seen = []

    @adapter.on_text()
    def speak(call):
        seen.append(call.text)
        return {"audio_b64": ""}

    with TestClient(adapter.app) as client:
        client.post("/infer", json={"text": "hello"})
        client.post("/infer", json={"prompt": "hello again"})
    assert seen == ["hello", "hello again"]


# ── Lifecycle ───────────────────────────────────────────────────────


def test_shutdown_runs_with_the_model():
    adapter = detector()
    adapter.on_image()(lambda call: [])
    released = []

    @adapter.load()
    def load():
        return "session"

    @adapter.on_shutdown()
    def release(model):
        released.append(model)

    service = adapter.service
    service.load()
    service.shutdown()
    assert released == ["session"]


def test_declaring_a_stream_handler_advertises_streaming():
    plain = detector()
    plain.on_image()(lambda call: [])
    with TestClient(plain.app) as client:
        assert client.get("/capabilities").json()["endpoints"]["infer_stream"]["supported"] \
            is False

    streaming = detector()
    streaming.on_image()(lambda call: [])
    streaming.on_stream()(lambda ws: None)
    with TestClient(streaming.app) as client:
        caps = client.get("/capabilities").json()
        assert caps["endpoints"]["infer_stream"]["supported"] is True
        assert client.get("/asyncapi.json").json()["channels"]
