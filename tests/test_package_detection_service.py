# Copyright (c) 2026 OpenNVR
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Integration tests for the package-detection contract service
(adapters/package_detection/main.py) and its ``PackageDetectionService``.

Covers:
  - load lifecycle (fresh → loaded → idempotent → failed weights →
    never refetch present weights)
  - HTTP /health, /capabilities, /hardware/evaluation, /metrics
  - /infer output shape: §5.1 ``detections`` with the ``package``
    label, normalized boxes un-letterboxed into the SOURCE frame on a
    16:9 camera, NMS, ``count`` and ``labels`` echo, the conf / iou /
    imgsz / max_detections params
  - multi-class exports and the label map from the environment
  - the wrong-model typed failure
  - malformed-input rejection for every caller-supplied param
  - fingerprint stability and drift
  - auth + correlation_id
  - build-accurate permissions (egress derived from the model URL)

The model itself is stubbed (tests/_package_detection_service_fixtures.py)
— no ONNX weights, no network, no GPU.

Run with:

    cd ai-adapter && pytest tests/test_package_detection_service.py -v
"""
from __future__ import annotations

import importlib
import json
import os

import pytest

from opennvr_adapter_sdk import ErrorCategory, HardwareVerdict, ServiceError
from opennvr_adapter_sdk.contract import (
    CapabilitiesResponse,
    FailureEnvelope,
    HardwareEvaluationResponse,
    HealthResponse,
    InferResponse,
)
from tests._package_detection_service_fixtures import (  # noqa: F401
    FAKE_IMGSZ,
    install_fake_onnxruntime,
    package_detection_app,
    package_detection_app_with_auth,
    package_detection_environment,
    service,
    square_jpeg,
    wide_jpeg,
)


def _detections(response) -> list[dict]:
    assert response.status_code == 200, response.text
    infer = InferResponse.model_validate(response.json())
    return infer.result["detections"]


def _infer(client, jpeg: bytes, **params):
    files = {"frame": ("frame.jpg", jpeg, "image/jpeg")}
    data = {"params": json.dumps(params)} if params else None
    return client.post("/infer", files=files, data=data)


# ── Load lifecycle ─────────────────────────────────────────────────


class TestLoadLifecycle:

    def test_fresh_service_is_not_ready(self, package_detection_environment):
        from adapters.package_detection.service import PackageDetectionService

        assert not PackageDetectionService().is_ready()

    def test_load_marks_service_ready(self, service):
        assert service.is_ready()

    def test_load_is_idempotent(self, service):
        first_session = service._session
        service.load()
        assert service.is_ready()
        assert service._session is first_session

    def test_missing_weights_with_no_url_fails_typed(
        self, package_detection_environment
    ):
        """No weights and no URL is a clean load failure, not a crash
        and not a silent download."""
        import adapters.package_detection.service as service_module

        package_detection_environment["weights_path"].unlink()
        svc = service_module.PackageDetectionService()
        svc.load()

        assert not svc.is_ready()
        assert svc.hardware_evaluation().verdict == HardwareVerdict.BLOCKED
        assert "no download URL" in (svc._load_error or "")

        with pytest.raises(ServiceError) as exc_info:
            svc.infer({"__file__": b"\xff\xd8\xff"})
        envelope = exc_info.value.envelope()
        assert envelope.error.category == ErrorCategory.MODEL_ERROR
        assert envelope.error.code == "weights_missing"

    def test_present_weights_are_never_refetched(self, package_detection_environment):
        import adapters.package_detection.service as service_module

        os.environ["PACKAGE_DETECTION_MODEL_URL"] = "http://256.0.0.1/never-reachable"
        try:
            svc = service_module.PackageDetectionService()
            svc.load()
        finally:
            os.environ["PACKAGE_DETECTION_MODEL_URL"] = ""
        assert svc.is_ready()

    def test_infer_before_load_raises_model_error(self, package_detection_environment):
        from adapters.package_detection.service import PackageDetectionService

        svc = PackageDetectionService()
        with pytest.raises(ServiceError) as exc_info:
            svc.infer({"__file__": b"\xff\xd8\xff"})
        envelope = exc_info.value.envelope()
        assert envelope.error.category == ErrorCategory.MODEL_ERROR
        assert envelope.error.transient is True


# ── Fingerprint (§11.3 drift detection) ────────────────────────────


class TestFingerprint:

    def test_fingerprint_is_stable_across_loads(self, package_detection_environment):
        import adapters.package_detection.service as service_module

        a = service_module.PackageDetectionService()
        a.load()
        b = service_module.PackageDetectionService()
        b.load()
        assert a.fingerprint() == b.fingerprint()
        assert a.fingerprint().startswith("sha256:")

    def test_fingerprint_moves_when_weights_change(self, package_detection_environment, service):
        before = service.fingerprint()
        package_detection_environment["weights_path"].write_bytes(b"ROTATED_WEIGHTS")
        assert service.fingerprint() != before


# ── /health, /capabilities, /hardware/evaluation, /metrics ─────────


def test_health_returns_valid_HealthResponse(package_detection_app):
    response = package_detection_app.get("/health")
    assert response.status_code == 200
    health = HealthResponse.model_validate(response.json())
    assert health.status.value == "ok"
    assert health.adapter_name == "package-detection"


def test_capabilities_advertises_only_package_detection(package_detection_app):
    """The one class this model knows must not let it stand in for a
    general detector: ``object_detection`` is deliberately absent."""
    caps = CapabilitiesResponse.model_validate(package_detection_app.get("/capabilities").json())
    assert caps.tasks_advertised == ["package_detection"]
    assert caps.model.name == "yolov8n-package"
    assert caps.model.modalities_out == ["bbox_classes"]
    assert caps.model.fingerprint and caps.model.fingerprint.startswith("sha256:")


def test_capabilities_declares_no_streaming(package_detection_app):
    caps = CapabilitiesResponse.model_validate(package_detection_app.get("/capabilities").json())
    assert caps.endpoints.infer.supported is True
    assert caps.endpoints.infer_stream.supported is False


def test_capabilities_declares_build_accurate_permissions(package_detection_app):
    caps = CapabilitiesResponse.model_validate(package_detection_app.get("/capabilities").json())
    assert caps.permissions.gpu is False          # CPU-only fake providers
    assert caps.permissions.network_egress == []   # no model URL configured
    assert caps.permissions.host_filesystem == []


def test_hardware_evaluation_treats_cpu_as_the_design_target(package_detection_app):
    hwe = HardwareEvaluationResponse.model_validate(
        package_detection_app.get("/hardware/evaluation").json())
    assert hwe.verdict == HardwareVerdict.OK
    assert hwe.details["gpu_required"] is False
    assert hwe.details["labels"] == ["package"]


def test_metrics_emits_baseline_and_domain_metrics(package_detection_app, square_jpeg):
    assert _infer(package_detection_app, square_jpeg).status_code == 200
    body = package_detection_app.get("/metrics").text
    for name in ("adapter_infer_total", "adapter_infer_latency_seconds",
                 "adapter_model_loaded", "adapter_inflight_requests"):
        assert name in body, name
    assert 'adapter_package_frames_total{result="packages"} 1' in body
    assert 'adapter_packages_total{label="package"} 2' in body


# ── /infer output shape ────────────────────────────────────────────


class TestOutputShape:

    def test_two_parcels_found_with_defaults(self, package_detection_app, square_jpeg):
        """Prediction 1 is NMS-suppressed, 3 is below conf: two remain,
        strongest first, both labelled ``package``."""
        response = _infer(package_detection_app, square_jpeg)
        dets = _detections(response)
        assert [d["label"] for d in dets] == ["package", "package"]
        assert dets[0]["confidence"] == pytest.approx(0.91)
        assert dets[1]["confidence"] == pytest.approx(0.55)
        result = response.json()["result"]
        assert result["count"] == 2
        assert result["labels"] == ["package"]
        assert result["frame_dimensions"] == {"w": FAKE_IMGSZ, "h": FAKE_IMGSZ}
        assert dets[0]["attributes"] == {"class_id": 0}

    def test_boxes_are_normalized_to_the_source_frame(self, package_detection_app, square_jpeg):
        """416 canvas == 416 source: a 100×60 box centred at (208,208)
        is x=(208-50)/416, w=100/416."""
        box = _detections(_infer(package_detection_app, square_jpeg))[0]["bbox"]
        assert box["x"] == pytest.approx(158 / 416, abs=1e-4)
        assert box["y"] == pytest.approx(178 / 416, abs=1e-4)
        assert box["w"] == pytest.approx(100 / 416, abs=1e-4)
        assert box["h"] == pytest.approx(60 / 416, abs=1e-4)

    def test_letterbox_is_unmapped_on_a_wide_frame(self, package_detection_app, wide_jpeg):
        """640×360 letterboxed into 416: scale 0.65, pad_y = (416-234)/2
        = 91. The centred parcel must land at the centre of the source
        frame, and its width must be 100/0.65 px = 0.2404 of 640."""
        box = _detections(_infer(package_detection_app, wide_jpeg))[0]["bbox"]
        cx = box["x"] + box["w"] / 2
        cy = box["y"] + box["h"] / 2
        assert cx == pytest.approx(0.5, abs=1e-3)
        assert cy == pytest.approx(0.5, abs=1e-3)
        assert box["w"] == pytest.approx((100 / 0.65) / 640, abs=1e-3)
        assert box["h"] == pytest.approx((60 / 0.65) / 360, abs=1e-3)

    def test_response_validates_as_InferResponse(self, package_detection_app, square_jpeg):
        infer = InferResponse.model_validate(_infer(package_detection_app, square_jpeg).json())
        assert infer.model_name == "yolov8n-package"
        assert infer.inference_ms >= 0


class TestParams:

    def test_lower_conf_reveals_the_faint_box(self, package_detection_app, square_jpeg):
        dets = _detections(_infer(package_detection_app, square_jpeg, conf=0.1))
        assert len(dets) == 3
        assert dets[-1]["confidence"] == pytest.approx(0.20)

    def test_confidence_threshold_alias_is_honoured(self, package_detection_app, square_jpeg):
        """The yolov8 adapter's spelling — a caller must not have to
        know which vision adapter it is talking to."""
        dets = _detections(_infer(package_detection_app, square_jpeg, confidence_threshold=0.1))
        assert len(dets) == 3

    def test_iou_one_keeps_the_duplicate(self, package_detection_app, square_jpeg):
        dets = _detections(_infer(package_detection_app, square_jpeg, iou=1.0))
        assert len(dets) == 3
        assert dets[1]["confidence"] == pytest.approx(0.70)

    def test_max_detections_caps_the_list(self, package_detection_app, square_jpeg):
        dets = _detections(_infer(package_detection_app, square_jpeg, max_detections=1))
        assert len(dets) == 1
        assert dets[0]["confidence"] == pytest.approx(0.91)

    def test_imgsz_changes_the_blob_fed_to_the_session(self, service, square_jpeg):
        service.infer({"__file__": square_jpeg, "imgsz": 320})
        blob = service._session.received_blobs[-1]
        assert blob.shape == (1, 3, 320, 320)
        assert blob.dtype.name == "float32"
        assert 0.0 <= float(blob.min()) and float(blob.max()) <= 1.0

    def test_fixed_size_export_dictates_imgsz(self, package_detection_environment):
        import adapters.package_detection.service as service_module

        install_fake_onnxruntime(input_shape=[1, 3, 640, 640])
        importlib.reload(service_module)
        svc = service_module.PackageDetectionService()
        svc.load()
        assert svc._static_imgsz == 640
        with pytest.raises(ServiceError) as exc_info:
            svc.infer({"__file__": b"\xff\xd8\xff", "imgsz": 416})
        assert exc_info.value.envelope().error.code == "malformed_input"
        assert "fixed 640px" in exc_info.value.envelope().error.message


class TestLabelMap:

    def test_multi_class_export_uses_the_env_label_map(
        self, package_detection_environment, monkeypatch, square_jpeg
    ):
        """An operator's own fine-tune with two classes: the second
        class score column is read and its label comes from the env."""
        import numpy as np
        import adapters.package_detection.service as service_module

        module = install_fake_onnxruntime(features=6)
        monkeypatch.setenv("PACKAGE_DETECTION_LABELS", "package, envelope")
        importlib.reload(service_module)
        svc = service_module.PackageDetectionService()
        svc.load()
        # Make prediction 2 an envelope: class-1 score above class-0.
        preds = svc._session._preds
        preds[0, 5, 2] = 0.8
        preds[0, 4, 2] = 0.1
        infer = svc.infer({"__file__": square_jpeg})
        labels = [d["label"] for d in infer.result["detections"]]
        assert labels == ["package", "envelope"]
        assert infer.result["labels"] == ["package", "envelope"]
        assert isinstance(preds, np.ndarray)

    def test_index_past_the_label_map_is_named_not_dropped(
        self, package_detection_environment, square_jpeg
    ):
        import adapters.package_detection.service as service_module

        install_fake_onnxruntime(features=6)   # two classes, one label
        importlib.reload(service_module)
        svc = service_module.PackageDetectionService()
        svc.load()
        preds = svc._session._preds
        preds[0, 5, 2] = 0.8
        preds[0, 4, 2] = 0.1
        labels = [d["label"] for d in svc.infer({"__file__": square_jpeg}).result["detections"]]
        assert labels == ["package", "class_1"]

    def test_wrong_model_export_fails_typed(self, package_detection_environment, square_jpeg):
        """A graph with fewer than 5 features per row cannot be a
        detection export — a misconfiguration, reported as such."""
        import adapters.package_detection.service as service_module

        install_fake_onnxruntime(features=3)
        importlib.reload(service_module)
        svc = service_module.PackageDetectionService()
        svc.load()
        with pytest.raises(ServiceError) as exc_info:
            svc.infer({"__file__": square_jpeg})
        envelope = exc_info.value.envelope()
        assert envelope.error.category == ErrorCategory.MODEL_ERROR
        assert envelope.error.code == "package_detection.unexpected_model_output"


# ── Malformed input ────────────────────────────────────────────────


class TestMalformedInput:

    def test_missing_frame_is_rejected(self, package_detection_app):
        response = package_detection_app.post(
            "/infer", data={"params": json.dumps({})},
            files={"_marker": ("", b"", "text/plain")},
        )
        assert response.status_code == 400
        FailureEnvelope.model_validate(response.json())

    def test_undecodable_frame_is_rejected(self, package_detection_app):
        response = package_detection_app.post(
            "/infer", files={"frame": ("not.jpg", b"this is not an image", "image/jpeg")})
        assert response.status_code == 400
        envelope = FailureEnvelope.model_validate(response.json())
        assert envelope.error.code == "malformed_input"

    def test_oversized_frame_is_rejected(self, package_detection_app):
        huge = bytes(8 * 1024 * 1024 + 1)
        response = package_detection_app.post(
            "/infer", files={"frame": ("huge.jpg", huge, "image/jpeg")})
        assert response.status_code == 413

    @pytest.mark.parametrize("params", [
        {"conf": "not-a-number"}, {"conf": 1.5}, {"conf": -0.1}, {"conf": True},
        {"iou": "high"}, {"iou": 2},
        {"imgsz": 100}, {"imgsz": 417}, {"imgsz": 4096}, {"imgsz": "big"},
        {"max_detections": 0}, {"max_detections": 10_000}, {"max_detections": "many"},
    ])
    def test_bad_params_are_typed_400s(self, package_detection_app, square_jpeg, params):
        response = _infer(package_detection_app, square_jpeg, **params)
        assert response.status_code == 400, response.text
        envelope = FailureEnvelope.model_validate(response.json())
        assert envelope.error.code == "malformed_input"


# ── Auth + correlation_id ──────────────────────────────────────────


def test_auth_rejects_missing_token_on_infer(package_detection_app_with_auth, square_jpeg):
    client, _ = package_detection_app_with_auth
    assert _infer(client, square_jpeg).status_code == 401


def test_auth_accepts_valid_token_on_infer(package_detection_app_with_auth, square_jpeg):
    client, token = package_detection_app_with_auth
    response = client.post(
        "/infer", files={"frame": ("frame.jpg", square_jpeg, "image/jpeg")},
        headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 200


def test_correlation_id_echoed_on_capabilities(package_detection_app):
    response = package_detection_app.get("/capabilities", headers={"X-Correlation-Id": "abc-123"})
    assert response.headers.get("X-Correlation-Id") == "abc-123"


# ── Build-accurate permissions ─────────────────────────────────────


def test_network_egress_is_empty_without_a_configured_fetch(monkeypatch):
    monkeypatch.delenv("PACKAGE_DETECTION_MODEL_URL", raising=False)
    from adapters.package_detection.main import _model_fetch_egress
    assert _model_fetch_egress() == []


def test_configured_model_url_is_declared_as_egress(monkeypatch):
    monkeypatch.setenv(
        "PACKAGE_DETECTION_MODEL_URL",
        "https://github.com/open-nvr/ai-adapter/releases/download/x/yolov8n-package.onnx")
    from adapters.package_detection.main import _model_fetch_egress
    assert _model_fetch_egress() == ["github.com"]
