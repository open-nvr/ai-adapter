# Copyright (c) 2026 OpenNVR
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Unit tests for FastPlateOcrService at the AdapterService level
(no HTTP). Mirrors test_piper_service.py / test_whisper_service.py.
"""
from __future__ import annotations

import importlib
import sys

import pytest

from opennvr_adapter_sdk import ErrorCategory, HardwareVerdict, ServiceError
from tests._fast_plate_ocr_service_fixtures import (  # noqa: F401
    fast_plate_ocr_environment,
    install_fake_fast_plate_ocr,
    sample_jpeg,
)


def _build_service(env):
    """Force-reload the service module so the patched ``fast_plate_ocr``
    import in the fixture takes effect, then construct a service."""
    if "adapters.fast_plate_ocr.service" in sys.modules:
        importlib.reload(sys.modules["adapters.fast_plate_ocr.service"])
    from adapters.fast_plate_ocr.service import FastPlateOcrService

    return FastPlateOcrService()


# ── load / readiness ─────────────────────────────────────────────────


def test_load_marks_service_ready(fast_plate_ocr_environment):
    svc = _build_service(fast_plate_ocr_environment)
    assert not svc.is_ready()
    svc.load()
    assert svc.is_ready()


def test_load_is_idempotent(fast_plate_ocr_environment):
    svc = _build_service(fast_plate_ocr_environment)
    svc.load()
    svc.load()  # must not raise
    assert svc.is_ready()


def test_load_failure_keeps_service_unhealthy(
    fast_plate_ocr_environment, monkeypatch
):
    svc = _build_service(fast_plate_ocr_environment)
    import adapters.fast_plate_ocr.service as service_module

    class _Boom:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("simulated load failure")

    fake_module = sys.modules["fast_plate_ocr"]
    monkeypatch.setattr(fake_module, "LicensePlateRecognizer", _Boom)

    svc.load()
    assert not svc.is_ready()
    eval_resp = svc.hardware_evaluation()
    assert eval_resp.verdict == HardwareVerdict.BLOCKED
    assert "simulated load failure" in (eval_resp.reasoning or "")


# ── fingerprint ──────────────────────────────────────────────────────


def test_fingerprint_uses_on_disk_sha256_when_model_path_set(
    fast_plate_ocr_environment,
):
    svc = _build_service(fast_plate_ocr_environment)
    svc.load()
    fp = svc.fingerprint()
    assert fp is not None
    # When the recognizer exposes ``model_path``, we hash the file.
    assert fp.startswith("sha256:")
    # 64 hex digits after the prefix.
    assert len(fp) == len("sha256:") + 64


def test_fingerprint_falls_back_to_synthetic_when_path_absent(monkeypatch):
    """If the recognizer doesn't expose any documented model-path
    attribute, fingerprint() returns the identifier-derived synthetic
    so KAI-C still gets a stable identity."""
    # Install a fake that doesn't set model_path.
    import types as _types
    import sys as _sys

    class _NoPathRecognizer:
        def __init__(self, model_id, *a, **kw):
            self.model_id = model_id
        def run(self, *a, **kw):
            return [("X", 1.0)]

    mod = _types.ModuleType("fast_plate_ocr")
    mod.LicensePlateRecognizer = _NoPathRecognizer
    _sys.modules["fast_plate_ocr"] = mod

    if "adapters.fast_plate_ocr.service" in _sys.modules:
        importlib.reload(_sys.modules["adapters.fast_plate_ocr.service"])
    from adapters.fast_plate_ocr.service import FastPlateOcrService

    svc = FastPlateOcrService()
    svc.load()
    fp = svc.fingerprint()
    assert fp is not None
    assert fp.startswith("sha256-id:")


# ── model_info / hardware_evaluation ─────────────────────────────────


def test_model_info_shape(fast_plate_ocr_environment):
    svc = _build_service(fast_plate_ocr_environment)
    svc.load()
    info = svc.model_info()
    assert info.framework == "onnx"
    assert info.modalities_in == ["image"]
    assert info.modalities_out == ["text"]
    assert info.fingerprint is not None


def test_hardware_evaluation_ok_after_load(fast_plate_ocr_environment):
    svc = _build_service(fast_plate_ocr_environment)
    svc.load()
    resp = svc.hardware_evaluation()
    assert resp.verdict == HardwareVerdict.OK
    assert resp.details["model_id"] == "cct-xs-v1-global-model"


def test_hardware_evaluation_warn_before_load(fast_plate_ocr_environment):
    svc = _build_service(fast_plate_ocr_environment)
    resp = svc.hardware_evaluation()
    assert resp.verdict == HardwareVerdict.WARN


# ── infer happy path ────────────────────────────────────────────────


def test_infer_returns_plate_text_and_characters(
    fast_plate_ocr_environment, sample_jpeg
):
    svc = _build_service(fast_plate_ocr_environment)
    svc.load()
    resp = svc.infer({"__file__": sample_jpeg})
    assert resp.result["plate_text"] == "ABC1234"
    assert resp.result["confidence"] == pytest.approx(0.93, rel=1e-3)
    assert resp.result["accepted"] is True
    assert resp.result["min_confidence_applied"] == pytest.approx(0.30)
    assert len(resp.result["characters"]) == len("ABC1234")
    assert all("char" in c and "confidence" in c for c in resp.result["characters"])
    assert resp.result["model_id"] == "cct-xs-v1-global-model"
    # inference_ms is a top-level field on InferResponse (per §3.5),
    # not a nested result key.
    assert resp.inference_ms >= 0
    assert resp.model_name == "cct-xs-v1-global-model"
    assert resp.model_version.startswith("fast-plate-ocr/")


def test_infer_per_call_min_confidence_marks_low_score_not_accepted(
    fast_plate_ocr_environment, sample_jpeg
):
    svc = _build_service(fast_plate_ocr_environment)
    svc.load()
    # Mutate the fake recognizer's next output to a low-confidence
    # candidate, then ask for a strict threshold.
    fast_plate_ocr_environment["fake_recognizer_cls"].next_output = ("XY99", 0.25)
    resp = svc.infer({"__file__": sample_jpeg, "min_confidence": 0.5})
    assert resp.result["plate_text"] == "XY99"
    assert resp.result["confidence"] == pytest.approx(0.25, rel=1e-3)
    assert resp.result["accepted"] is False
    assert resp.result["min_confidence_applied"] == pytest.approx(0.5)


# ── infer error paths ───────────────────────────────────────────────


def test_infer_before_load_returns_model_error(fast_plate_ocr_environment, sample_jpeg):
    svc = _build_service(fast_plate_ocr_environment)
    with pytest.raises(ServiceError) as excinfo:
        svc.infer({"__file__": sample_jpeg})
    assert excinfo.value.category == ErrorCategory.MODEL_ERROR
    assert excinfo.value.code == "model_not_loaded"
    # Before load(), the service is in HealthStatus.LOADING, so the
    # error should be transient and carry a retry hint.
    assert excinfo.value.transient is True


def test_infer_missing_image_returns_transport_error(fast_plate_ocr_environment):
    svc = _build_service(fast_plate_ocr_environment)
    svc.load()
    with pytest.raises(ServiceError) as excinfo:
        svc.infer({})
    assert excinfo.value.category == ErrorCategory.TRANSPORT_ERROR
    assert excinfo.value.code == "missing_image"


def test_infer_garbage_bytes_returns_transport_error(fast_plate_ocr_environment):
    """Request body that isn't a valid image must surface as
    TRANSPORT_ERROR(invalid_image), not bubble up as MODEL_ERROR from
    inside the recognizer (which would happen if the decode step
    were missing and we passed raw bytes through)."""
    svc = _build_service(fast_plate_ocr_environment)
    svc.load()
    with pytest.raises(ServiceError) as excinfo:
        svc.infer({"__file__": b"this-is-not-a-jpeg-or-anything-else"})
    assert excinfo.value.category == ErrorCategory.TRANSPORT_ERROR
    assert excinfo.value.code == "invalid_image"
    assert excinfo.value.http_status == 400


def test_infer_passes_decoded_ndarray_to_recognizer(
    fast_plate_ocr_environment, sample_jpeg,
):
    """Regression: the adapter must call recognizer.run() with a
    decoded numpy.ndarray, NOT the raw request bytes.
    fast-plate-ocr 1.x's run() signature is
    ``source: str | ndarray | list[...]`` — passing bytes used to
    explode inside the library. Capture what we sent and assert."""
    import numpy as np

    svc = _build_service(fast_plate_ocr_environment)
    svc.load()
    fake_cls = fast_plate_ocr_environment["fake_recognizer_cls"]
    fake_cls.last_run_source = None  # reset between tests

    svc.infer({"__file__": sample_jpeg})

    received = fake_cls.last_run_source
    assert received is not None, "recognizer.run() was never called"
    assert isinstance(received, np.ndarray), (
        f"recognizer.run() got {type(received).__name__}, expected ndarray"
    )
    # cv2.IMREAD_COLOR → 3-channel HxWx3 BGR. _tiny_jpeg_bytes() is
    # a 64x32 Pillow-generated JPEG.
    assert received.ndim == 3
    assert received.shape == (32, 64, 3), (
        f"expected (32, 64, 3) BGR ndarray; got {received.shape}"
    )


@pytest.mark.parametrize("bad_value", ["not-a-number", None, [], {}])
def test_infer_invalid_min_confidence_type_returns_transport_error(
    fast_plate_ocr_environment, sample_jpeg, bad_value
):
    svc = _build_service(fast_plate_ocr_environment)
    svc.load()
    with pytest.raises(ServiceError) as excinfo:
        svc.infer({"__file__": sample_jpeg, "min_confidence": bad_value})
    assert excinfo.value.category == ErrorCategory.TRANSPORT_ERROR
    assert excinfo.value.code == "invalid_min_confidence"


@pytest.mark.parametrize("bad_value", [-0.01, 1.01, 5.0, -1.0])
def test_infer_min_confidence_out_of_range_returns_transport_error(
    fast_plate_ocr_environment, sample_jpeg, bad_value
):
    svc = _build_service(fast_plate_ocr_environment)
    svc.load()
    with pytest.raises(ServiceError) as excinfo:
        svc.infer({"__file__": sample_jpeg, "min_confidence": bad_value})
    assert excinfo.value.category == ErrorCategory.TRANSPORT_ERROR
    assert excinfo.value.code == "min_confidence_out_of_range"


def test_infer_recognizer_raises_returns_model_error(
    fast_plate_ocr_environment, sample_jpeg, monkeypatch
):
    svc = _build_service(fast_plate_ocr_environment)
    svc.load()

    def _boom(*a, **kw):
        raise RuntimeError("ONNX session died")

    monkeypatch.setattr(svc._recognizer, "run", _boom)

    with pytest.raises(ServiceError) as excinfo:
        svc.infer({"__file__": sample_jpeg})
    assert excinfo.value.category == ErrorCategory.MODEL_ERROR
    assert excinfo.value.code == "ocr_failed"
    assert excinfo.value.transient is True
    assert "ONNX session died" in (excinfo.value.message or "")


# ── output parser ───────────────────────────────────────────────────


def test_parse_recognizer_output_handles_list_of_tuples(
    fast_plate_ocr_environment, sample_jpeg
):
    """fast-plate-ocr returns various shapes across versions; the
    parser must cope with each."""
    from adapters.fast_plate_ocr.service import _parse_recognizer_output

    text, chars, conf = _parse_recognizer_output([("ABC123", 0.88)])
    assert text == "ABC123"
    assert conf == pytest.approx(0.88)
    assert [c["char"] for c in chars] == list("ABC123")


def test_parse_recognizer_output_handles_bare_string():
    from adapters.fast_plate_ocr.service import _parse_recognizer_output

    text, chars, conf = _parse_recognizer_output("DEF456")
    assert text == "DEF456"
    assert conf == pytest.approx(1.0)


def test_parse_recognizer_output_strips_whitespace():
    from adapters.fast_plate_ocr.service import _parse_recognizer_output

    text, _, _ = _parse_recognizer_output("  ABC1234  ")
    assert text == "ABC1234"


def test_parse_recognizer_output_handles_plate_prediction_object():
    """fast-plate-ocr v2 returns a list of ``PlatePrediction`` objects
    with ``.plate`` + ``.confidence`` attributes — the real upstream
    default. Stubs that hand back tuples won't catch this; a
    duck-typed object will."""
    from adapters.fast_plate_ocr.service import _parse_recognizer_output

    class _FakePrediction:
        def __init__(self, plate: str, confidence: float):
            self.plate = plate
            self.confidence = confidence

    text, chars, conf = _parse_recognizer_output([_FakePrediction("XY1234", 0.81)])
    assert text == "XY1234"
    assert conf == pytest.approx(0.81)
    assert [c["char"] for c in chars] == list("XY1234")


def test_parse_recognizer_output_handles_dict_shape():
    """Some forks return prediction dicts. Same as the attribute
    path but addressed by keys."""
    from adapters.fast_plate_ocr.service import _parse_recognizer_output

    text, _, conf = _parse_recognizer_output(
        [{"plate": "AAA-000", "confidence": 0.77}]
    )
    assert text == "AAA-000"
    assert conf == pytest.approx(0.77)


def test_parse_recognizer_output_handles_parallel_arrays_shape():
    """Some library variants return ``([texts], [confs])``."""
    from adapters.fast_plate_ocr.service import _parse_recognizer_output

    text, _, conf = _parse_recognizer_output((["BBB-111", "CCC-222"], [0.66, 0.55]))
    assert text == "BBB-111"
    assert conf == pytest.approx(0.66)


# ── Plate localization (v1.1 — the vehicle-crop fix) ─────────────────
#
# Field failure: the platform sends whole-VEHICLE crops, and pure OCR
# on those hallucinates ("1023" off a van with an illegible plate)
# while visible plates go unread. The detector stage crops the plate
# first; when nothing is found, whole-image OCR runs at a RAISED floor
# so scene noise can't pass as a read.


def test_detected_plate_is_cropped_before_ocr(
    fast_plate_ocr_environment, sample_jpeg,
):
    det_cls = fast_plate_ocr_environment["fake_detector_cls"]
    det_cls.next_detections = [(0.8, (16, 8, 48, 24))]
    svc = _build_service(fast_plate_ocr_environment)
    svc.load()
    fake_cls = fast_plate_ocr_environment["fake_recognizer_cls"]
    fake_cls.last_run_source = None

    resp = svc.infer({"__file__": sample_jpeg})

    received = fake_cls.last_run_source
    # box 32x16 + margins (10% x → 3px, 20% y → 3px) = 38x22 crop
    assert received.shape == (22, 38, 3), (
        f"OCR must run on the plate crop, got {received.shape}")
    det = resp.result["plate_detection"]
    assert det["found"] is True
    assert det["box"] == [16, 8, 48, 24]
    assert det["confidence"] == 0.8
    # Localized read keeps the caller's floor.
    assert resp.result["min_confidence_applied"] == 0.30


def test_no_detection_raises_the_floor(
    fast_plate_ocr_environment, sample_jpeg,
):
    """No plate localized → whole-image OCR at the raised floor: a
    0.93 'read' passes (a genuine tight crop), a 0.5 one — the
    hallucination band — does not, even though it clears the default
    0.30 caller floor."""
    det_cls = fast_plate_ocr_environment["fake_detector_cls"]
    det_cls.next_detections = []
    svc = _build_service(fast_plate_ocr_environment)
    svc.load()
    fake_cls = fast_plate_ocr_environment["fake_recognizer_cls"]

    fake_cls.next_output = ("ABC1234", 0.93)
    resp = svc.infer({"__file__": sample_jpeg})
    assert resp.result["min_confidence_applied"] == 0.75
    assert resp.result["accepted"] is True
    assert resp.result["plate_detection"]["found"] is False
    assert resp.result["plate_detection"]["attempted"] is True

    fake_cls.next_output = ("1023", 0.5)     # the exact field garbage
    resp = svc.infer({"__file__": sample_jpeg})
    assert resp.result["accepted"] is False
    fake_cls.next_output = ("ABC1234", 0.93)  # restore for other tests


def test_detection_below_confidence_is_ignored(
    fast_plate_ocr_environment, sample_jpeg,
):
    det_cls = fast_plate_ocr_environment["fake_detector_cls"]
    det_cls.next_detections = [(0.10, (16, 8, 48, 24))]  # under 0.35
    svc = _build_service(fast_plate_ocr_environment)
    svc.load()
    resp = svc.infer({"__file__": sample_jpeg})
    assert resp.result["plate_detection"]["found"] is False
    assert resp.result["min_confidence_applied"] == 0.75


def test_detector_disabled_keeps_callers_floor(
    fast_plate_ocr_environment, sample_jpeg, monkeypatch,
):
    """OPENNVR_LPR_DETECTOR='' is an explicit operator assertion that
    inputs are tight plate crops — pure-OCR mode with the caller's
    floor, exactly the pre-1.1 behaviour."""
    monkeypatch.setenv("OPENNVR_LPR_DETECTOR", "")
    svc = _build_service(fast_plate_ocr_environment)
    svc.load()
    resp = svc.infer({"__file__": sample_jpeg})
    assert resp.result["plate_detection"]["attempted"] is False
    assert resp.result["min_confidence_applied"] == 0.30
    assert resp.result["accepted"] is True


def test_detector_load_failure_degrades_not_dies(
    fast_plate_ocr_environment, sample_jpeg,
):
    """A configured detector that cannot load must NOT take the
    adapter down (OCR is the core; detection is an enhancer) — but
    degraded mode keeps the raised floor, because nobody vouched for
    the inputs being crops."""
    det_cls = fast_plate_ocr_environment["fake_detector_cls"]
    det_cls.raise_on_create = True
    svc = _build_service(fast_plate_ocr_environment)
    svc.load()
    assert svc.is_ready() is True
    resp = svc.infer({"__file__": sample_jpeg})
    assert resp.result["plate_detection"]["attempted"] is False
    assert resp.result["min_confidence_applied"] == 0.75


def test_edge_detection_box_clamps_to_frame(
    fast_plate_ocr_environment, sample_jpeg,
):
    """A detection at the frame corner must clamp, not crash or wrap."""
    det_cls = fast_plate_ocr_environment["fake_detector_cls"]
    det_cls.next_detections = [(0.9, (0, 0, 20, 10))]
    svc = _build_service(fast_plate_ocr_environment)
    svc.load()
    fake_cls = fast_plate_ocr_environment["fake_recognizer_cls"]
    fake_cls.last_run_source = None
    resp = svc.infer({"__file__": sample_jpeg})
    received = fake_cls.last_run_source
    # margins: mx=2, my=2 → x1,y1 clamp at 0; x2=22, y2=12
    assert received.shape == (12, 22, 3)
    assert resp.result["plate_detection"]["found"] is True
