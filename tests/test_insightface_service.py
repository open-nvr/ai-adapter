# Copyright (c) 2026 OpenNVR
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Unit tests for InsightFaceService at the AdapterService level."""
from __future__ import annotations

import importlib
import sys

import pytest

from opennvr_adapter_sdk import ErrorCategory, HardwareVerdict, ServiceError
from tests._insightface_service_fixtures import (  # noqa: F401
    insightface_environment,
    install_fake_insightface,
    sample_jpeg,
)


def _build_service(env):
    if "adapters.insightface.service" in sys.modules:
        importlib.reload(sys.modules["adapters.insightface.service"])
    from adapters.insightface.face_db import FaceDB
    from adapters.insightface.service import InsightFaceService

    db = FaceDB(storage_path=str(env["db_path"]))
    return InsightFaceService(face_db=db)


# ── Load / readiness ───────────────────────────────────────────────


def test_load_marks_service_ready(insightface_environment):
    svc = _build_service(insightface_environment)
    assert not svc.is_ready()
    svc.load()
    assert svc.is_ready()


def test_load_failure_keeps_service_unhealthy(insightface_environment, monkeypatch):
    svc = _build_service(insightface_environment)
    fake_mod = sys.modules["insightface.app"]

    class _Boom:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("simulated load failure")

    monkeypatch.setattr(fake_mod, "FaceAnalysis", _Boom)
    svc.load()
    assert not svc.is_ready()
    eval_resp = svc.hardware_evaluation()
    assert eval_resp.verdict == HardwareVerdict.BLOCKED


# ── Model info / hardware evaluation ───────────────────────────────


def test_model_info_shape(insightface_environment):
    svc = _build_service(insightface_environment)
    svc.load()
    info = svc.model_info()
    assert info.framework == "onnx"
    assert info.modalities_in == ["image"]
    assert "bbox_classes" in info.modalities_out
    assert info.fingerprint is not None


def test_hardware_evaluation_includes_registered_face_count(insightface_environment):
    svc = _build_service(insightface_environment)
    svc.load()
    resp = svc.hardware_evaluation()
    assert resp.verdict == HardwareVerdict.OK
    assert resp.details["registered_faces"] == 0


# ── infer: face_detection ───────────────────────────────────────────


def test_face_detection_returns_faces_without_embedding(insightface_environment, sample_jpeg):
    svc = _build_service(insightface_environment)
    svc.load()
    resp = svc.infer({"__file__": sample_jpeg, "task": "face_detection"})
    assert resp.result["task"] == "face_detection"
    assert resp.result["face_count"] == 1
    face = resp.result["faces"][0]
    assert face["confidence"] >= 0.5
    # Embedding is stripped from the detection-task response
    # (it's a 512-d vector — clients that want it ask the
    # face_embedding task explicitly).
    assert "embedding" not in face
    assert "bbox" in face


# ── infer: face_embedding ───────────────────────────────────────────


def test_face_embedding_returns_normalised_vector(insightface_environment, sample_jpeg):
    svc = _build_service(insightface_environment)
    svc.load()
    resp = svc.infer({"__file__": sample_jpeg, "task": "face_embedding"})
    assert resp.result["task"] == "face_embedding"
    embedding = resp.result["embedding"]
    assert embedding is not None
    assert len(embedding) == 512
    # face_count exposed so /faces/register can reject multi-face uploads.
    assert resp.result["face_count"] == 1


def test_face_embedding_returns_null_when_no_face(insightface_environment, sample_jpeg, monkeypatch):
    """If the fake FaceAnalysis returns no faces, the embedding-task
    path must produce a clean null response, not crash."""
    svc = _build_service(insightface_environment)
    svc.load()
    monkeypatch.setattr(svc._face_app, "get", lambda img: [])
    resp = svc.infer({"__file__": sample_jpeg, "task": "face_embedding"})
    assert resp.result["embedding"] is None
    assert resp.result["face_bbox"] is None
    assert resp.result["face_count"] == 0
    assert "no face detected" in resp.result["message"]


def test_face_embedding_reports_face_count_for_multi_face_image(
    insightface_environment, sample_jpeg, monkeypatch
):
    """When the image contains multiple faces, the embedding task
    still returns the highest-confidence face but exposes the total
    count so /faces/register can refuse multi-face enrolment."""
    svc = _build_service(insightface_environment)
    svc.load()
    FakeFace = insightface_environment["FakeFace"]
    monkeypatch.setattr(
        svc._face_app,
        "get",
        lambda img: [
            FakeFace(det_score=0.95, embedding=insightface_environment["embedding_alice"]),
            FakeFace(det_score=0.92, embedding=insightface_environment["embedding_bob"]),
        ],
    )
    resp = svc.infer({"__file__": sample_jpeg, "task": "face_embedding"})
    assert resp.result["embedding"] is not None
    assert resp.result["face_count"] == 2


# ── infer: face_recognition (the path Smart Doorbell drives) ─────


def test_face_recognition_unknown_when_db_empty(insightface_environment, sample_jpeg):
    svc = _build_service(insightface_environment)
    svc.load()
    resp = svc.infer({"__file__": sample_jpeg, "task": "face_recognition"})
    assert resp.result["task"] == "face_recognition"
    assert resp.result["recognized"] is False
    assert resp.result["registered_faces"] == 0


def test_face_recognition_matches_registered_person(insightface_environment, sample_jpeg):
    svc = _build_service(insightface_environment)
    svc.load()
    # Register Alice with the same embedding the fake returns.
    svc.face_db.register(
        person_id="alice",
        name="Alice Smith",
        embedding=insightface_environment["embedding_alice"],
        category="family",
    )
    resp = svc.infer({"__file__": sample_jpeg, "task": "face_recognition"})
    assert resp.result["recognized"] is True
    assert resp.result["person_id"] == "alice"
    assert resp.result["name"] == "Alice Smith"
    assert resp.result["category"] == "family"
    assert resp.result["similarity"] >= 0.99


def test_face_recognition_unknown_when_only_other_person_registered(
    insightface_environment, sample_jpeg
):
    svc = _build_service(insightface_environment)
    svc.load()
    # Register only Bob; the fake face returns Alice's embedding —
    # similarity should be below threshold → unknown.
    svc.face_db.register(
        person_id="bob",
        name="Bob Jones",
        embedding=insightface_environment["embedding_bob"],
        category="family",
    )
    resp = svc.infer({"__file__": sample_jpeg, "task": "face_recognition"})
    assert resp.result["recognized"] is False
    assert resp.result["registered_faces"] == 1


def test_face_recognition_returns_no_face_when_none_detected(
    insightface_environment, sample_jpeg, monkeypatch
):
    svc = _build_service(insightface_environment)
    svc.load()
    monkeypatch.setattr(svc._face_app, "get", lambda img: [])
    resp = svc.infer({"__file__": sample_jpeg, "task": "face_recognition"})
    assert resp.result["recognized"] is False
    assert resp.result["face_bbox"] is None
    assert "no face detected" in resp.result["message"]


def test_face_recognition_honours_per_call_threshold(
    insightface_environment, sample_jpeg
):
    svc = _build_service(insightface_environment)
    svc.load()
    svc.face_db.register(
        person_id="alice",
        name="Alice Smith",
        embedding=insightface_environment["embedding_alice"],
    )
    # Tight threshold should still match (same embedding → cos = 1.0).
    strict = svc.infer({
        "__file__": sample_jpeg, "task": "face_recognition", "threshold": 0.95
    })
    assert strict.result["recognized"] is True
    assert strict.result["threshold"] == pytest.approx(0.95)


# ── Error paths ────────────────────────────────────────────────────


def test_infer_before_load_returns_transient_envelope(insightface_environment, sample_jpeg):
    svc = _build_service(insightface_environment)
    with pytest.raises(ServiceError) as excinfo:
        svc.infer({"__file__": sample_jpeg, "task": "face_detection"})
    assert excinfo.value.category == ErrorCategory.MODEL_ERROR
    assert excinfo.value.code == "model_not_loaded"
    assert excinfo.value.transient is True


def test_infer_missing_image_returns_transport_error(insightface_environment):
    svc = _build_service(insightface_environment)
    svc.load()
    with pytest.raises(ServiceError) as excinfo:
        svc.infer({"task": "face_recognition"})
    assert excinfo.value.category == ErrorCategory.TRANSPORT_ERROR
    assert excinfo.value.code == "missing_image"


def test_infer_unknown_task_returns_not_supported(insightface_environment, sample_jpeg):
    svc = _build_service(insightface_environment)
    svc.load()
    with pytest.raises(ServiceError) as excinfo:
        svc.infer({"__file__": sample_jpeg, "task": "face_verify"})  # not in v1
    assert excinfo.value.category == ErrorCategory.NOT_SUPPORTED
    assert excinfo.value.code == "unsupported_task"


@pytest.mark.parametrize("bad", ["not-a-number", [], {}])
def test_infer_invalid_threshold_type_returns_transport_error(
    insightface_environment, sample_jpeg, bad
):
    svc = _build_service(insightface_environment)
    svc.load()
    with pytest.raises(ServiceError) as excinfo:
        svc.infer({"__file__": sample_jpeg, "task": "face_recognition", "threshold": bad})
    assert excinfo.value.category == ErrorCategory.TRANSPORT_ERROR
    assert excinfo.value.code == "invalid_threshold"


def test_infer_threshold_none_uses_service_default(insightface_environment, sample_jpeg):
    """``threshold: None`` (or omitted) is NOT an error — it falls
    back to the service default. Matches REST 'missing field' semantics."""
    svc = _build_service(insightface_environment)
    svc.load()
    resp = svc.infer({"__file__": sample_jpeg, "task": "face_recognition", "threshold": None})
    assert resp.result["threshold"] == svc._recognition_threshold


@pytest.mark.parametrize("bad", [-0.01, 0.0, 1.01, 5.0])
def test_infer_threshold_out_of_range_returns_transport_error(
    insightface_environment, sample_jpeg, bad
):
    svc = _build_service(insightface_environment)
    svc.load()
    with pytest.raises(ServiceError) as excinfo:
        svc.infer({"__file__": sample_jpeg, "task": "face_recognition", "threshold": bad})
    assert excinfo.value.category == ErrorCategory.TRANSPORT_ERROR
    assert excinfo.value.code == "threshold_out_of_range"


def test_infer_decode_failure_returns_transport_error(
    insightface_environment, sample_jpeg, monkeypatch
):
    """If cv2.imdecode returns None (bad bytes), the service must
    raise TRANSPORT_ERROR with invalid_image, not crash."""
    svc = _build_service(insightface_environment)
    svc.load()
    cv2_mod = sys.modules["cv2"]
    monkeypatch.setattr(cv2_mod, "imdecode", lambda arr, flags: None)
    with pytest.raises(ServiceError) as excinfo:
        svc.infer({"__file__": sample_jpeg, "task": "face_detection"})
    assert excinfo.value.category == ErrorCategory.TRANSPORT_ERROR
    assert excinfo.value.code == "invalid_image"


def test_infer_model_runtime_failure_returns_model_error(
    insightface_environment, sample_jpeg, monkeypatch
):
    svc = _build_service(insightface_environment)
    svc.load()
    monkeypatch.setattr(svc._face_app, "get", lambda img: (_ for _ in ()).throw(RuntimeError("ONNX session died")))
    with pytest.raises(ServiceError) as excinfo:
        svc.infer({"__file__": sample_jpeg, "task": "face_detection"})
    assert excinfo.value.category == ErrorCategory.MODEL_ERROR
    assert excinfo.value.code == "inference_failed"


# ── Filtering by min_face_confidence ──────────────────────────────


def test_low_confidence_faces_are_filtered_before_recognition(
    insightface_environment, sample_jpeg, monkeypatch
):
    """Faces below the service's confidence floor must be dropped
    before the recognition / DB lookup. Otherwise we'd 'recognize'
    a wall texture."""
    svc = _build_service(insightface_environment)
    svc.load()
    # Fake returns one face with confidence well below the 0.5 floor.
    FakeFace = insightface_environment["FakeFace"]
    monkeypatch.setattr(svc._face_app, "get", lambda img: [FakeFace(det_score=0.10)])
    resp = svc.infer({"__file__": sample_jpeg, "task": "face_detection"})
    assert resp.result["face_count"] == 0
