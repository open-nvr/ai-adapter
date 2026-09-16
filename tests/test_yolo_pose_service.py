# Copyright (c) 2026 OpenNVR
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Integration tests for the YOLO-pose contract service
(adapters/yolo_pose/main.py) and its ``YoloPoseService``.

Covers:
  - load lifecycle (fresh → loaded → idempotent → failed weights)
  - HTTP /health, /capabilities, /hardware/evaluation, /metrics
  - /infer output shape: the documented ``persons`` / ``keypoints``
    contract, COCO-17 ordering, pixel coordinates un-letterboxed back
    into the SOURCE frame, NMS, conf / iou / imgsz / max_persons
  - malformed-input rejection for every caller-supplied param
  - /infer/stream — §6 WebSocket protocol roundtrip
  - fingerprint stability and drift
  - auth + correlation_id (mirroring the yolov8 suite)

The model itself is stubbed (tests/_yolo_pose_service_fixtures.py) —
no ONNX weights, no network, no GPU. What's under test is the wrapper:
everything between the request bytes and the response JSON.

Run with:

    cd ai-adapter && pytest tests/test_yolo_pose_service.py -v
"""
from __future__ import annotations

import base64
import json

import pytest

from opennvr_adapter_sdk import ErrorCategory, HardwareVerdict, ServiceError
from opennvr_adapter_sdk.contract import (
    CapabilitiesResponse,
    FailureEnvelope,
    HardwareEvaluationResponse,
    HealthResponse,
    InferResponse,
    StreamCloseCode,
)
from adapters.yolo_pose.coco_keypoints import COCO_KEYPOINTS
from tests._yolo_pose_service_fixtures import (  # noqa: F401
    FAKE_IMGSZ,
    LEFT_WRIST,
    NOSE,
    RIGHT_WRIST,
    install_fake_onnxruntime,
    service,
    square_jpeg,
    wide_jpeg,
    yolo_pose_app,
    yolo_pose_app_with_auth,
    yolo_pose_environment,
)


def _persons(response) -> list[dict]:
    """Pull the ``persons`` list out of a 200 /infer response."""
    assert response.status_code == 200, response.text
    infer = InferResponse.model_validate(response.json())
    return infer.result["persons"]


def _infer(client, jpeg: bytes, **params):
    files = {"frame": ("frame.jpg", jpeg, "image/jpeg")}
    data = {"params": json.dumps(params)} if params else None
    return client.post("/infer", files=files, data=data)


# ── Load lifecycle ─────────────────────────────────────────────────


class TestLoadLifecycle:

    def test_fresh_service_is_not_ready(self, yolo_pose_environment):
        from adapters.yolo_pose.service import YoloPoseService

        assert not YoloPoseService().is_ready()

    def test_load_marks_service_ready(self, service):
        assert service.is_ready()

    def test_load_is_idempotent(self, service):
        """A second load() must not rebuild the session — KAI-C retries
        /health during startup and the SDK may call load() again."""
        first_session = service._session
        service.load()
        assert service.is_ready()
        assert service._session is first_session

    def test_missing_weights_with_no_url_fails_typed(
        self, yolo_pose_environment, monkeypatch
    ):
        """The sovereignty posture: no weights file and no configured
        URL is a clean load failure (ensure_model_file raises
        FileNotFoundError), not a crash and not a silent download."""
        import adapters.yolo_pose.service as service_module

        yolo_pose_environment["weights_path"].unlink()
        svc = service_module.YoloPoseService()
        svc.load()

        assert not svc.is_ready()
        evaluation = svc.hardware_evaluation()
        assert evaluation.verdict == HardwareVerdict.BLOCKED
        assert "no download URL" in (svc._load_error or "")

        with pytest.raises(ServiceError) as exc_info:
            svc.infer({"__file__": b"\xff\xd8\xff"})
        envelope = exc_info.value.envelope()
        assert envelope.error.category == ErrorCategory.MODEL_ERROR
        assert envelope.error.code == "weights_missing"

    def test_present_weights_are_never_refetched(self, yolo_pose_environment):
        """A file already at the weights path short-circuits
        ``ensure_model_file`` before any network access — asserted by
        pointing the URL at an unroutable host and still loading."""
        import adapters.yolo_pose.service as service_module

        import os
        os.environ["YOLO_POSE_MODEL_URL"] = "http://256.0.0.1/never-reachable"
        try:
            svc = service_module.YoloPoseService()
            svc.load()
        finally:
            os.environ["YOLO_POSE_MODEL_URL"] = ""
        assert svc.is_ready()

    def test_infer_before_load_raises_model_error(self, yolo_pose_environment):
        """KAI-C polls /infer during startup — it must get a typed
        ServiceError, not a generic crash."""
        from adapters.yolo_pose.service import YoloPoseService

        svc = YoloPoseService()  # NOT loaded
        with pytest.raises(ServiceError) as exc_info:
            svc.infer({"__file__": b"\xff\xd8\xff"})
        envelope = exc_info.value.envelope()
        assert envelope.error.category == ErrorCategory.MODEL_ERROR
        assert envelope.error.transient is True  # still loading → retryable


# ── Fingerprint (§11.3 drift detection) ────────────────────────────


class TestFingerprint:

    def test_fingerprint_is_stable_across_loads(self, yolo_pose_environment):
        """Same weights file → same fingerprint, every load, every
        call. A fingerprint that moves on its own makes drift
        detection useless."""
        import adapters.yolo_pose.service as service_module

        first = service_module.YoloPoseService()
        first.load()
        second = service_module.YoloPoseService()
        second.load()

        assert first.fingerprint() == second.fingerprint()
        assert first.fingerprint() == first.fingerprint()
        assert first.fingerprint().startswith("sha256:")

    def test_fingerprint_tracks_the_weights_file(self, service, yolo_pose_environment):
        before = service.fingerprint()
        yolo_pose_environment["weights_path"].write_bytes(b"ROTATED_WEIGHTS_v2")
        assert service.fingerprint() != before

    def test_model_info_reports_pose_modalities(self, service):
        info = service.model_info()
        assert info.name == "yolo11n-pose"
        assert info.framework == "onnxruntime"
        assert info.modalities_in == ["image"]
        # Must match server/config/adapters_index.yml's
        # modalities_out: [keypoints] — one spelling, not two.
        assert info.modalities_out == ["keypoints"]
        assert info.fingerprint == service.fingerprint()


# ── /health, /capabilities, /hardware/evaluation, /metrics ─────────


def test_health_returns_valid_HealthResponse(yolo_pose_app):
    response = yolo_pose_app.get("/health")
    assert response.status_code == 200
    health = HealthResponse.model_validate(response.json())
    assert health.status.value == "ok"
    assert health.adapter_name == "yolo-pose-estimation"


def test_capabilities_advertises_pose_estimation(yolo_pose_app):
    caps = CapabilitiesResponse.model_validate(yolo_pose_app.get("/capabilities").json())
    assert "pose_estimation" in caps.tasks_advertised
    assert caps.endpoints.infer.supported is True
    assert "multipart/form-data" in caps.endpoints.infer.input_content_types
    assert "application/json" in caps.endpoints.infer.input_content_types


def test_capabilities_declares_streaming_support(yolo_pose_app):
    """The wand-compliance app drives this adapter at 10 fps over the
    §6 WS path — if the capability says otherwise, KAI-C never opens
    a stream."""
    caps = CapabilitiesResponse.model_validate(yolo_pose_app.get("/capabilities").json())
    assert caps.endpoints.infer_stream.supported is True
    assert caps.endpoints.infer_stream.max_concurrent_streams >= 1
    assert caps.endpoints.infer_stream.supports_shared_memory is False


def test_capabilities_declares_build_accurate_permissions(yolo_pose_app):
    """The fixture's fake onnxruntime is CPU-only, so this exercises
    the stock CPU image: gpu=False, no egress, no host_filesystem."""
    caps = CapabilitiesResponse.model_validate(yolo_pose_app.get("/capabilities").json())
    assert caps.permissions.gpu is False
    assert caps.permissions.network_egress == []
    assert caps.permissions.host_filesystem == []
    assert caps.scheduling.fair_queuing.value == "per_camera"
    assert caps.scheduling.max_inflight == 1


def test_capabilities_exposes_model_fingerprint(yolo_pose_app):
    caps = CapabilitiesResponse.model_validate(yolo_pose_app.get("/capabilities").json())
    assert caps.model.fingerprint is not None
    assert caps.model.fingerprint.startswith("sha256:")


def test_hardware_evaluation_treats_cpu_as_supported(yolo_pose_app):
    """Unlike the yolov8 adapter, a CPU-only host is not a degraded
    mode here — it's the design target. On a machine with >= 4 cores
    the verdict is ok (warn only when the core count can't hold 10
    fps)."""
    hwe = HardwareEvaluationResponse.model_validate(
        yolo_pose_app.get("/hardware/evaluation").json()
    )
    assert hwe.verdict.value in ("ok", "warn")
    if (hwe.details or {}).get("cpu_count", 0) >= 4:
        assert hwe.verdict.value == "ok"
    assert hwe.details["gpu_required"] is False
    assert hwe.details["default_imgsz"] == FAKE_IMGSZ


def test_metrics_emits_prometheus_baseline_and_domain_metrics(
    yolo_pose_app, square_jpeg
):
    assert _infer(yolo_pose_app, square_jpeg).status_code == 200
    body = yolo_pose_app.get("/metrics").text
    for name in (
        "adapter_infer_total",
        "adapter_infer_latency_seconds",
        "adapter_model_loaded",
        "adapter_stream_connections_active",
        "adapter_inflight_requests",
    ):
        assert name in body
    assert "adapter_model_loaded 1" in body
    # One person survived NMS on the stubbed frame.
    assert "adapter_pose_persons_total 1.0" in body
    # Per-joint visibility: the high-confidence left wrist is counted,
    # the occluded right wrist (conf 0.2) is not — that asymmetry is
    # the whole point of the metric.
    assert 'adapter_pose_keypoints_visible_total{keypoint="left_wrist"} 1.0' in body
    assert 'keypoint="right_wrist"' not in body


# ── /infer — the documented output shape ───────────────────────────


class TestOutputShape:

    def test_persons_carry_bbox_score_and_17_keypoints(
        self, yolo_pose_app, square_jpeg
    ):
        persons = _persons(_infer(yolo_pose_app, square_jpeg))
        assert len(persons) == 1  # the duplicate prediction was NMS-ed
        person = persons[0]
        assert sorted(person.keys()) == ["bbox", "keypoints", "score"]
        assert len(person["bbox"]) == 4
        assert person["score"] == pytest.approx(0.92, abs=1e-3)
        assert len(person["keypoints"]) == 17
        for keypoint in person["keypoints"]:
            assert len(keypoint) == 3        # [x, y, conf]
            assert 0.0 <= keypoint[2] <= 1.0

    def test_result_documents_the_keypoint_order_and_frame(
        self, yolo_pose_app, wide_jpeg
    ):
        """``keypoint_names`` is part of the response so a consumer
        never has to hard-code that slot 9 is the left wrist."""
        response = _infer(yolo_pose_app, wide_jpeg)
        result = InferResponse.model_validate(response.json()).result
        assert result["keypoint_names"] == list(COCO_KEYPOINTS)
        assert result["keypoint_names"][:5] == [
            "nose", "left_eye", "right_eye", "left_ear", "right_ear",
        ]
        assert result["keypoint_names"][NOSE] == "nose"
        assert result["keypoint_names"][LEFT_WRIST] == "left_wrist"
        assert result["keypoint_names"][RIGHT_WRIST] == "right_wrist"
        assert result["frame_dimensions"] == {"w": 640, "h": 360}

    def test_coordinates_are_pixels_in_the_source_frame(
        self, yolo_pose_app, square_jpeg
    ):
        """Square input: the model canvas and the source frame
        coincide, so the centred nose lands at the centre pixel and
        the box is exactly the model's box."""
        person = _persons(_infer(yolo_pose_app, square_jpeg))[0]
        assert person["bbox"] == pytest.approx([174.0, 124.0, 274.0, 324.0])
        nose = person["keypoints"][NOSE]
        assert nose[0] == pytest.approx(FAKE_IMGSZ / 2)
        assert nose[1] == pytest.approx(FAKE_IMGSZ / 2)
        assert nose[2] == pytest.approx(0.90, abs=1e-3)

    def test_letterbox_padding_is_unmapped_for_wide_frames(
        self, yolo_pose_app, wide_jpeg
    ):
        """The regression that matters most: a 16:9 frame is padded to
        a square before inference, and every coordinate has to come
        back out of that padding. The stub puts the nose at the exact
        centre of the model canvas, so it must land at the exact
        centre of the 640×360 source — (320, 180). Getting the pad
        arithmetic wrong shifts it by ~140 px vertically, which is a
        whole torso."""
        person = _persons(_infer(yolo_pose_app, wide_jpeg))[0]
        nose = person["keypoints"][NOSE]
        assert nose[0] == pytest.approx(320.0, abs=0.5)
        assert nose[1] == pytest.approx(180.0, abs=0.5)

    def test_coordinates_stay_inside_the_frame(self, yolo_pose_app, wide_jpeg):
        person = _persons(_infer(yolo_pose_app, wide_jpeg))[0]
        x1, y1, x2, y2 = person["bbox"]
        assert 0.0 <= x1 < x2 <= 640.0
        assert 0.0 <= y1 < y2 <= 360.0
        for x, y, _conf in person["keypoints"]:
            assert 0.0 <= x <= 640.0
            assert 0.0 <= y <= 360.0

    def test_json_base64_path_returns_the_same_shape(
        self, yolo_pose_app, square_jpeg
    ):
        body = {"frame_b64": base64.b64encode(square_jpeg).decode("ascii")}
        response = yolo_pose_app.post("/infer", json=body)
        assert response.status_code == 200, response.text
        result = InferResponse.model_validate(response.json()).result
        assert len(result["persons"]) == 1
        assert len(result["persons"][0]["keypoints"]) == 17


# ── /infer — caller params ─────────────────────────────────────────


class TestParams:

    def test_default_conf_drops_the_weak_detection(self, yolo_pose_app, square_jpeg):
        """The third stub prediction scores 0.30, below the 0.4
        default."""
        assert len(_persons(_infer(yolo_pose_app, square_jpeg))) == 1

    def test_lower_conf_admits_the_weak_detection(self, yolo_pose_app, square_jpeg):
        persons = _persons(_infer(yolo_pose_app, square_jpeg, conf=0.2))
        assert len(persons) == 2
        assert [p["score"] for p in persons] == sorted(
            [p["score"] for p in persons], reverse=True
        ), "persons must come back in descending-score order"

    def test_confidence_threshold_alias_is_accepted(self, yolo_pose_app, square_jpeg):
        """yolov8 spells this knob ``confidence_threshold``; a caller
        that talks to both adapters shouldn't have to branch."""
        persons = _persons(
            _infer(yolo_pose_app, square_jpeg, confidence_threshold=0.2)
        )
        assert len(persons) == 2

    def test_iou_one_disables_suppression(self, yolo_pose_app, square_jpeg):
        """The near-duplicate prediction is suppressed at the default
        IoU and survives at iou=1.0 — proof NMS is actually running
        rather than the fixture being short by one row."""
        assert len(_persons(_infer(yolo_pose_app, square_jpeg, iou=1.0))) == 2

    def test_max_persons_caps_the_payload(self, yolo_pose_app, square_jpeg):
        persons = _persons(
            _infer(yolo_pose_app, square_jpeg, conf=0.2, max_persons=1)
        )
        assert len(persons) == 1
        assert persons[0]["score"] == pytest.approx(0.92, abs=1e-3)  # the best one

    def test_fixed_size_export_dictates_imgsz(self, yolo_pose_environment, square_jpeg):
        """An Ultralytics export is fixed-size unless it was exported
        with dynamic=True. A 640 export must just work — the adapter
        uses the size the graph demands instead of its own 448 default
        and letting onnxruntime throw a shape error."""
        import importlib

        install_fake_onnxruntime(input_shape=[1, 3, 640, 640])
        import adapters.yolo_pose.service as service_module
        importlib.reload(service_module)
        svc = service_module.YoloPoseService()
        svc.load()
        try:
            svc.infer({"__file__": square_jpeg})
            assert svc._session.received_blobs[-1].shape == (1, 3, 640, 640)

            # But a caller who explicitly asks for something else is
            # told why, rather than getting a 500 from inside the
            # session.
            with pytest.raises(ServiceError) as exc_info:
                svc.infer({"__file__": square_jpeg, "imgsz": 448})
            envelope = exc_info.value.envelope()
            assert envelope.error.code == "malformed_input"
            assert "dynamic=True" in envelope.error.message
        finally:
            install_fake_onnxruntime()  # restore for the next test

    def test_imgsz_changes_the_preprocessed_blob(self, service, square_jpeg):
        """imgsz is the CPU/accuracy dial. Assert it reaches the
        session rather than being quietly ignored — the stub records
        every blob it is handed."""
        service.infer({"__file__": square_jpeg, "imgsz": 320})
        blob = service._session.received_blobs[-1]
        assert blob.shape == (1, 3, 320, 320)
        assert blob.dtype.name == "float32"
        assert 0.0 <= float(blob.min()) and float(blob.max()) <= 1.0


# ── /infer — malformed input ───────────────────────────────────────


class TestMalformedInput:

    def test_missing_frame_is_rejected(self, yolo_pose_app):
        response = yolo_pose_app.post(
            "/infer",
            data={"params": json.dumps({})},
            files={"_marker": ("", b"", "text/plain")},  # forces multipart
        )
        assert response.status_code == 400
        FailureEnvelope.model_validate(response.json())

    def test_undecodable_frame_is_rejected(self, yolo_pose_app):
        response = yolo_pose_app.post(
            "/infer",
            files={"frame": ("not.jpg", b"this is not an image", "image/jpeg")},
        )
        assert response.status_code == 400
        envelope = FailureEnvelope.model_validate(response.json())
        assert envelope.error.code == "malformed_input"
        assert envelope.error.category.value == "transport_error"

    def test_oversized_frame_is_rejected(self, yolo_pose_app):
        huge = bytes(8 * 1024 * 1024 + 1)
        response = yolo_pose_app.post(
            "/infer", files={"frame": ("huge.jpg", huge, "image/jpeg")}
        )
        assert response.status_code == 413

    def test_unsupported_content_type_is_rejected(self, yolo_pose_app):
        response = yolo_pose_app.post(
            "/infer",
            content=b"raw bytes",
            headers={"Content-Type": "application/octet-stream"},
        )
        assert response.status_code == 415
        envelope = FailureEnvelope.model_validate(response.json())
        assert envelope.error.code == "unsupported_content_type"

    @pytest.mark.parametrize(
        "params",
        [
            {"conf": "not-a-number"},
            {"conf": 1.5},
            {"conf": -0.1},
            {"iou": "high"},
            {"iou": 2.0},
            {"imgsz": "big"},
            {"imgsz": 64},        # below MIN_IMGSZ
            {"imgsz": 4096},      # above MAX_IMGSZ
            {"imgsz": 450},       # not a multiple of the stride
            {"max_persons": 0},
            {"max_persons": "lots"},
            {"max_persons": 10_000},
            # bool is an int subclass: without an explicit guard these
            # slide through as 1.0 / 1 and the caller never learns their
            # threshold was a typo.
            {"conf": True},
            {"iou": False},
            {"max_persons": True},
        ],
    )
    def test_bad_params_return_typed_400(self, yolo_pose_app, square_jpeg, params):
        """Every caller-supplied number is validated before it reaches
        numpy — a bad value is a 400 with a §7 envelope, never a 500."""
        response = _infer(yolo_pose_app, square_jpeg, **params)
        assert response.status_code == 400, response.text
        envelope = FailureEnvelope.model_validate(response.json())
        assert envelope.error.code == "malformed_input"
        assert envelope.error.transient is False

    def test_off_stride_imgsz_message_suggests_a_valid_value(
        self, yolo_pose_app, square_jpeg
    ):
        response = _infer(yolo_pose_app, square_jpeg, imgsz=450)
        envelope = FailureEnvelope.model_validate(response.json())
        assert "multiple of 32" in envelope.error.message
        assert "448" in envelope.error.message

    def test_wrong_model_export_fails_typed(self, yolo_pose_environment, square_jpeg):
        """Mounting a detection ONNX (84 features) instead of a pose
        export must say so, not return silently truncated keypoints."""
        import importlib

        install_fake_onnxruntime(features=84)
        import adapters.yolo_pose.service as service_module
        importlib.reload(service_module)
        svc = service_module.YoloPoseService()
        svc.load()

        with pytest.raises(ServiceError) as exc_info:
            svc.infer({"__file__": square_jpeg})
        envelope = exc_info.value.envelope()
        assert envelope.error.category == ErrorCategory.MODEL_ERROR
        assert envelope.error.code == "yolo_pose.unexpected_model_output"
        assert "84" in envelope.error.message

        install_fake_onnxruntime()  # restore for the next test


# ── /infer/stream — §6 WebSocket protocol ──────────────────────────


class TestStreaming:

    def test_handshake_returns_ack(self, yolo_pose_app):
        with yolo_pose_app.websocket_connect("/infer/stream") as ws:
            ws.send_text(json.dumps({
                "type": "handshake", "client_id": "wand-app",
                "camera_id": "entrance-1", "frame_transport": "websocket",
            }))
            ack = json.loads(ws.receive_text())
            assert ack["type"] == "handshake_ack"
            assert ack["frame_transport"] == "websocket"
            assert ack["session_id"]
            ws.send_text(json.dumps({"type": "close", "reason": "done"}))

    def test_shared_memory_offer_downgrades_to_websocket(self, yolo_pose_app):
        with yolo_pose_app.websocket_connect("/infer/stream") as ws:
            ws.send_text(json.dumps({
                "type": "handshake", "client_id": "c", "camera_id": "cam-2",
                "frame_transport": "shared_memory",
                "shared_memory_root": "/dev/shm/x",
            }))
            ack = json.loads(ws.receive_text())
            assert ack["frame_transport"] == "websocket"

    def test_frame_roundtrip_returns_persons(self, yolo_pose_app, wide_jpeg):
        """The 10 fps path: metadata + bytes in, one result message
        out, same payload as HTTP /infer."""
        with yolo_pose_app.websocket_connect("/infer/stream") as ws:
            ws.send_text(json.dumps({
                "type": "handshake", "client_id": "wand-app",
                "camera_id": "entrance-1", "frame_transport": "websocket",
            }))
            json.loads(ws.receive_text())  # ack
            ws.send_text(json.dumps({
                "type": "frame", "seq": 42, "ts_ms": 1716000000123,
                "content_type": "image/jpeg",
            }))
            ws.send_bytes(wide_jpeg)
            result = json.loads(ws.receive_text())

            assert result["type"] == "result"
            assert result["seq"] == 42
            assert result["ts_ms"] == 1716000000123
            persons = result["result"]["persons"]
            assert len(persons) == 1
            assert len(persons[0]["keypoints"]) == 17
            assert result["result"]["frame_dimensions"] == {"w": 640, "h": 360}

    def test_pause_resume_swallows_frames(self, yolo_pose_app, square_jpeg):
        with yolo_pose_app.websocket_connect("/infer/stream") as ws:
            ws.send_text(json.dumps({
                "type": "handshake", "client_id": "c", "camera_id": "cam-p",
                "frame_transport": "websocket",
            }))
            json.loads(ws.receive_text())  # ack
            ws.send_text(json.dumps({"type": "pause"}))
            ws.send_text(json.dumps({
                "type": "frame", "seq": 1, "ts_ms": 0, "content_type": "image/jpeg",
            }))
            ws.send_bytes(square_jpeg)
            ws.send_text(json.dumps({"type": "resume"}))
            ws.send_text(json.dumps({
                "type": "frame", "seq": 2, "ts_ms": 0, "content_type": "image/jpeg",
            }))
            ws.send_bytes(square_jpeg)
            result = json.loads(ws.receive_text())
            assert result["seq"] == 2  # the paused frame was dropped

    def test_stats_reports_this_sessions_real_numbers(
        self, yolo_pose_app, square_jpeg
    ):
        """§6.4's stats reply carries live values, not zeros: after one
        inferred frame the session's fps is positive, and inflight
        reflects the SDK gauge (zero between frames, since inference is
        serial and already finished)."""
        with yolo_pose_app.websocket_connect("/infer/stream") as ws:
            ws.send_text(json.dumps({
                "type": "handshake", "client_id": "c", "camera_id": "cam-stats",
                "frame_transport": "websocket",
            }))
            json.loads(ws.receive_text())  # ack

            ws.send_text(json.dumps({"type": "stats"}))
            before = json.loads(ws.receive_text())
            assert before["fps"] == 0.0        # nothing inferred yet

            ws.send_text(json.dumps({
                "type": "frame", "seq": 1, "ts_ms": 0,
                "content_type": "image/jpeg",
            }))
            ws.send_bytes(square_jpeg)
            json.loads(ws.receive_text())      # result

            ws.send_text(json.dumps({"type": "stats"}))
            after = json.loads(ws.receive_text())
            assert after["fps"] > 0.0
            assert after["inflight"] == 0
            assert after["queue_depth"] == 0
            ws.send_text(json.dumps({"type": "close", "reason": "done"}))

    def test_stats_message_returns_stats(self, yolo_pose_app):
        with yolo_pose_app.websocket_connect("/infer/stream") as ws:
            ws.send_text(json.dumps({
                "type": "handshake", "client_id": "c", "camera_id": "cam-s",
                "frame_transport": "websocket",
            }))
            json.loads(ws.receive_text())  # ack
            ws.send_text(json.dumps({"type": "stats"}))
            stats = json.loads(ws.receive_text())
            assert stats["type"] == "stats"
            assert {"inflight", "queue_depth", "fps"} <= set(stats)

    def test_bad_handshake_closes_with_policy_refused(self, yolo_pose_app):
        from starlette.websockets import WebSocketDisconnect

        with yolo_pose_app.websocket_connect("/infer/stream") as ws:
            ws.send_text("not json at all")
            with pytest.raises(WebSocketDisconnect) as exc_info:
                ws.receive_text()
            assert exc_info.value.code == StreamCloseCode.POLICY_REFUSED.value

    def test_frame_without_binary_closes_with_policy_refused(self, yolo_pose_app):
        from starlette.websockets import WebSocketDisconnect

        with yolo_pose_app.websocket_connect("/infer/stream") as ws:
            ws.send_text(json.dumps({
                "type": "handshake", "client_id": "c", "camera_id": "cam-bad",
                "frame_transport": "websocket",
            }))
            json.loads(ws.receive_text())
            ws.send_text(json.dumps({
                "type": "frame", "seq": 1, "ts_ms": 0, "content_type": "image/jpeg",
            }))
            # A second text message instead of the promised bytes.
            ws.send_text(json.dumps({
                "type": "frame", "seq": 2, "ts_ms": 0, "content_type": "image/jpeg",
            }))
            with pytest.raises(WebSocketDisconnect) as exc_info:
                ws.receive_text()
            assert exc_info.value.code == StreamCloseCode.POLICY_REFUSED.value

    def test_stream_error_uses_failure_envelope_shape(self, yolo_pose_app):
        """A bad frame mid-stream must not kill the stream, and the
        embedded error must be the same §7 envelope the HTTP path
        returns — one parser for both."""
        with yolo_pose_app.websocket_connect("/infer/stream") as ws:
            ws.send_text(json.dumps({
                "type": "handshake", "client_id": "c", "camera_id": "cam-err",
                "frame_transport": "websocket",
            }))
            json.loads(ws.receive_text())  # ack
            ws.send_text(json.dumps({
                "type": "frame", "seq": 7, "ts_ms": 0, "content_type": "image/jpeg",
            }))
            ws.send_bytes(b"not a real image")
            result = json.loads(ws.receive_text())

            assert result["type"] == "result"
            assert result["seq"] == 7
            envelope = result["result"]
            FailureEnvelope.model_validate(envelope)
            assert envelope["error"]["category"] == "transport_error"
            assert envelope["error"]["code"] == "malformed_input"


# ── Auth + correlation_id (mirrors the yolov8 suite) ───────────────


def test_auth_rejects_missing_token_on_infer(yolo_pose_app_with_auth, square_jpeg):
    client, _token = yolo_pose_app_with_auth
    response = client.post(
        "/infer", files={"frame": ("frame.jpg", square_jpeg, "image/jpeg")}
    )
    assert response.status_code == 401


def test_auth_accepts_valid_token_on_infer(yolo_pose_app_with_auth, square_jpeg):
    client, token = yolo_pose_app_with_auth
    response = client.post(
        "/infer",
        files={"frame": ("frame.jpg", square_jpeg, "image/jpeg")},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200, response.text


def test_correlation_id_echoed_on_capabilities(yolo_pose_app):
    response = yolo_pose_app.get(
        "/capabilities", headers={"X-Correlation-Id": "pose-corr-1"}
    )
    assert response.headers.get("X-Correlation-Id") == "pose-corr-1"


def test_correlation_id_minted_when_absent(yolo_pose_app):
    assert yolo_pose_app.get("/capabilities").headers.get("X-Correlation-Id")


# ── Unexpected-failure containment ─────────────────────────────────
#
# Every failure _infer_image_bytes knows about is already a typed
# ServiceError. These two tests pin the failure it does NOT know about:
# post-processing raising something unforeseen. Shaping the response used
# to sit outside the guarded block, so such an error escaped as a bare
# exception — a 500 with no §7 envelope on HTTP, and a torn-down socket
# mid-session on the stream, in both cases uncounted by record_infer.


def _explode(*_args, **_kwargs):
    raise RuntimeError("post-processing blew up")


def test_unexpected_shaping_failure_is_a_typed_envelope(
    yolo_pose_app, square_jpeg, monkeypatch
):
    """HTTP: an unforeseen error in post-processing is a §7 envelope
    with a 500, not an untyped framework error page."""
    from adapters.yolo_pose.service import YoloPoseService

    monkeypatch.setattr(YoloPoseService, "_shape_persons", _explode)
    response = _infer(yolo_pose_app, square_jpeg)
    assert response.status_code == 500, response.text
    envelope = FailureEnvelope.model_validate(response.json())
    assert envelope.error.category == ErrorCategory.MODEL_ERROR
    assert envelope.error.code == "inference_runtime_crash"


def test_unexpected_shaping_failure_does_not_kill_the_stream(
    yolo_pose_app, square_jpeg, monkeypatch
):
    """WS: the same failure comes back as a result message carrying the
    envelope, and the session stays open for the next frame. One bad
    frame must not end a camera's stream."""
    from adapters.yolo_pose.service import YoloPoseService

    monkeypatch.setattr(YoloPoseService, "_shape_persons", _explode)
    with yolo_pose_app.websocket_connect("/infer/stream") as ws:
        ws.send_text(json.dumps({
            "type": "handshake", "client_id": "c", "camera_id": "cam-boom",
            "frame_transport": "websocket",
        }))
        json.loads(ws.receive_text())  # ack

        for seq in (1, 2):
            ws.send_text(json.dumps({
                "type": "frame", "seq": seq, "ts_ms": 0,
                "content_type": "image/jpeg",
            }))
            ws.send_bytes(square_jpeg)
            message = json.loads(ws.receive_text())
            assert message["type"] == "result"
            assert message["seq"] == seq
            envelope = FailureEnvelope.model_validate(message["result"])
            assert envelope.error.code == "inference_runtime_crash"

        # Still a live session: a control message is still answered.
        ws.send_text(json.dumps({"type": "stats"}))
        assert json.loads(ws.receive_text())["type"] == "stats"
        ws.send_text(json.dumps({"type": "close", "reason": "done"}))


# ── §8 permissions are derived from the build AND the config ───────


def test_network_egress_is_empty_without_a_configured_fetch(monkeypatch):
    """The default posture: weights are mounted, nothing is declared,
    and the adapter stays clean under sovereignty=local_only."""
    import adapters.yolo_pose.main as pose_main

    monkeypatch.delenv("YOLO_POSE_MODEL_URL", raising=False)
    assert pose_main._model_fetch_egress() == []


def test_configured_model_url_is_declared_as_egress(monkeypatch):
    """An operator who turns on the first-boot fetch gets that one host
    declared — §8 treats an undeclared egress host as grounds for
    removing the adapter."""
    import adapters.yolo_pose.main as pose_main

    monkeypatch.setenv(
        "YOLO_POSE_MODEL_URL", "https://artifacts.example.com/pose/yolo11n-pose.onnx"
    )
    assert pose_main._model_fetch_egress() == ["artifacts.example.com"]
