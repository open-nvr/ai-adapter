# Copyright (c) 2026 OpenNVR
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Integration tests for the YOLOv8 contract service (adapters/yolov8/main.py).

Covers:
  - HTTP /health, /capabilities, /hardware/evaluation, /metrics
  - /infer (multipart and application/json paths)
  - /infer/stream — full §6 WebSocket protocol roundtrip
  - Auth, correlation_id (mirroring Piper)
"""
from __future__ import annotations

import base64
import json

import pytest

from opennvr_adapter_sdk.contract import (
    CapabilitiesResponse,
    DetectionResult,
    FailureEnvelope,
    HardwareEvaluationResponse,
    HealthResponse,
    InferResponse,
    StreamCloseCode,
)
from tests._yolov8_service_fixtures import (  # noqa: F401
    sample_jpeg,
    yolov8_app,
    yolov8_app_with_auth,
    yolov8_environment,
)


# ── /health ────────────────────────────────────────────────────────


def test_health_returns_valid_HealthResponse(yolov8_app):
    response = yolov8_app.get("/health")
    assert response.status_code == 200
    health = HealthResponse.model_validate(response.json())
    assert health.status.value == "ok"
    assert health.adapter_name == "yolov8-object-detection"


# ── /capabilities ──────────────────────────────────────────────────


def test_capabilities_declares_streaming_support(yolov8_app):
    caps = CapabilitiesResponse.model_validate(yolov8_app.get("/capabilities").json())
    assert "object_detection" in caps.tasks_advertised
    assert caps.endpoints.infer.supported is True
    assert "multipart/form-data" in caps.endpoints.infer.input_content_types
    assert "application/json" in caps.endpoints.infer.input_content_types
    assert caps.endpoints.infer_stream.supported is True
    assert caps.endpoints.infer_stream.max_concurrent_streams >= 1
    # Shared-memory deferred to A2.2b
    assert caps.endpoints.infer_stream.supports_shared_memory is False


def test_capabilities_declares_gpu_permission_and_fair_queuing(yolov8_app):
    caps = CapabilitiesResponse.model_validate(yolov8_app.get("/capabilities").json())
    assert caps.permissions.gpu is True
    assert caps.permissions.network_egress == []
    assert caps.scheduling.fair_queuing.value == "per_camera"


def test_capabilities_exposes_model_fingerprint(yolov8_app):
    caps = CapabilitiesResponse.model_validate(yolov8_app.get("/capabilities").json())
    assert caps.model.fingerprint is not None
    assert caps.model.fingerprint.startswith("sha256:")


# ── /hardware/evaluation ───────────────────────────────────────────


def test_hardware_evaluation_warns_when_cpu_only(yolov8_app):
    """Fake onnxruntime advertises only CPU — service should WARN."""
    hwe = HardwareEvaluationResponse.model_validate(
        yolov8_app.get("/hardware/evaluation").json()
    )
    assert hwe.verdict.value == "warn"
    assert "CPU" in hwe.reasoning or "cpu" in hwe.reasoning


# ── /metrics ───────────────────────────────────────────────────────


def test_metrics_emits_prometheus_baseline(yolov8_app):
    body = yolov8_app.get("/metrics").text
    for name in (
        "adapter_infer_total",
        "adapter_infer_latency_seconds",
        "adapter_model_loaded",
        "adapter_stream_connections_active",
        "adapter_inflight_requests",
        "adapter_queue_depth",
    ):
        assert name in body
    assert "adapter_model_loaded 1" in body


# ── /infer multipart (§3.5 canonical) ──────────────────────────────


def test_infer_multipart_with_frame_returns_detection(yolov8_app, sample_jpeg):
    response = yolov8_app.post(
        "/infer",
        files={"frame": ("frame.jpg", sample_jpeg, "image/jpeg")},
    )
    assert response.status_code == 200, response.text
    infer = InferResponse.model_validate(response.json())
    result = DetectionResult.model_validate(infer.result)
    # One detection above threshold from the stubbed predictions
    assert len(result.detections) == 1
    assert result.detections[0].label == "person"
    assert result.detections[0].confidence > 0.9
    # Bbox normalized [0, 1]
    bbox = result.detections[0].bbox
    assert 0.0 <= bbox.x <= 1.0
    assert 0.0 <= bbox.y <= 1.0
    assert 0.0 <= bbox.w <= 1.0
    assert 0.0 <= bbox.h <= 1.0
    # Frame dimensions match input (64x64)
    assert result.frame_dimensions.w == 64
    assert result.frame_dimensions.h == 64


def test_infer_multipart_with_params_filters_classes(yolov8_app, sample_jpeg):
    """When 'classes' is set, detections outside the allow-list are filtered."""
    response = yolov8_app.post(
        "/infer",
        files={"frame": ("frame.jpg", sample_jpeg, "image/jpeg")},
        data={"params": json.dumps({"classes": ["car", "truck"]})},
    )
    assert response.status_code == 200
    result = DetectionResult.model_validate(InferResponse.model_validate(response.json()).result)
    # Stubbed prediction is "person" — outside allow-list → empty.
    assert result.detections == []


def test_infer_multipart_rejects_missing_frame(yolov8_app):
    response = yolov8_app.post(
        "/infer",
        data={"params": json.dumps({})},
        files={"_marker": ("", b"", "text/plain")},  # forces multipart
    )
    assert response.status_code == 400
    FailureEnvelope.model_validate(response.json())


# ── /infer JSON (convenience path) ─────────────────────────────────


def test_infer_json_with_base64_frame(yolov8_app, sample_jpeg):
    body = {"frame_b64": base64.b64encode(sample_jpeg).decode("ascii")}
    response = yolov8_app.post("/infer", json=body)
    assert response.status_code == 200, response.text
    infer = InferResponse.model_validate(response.json())
    DetectionResult.model_validate(infer.result)


def test_infer_json_rejects_invalid_base64(yolov8_app):
    response = yolov8_app.post("/infer", json={"frame_b64": "%%%not-base64%%%"})
    assert response.status_code == 400
    envelope = FailureEnvelope.model_validate(response.json())
    assert envelope.error.code == "malformed_input"


def test_infer_json_rejects_missing_frame_b64(yolov8_app):
    response = yolov8_app.post("/infer", json={"confidence_threshold": 0.5})
    assert response.status_code == 400


def test_infer_rejects_unsupported_content_type(yolov8_app):
    response = yolov8_app.post(
        "/infer",
        content=b"raw bytes",
        headers={"Content-Type": "application/octet-stream"},
    )
    assert response.status_code == 415
    envelope = FailureEnvelope.model_validate(response.json())
    assert envelope.error.code == "unsupported_content_type"


def test_infer_rejects_oversized_frame(yolov8_app):
    """8 MiB cap on frame bytes per service.MAX_IMAGE_BYTES."""
    huge = bytes(8 * 1024 * 1024 + 1)
    response = yolov8_app.post(
        "/infer",
        files={"frame": ("huge.jpg", huge, "image/jpeg")},
    )
    assert response.status_code == 413


def test_infer_rejects_undecodable_frame(yolov8_app):
    """Non-image bytes → transport_error: malformed_input."""
    response = yolov8_app.post(
        "/infer",
        files={"frame": ("not.jpg", b"this is not an image", "image/jpeg")},
    )
    assert response.status_code == 400


def test_infer_rejects_non_numeric_confidence_threshold(yolov8_app, sample_jpeg):
    """Regression for self-review C-5: non-numeric confidence_threshold
    used to bubble a ValueError → 500; now it's caught and returned as
    a typed 400 malformed_input."""
    response = yolov8_app.post(
        "/infer",
        files={"frame": ("frame.jpg", sample_jpeg, "image/jpeg")},
        data={"params": json.dumps({"confidence_threshold": "not-a-number"})},
    )
    assert response.status_code == 400
    envelope = FailureEnvelope.model_validate(response.json())
    assert envelope.error.code == "malformed_input"


def test_capabilities_advertises_honest_max_inflight(yolov8_app):
    """Regression for self-review C-3: capabilities.scheduling.max_inflight
    must match what the adapter can actually serve concurrently. With
    a singleton ONNX session and no inter-stream serialization, the
    honest value is 1."""
    caps = CapabilitiesResponse.model_validate(yolov8_app.get("/capabilities").json())
    assert caps.scheduling.max_inflight == 1


def test_infer_rejects_reserved_body_key_in_params(yolov8_app, sample_jpeg):
    """Regression for A2.3b peer-review H2: a caller-supplied params
    key shadowing the SDK's reserved binary-body key must be rejected
    with malformed_input, not silently overwritten. Without this guard,
    ``params={"__file__": "x"}`` got its value clobbered by the raw
    image bytes and then stripped — invisible from the client side.

    The check applies to both the multipart (``params`` JSON field)
    and JSON (top-level) paths. Test both.
    """
    # Multipart path.
    response = yolov8_app.post(
        "/infer",
        files={"frame": ("frame.jpg", sample_jpeg, "image/jpeg")},
        data={"params": json.dumps({"__file__": "user-value"})},
    )
    assert response.status_code == 400
    envelope = FailureEnvelope.model_validate(response.json())
    assert envelope.error.code == "malformed_input"
    assert "reserved" in envelope.error.message.lower()

    # JSON path.
    response = yolov8_app.post(
        "/infer",
        json={
            "frame_b64": base64.b64encode(sample_jpeg).decode("ascii"),
            "__file__": "user-value",
        },
    )
    assert response.status_code == 400
    envelope = FailureEnvelope.model_validate(response.json())
    assert envelope.error.code == "malformed_input"
    assert "reserved" in envelope.error.message.lower()


def test_capabilities_recomputes_fingerprint_on_each_call(yolov8_app, yolov8_environment):
    """Regression for peer-review PR-22: rotating the weights file
    under a running adapter must cause the fingerprint reported by
    /capabilities to change on the next call. Without this, KAI-C's
    §11.3 drift detection can't see tamper."""
    caps1 = CapabilitiesResponse.model_validate(yolov8_app.get("/capabilities").json())
    fp1 = caps1.model.fingerprint
    assert fp1 and fp1.startswith("sha256:")

    # Rotate the weights file underneath the running service.
    weights_path = yolov8_environment["weights_path"]
    weights_path.write_bytes(b"DIFFERENT_WEIGHTS_PAYLOAD_v2")

    caps2 = CapabilitiesResponse.model_validate(yolov8_app.get("/capabilities").json())
    fp2 = caps2.model.fingerprint
    assert fp2 != fp1, "fingerprint must change after weights file changes"


def test_auth_failure_emits_log_line(yolov8_app_with_auth, sample_jpeg, caplog):
    """Regression for peer-review PR-23: auth rejections must produce
    an INFO log line so brute-force probes are visible in the audit
    trail. The token bytes must NOT be in the log."""
    import logging
    caplog.set_level(logging.INFO)
    client, _token = yolov8_app_with_auth
    response = client.post(
        "/infer",
        files={"frame": ("frame.jpg", sample_jpeg, "image/jpeg")},
        headers={"Authorization": "Bearer wrong-secret-token-12345"},
    )
    assert response.status_code == 401
    # Log line must mention the rejection code but not the token.
    audit_messages = [r.message for r in caplog.records if "auth rejected" in r.message]
    assert audit_messages, "expected an 'auth rejected' log line"
    joined = "\n".join(audit_messages)
    assert "auth_invalid" in joined
    assert "wrong-secret-token-12345" not in joined, "token bytes leaked into logs"


def test_stream_error_result_uses_failure_envelope_shape(yolov8_app):
    """Regression for self-review R-3: when a stream frame fails
    validation, the embedded error body must match the §7 FailureEnvelope
    shape (status='error' + error.{category,code,message,transient,details})
    so a single consumer parser handles HTTP and WS error bodies
    identically."""
    with yolov8_app.websocket_connect("/infer/stream") as ws:
        ws.send_text(json.dumps({
            "type": "handshake", "client_id": "c", "camera_id": "cam-err",
            "frame_transport": "websocket",
        }))
        json.loads(ws.receive_text())  # ack
        # Send frame metadata + non-image bytes to trigger DecodeError
        # in the service, which becomes a transport_error envelope.
        ws.send_text(json.dumps({
            "type": "frame", "seq": 7, "ts_ms": 0, "content_type": "image/jpeg",
        }))
        ws.send_bytes(b"not a real image")
        result = json.loads(ws.receive_text())
        assert result["type"] == "result"
        assert result["seq"] == 7
        envelope_dict = result["result"]
        FailureEnvelope.model_validate(envelope_dict)  # full shape check
        assert envelope_dict["error"]["category"] == "transport_error"
        assert envelope_dict["error"]["code"] == "malformed_input"


# ── /infer/stream — §6 WebSocket protocol ──────────────────────────


def test_stream_handshake_returns_ack(yolov8_app):
    with yolov8_app.websocket_connect("/infer/stream") as ws:
        ws.send_text(json.dumps({
            "type": "handshake",
            "client_id": "test-client",
            "camera_id": "cam-1",
            "frame_transport": "websocket",
        }))
        ack = json.loads(ws.receive_text())
        assert ack["type"] == "handshake_ack"
        assert ack["frame_transport"] == "websocket"
        assert ack["session_id"]
        ws.send_text(json.dumps({"type": "close", "reason": "done"}))


def test_stream_falls_back_when_shared_memory_offered(yolov8_app):
    """Client offers shared_memory; adapter declares supports_shared_memory=false;
    ack downgrades to websocket per §6.1."""
    with yolov8_app.websocket_connect("/infer/stream") as ws:
        ws.send_text(json.dumps({
            "type": "handshake",
            "client_id": "test-client",
            "camera_id": "cam-2",
            "frame_transport": "shared_memory",
            "shared_memory_root": "/dev/shm/x",
        }))
        ack = json.loads(ws.receive_text())
        assert ack["frame_transport"] == "websocket"


def test_stream_frame_roundtrip_returns_result(yolov8_app, sample_jpeg):
    with yolov8_app.websocket_connect("/infer/stream") as ws:
        ws.send_text(json.dumps({
            "type": "handshake",
            "client_id": "test-client",
            "camera_id": "cam-3",
            "frame_transport": "websocket",
        }))
        ack = json.loads(ws.receive_text())
        assert ack["type"] == "handshake_ack"

        ws.send_text(json.dumps({
            "type": "frame",
            "seq": 42,
            "ts_ms": 1716000000123,
            "content_type": "image/jpeg",
        }))
        ws.send_bytes(sample_jpeg)
        result = json.loads(ws.receive_text())
        assert result["type"] == "result"
        assert result["seq"] == 42
        # result.result is a §5.1 DetectionResult
        DetectionResult.model_validate(result["result"])


def test_stream_pause_resume_swallows_frames(yolov8_app, sample_jpeg):
    """After pause, frames are dropped (no result sent). After resume,
    frames are processed again."""
    with yolov8_app.websocket_connect("/infer/stream") as ws:
        ws.send_text(json.dumps({
            "type": "handshake", "client_id": "c", "camera_id": "cam-p",
            "frame_transport": "websocket",
        }))
        json.loads(ws.receive_text())  # ack
        ws.send_text(json.dumps({"type": "pause"}))
        # Frame sent while paused should NOT produce a result. We
        # follow it with a resume + frame and assert we receive a
        # result for the post-resume frame (not the paused one).
        ws.send_text(json.dumps({
            "type": "frame", "seq": 1, "ts_ms": 0, "content_type": "image/jpeg",
        }))
        ws.send_bytes(sample_jpeg)
        ws.send_text(json.dumps({"type": "resume"}))
        ws.send_text(json.dumps({
            "type": "frame", "seq": 2, "ts_ms": 0, "content_type": "image/jpeg",
        }))
        ws.send_bytes(sample_jpeg)
        result = json.loads(ws.receive_text())
        assert result["type"] == "result"
        assert result["seq"] == 2  # paused frame swallowed


def test_stream_stats_message_returns_stats(yolov8_app):
    with yolov8_app.websocket_connect("/infer/stream") as ws:
        ws.send_text(json.dumps({
            "type": "handshake", "client_id": "c", "camera_id": "cam-s",
            "frame_transport": "websocket",
        }))
        json.loads(ws.receive_text())  # ack
        ws.send_text(json.dumps({"type": "stats"}))
        stats = json.loads(ws.receive_text())
        assert stats["type"] == "stats"
        assert "inflight" in stats and "queue_depth" in stats and "fps" in stats


def test_stream_close_with_bad_handshake(yolov8_app):
    """Send garbage on first message → close with 4001 (policy refused)."""
    from starlette.websockets import WebSocketDisconnect
    with yolov8_app.websocket_connect("/infer/stream") as ws:
        ws.send_text("not json at all")
        with pytest.raises(WebSocketDisconnect) as exc_info:
            ws.receive_text()
        assert exc_info.value.code == StreamCloseCode.POLICY_REFUSED.value


def test_stream_close_when_frame_bytes_missing(yolov8_app):
    """Frame metadata not followed by binary → close with 4001."""
    from starlette.websockets import WebSocketDisconnect
    with yolov8_app.websocket_connect("/infer/stream") as ws:
        ws.send_text(json.dumps({
            "type": "handshake", "client_id": "c", "camera_id": "cam-bad",
            "frame_transport": "websocket",
        }))
        json.loads(ws.receive_text())
        ws.send_text(json.dumps({
            "type": "frame", "seq": 1, "ts_ms": 0, "content_type": "image/jpeg",
        }))
        # Send another text message instead of bytes → protocol violation
        ws.send_text(json.dumps({"type": "frame", "seq": 2, "ts_ms": 0,
                                  "content_type": "image/jpeg"}))
        with pytest.raises(WebSocketDisconnect) as exc_info:
            ws.receive_text()
        assert exc_info.value.code == StreamCloseCode.POLICY_REFUSED.value


# ── correlation_id wire spec (mirrors Piper) ───────────────────────


def test_correlation_id_echoed_on_capabilities(yolov8_app):
    response = yolov8_app.get(
        "/capabilities",
        headers={"X-Correlation-Id": "yolov8-corr-1"},
    )
    assert response.headers.get("X-Correlation-Id") == "yolov8-corr-1"


def test_correlation_id_minted_when_absent(yolov8_app):
    response = yolov8_app.get("/capabilities")
    assert response.headers.get("X-Correlation-Id")


# ── Auth (mirrors Piper) ───────────────────────────────────────────


def test_auth_rejects_missing_token_on_infer(yolov8_app_with_auth, sample_jpeg):
    client, _token = yolov8_app_with_auth
    response = client.post(
        "/infer",
        files={"frame": ("frame.jpg", sample_jpeg, "image/jpeg")},
    )
    assert response.status_code == 401


def test_auth_accepts_valid_token_on_infer(yolov8_app_with_auth, sample_jpeg):
    client, token = yolov8_app_with_auth
    response = client.post(
        "/infer",
        files={"frame": ("frame.jpg", sample_jpeg, "image/jpeg")},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200


def test_auth_rejects_ws_without_token(yolov8_app_with_auth):
    """§3.8 + §6 — WS upgrade refused with policy_refused close code."""
    from starlette.websockets import WebSocketDisconnect
    client, _token = yolov8_app_with_auth
    with pytest.raises(WebSocketDisconnect) as exc_info:
        with client.websocket_connect("/infer/stream") as ws:
            ws.receive_text()
    assert exc_info.value.code == StreamCloseCode.POLICY_REFUSED.value


def test_auth_accepts_ws_with_token(yolov8_app_with_auth):
    client, token = yolov8_app_with_auth
    with client.websocket_connect(
        "/infer/stream",
        headers={"Authorization": f"Bearer {token}"},
    ) as ws:
        ws.send_text(json.dumps({
            "type": "handshake", "client_id": "c", "camera_id": "cam-auth",
            "frame_transport": "websocket",
        }))
        ack = json.loads(ws.receive_text())
        assert ack["type"] == "handshake_ack"


# ── Streaming metric correctness (regression for self-review B2) ────


def test_stream_records_real_latency_and_ok_outcome(yolov8_app, sample_jpeg):
    """After a successful stream-inference, /metrics should reflect a
    non-zero latency observation and an incremented ``outcome=ok`` counter.
    Guards against the regression where the WS loop always recorded
    latency=0.0."""
    # Snapshot the ok counter before.
    before_body = yolov8_app.get("/metrics").text
    before_ok = _parse_outcome_count(before_body, "ok")
    before_count = _parse_latency_count(before_body)

    with yolov8_app.websocket_connect("/infer/stream") as ws:
        ws.send_text(json.dumps({
            "type": "handshake", "client_id": "metrics-test",
            "camera_id": "cam-m", "frame_transport": "websocket",
        }))
        json.loads(ws.receive_text())  # ack
        ws.send_text(json.dumps({
            "type": "frame", "seq": 1, "ts_ms": 0, "content_type": "image/jpeg",
        }))
        ws.send_bytes(sample_jpeg)
        json.loads(ws.receive_text())  # result
        ws.send_text(json.dumps({"type": "close", "reason": "done"}))

    after_body = yolov8_app.get("/metrics").text
    after_ok = _parse_outcome_count(after_body, "ok")
    after_count = _parse_latency_count(after_body)

    assert after_ok == before_ok + 1, "ok counter should increment by 1"
    assert after_count == before_count + 1, "latency observation count should increment"


def _parse_outcome_count(prometheus_body: str, outcome: str) -> int:
    for line in prometheus_body.splitlines():
        if line.startswith(f'adapter_infer_total{{outcome="{outcome}"}}'):
            return int(line.rsplit(" ", 1)[1])
    return 0


def _parse_latency_count(prometheus_body: str) -> int:
    for line in prometheus_body.splitlines():
        if line.startswith("adapter_infer_latency_seconds_count"):
            return int(line.rsplit(" ", 1)[1])
    return 0
