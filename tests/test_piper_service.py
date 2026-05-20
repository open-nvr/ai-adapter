# Copyright (c) 2026 OpenNVR
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Integration tests for the Piper contract service
(adapters/piper/main.py).

Asserts wire shapes match the Pydantic contract types and that auth +
correlation-id plumbing behave per §3.8.
"""
from __future__ import annotations

# Re-export fixtures from the shared module so pytest discovers them
# in this test file's scope.
from tests._piper_service_fixtures import (  # noqa: F401
    piper_app,
    piper_app_with_auth,
    piper_environment,
)
from opennvr_adapter_sdk.contract import (
    CapabilitiesResponse,
    FailureEnvelope,
    HardwareEvaluationResponse,
    HealthResponse,
    InferResponse,
)


# ── /health ────────────────────────────────────────────────────────


def test_health_returns_valid_HealthResponse(piper_app):
    response = piper_app.get("/health")
    assert response.status_code == 200
    body = response.json()
    health = HealthResponse.model_validate(body)
    assert health.status.value == "ok"
    assert health.adapter_name == "piper-tts"
    assert health.adapter_version
    assert health.uptime_seconds >= 0


def test_health_open_without_auth_even_when_required(piper_app_with_auth):
    """§3.8 — /health is always open, even after the grace window."""
    client, _token = piper_app_with_auth
    response = client.get("/health")
    assert response.status_code == 200


# ── /capabilities ──────────────────────────────────────────────────


def test_capabilities_returns_valid_CapabilitiesResponse(piper_app):
    response = piper_app.get("/capabilities")
    assert response.status_code == 200
    caps = CapabilitiesResponse.model_validate(response.json())
    assert caps.adapter.name == "piper-tts"
    assert "1" in caps.adapter.supported_contract_versions
    assert "speech_synthesis" in caps.tasks_advertised
    assert caps.endpoints.infer.supported is True
    assert caps.endpoints.infer_stream.supported is False
    assert caps.model.fingerprint is not None
    assert caps.model.fingerprint.startswith("sha256:")
    assert caps.permissions.gpu is False
    assert caps.permissions.network_egress == []


# ── /hardware/evaluation ───────────────────────────────────────────


def test_hardware_evaluation_returns_ok_when_voice_loaded(piper_app):
    response = piper_app.get("/hardware/evaluation")
    assert response.status_code == 200
    hwe = HardwareEvaluationResponse.model_validate(response.json())
    assert hwe.verdict.value == "ok"
    assert hwe.details["gpu_required"] is False


# ── /metrics ───────────────────────────────────────────────────────


def test_metrics_emits_prometheus_baseline(piper_app):
    response = piper_app.get("/metrics")
    assert response.status_code == 200
    body = response.text
    for name in (
        "adapter_infer_total",
        "adapter_infer_latency_seconds",
        "adapter_model_loaded",
        "adapter_stream_connections_active",
        "adapter_inflight_requests",
        "adapter_queue_depth",
    ):
        assert name in body, f"missing metric {name}"
    assert "adapter_model_loaded 1" in body


# ── /infer ─────────────────────────────────────────────────────────


def test_infer_happy_path_returns_InferResponse(piper_app):
    response = piper_app.post(
        "/infer",
        json={"text": "Hello, conformance!"},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    infer = InferResponse.model_validate(body)
    assert infer.model_name == "test-voice"
    assert infer.result["audio_uri"].startswith("opennvr://audio/tts/")
    assert infer.result["sample_rate"] == 22050


def test_infer_rejects_missing_text(piper_app):
    response = piper_app.post("/infer", json={})
    assert response.status_code == 400
    envelope = FailureEnvelope.model_validate(response.json())
    assert envelope.error.category.value == "transport_error"
    assert envelope.error.code == "malformed_input"


def test_infer_rejects_oversized_text(piper_app):
    response = piper_app.post("/infer", json={"text": "x" * 10_001})
    assert response.status_code == 400
    envelope = FailureEnvelope.model_validate(response.json())
    assert envelope.error.code == "malformed_input"


def test_infer_rejects_wrong_content_type(piper_app):
    response = piper_app.post(
        "/infer",
        content=b"text=hello",
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    assert response.status_code == 415
    envelope = FailureEnvelope.model_validate(response.json())
    assert envelope.error.code == "unsupported_content_type"


def test_infer_rejects_non_json_body(piper_app):
    response = piper_app.post(
        "/infer",
        content=b"not json",
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 400
    envelope = FailureEnvelope.model_validate(response.json())
    assert envelope.error.code == "malformed_input"


def test_infer_rejects_non_object_body(piper_app):
    response = piper_app.post(
        "/infer",
        content=b'"just a string"',
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 400


# §3.5 — multipart support is MANDATORY for contract conformance


def test_infer_accepts_multipart_with_params_field(piper_app):
    """§3.5 canonical pattern: multipart with a single ``params`` JSON blob."""
    import json
    response = piper_app.post(
        "/infer",
        files={"params": (None, json.dumps({"text": "Multipart hello"}), "application/json")},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    infer = InferResponse.model_validate(body)
    assert infer.result["audio_uri"].startswith("opennvr://audio/tts/")


def test_infer_accepts_multipart_with_individual_fields(piper_app):
    """Convenience path: per-field form data for readable curl examples.

    httpx ``data=`` alone serializes as ``application/x-www-form-urlencoded``;
    add a ``files=`` argument with an empty placeholder to force the
    request to multipart/form-data instead.
    """
    response = piper_app.post(
        "/infer",
        data={"text": "Multipart via individual fields", "length_scale": "1.1"},
        files={"_marker": ("", b"", "text/plain")},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    InferResponse.model_validate(body)


def test_infer_multipart_rejects_empty_form(piper_app):
    response = piper_app.post("/infer", files={"unrelated": ("x.txt", b"", "text/plain")})
    assert response.status_code == 400
    envelope = FailureEnvelope.model_validate(response.json())
    assert envelope.error.code == "malformed_input"


def test_capabilities_declares_both_content_types(piper_app):
    """Verify §3.5 — multipart must be advertised alongside JSON."""
    response = piper_app.get("/capabilities")
    caps = CapabilitiesResponse.model_validate(response.json())
    ct = caps.endpoints.infer.input_content_types
    assert "multipart/form-data" in ct
    assert "application/json" in ct


# §7.1 — adapter-specific codes are prefix-namespaced


def test_loading_state_returns_prefix_namespaced_code(piper_app, monkeypatch):
    """When the service is still loading, infer returns 503 with the
    prefix-namespaced ``piper.model_loading`` code, not the un-namespaced
    string. Catches regression of B18."""
    import adapters.piper.main as main_module
    from opennvr_adapter_sdk.contract import HealthStatus

    service = main_module._service
    # Simulate LOADING (state between init and successful load).
    monkeypatch.setattr(service, "_load_state", HealthStatus.LOADING)
    response = piper_app.post("/infer", json={"text": "hi"})
    assert response.status_code == 503
    envelope = FailureEnvelope.model_validate(response.json())
    assert envelope.error.code == "piper.model_loading"
    assert envelope.error.transient is True
    assert envelope.error.retry_after_ms == 2000


# §3.8 — case-insensitive Bearer scheme per RFC 7235


def test_auth_accepts_lowercase_bearer(piper_app_with_auth):
    client, token = piper_app_with_auth
    response = client.post(
        "/infer",
        json={"text": "hi"},
        headers={"Authorization": f"bearer {token}"},  # lowercase scheme
    )
    assert response.status_code == 200, response.text


# ── /infer/stream — not supported ──────────────────────────────────


def test_infer_stream_http_probe_returns_501(piper_app):
    response = piper_app.get("/infer/stream")
    assert response.status_code == 501
    envelope = FailureEnvelope.model_validate(response.json())
    assert envelope.error.category.value == "not_supported"
    assert envelope.error.code == "stream_not_supported"


def test_infer_stream_post_probe_returns_501(piper_app):
    response = piper_app.post("/infer/stream")
    assert response.status_code == 501


# ── correlation_id wire spec ───────────────────────────────────────


def test_correlation_id_echoed_when_supplied(piper_app):
    response = piper_app.get(
        "/capabilities",
        headers={"X-Correlation-Id": "test-corr-1234"},
    )
    assert response.headers.get("X-Correlation-Id") == "test-corr-1234"


def test_correlation_id_minted_when_absent(piper_app):
    response = piper_app.get("/capabilities")
    minted = response.headers.get("X-Correlation-Id")
    assert minted and len(minted) >= 8


def test_correlation_id_echoed_on_failure(piper_app):
    response = piper_app.post(
        "/infer",
        json={},
        headers={"X-Correlation-Id": "test-corr-fail"},
    )
    assert response.status_code == 400
    assert response.headers.get("X-Correlation-Id") == "test-corr-fail"


# ── Auth (§3.8) ────────────────────────────────────────────────────


def test_auth_rejects_missing_token(piper_app_with_auth):
    client, _token = piper_app_with_auth
    response = client.post("/infer", json={"text": "hi"})
    assert response.status_code == 401
    envelope = FailureEnvelope.model_validate(response.json())
    assert envelope.error.code == "auth_missing"


def test_auth_rejects_wrong_token(piper_app_with_auth):
    client, _token = piper_app_with_auth
    response = client.post(
        "/infer",
        json={"text": "hi"},
        headers={"Authorization": "Bearer wrong-token"},
    )
    assert response.status_code == 401
    envelope = FailureEnvelope.model_validate(response.json())
    assert envelope.error.code == "auth_invalid"


def test_auth_accepts_valid_token(piper_app_with_auth):
    client, token = piper_app_with_auth
    response = client.post(
        "/infer",
        json={"text": "hi"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200, response.text


def test_capabilities_open_during_grace_window(piper_app_with_auth):
    """During the 5-minute grace window /capabilities accepts probes."""
    client, _token = piper_app_with_auth
    response = client.get("/capabilities")
    assert response.status_code == 200


def test_failure_envelope_carries_correlation_id_on_auth_reject(piper_app_with_auth):
    client, _token = piper_app_with_auth
    response = client.post(
        "/infer",
        json={"text": "hi"},
        headers={"X-Correlation-Id": "auth-reject-corr"},
    )
    assert response.status_code == 401
    assert response.headers.get("X-Correlation-Id") == "auth-reject-corr"


# ── /voices — extra endpoint advertised in /capabilities ───────────


def test_voices_lists_installed_voices(piper_app):
    response = piper_app.get("/voices")
    assert response.status_code == 200
    body = response.json()
    assert body["default"] == "test-voice"
    assert any(v["name"] == "test-voice" for v in body["voices"])
