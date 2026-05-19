# Copyright (c) 2026 OpenNVR
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Integration tests for the Whisper contract service.

Covers:
  - HTTP /health, /capabilities, /hardware/evaluation, /metrics
  - /infer multipart (audio file) + application/json (audio_b64)
  - §5.3 ASR output shape (transcript + language + segments with
    int ms timestamps)
  - audio_translation forces language = "en"
  - Adapter-specific extras (language_confidence, duration_seconds,
    translated_to_english, model) ride alongside the canonical
    §5.3 keys
  - /infer/stream refused with HTTP 501 (streaming ASR is deferred)
  - Auth + correlation_id (same shape as Piper/YOLOv8)
"""
from __future__ import annotations

import base64
import json

from app.interfaces.contract import (
    AsrResult,
    CapabilitiesResponse,
    FailureEnvelope,
    HardwareEvaluationResponse,
    HealthResponse,
    InferResponse,
)
from tests._whisper_service_fixtures import (  # noqa: F401
    sample_wav,
    whisper_app,
    whisper_app_with_auth,
    whisper_environment,
)


# ── /health ────────────────────────────────────────────────────────


def test_health_returns_valid_HealthResponse(whisper_app):
    response = whisper_app.get("/health")
    assert response.status_code == 200
    health = HealthResponse.model_validate(response.json())
    assert health.status.value == "ok"
    assert health.adapter_name == "whisper-asr"


# ── /capabilities ──────────────────────────────────────────────────


def test_capabilities_declares_asr_tasks(whisper_app):
    caps = CapabilitiesResponse.model_validate(whisper_app.get("/capabilities").json())
    assert "audio_transcription" in caps.tasks_advertised
    assert "audio_translation" in caps.tasks_advertised
    assert caps.endpoints.infer.supported is True
    assert "multipart/form-data" in caps.endpoints.infer.input_content_types
    assert "application/json" in caps.endpoints.infer.input_content_types
    # Streaming ASR deferred
    assert caps.endpoints.infer_stream.supported is False
    # Audio modality
    assert caps.model.modalities_in == ["audio"]
    assert caps.model.modalities_out == ["text"]


def test_capabilities_declares_gpu_permission(whisper_app):
    caps = CapabilitiesResponse.model_validate(whisper_app.get("/capabilities").json())
    assert caps.permissions.gpu is True
    assert caps.scheduling.fair_queuing.value == "per_camera"
    assert caps.scheduling.max_inflight == 1  # honest singleton concurrency


def test_capabilities_exposes_model_fingerprint(whisper_app):
    caps = CapabilitiesResponse.model_validate(whisper_app.get("/capabilities").json())
    assert caps.model.fingerprint is not None
    assert caps.model.fingerprint.startswith("sha256:")


def test_capabilities_advertises_whisper_subdir_in_host_filesystem(whisper_app):
    """Regression for A2.3c peer-review M1: capabilities.permissions
    .host_filesystem must advertise the Whisper-specific subdir, NOT
    the parent ``MODEL_WEIGHTS_DIR``. KAI-C's §8 operator-policy
    comparator does string-equality, so widening to the parent silently
    breaks any prior approval of the narrower path."""
    caps = CapabilitiesResponse.model_validate(whisper_app.get("/capabilities").json())
    assert len(caps.permissions.host_filesystem) == 1
    declared = caps.permissions.host_filesystem[0]
    # Must end in 'whisper' — i.e., be the Whisper-specific subdir,
    # not the parent weights root.
    assert declared.endswith("whisper"), (
        f"host_filesystem widened to weights parent: {declared!r}; "
        "should be the Whisper-specific subdir."
    )


# ── /hardware/evaluation ───────────────────────────────────────────


def test_hardware_evaluation_warns_on_cpu(whisper_app):
    hwe = HardwareEvaluationResponse.model_validate(
        whisper_app.get("/hardware/evaluation").json()
    )
    # Test fixture uses device=cpu, so verdict is warn (CPU works
    # but slower than CUDA).
    assert hwe.verdict.value in ("warn", "ok")
    assert hwe.details["model_size"] == "tiny"


# ── /metrics ───────────────────────────────────────────────────────


def test_metrics_emits_prometheus_baseline(whisper_app):
    body = whisper_app.get("/metrics").text
    for name in (
        "adapter_infer_total",
        "adapter_infer_latency_seconds",
        "adapter_model_loaded",
        "adapter_inflight_requests",
        "adapter_queue_depth",
    ):
        assert name in body
    assert "adapter_model_loaded 1" in body


# ── /infer multipart (§3.5 canonical) ──────────────────────────────


def test_infer_multipart_audio_returns_asr_result(whisper_app, sample_wav):
    response = whisper_app.post(
        "/infer",
        files={"audio": ("clip.wav", sample_wav, "audio/wav")},
    )
    assert response.status_code == 200, response.text
    infer = InferResponse.model_validate(response.json())
    result = infer.result
    # §5.3 shape — transcript + language + segments with int ms
    asr = AsrResult.model_validate(result)
    assert asr.language == "en"
    assert asr.transcript == "Hello world"  # third segment was empty-text → dropped
    assert len(asr.segments) == 2
    assert asr.segments[0].start_ms == 0
    assert asr.segments[0].end_ms == 1500
    assert asr.segments[0].text == "Hello"
    assert asr.segments[1].start_ms == 1500
    assert asr.segments[1].end_ms == 3000

    # Adapter-specific extras
    assert result["language_confidence"] == 0.97
    assert result["duration_seconds"] == 4.2
    assert result["translated_to_english"] is False
    assert result["model"] == "whisper-tiny"


def test_infer_multipart_translation_forces_english(whisper_app, sample_wav):
    response = whisper_app.post(
        "/infer",
        files={"audio": ("clip.wav", sample_wav, "audio/wav")},
        data={"params": json.dumps({"task": "audio_translation"})},
    )
    assert response.status_code == 200
    result = InferResponse.model_validate(response.json()).result
    assert result["language"] == "en"
    assert result["translated_to_english"] is True


def test_infer_multipart_rejects_missing_audio(whisper_app):
    response = whisper_app.post(
        "/infer",
        data={"params": json.dumps({})},
        files={"_marker": ("", b"", "text/plain")},
    )
    assert response.status_code == 400
    envelope = FailureEnvelope.model_validate(response.json())
    assert envelope.error.code == "malformed_input"


# ── /infer JSON (convenience path) ─────────────────────────────────


def test_infer_json_with_base64_audio(whisper_app, sample_wav):
    body = {"audio_b64": base64.b64encode(sample_wav).decode("ascii")}
    response = whisper_app.post("/infer", json=body)
    assert response.status_code == 200, response.text
    InferResponse.model_validate(response.json())


def test_infer_json_rejects_invalid_base64(whisper_app):
    response = whisper_app.post("/infer", json={"audio_b64": "%%%not-base64%%%"})
    assert response.status_code == 400
    envelope = FailureEnvelope.model_validate(response.json())
    assert envelope.error.code == "malformed_input"


def test_infer_json_rejects_missing_audio_b64(whisper_app):
    response = whisper_app.post("/infer", json={"task": "audio_transcription"})
    assert response.status_code == 400


def test_infer_rejects_unsupported_content_type(whisper_app):
    response = whisper_app.post(
        "/infer",
        content=b"some bytes",
        headers={"Content-Type": "application/octet-stream"},
    )
    assert response.status_code == 415


def test_infer_rejects_oversized_audio(whisper_app):
    huge = bytes(25 * 1024 * 1024 + 1)
    response = whisper_app.post(
        "/infer",
        files={"audio": ("huge.wav", huge, "audio/wav")},
    )
    assert response.status_code == 413


# ── Params validation ──────────────────────────────────────────────


def test_infer_rejects_unknown_task(whisper_app, sample_wav):
    response = whisper_app.post(
        "/infer",
        files={"audio": ("clip.wav", sample_wav, "audio/wav")},
        data={"params": json.dumps({"task": "audio_singalong"})},
    )
    assert response.status_code == 400
    envelope = FailureEnvelope.model_validate(response.json())
    assert envelope.error.code == "malformed_input"
    assert "audio_transcription" in envelope.error.message


def test_infer_rejects_non_integer_beam_size(whisper_app, sample_wav):
    response = whisper_app.post(
        "/infer",
        files={"audio": ("clip.wav", sample_wav, "audio/wav")},
        data={"params": json.dumps({"beam_size": "not-an-int"})},
    )
    assert response.status_code == 400


def test_infer_rejects_out_of_range_beam_size(whisper_app, sample_wav):
    response = whisper_app.post(
        "/infer",
        files={"audio": ("clip.wav", sample_wav, "audio/wav")},
        data={"params": json.dumps({"beam_size": 99})},
    )
    assert response.status_code == 400


# ── /infer/stream — not supported ──────────────────────────────────


def test_infer_stream_http_probe_returns_501(whisper_app):
    response = whisper_app.get("/infer/stream")
    assert response.status_code == 501
    envelope = FailureEnvelope.model_validate(response.json())
    assert envelope.error.category.value == "not_supported"
    assert envelope.error.code == "stream_not_supported"


# ── correlation_id wire spec ───────────────────────────────────────


def test_correlation_id_echoed_when_supplied(whisper_app):
    response = whisper_app.get(
        "/capabilities",
        headers={"X-Correlation-Id": "whisper-corr-1"},
    )
    assert response.headers.get("X-Correlation-Id") == "whisper-corr-1"


def test_correlation_id_minted_when_absent(whisper_app):
    response = whisper_app.get("/capabilities")
    assert response.headers.get("X-Correlation-Id")


# ── Auth ───────────────────────────────────────────────────────────


def test_auth_rejects_missing_token_on_infer(whisper_app_with_auth, sample_wav):
    client, _ = whisper_app_with_auth
    response = client.post(
        "/infer",
        files={"audio": ("clip.wav", sample_wav, "audio/wav")},
    )
    assert response.status_code == 401


def test_auth_accepts_valid_token_on_infer(whisper_app_with_auth, sample_wav):
    client, token = whisper_app_with_auth
    response = client.post(
        "/infer",
        files={"audio": ("clip.wav", sample_wav, "audio/wav")},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200
