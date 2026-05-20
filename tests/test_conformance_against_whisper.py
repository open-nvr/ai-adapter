# Copyright (c) 2026 OpenNVR
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
End-to-end: ConformanceRunner pointed at the Whisper service.

Validates that the conformance kit handles the audio modality
correctly — it should detect ``modalities_in: ["audio"]`` on
/capabilities and drive /infer with multipart-audio rather than
JSON. Green here = "Whisper conforms" + "conformance kit supports
audio adapters."
"""
from __future__ import annotations

import pytest

from conformance.runner import CheckOutcome, ConformanceRunner
from tests._whisper_service_fixtures import (  # noqa: F401
    whisper_app,
    whisper_environment,
)


@pytest.fixture
def runner_against_whisper(whisper_app):
    runner = ConformanceRunner(base_url="", client=whisper_app)
    try:
        yield runner
    finally:
        runner.close()


def test_whisper_conforms_to_contract_v1(runner_against_whisper):
    report = runner_against_whisper.run_all()
    failures = [r for r in report.results if r.outcome == CheckOutcome.FAIL]
    assert not failures, "FAIL items:\n" + "\n".join(
        f"  {r.name}: {r.detail}" for r in failures
    )
    assert report.is_green


def test_whisper_capabilities_advertises_asr_tasks(runner_against_whisper):
    report = runner_against_whisper.run_all()
    caps_result = next(r for r in report.results if r.name == "capabilities")
    assert caps_result.outcome in (CheckOutcome.PASS, CheckOutcome.WARN)
    tasks = caps_result.evidence.get("tasks_advertised", [])
    assert "audio_transcription" in tasks
    assert "audio_translation" in tasks


def test_whisper_infer_check_uses_multipart_audio(runner_against_whisper):
    """Regression: the conformance kit must pick the multipart path
    for audio-modality adapters (added this commit alongside
    image-modality multipart from A2.2)."""
    report = runner_against_whisper.run_all()
    infer = next(r for r in report.results if r.name == "infer")
    assert infer.outcome == CheckOutcome.PASS, infer.detail


def test_whisper_stream_check_passes_for_unsupported_stream(runner_against_whisper):
    """Whisper declares infer_stream.supported=false and returns
    HTTP 501 with the canonical envelope — conformance check should
    mark this PASS (same as Piper)."""
    report = runner_against_whisper.run_all()
    stream_result = next(r for r in report.results if r.name == "infer_stream")
    assert stream_result.outcome == CheckOutcome.PASS, stream_result.detail


def test_whisper_metrics_and_hardware_checks(runner_against_whisper):
    report = runner_against_whisper.run_all()
    metrics = next(r for r in report.results if r.name == "metrics")
    assert metrics.outcome == CheckOutcome.PASS
    hwe = next(r for r in report.results if r.name == "hardware_evaluation")
    assert hwe.outcome == CheckOutcome.PASS
