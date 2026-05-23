# Copyright (c) 2026 OpenNVR
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
End-to-end: ConformanceRunner pointed at the InsightFace service.

Validates that the kit handles a multi-task face adapter and that
the SDK-migrated InsightFace adapter declares the right contract
surface — image-in, multiple tasks advertised, no streaming.
"""
from __future__ import annotations

import pytest

from conformance.runner import CheckOutcome, ConformanceRunner
from tests._insightface_service_fixtures import (  # noqa: F401
    insightface_app,
    insightface_environment,
)


@pytest.fixture
def runner_against_insightface(insightface_app):
    runner = ConformanceRunner(base_url="", client=insightface_app)
    try:
        yield runner
    finally:
        runner.close()


def test_insightface_conforms_to_contract_v1(runner_against_insightface):
    report = runner_against_insightface.run_all()
    failures = [r for r in report.results if r.outcome == CheckOutcome.FAIL]
    assert not failures, "FAIL items:\n" + "\n".join(
        f"  {r.name}: {r.detail}" for r in failures
    )
    assert report.is_green


def test_insightface_advertises_three_face_tasks(runner_against_insightface):
    report = runner_against_insightface.run_all()
    caps_result = next(r for r in report.results if r.name == "capabilities")
    assert caps_result.outcome in (CheckOutcome.PASS, CheckOutcome.WARN)
    tasks = caps_result.evidence.get("tasks_advertised", [])
    assert "face_detection" in tasks
    assert "face_recognition" in tasks
    assert "face_embedding" in tasks


def test_insightface_infer_check_uses_multipart_image(runner_against_insightface):
    report = runner_against_insightface.run_all()
    infer = next(r for r in report.results if r.name == "infer")
    assert infer.outcome == CheckOutcome.PASS, infer.detail


def test_insightface_stream_check_passes_for_unsupported_stream(runner_against_insightface):
    """The InsightFace adapter declares supports_stream=False (face
    recognition is event-driven, not frame-rate). The conformance
    kit should mark this PASS — HTTP 501 with the canonical envelope."""
    report = runner_against_insightface.run_all()
    stream_result = next(r for r in report.results if r.name == "infer_stream")
    assert stream_result.outcome == CheckOutcome.PASS, stream_result.detail


def test_insightface_metrics_and_hardware_checks(runner_against_insightface):
    report = runner_against_insightface.run_all()
    metrics = next(r for r in report.results if r.name == "metrics")
    assert metrics.outcome == CheckOutcome.PASS
    hwe = next(r for r in report.results if r.name == "hardware_evaluation")
    assert hwe.outcome == CheckOutcome.PASS
