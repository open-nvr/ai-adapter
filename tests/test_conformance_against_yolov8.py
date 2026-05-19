# Copyright (c) 2026 OpenNVR
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
End-to-end: ConformanceRunner pointed at the YOLOv8 service.

This is the first conformance test that exercises the §6 WebSocket
streaming protocol roundtrip — handshake → frame → result → close.
Green here is the acceptance criterion for "YOLOv8 conforms to the
AI Adapter Contract v1."
"""
from __future__ import annotations

import pytest

from conformance.runner import CheckOutcome, ConformanceRunner
from tests._yolov8_service_fixtures import (  # noqa: F401
    sample_jpeg,
    yolov8_app,
    yolov8_environment,
)


@pytest.fixture
def runner_against_yolov8(yolov8_app):
    runner = ConformanceRunner(
        base_url="",  # in-process TestClient
        client=yolov8_app,
    )
    try:
        yield runner
    finally:
        runner.close()


def test_yolov8_conforms_to_contract_v1(runner_against_yolov8):
    report = runner_against_yolov8.run_all()
    failures = [r for r in report.results if r.outcome == CheckOutcome.FAIL]
    assert not failures, "FAIL items:\n" + "\n".join(
        f"  {r.name}: {r.detail}" for r in failures
    )
    assert report.is_green


def test_yolov8_capabilities_includes_object_detection(runner_against_yolov8):
    report = runner_against_yolov8.run_all()
    caps_result = next(r for r in report.results if r.name == "capabilities")
    assert caps_result.outcome in (CheckOutcome.PASS, CheckOutcome.WARN)
    assert "object_detection" in caps_result.evidence.get("tasks_advertised", [])


def test_yolov8_stream_check_exercises_full_roundtrip(runner_against_yolov8):
    """The new §6 conformance check: real handshake → frame → result."""
    report = runner_against_yolov8.run_all()
    stream_result = next(r for r in report.results if r.name == "infer_stream")
    assert stream_result.outcome == CheckOutcome.PASS, stream_result.detail
    assert "roundtrip" in stream_result.detail.lower()


def test_yolov8_infer_check_passes(runner_against_yolov8):
    report = runner_against_yolov8.run_all()
    infer = next(r for r in report.results if r.name == "infer")
    assert infer.outcome == CheckOutcome.PASS, infer.detail


def test_yolov8_metrics_check_passes(runner_against_yolov8):
    report = runner_against_yolov8.run_all()
    metrics = next(r for r in report.results if r.name == "metrics")
    assert metrics.outcome == CheckOutcome.PASS


def test_yolov8_hardware_check_passes(runner_against_yolov8):
    """Verdict may be PASS or WARN — both are acceptable for CPU-only test env."""
    report = runner_against_yolov8.run_all()
    hwe = next(r for r in report.results if r.name == "hardware_evaluation")
    assert hwe.outcome == CheckOutcome.PASS
