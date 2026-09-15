# Copyright (c) 2026 OpenNVR
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
End-to-end: ConformanceRunner pointed at the YOLO-pose service.

Green here is the acceptance criterion for "the pose adapter conforms
to the AI Adapter Contract v1" — including the §6 WebSocket streaming
roundtrip (handshake → frame → result → close), which is the path the
wand-compliance app runs at 10 fps.

Mirrors tests/test_conformance_against_yolov8.py; the model is stubbed
by the shared fixtures, so this needs no weights and no network.
"""
from __future__ import annotations

import pytest

from conformance.runner import CheckOutcome, ConformanceRunner
from tests._yolo_pose_service_fixtures import (  # noqa: F401
    square_jpeg,
    yolo_pose_app,
    yolo_pose_environment,
)


@pytest.fixture
def runner_against_yolo_pose(yolo_pose_app):
    runner = ConformanceRunner(
        base_url="",  # in-process TestClient
        client=yolo_pose_app,
    )
    try:
        yield runner
    finally:
        runner.close()


def test_yolo_pose_conforms_to_contract_v1(runner_against_yolo_pose):
    report = runner_against_yolo_pose.run_all()
    failures = [r for r in report.results if r.outcome == CheckOutcome.FAIL]
    assert not failures, "FAIL items:\n" + "\n".join(
        f"  {r.name}: {r.detail}" for r in failures
    )
    assert report.is_green


def test_yolo_pose_capabilities_includes_pose_estimation(runner_against_yolo_pose):
    report = runner_against_yolo_pose.run_all()
    caps_result = next(r for r in report.results if r.name == "capabilities")
    assert caps_result.outcome in (CheckOutcome.PASS, CheckOutcome.WARN)
    assert "pose_estimation" in caps_result.evidence.get("tasks_advertised", [])


def test_yolo_pose_infer_check_passes(runner_against_yolo_pose):
    """A registered sample frame for ``pose_estimation`` means the kit
    actually POSTs bytes rather than warning that it has nothing to
    send (conformance/runner.py SAMPLE_STREAM_FRAMES)."""
    report = runner_against_yolo_pose.run_all()
    infer = next(r for r in report.results if r.name == "infer")
    assert infer.outcome == CheckOutcome.PASS, infer.detail


def test_yolo_pose_stream_check_exercises_full_roundtrip(runner_against_yolo_pose):
    report = runner_against_yolo_pose.run_all()
    stream_result = next(r for r in report.results if r.name == "infer_stream")
    assert stream_result.outcome == CheckOutcome.PASS, stream_result.detail
    assert "roundtrip" in stream_result.detail.lower()


def test_yolo_pose_metrics_check_passes(runner_against_yolo_pose):
    report = runner_against_yolo_pose.run_all()
    metrics = next(r for r in report.results if r.name == "metrics")
    assert metrics.outcome == CheckOutcome.PASS


def test_yolo_pose_hardware_check_passes(runner_against_yolo_pose):
    report = runner_against_yolo_pose.run_all()
    hwe = next(r for r in report.results if r.name == "hardware_evaluation")
    assert hwe.outcome == CheckOutcome.PASS
