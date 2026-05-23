# Copyright (c) 2026 OpenNVR
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
End-to-end: ConformanceRunner pointed at the fast-plate-ocr service.

Validates that the kit handles a third image-modality adapter
(in addition to YOLOv8) and that the LPR adapter declares the right
contract surface — image-in, text-out, no streaming.
"""
from __future__ import annotations

import pytest

from conformance.runner import CheckOutcome, ConformanceRunner
from tests._fast_plate_ocr_service_fixtures import (  # noqa: F401
    fast_plate_ocr_app,
    fast_plate_ocr_environment,
)


@pytest.fixture
def runner_against_lpr(fast_plate_ocr_app):
    runner = ConformanceRunner(base_url="", client=fast_plate_ocr_app)
    try:
        yield runner
    finally:
        runner.close()


def test_lpr_conforms_to_contract_v1(runner_against_lpr):
    report = runner_against_lpr.run_all()
    failures = [r for r in report.results if r.outcome == CheckOutcome.FAIL]
    assert not failures, "FAIL items:\n" + "\n".join(
        f"  {r.name}: {r.detail}" for r in failures
    )
    assert report.is_green


def test_lpr_capabilities_advertises_license_plate_recognition(runner_against_lpr):
    report = runner_against_lpr.run_all()
    caps_result = next(r for r in report.results if r.name == "capabilities")
    assert caps_result.outcome in (CheckOutcome.PASS, CheckOutcome.WARN)
    tasks = caps_result.evidence.get("tasks_advertised", [])
    assert "license_plate_recognition" in tasks


def test_lpr_infer_check_uses_multipart_image(runner_against_lpr):
    """LPR declares modalities_in=['image']; the conformance runner
    should pick the multipart-image path, not JSON."""
    report = runner_against_lpr.run_all()
    infer = next(r for r in report.results if r.name == "infer")
    assert infer.outcome == CheckOutcome.PASS, infer.detail


def test_lpr_stream_check_passes_for_unsupported_stream(runner_against_lpr):
    """The LPR adapter declares supports_stream=False; the conformance
    check should mark this PASS (HTTP 501 with the canonical envelope)."""
    report = runner_against_lpr.run_all()
    stream_result = next(r for r in report.results if r.name == "infer_stream")
    assert stream_result.outcome == CheckOutcome.PASS, stream_result.detail


def test_lpr_metrics_and_hardware_checks(runner_against_lpr):
    report = runner_against_lpr.run_all()
    metrics = next(r for r in report.results if r.name == "metrics")
    assert metrics.outcome == CheckOutcome.PASS
    hwe = next(r for r in report.results if r.name == "hardware_evaluation")
    assert hwe.outcome == CheckOutcome.PASS
