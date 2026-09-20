# Copyright (c) 2026 OpenNVR
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
End-to-end: ConformanceRunner pointed at the package-detection service.

Green here is the acceptance criterion for "the package adapter
conforms to the AI Adapter Contract v1". The adapter advertises no
stream, so the §6 check is expected to report that honestly rather
than fail. The model is stubbed by the shared fixtures — no weights,
no network.
"""
from __future__ import annotations

import pytest

from conformance.runner import CheckOutcome, ConformanceRunner
from tests._package_detection_service_fixtures import (  # noqa: F401
    package_detection_app,
    package_detection_environment,
    square_jpeg,
)


@pytest.fixture
def runner_against_package_detection(package_detection_app):
    runner = ConformanceRunner(base_url="", client=package_detection_app)
    try:
        yield runner
    finally:
        runner.close()


def test_package_detection_conforms_to_contract_v1(runner_against_package_detection):
    report = runner_against_package_detection.run_all()
    failures = [r for r in report.results if r.outcome == CheckOutcome.FAIL]
    assert not failures, "FAIL items:\n" + "\n".join(
        f"  {r.name}: {r.detail}" for r in failures)
    assert report.is_green


def test_package_detection_capabilities_advertise_the_task(runner_against_package_detection):
    report = runner_against_package_detection.run_all()
    caps = next(r for r in report.results if r.name == "capabilities")
    assert caps.outcome in (CheckOutcome.PASS, CheckOutcome.WARN)
    assert caps.evidence.get("tasks_advertised") == ["package_detection"]


def test_package_detection_infer_check_passes(runner_against_package_detection):
    """A registered sample frame for ``package_detection`` means the
    kit actually POSTs bytes (conformance runner SAMPLE tables)."""
    report = runner_against_package_detection.run_all()
    infer = next(r for r in report.results if r.name == "infer")
    assert infer.outcome == CheckOutcome.PASS, infer.detail


def test_package_detection_metrics_and_hardware_checks_pass(runner_against_package_detection):
    report = runner_against_package_detection.run_all()
    for name in ("metrics", "hardware_evaluation"):
        result = next(r for r in report.results if r.name == name)
        assert result.outcome == CheckOutcome.PASS, result.detail
