# Copyright (c) 2026 OpenNVR
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
End-to-end: ConformanceRunner pointed at the Piper service.

Green here is the only acceptance criterion for "Piper conforms to
the AI Adapter Contract v1." Future adapters reuse this fixture
pattern, swapping out the service fixture.
"""
from __future__ import annotations

import pytest

from conformance.runner import CheckOutcome, ConformanceRunner
from tests._piper_service_fixtures import (  # noqa: F401
    piper_app,
    piper_environment,
)


@pytest.fixture
def runner_against_piper(piper_app):
    """
    Build a ConformanceRunner whose client IS the FastAPI TestClient —
    no real network, no real port. The runner is duck-typed on the
    client's get/post methods so TestClient and httpx.Client are
    interchangeable.
    """
    runner = ConformanceRunner(
        base_url="",  # TestClient resolves relative paths itself
        client=piper_app,
    )
    try:
        yield runner
    finally:
        runner.close()


def test_piper_conforms_to_contract_v1(runner_against_piper):
    report = runner_against_piper.run_all()
    failures = [r for r in report.results if r.outcome == CheckOutcome.FAIL]
    assert not failures, "FAIL items:\n" + "\n".join(
        f"  {r.name}: {r.detail}" for r in failures
    )
    assert report.is_green


def test_piper_capabilities_check_records_tasks(runner_against_piper):
    report = runner_against_piper.run_all()
    caps_result = next(r for r in report.results if r.name == "capabilities")
    assert caps_result.outcome in (CheckOutcome.PASS, CheckOutcome.WARN)
    assert "speech_synthesis" in caps_result.evidence.get("tasks_advertised", [])


def test_piper_stream_check_passes_for_unsupported_stream(runner_against_piper):
    """Piper declares infer_stream.supported=false + returns HTTP 501 +
    FailureEnvelope — runner should mark this PASS."""
    report = runner_against_piper.run_all()
    stream_result = next(r for r in report.results if r.name == "infer_stream")
    assert stream_result.outcome == CheckOutcome.PASS, stream_result.detail


def test_piper_health_check_passes(runner_against_piper):
    report = runner_against_piper.run_all()
    health = next(r for r in report.results if r.name == "health")
    assert health.outcome == CheckOutcome.PASS


def test_piper_metrics_check_passes(runner_against_piper):
    report = runner_against_piper.run_all()
    metrics = next(r for r in report.results if r.name == "metrics")
    assert metrics.outcome == CheckOutcome.PASS


def test_piper_infer_check_passes(runner_against_piper):
    report = runner_against_piper.run_all()
    infer = next(r for r in report.results if r.name == "infer")
    assert infer.outcome == CheckOutcome.PASS, infer.detail
