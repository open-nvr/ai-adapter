# Copyright (c) 2026 OpenNVR
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Negative-path regression for the conformance kit (conformance/runner.py).

Every existing conformance test drives a *conforming* adapter green.
That proves the runner accepts good adapters but says nothing about
whether it actually DETECTS a bad one — and detection is the kit's
entire purpose. Here we stand up a deliberately non-conforming
in-process adapter (its /capabilities body carries an unknown field,
violating the extra='forbid' envelope) and assert the runner reports
red: ``report.is_green is False`` and the ``capabilities`` check is a
hard FAIL.

Framework-free: a couple of stub FastAPI routes, no torch/ultralytics/
onnxruntime, no real weights.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest
from fastapi import FastAPI
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.testclient import TestClient

from conformance.runner import CheckOutcome, ConformanceRunner


# A fully-valid /capabilities body; the fixture injects one bad key.
_VALID_CAPABILITIES = {
    "adapter": {
        "name": "nonconforming-adapter",
        "version": "1.0.0",
        "vendor": "open-nvr",
        "license": "AGPL-3.0",
        "supported_contract_versions": ["1"],
    },
    "model": {
        "name": "stub-model",
        "version": "1",
        "framework": "test",
        "fingerprint": "sha256:deadbeef",
    },
    "endpoints": {
        "infer": {"supported": True},
        "infer_stream": {"supported": False},
    },
    "tasks_advertised": ["object_detection"],
    "scheduling": {},
}


def _build_nonconforming_app() -> FastAPI:
    """A stub adapter that conforms on every endpoint EXCEPT
    /capabilities, which returns an unknown top-level field. That single
    violation is what the runner must catch."""
    app = FastAPI()

    @app.get("/health")
    def health() -> JSONResponse:
        return JSONResponse(
            {
                "status": "ok",
                "adapter_name": "nonconforming-adapter",
                "adapter_version": "1.0.0",
                "model_name": "stub-model",
                "model_version": "1",
                "started_at": datetime.now(timezone.utc).isoformat(),
                "uptime_seconds": 0,
            }
        )

    @app.get("/capabilities")
    def capabilities() -> JSONResponse:
        # ``surprise`` is not part of CapabilitiesResponse — extra='forbid'
        # means this body must fail validation inside the runner.
        return JSONResponse({**_VALID_CAPABILITIES, "surprise": 1})

    @app.get("/hardware/evaluation")
    def hardware_evaluation() -> JSONResponse:
        return JSONResponse(
            {
                "verdict": "ok",
                "reasoning": "stub always ok",
                "checked_at": datetime.now(timezone.utc).isoformat(),
                "details": {},
            }
        )

    @app.get("/metrics")
    def metrics() -> PlainTextResponse:
        return PlainTextResponse(
            "adapter_infer_total 0\n"
            "adapter_infer_latency_seconds 0\n"
            "adapter_model_loaded 1\n"
        )

    return app


@pytest.fixture
def runner_against_nonconforming():
    app = TestClient(_build_nonconforming_app())
    runner = ConformanceRunner(base_url="", client=app)
    try:
        yield runner
    finally:
        runner.close()


def test_runner_reports_red_for_nonconforming_adapter(runner_against_nonconforming):
    """The kit must NOT be green when an adapter violates the contract."""
    report = runner_against_nonconforming.run_all()
    assert report.is_green is False
    assert report.failed >= 1


def test_runner_flags_capabilities_check_as_fail(runner_against_nonconforming):
    """The specific violation (bad /capabilities body) surfaces as a
    FAIL on the capabilities check, not a WARN or a swallowed error."""
    report = runner_against_nonconforming.run_all()
    caps = next(r for r in report.results if r.name == "capabilities")
    assert caps.outcome == CheckOutcome.FAIL
    assert "CapabilitiesResponse" in caps.detail
