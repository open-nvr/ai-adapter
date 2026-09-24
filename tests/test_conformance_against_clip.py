# Copyright (c) 2026 OpenNVR
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
End-to-end: ConformanceRunner pointed at the CLIP embedding service.

Green here is the acceptance criterion for "the CLIP adapter conforms
to the AI Adapter Contract v1". The adapter advertises no stream, so
the §6 check is expected to report that honestly rather than fail. The
model is stubbed by the shared fixtures — no weights, no network.
"""
from __future__ import annotations

import pytest

from conformance.runner import CheckOutcome, ConformanceRunner
from tests._clip_service_fixtures import (  # noqa: F401
    clip_app,
    clip_environment,
    small_jpeg,
)


@pytest.fixture
def runner_against_clip(clip_app):
    runner = ConformanceRunner(base_url="", client=clip_app)
    try:
        yield runner
    finally:
        runner.close()


def test_clip_conforms_to_contract_v1(runner_against_clip):
    report = runner_against_clip.run_all()
    failures = [r for r in report.results if r.outcome == CheckOutcome.FAIL]
    assert not failures, "FAIL items:\n" + "\n".join(
        f"  {r.name}: {r.detail}" for r in failures)
    assert report.is_green
