"""The conformance kit against the (faked) Parakeet adapter: the second
audio adapter must pass the same checks the whisper one does."""
from __future__ import annotations

import pytest

from conformance.runner import CheckOutcome, ConformanceRunner
from tests._parakeet_service_fixtures import (  # noqa: F401
    parakeet_app, parakeet_environment, sample_wav,
)


@pytest.fixture
def runner_against_parakeet(parakeet_app):
    runner = ConformanceRunner(base_url="", client=parakeet_app)
    try:
        yield runner
    finally:
        runner.close()


def test_parakeet_conforms_to_contract_v1(runner_against_parakeet):
    report = runner_against_parakeet.run_all()
    failures = [r for r in report.results if r.outcome == CheckOutcome.FAIL]
    assert not failures, "FAIL items:\n" + "\n".join(f"  {r.name}: {r.detail}" for r in failures)
    assert report.is_green
