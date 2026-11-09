"""The conformance kit against the (faked) Whisper-QNN adapter: the third
audio adapter must pass the same checks the whisper and parakeet ones do."""
from __future__ import annotations

import pytest

from conformance.runner import CheckOutcome, ConformanceRunner
from tests._whisper_qnn_fixtures import (  # noqa: F401
    sample_wav, whisper_qnn_app, whisper_qnn_environment,
)


@pytest.fixture
def runner_against_whisper_qnn(whisper_qnn_app):
    runner = ConformanceRunner(base_url="", client=whisper_qnn_app)
    try:
        yield runner
    finally:
        runner.close()


def test_whisper_qnn_conforms_to_contract_v1(runner_against_whisper_qnn):
    report = runner_against_whisper_qnn.run_all()
    failures = [r for r in report.results if r.outcome == CheckOutcome.FAIL]
    assert not failures, "FAIL items:\n" + "\n".join(f"  {r.name}: {r.detail}" for r in failures)
    assert report.is_green
