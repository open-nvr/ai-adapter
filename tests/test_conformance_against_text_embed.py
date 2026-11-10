"""The conformance kit against the (faked) text-embed adapter."""
from __future__ import annotations

import pytest

from conformance.runner import CheckOutcome, ConformanceRunner
from tests._text_embed_fixtures import (  # noqa: F401
    text_embed_app,
    text_embed_environment,
)


@pytest.fixture
def runner_against_text_embed(text_embed_app):
    runner = ConformanceRunner(base_url="", client=text_embed_app)
    try:
        yield runner
    finally:
        runner.close()


def test_text_embed_conforms_to_contract_v1(runner_against_text_embed):
    report = runner_against_text_embed.run_all()
    failures = [r for r in report.results if r.outcome == CheckOutcome.FAIL]
    assert not failures, "FAIL items:\n" + "\n".join(f"  {r.name}: {r.detail}" for r in failures)
    assert report.is_green
