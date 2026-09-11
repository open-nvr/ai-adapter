# Copyright (c) 2026 OpenNVR
# SPDX-License-Identifier: Apache-2.0

"""
``opennvr-adapter validate`` — the conformance run, without a server.

``conform`` points at a running adapter; ``validate`` points at a
directory and drives the adapter in-process through FastAPI's test
client. Same checks, same verdicts, no port to bind and nothing to
start — which is what lets it run in CI on every commit rather than
only after a deployment.

A green run means KAI-C will accept the adapter.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from opennvr_adapter_sdk.conformance.runner import CheckOutcome, ConformanceRunner
from opennvr_adapter_sdk.discovery import NotAnAdapter, asgi_app, identity, load

_SYMBOL = {
    CheckOutcome.PASS: "✓",
    CheckOutcome.WARN: "!",
    CheckOutcome.FAIL: "✗",
    CheckOutcome.SKIP: "·",
}


def run_validate(directory: Path, *, as_json: bool = False) -> int:
    """Run the conformance suite against the adapter in ``directory``."""
    try:
        module, _name = load(Path(directory))
        app = asgi_app(module)
    except NotAnAdapter as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    try:
        from fastapi.testclient import TestClient
    except ImportError:  # pragma: no cover — fastapi is an SDK dependency
        print("error: fastapi is required to run `validate`", file=sys.stderr)
        return 2

    name, version, _tasks = identity(module)
    # The runner's client is duck-typed, and TestClient speaks the same
    # httpx surface — so the checks run against the real ASGI app with
    # no port bound and no server to start.
    base_url = "http://adapter.test"

    with TestClient(app, base_url=base_url) as client:
        report = ConformanceRunner(base_url, client=client).run_all()

    if as_json:
        print(json.dumps(_as_dict(report, name, version), indent=2))
        return 0 if report.is_green else 1

    print(f"opennvr-adapter validate — {name} {version}")
    for result in report.results:
        symbol = _SYMBOL.get(result.outcome, "?")
        print(f"  {symbol} {result.name:24} {result.detail}")
    print()
    if report.is_green:
        print(f"  OK — {report.passed} passed, {report.warned} warning(s), "
              f"{report.skipped} skipped. KAI-C will accept this adapter.")
        return 0
    print(f"  {report.failed} check(s) FAILED — KAI-C would refuse this "
          f"adapter. The contract is docs/AI_ADAPTER_CONTRACT.md.")
    return 1


def _as_dict(report, name: str, version: str) -> dict:
    return {
        "adapter": {"name": name, "version": version},
        "ok": report.is_green,
        "summary": {"passed": report.passed, "warned": report.warned,
                    "failed": report.failed, "skipped": report.skipped},
        "checks": [
            {"name": r.name, "outcome": r.outcome.value, "detail": r.detail,
             "latency_ms": r.latency_ms}
            for r in report.results
        ],
    }


__all__ = ["run_validate"]
