# Copyright (c) 2026 OpenNVR
# SPDX-License-Identifier: AGPL-3.0-or-later

"""CLI entry-point for ``python -m conformance``."""
from __future__ import annotations

import argparse
import json
import os
import sys

from conformance.runner import CheckOutcome, ConformanceRunner


# Terminal colour escapes; disabled when stdout is not a TTY.
_COLOUR: dict[CheckOutcome, str] = {
    CheckOutcome.PASS: "\033[32m",   # green
    CheckOutcome.WARN: "\033[33m",   # yellow
    CheckOutcome.FAIL: "\033[31m",   # red
    CheckOutcome.SKIP: "\033[90m",   # grey
}
_RESET = "\033[0m"


def _colour(outcome: CheckOutcome, use_colour: bool) -> str:
    if not use_colour:
        return outcome.value
    return f"{_COLOUR[outcome]}{outcome.value:>4}{_RESET}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m conformance",
        description="Validate an AI adapter against the AI Adapter Contract v1.",
    )
    parser.add_argument("url", help="Adapter base URL, e.g. http://localhost:9001")
    parser.add_argument(
        "--token",
        default=os.environ.get("OPENNVR_ADAPTER_TOKEN"),
        help="Bearer token (or set OPENNVR_ADAPTER_TOKEN)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="emit_json",
        help="Emit the report as JSON to stdout (suppresses pretty output)",
    )
    parser.add_argument(
        "--no-colour",
        action="store_true",
        help="Disable ANSI colour codes",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=5.0,
        help="Per-request timeout in seconds (default: 5.0)",
    )
    args = parser.parse_args(argv)

    use_colour = (not args.no_colour) and sys.stdout.isatty() and not args.emit_json

    with ConformanceRunner(args.url, token=args.token, timeout_seconds=args.timeout) as runner:
        report = runner.run_all()

    if args.emit_json:
        json.dump(
            {
                "base_url": report.base_url,
                "results": [
                    {
                        "name": r.name,
                        "outcome": r.outcome.value,
                        "detail": r.detail,
                        "latency_ms": r.latency_ms,
                        "evidence": r.evidence,
                    }
                    for r in report.results
                ],
                "summary": {
                    "passed": report.passed,
                    "warned": report.warned,
                    "failed": report.failed,
                    "skipped": report.skipped,
                    "green": report.is_green,
                },
            },
            sys.stdout,
            indent=2,
        )
        sys.stdout.write("\n")
    else:
        print(f"\nOpenNVR Adapter Contract v1 — conformance report for {report.base_url}\n")
        for r in report.results:
            tag = _colour(r.outcome, use_colour)
            detail = r.detail or ""
            latency = f"{r.latency_ms:>5}ms" if r.latency_ms else " " * 7
            print(f"  [{tag}] {r.name:<22s} {latency}  {detail}")
        summary = (
            f"\n  {report.passed} pass · {report.warned} warn · "
            f"{report.failed} fail · {report.skipped} skip\n"
        )
        if report.is_green:
            verdict = "GREEN — KAI-C will accept this adapter."
        else:
            verdict = "RED — fix the FAIL items before submitting."
        if use_colour:
            verdict = (_COLOUR[CheckOutcome.PASS] if report.is_green else _COLOUR[CheckOutcome.FAIL]) + verdict + _RESET
        print(summary)
        print(f"  {verdict}\n")

    return 0 if report.is_green else 1


if __name__ == "__main__":
    raise SystemExit(main())
