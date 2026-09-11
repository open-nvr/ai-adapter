"""
opennvr-adapter-conformance — moved into the SDK.

The implementation now lives in :mod:`opennvr_adapter_sdk.conformance`
so that it ships in the wheel: a third-party model developer who has
only ``pip install opennvr-adapter-sdk`` can check their own adapter,
which was the whole point of having a conformance kit.

This module stays as a re-export so `python -m conformance`, this
repository's tests, and every README that names it keep working.
"""
from opennvr_adapter_sdk.conformance import (
    CheckOutcome,
    CheckResult,
    ConformanceReport,
    ConformanceRunner,
)

__all__ = ["ConformanceRunner", "ConformanceReport", "CheckOutcome", "CheckResult"]
