"""
opennvr-adapter-conformance — point at any adapter URL and check whether
it conforms to the AI Adapter Contract v1.

The full spec is ``open-nvr/docs/AI_ADAPTER_CONTRACT.md``. Wire shapes
are validated against the Pydantic models in
``ai-adapter/app/interfaces/contract.py``.

Usage:
    python -m conformance http://localhost:9001
    python -m conformance http://localhost:9001 --token <bearer>
"""
from conformance.runner import ConformanceRunner, CheckOutcome, CheckResult

__all__ = ["ConformanceRunner", "CheckOutcome", "CheckResult"]
