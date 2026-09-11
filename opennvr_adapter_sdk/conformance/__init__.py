# Copyright (c) 2026 OpenNVR
# SPDX-License-Identifier: Apache-2.0

"""
Conformance — point at any adapter and check it against the contract.

This is the answer to "is my adapter correct?", and it now ships INSIDE
the SDK. It used to live only in the ai-adapter repository, which meant
the one tool that tells a model developer their adapter will be accepted
was unreachable by anybody who had merely ``pip install
opennvr-adapter-sdk`` — exactly the audience it is for.

It probes every mandatory endpoint, validates the wire shapes against
the Pydantic models in :mod:`~.contract`, and reports PASS / WARN /
FAIL / SKIP. A green run means KAI-C will accept the adapter.

From the command line::

    opennvr-adapter conform http://localhost:9001 --token <bearer>
    opennvr-adapter validate .        # in-process, no server to start

In a test, hand it a client backed by FastAPI's ``TestClient`` and no
network is involved::

    report = ConformanceRunner(base_url, client=client).run()
    assert report.ok
"""
from opennvr_adapter_sdk.conformance.runner import (
    CheckOutcome,
    CheckResult,
    ConformanceReport,
    ConformanceRunner,
)

__all__ = [
    "ConformanceRunner",
    "ConformanceReport",
    "CheckOutcome",
    "CheckResult",
]
