# Copyright (c) 2026 OpenNVR
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Unit tests for __SERVICE_CLASS__.

These tests cover the AdapterService surface — load lifecycle,
input validation, and the error-envelope shape. They do NOT exercise
the actual model inference path, since that's adapter-specific.
Replace the toy assertions in TestInference with real ones for your
model: detection shape, output ordering, per-camera state, anything
your adapter promises.

Run with:

    cd ai-adapter && pytest tests/test___DIR_NAME___service.py -v
"""
from __future__ import annotations

import pytest

# TODO: if your adapter requires an optional ML library that may not
# be installed in the test venv, gate the entire module with:
#
#     my_ml_library = pytest.importorskip("my_ml_library")
#
# (See tests/test_bytetrack_service.py for the pattern.)

from opennvr_adapter_sdk import ErrorCategory, HardwareVerdict, ServiceError


# ────────────────────────────────────────────────────────────────────
# Fixtures
# ────────────────────────────────────────────────────────────────────


@pytest.fixture
def service():
    """A fresh, loaded __SERVICE_CLASS__."""
    from adapters.__DIR_NAME__.service import __SERVICE_CLASS__

    svc = __SERVICE_CLASS__()
    svc.load()
    assert svc.is_ready(), "__SERVICE_CLASS__ failed to load"
    return svc


# ────────────────────────────────────────────────────────────────────
# Load lifecycle
# ────────────────────────────────────────────────────────────────────


class TestLoadLifecycle:

    def test_fresh_service_is_not_ready(self):
        from adapters.__DIR_NAME__.service import __SERVICE_CLASS__

        svc = __SERVICE_CLASS__()
        assert not svc.is_ready()

    def test_load_marks_service_ready(self):
        from adapters.__DIR_NAME__.service import __SERVICE_CLASS__

        svc = __SERVICE_CLASS__()
        svc.load()
        assert svc.is_ready()

    def test_load_is_idempotent(self):
        """Calling load() twice must not double-initialise state."""
        from adapters.__DIR_NAME__.service import __SERVICE_CLASS__

        svc = __SERVICE_CLASS__()
        svc.load()
        svc.load()  # no-op
        assert svc.is_ready()

    def test_hardware_evaluation_ok_when_loaded(self, service):
        evaluation = service.hardware_evaluation()
        assert evaluation.verdict == HardwareVerdict.OK

    def test_infer_before_load_raises_loading(self):
        """KAI-C polls /infer during startup — it must get a typed
        ServiceError, not a generic crash."""
        from adapters.__DIR_NAME__.service import __SERVICE_CLASS__

        svc = __SERVICE_CLASS__()  # NOT loaded
        with pytest.raises(ServiceError) as exc_info:
            svc.infer({})
        # FailureEnvelope nests category/code under .error.
        env = exc_info.value.envelope()
        assert env.error.category == ErrorCategory.MODEL_ERROR


# ────────────────────────────────────────────────────────────────────
# Inference
# ────────────────────────────────────────────────────────────────────


class TestInference:
    """TODO: replace these toy assertions with what your adapter
    actually guarantees. The placeholder below covers only the
    contract envelope shape — your real tests should pin the
    detection format, error semantics, per-camera state, etc."""

    def test_valid_input_returns_envelope(self, service):
        # TODO: build a realistic payload for your adapter and assert
        # the result shape your model is supposed to produce.
        response = service.infer({"placeholder": "value"})
        assert response.model_name == "__SLUG__"
        assert response.inference_ms >= 0

    def test_malformed_input_returns_transport_error(self, service):
        # TODO: send input your adapter SHOULD reject and assert the
        # envelope shape (TRANSPORT_ERROR + a stable error code).
        # The placeholder fails loudly so the implementor can't merge
        # a scaffolded adapter without actually writing this test —
        # a vacuous ``pass`` would slip through every code review.
        pytest.fail(
            "TODO: send a payload your adapter rejects and assert "
            "ServiceError(TRANSPORT_ERROR, malformed_input). Remove "
            "this fail() once the assertion is real."
        )
