# Copyright (c) 2026 OpenNVR
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Regression tests for the strict-schema guarantee on the contract
envelopes (contract.py). Every wire-envelope model sets
``extra="forbid"`` — an adapter that sneaks an unknown field into
/capabilities (or its sub-shapes) MUST be rejected at validation, not
silently accepted. A silently-accepted unknown field is how a tampered
or version-drifted adapter slips past KAI-C's registration check, so
this invariant is security-relevant and worth pinning.
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from opennvr_adapter_sdk.contract import (
    CapabilitiesResponse,
    ModelInfo,
    Permissions,
)


def _valid_capabilities_payload() -> dict:
    """A minimal but fully-valid /capabilities body.

    ``scheduling`` is the only required sub-object with no model-level
    default; ``permissions``/``cost`` default in, and every declared
    field here is the minimum the envelope requires.
    """
    return {
        "adapter": {
            "name": "regression-adapter",
            "version": "1.0.0",
            "vendor": "open-nvr",
            "license": "AGPL-3.0",
            "supported_contract_versions": ["1"],
        },
        "model": {
            "name": "regression-model",
            "version": "1",
            "framework": "test",
        },
        "endpoints": {
            "infer": {"supported": True},
            "infer_stream": {"supported": False},
        },
        "scheduling": {},
    }


def test_valid_capabilities_payload_validates():
    """Sanity: the baseline payload really is valid, so the extra-key
    tests below prove the *extra key* is what triggers rejection."""
    caps = CapabilitiesResponse.model_validate(_valid_capabilities_payload())
    assert caps.adapter.name == "regression-adapter"
    assert caps.endpoints.infer.supported is True


def test_capabilities_rejects_unknown_top_level_field():
    """extra='forbid' on CapabilitiesResponse: a bogus top-level key
    like ``surprise`` must raise, not be silently dropped."""
    payload = {**_valid_capabilities_payload(), "surprise": 1}
    with pytest.raises(ValidationError):
        CapabilitiesResponse.model_validate(payload)


def test_model_info_rejects_unknown_field():
    """extra='forbid' on ModelInfo — an unknown field on the nested
    model block is rejected too, so drift can't hide inside a subobject."""
    payload = {
        "name": "regression-model",
        "version": "1",
        "framework": "test",
        "surprise": 1,
    }
    with pytest.raises(ValidationError):
        ModelInfo.model_validate(payload)


def test_permissions_rejects_unknown_field():
    """extra='forbid' on Permissions — the sandboxing declaration is
    strict so KAI-C can't be handed an unrecognized capability grant."""
    # Every Permissions field has a default, so an all-defaults dict is
    # valid; the ONLY thing that can trip validation here is the extra key.
    with pytest.raises(ValidationError):
        Permissions.model_validate({"surprise": 1})
