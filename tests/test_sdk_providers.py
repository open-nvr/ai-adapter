# Copyright (c) 2026 OpenNVR
# SPDX-License-Identifier: Apache-2.0
"""opennvr_adapter_sdk.providers — the one provider resolver every ONNX adapter uses."""
from __future__ import annotations

import logging

import pytest

from opennvr_adapter_sdk.providers import (
    accelerator_available,
    describe_providers,
    effective_providers,
    parse_provider_specs,
    requested_providers,
    resolve_providers,
    session_providers,
)


def test_parse_shorthand_and_multi_device_continuation():
    assert parse_provider_specs("OpenVINOExecutionProvider:NPU,CPUExecutionProvider") == [
        ("OpenVINOExecutionProvider", "NPU"), ("CPUExecutionProvider", None)]
    assert parse_provider_specs("OpenVINOExecutionProvider:AUTO:NPU,GPU,CPUExecutionProvider") == [
        ("OpenVINOExecutionProvider", "AUTO:NPU,GPU"), ("CPUExecutionProvider", None)]
    assert parse_provider_specs(["CUDAExecutionProvider:1"]) == [("CUDAExecutionProvider", "1")]
    assert parse_provider_specs([("OpenVINOExecutionProvider", {"device_type": "GPU"})]) == [
        ("OpenVINOExecutionProvider", "GPU")]
    assert parse_provider_specs(None) == [] and parse_provider_specs("") == []


def test_resolve_carries_device_options_and_keeps_cpu_last():
    provs = resolve_providers("OpenVINOExecutionProvider:NPU",
                              available=["OpenVINOExecutionProvider", "CPUExecutionProvider"])
    assert provs == [("OpenVINOExecutionProvider", {"device_type": "NPU"}), "CPUExecutionProvider"]
    assert resolve_providers("CUDAExecutionProvider:1", available=["CUDAExecutionProvider"]) == [
        ("CUDAExecutionProvider", {"device_id": "1"}), "CPUExecutionProvider"]


def test_resolve_skips_a_provider_this_build_lacks_and_never_fails():
    provs = resolve_providers("OpenVINOExecutionProvider:NPU,CUDAExecutionProvider",
                              available=["CPUExecutionProvider"])
    assert provs == ["CPUExecutionProvider"]
    assert resolve_providers(None, ["CUDAExecutionProvider", "CPUExecutionProvider"]) == [
        "CUDAExecutionProvider", "CPUExecutionProvider"]
    assert resolve_providers("CPUExecutionProvider:0", ["CPUExecutionProvider"]) == ["CPUExecutionProvider"]


def test_describe_is_human_readable():
    assert describe_providers([("OpenVINOExecutionProvider", {"device_type": "NPU"}),
                               "CPUExecutionProvider"]) == \
        "OpenVINOExecutionProvider(NPU), CPUExecutionProvider"
    assert describe_providers(None) == ""


def test_env_setting_and_default(monkeypatch):
    monkeypatch.delenv("ADAPTER_ONNX_PROVIDERS", raising=False)
    assert requested_providers() is None
    assert requested_providers("CUDAExecutionProvider") == "CUDAExecutionProvider"
    monkeypatch.setenv("ADAPTER_ONNX_PROVIDERS", " OpenVINOExecutionProvider:GPU ")
    assert requested_providers("x") == "OpenVINOExecutionProvider:GPU"


def test_accelerator_available_ignores_cpu_and_azure():
    assert accelerator_available(["CPUExecutionProvider"]) is False
    assert accelerator_available(["AzureExecutionProvider", "CPUExecutionProvider"]) is False
    assert accelerator_available(["OpenVINOExecutionProvider", "CPUExecutionProvider"]) is True
    assert accelerator_available(["CUDAExecutionProvider", "CPUExecutionProvider"]) is True


class _Sess:
    def __init__(self, provs): self._p = provs
    def get_providers(self): return list(self._p)


def test_effective_providers_warns_on_silent_fallback(caplog):
    req = [("OpenVINOExecutionProvider", {"device_type": "NPU"}), "CPUExecutionProvider"]
    with caplog.at_level(logging.WARNING, logger="opennvr_adapter_sdk.providers"):
        info = effective_providers(_Sess(["CPUExecutionProvider"]), req, adapter="yolov8")
    assert info["effective_providers"] == ["CPUExecutionProvider"]
    assert info["accelerator_in_use"] is False and info["accelerator_requested"] is True
    assert info["requested_providers"] == "OpenVINOExecutionProvider(NPU), CPUExecutionProvider"
    assert "yolov8: ONNX Runtime could not use OpenVINOExecutionProvider(NPU)" in caplog.text


def test_effective_providers_reports_accelerator_in_use(caplog):
    req = [("OpenVINOExecutionProvider", {"device_type": "NPU"}), "CPUExecutionProvider"]
    with caplog.at_level(logging.WARNING, logger="opennvr_adapter_sdk.providers"):
        info = effective_providers(_Sess(["OpenVINOExecutionProvider", "CPUExecutionProvider"]), req)
    assert info["accelerator_in_use"] is True
    assert "could not use" not in caplog.text


def test_session_providers_uses_env_against_this_build(monkeypatch):
    import opennvr_adapter_sdk.providers as m
    monkeypatch.setattr(m, "available_providers", lambda: ["OpenVINOExecutionProvider", "CPUExecutionProvider"])
    monkeypatch.setenv("ADAPTER_ONNX_PROVIDERS", "OpenVINOExecutionProvider:GPU")
    assert session_providers() == [("OpenVINOExecutionProvider", {"device_type": "GPU"}), "CPUExecutionProvider"]
    monkeypatch.delenv("ADAPTER_ONNX_PROVIDERS")
    assert session_providers() == ["OpenVINOExecutionProvider", "CPUExecutionProvider"]  # ORT's order
    assert session_providers("CUDAExecutionProvider") == ["CPUExecutionProvider"]        # not in this build
