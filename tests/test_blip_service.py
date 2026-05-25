# Copyright (c) 2026 OpenNVR
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Unit tests for BlipService at the AdapterService level."""
from __future__ import annotations

import importlib
import sys

import pytest

from opennvr_adapter_sdk import ErrorCategory, HardwareVerdict, ServiceError
from tests._blip_service_fixtures import (  # noqa: F401
    blip_environment,
    install_fake_transformers,
    sample_jpeg,
)


def _build_service(env, caption: str | None = None):
    if "adapters.blip.service" in sys.modules:
        importlib.reload(sys.modules["adapters.blip.service"])
    from adapters.blip.service import BlipService

    if caption is not None:
        # Re-install the transformers stub with the requested
        # caption so the test gets the value it expects back.
        install_fake_transformers(caption)
    return BlipService()


# ── Load / readiness ───────────────────────────────────────────────


def test_load_marks_service_ready(blip_environment):
    svc = _build_service(blip_environment)
    assert not svc.is_ready()
    svc.load()
    assert svc.is_ready()


def test_load_failure_keeps_service_unhealthy(blip_environment, monkeypatch):
    svc = _build_service(blip_environment)
    fake_mod = sys.modules["transformers"]

    class _Boom:
        @classmethod
        def from_pretrained(cls, model_id):
            raise RuntimeError("simulated load failure")

    monkeypatch.setattr(fake_mod, "BlipProcessor", _Boom)
    svc.load()
    assert not svc.is_ready()
    eval_resp = svc.hardware_evaluation()
    assert eval_resp.verdict == HardwareVerdict.BLOCKED


def test_load_is_idempotent(blip_environment):
    svc = _build_service(blip_environment)
    svc.load()
    svc.load()
    assert svc.is_ready()


# ── Model info / hardware evaluation ───────────────────────────────


def test_model_info_shape(blip_environment):
    svc = _build_service(blip_environment)
    svc.load()
    info = svc.model_info()
    assert info.framework == "transformers"
    assert info.modalities_in == ["image"]
    assert "text" in info.modalities_out
    assert info.fingerprint is not None
    assert info.fingerprint.startswith("sha256-id:")


def test_hardware_evaluation_ok_after_load(blip_environment):
    svc = _build_service(blip_environment)
    svc.load()
    resp = svc.hardware_evaluation()
    assert resp.verdict == HardwareVerdict.OK
    # Falls back to "cpu" in the stub since FakeCuda.is_available() is False.
    assert resp.details["device"] == "cpu"


def test_hardware_evaluation_warn_before_load(blip_environment):
    svc = _build_service(blip_environment)
    resp = svc.hardware_evaluation()
    assert resp.verdict == HardwareVerdict.WARN


# ── infer: happy path ──────────────────────────────────────────────


def test_infer_returns_caption(blip_environment, sample_jpeg):
    svc = _build_service(blip_environment, caption="a brown box on the doormat")
    svc.load()
    resp = svc.infer({"__file__": sample_jpeg})
    assert resp.result["task"] == "scene_caption"
    assert resp.result["caption"] == "a brown box on the doormat"
    assert resp.result["device"] == "cpu"
    assert resp.model_name.endswith("blip-image-captioning-base")


def test_infer_default_task_is_scene_caption(blip_environment, sample_jpeg):
    svc = _build_service(blip_environment)
    svc.load()
    # No 'task' field provided → defaults to scene_caption.
    resp = svc.infer({"__file__": sample_jpeg})
    assert resp.result["task"] == "scene_caption"


# ── infer: error envelopes ─────────────────────────────────────────


def test_infer_before_load_returns_model_error(blip_environment, sample_jpeg):
    svc = _build_service(blip_environment)
    with pytest.raises(ServiceError) as excinfo:
        svc.infer({"__file__": sample_jpeg})
    assert excinfo.value.category == ErrorCategory.MODEL_ERROR
    assert excinfo.value.code == "model_not_loaded"


def test_infer_missing_image_returns_transport_error(blip_environment):
    svc = _build_service(blip_environment)
    svc.load()
    with pytest.raises(ServiceError) as excinfo:
        svc.infer({})
    assert excinfo.value.category == ErrorCategory.TRANSPORT_ERROR
    assert excinfo.value.code == "missing_image"


def test_infer_unsupported_task_returns_not_supported(blip_environment, sample_jpeg):
    svc = _build_service(blip_environment)
    svc.load()
    with pytest.raises(ServiceError) as excinfo:
        svc.infer({"__file__": sample_jpeg, "task": "vqa"})
    assert excinfo.value.category == ErrorCategory.NOT_SUPPORTED
    assert excinfo.value.code == "unsupported_task"


def test_infer_invalid_image_returns_transport_error(blip_environment):
    svc = _build_service(blip_environment)
    svc.load()
    with pytest.raises(ServiceError) as excinfo:
        svc.infer({"__file__": b"this is not an image"})
    assert excinfo.value.category == ErrorCategory.TRANSPORT_ERROR
    assert excinfo.value.code == "invalid_image"
    assert excinfo.value.http_status == 400


def test_infer_inference_failure_returns_model_error(
    blip_environment, sample_jpeg, monkeypatch
):
    svc = _build_service(blip_environment)
    svc.load()
    monkeypatch.setattr(
        svc, "_run_blip",
        lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("CUDA OOM")),
    )
    with pytest.raises(ServiceError) as excinfo:
        svc.infer({"__file__": sample_jpeg})
    assert excinfo.value.category == ErrorCategory.MODEL_ERROR
    assert excinfo.value.code == "inference_failed"


# ── max_new_tokens validation ──────────────────────────────────────


@pytest.mark.parametrize("bad", ["abc", None, [], {}])
def test_infer_invalid_max_new_tokens_returns_transport_error(
    blip_environment, sample_jpeg, bad
):
    if bad is None:
        # None means "use default" — that's not an error.
        return
    svc = _build_service(blip_environment)
    svc.load()
    with pytest.raises(ServiceError) as excinfo:
        svc.infer({"__file__": sample_jpeg, "max_new_tokens": bad})
    assert excinfo.value.code == "invalid_max_new_tokens"


@pytest.mark.parametrize("bad", [0, -1, 257, 1000])
def test_infer_out_of_range_max_new_tokens_returns_transport_error(
    blip_environment, sample_jpeg, bad
):
    svc = _build_service(blip_environment)
    svc.load()
    with pytest.raises(ServiceError) as excinfo:
        svc.infer({"__file__": sample_jpeg, "max_new_tokens": bad})
    assert excinfo.value.code == "max_new_tokens_out_of_range"


def test_infer_valid_max_new_tokens_propagates(
    blip_environment, sample_jpeg, monkeypatch
):
    svc = _build_service(blip_environment)
    svc.load()
    captured: dict = {}

    def _capture(image, *, max_new_tokens):
        captured["max_new_tokens"] = max_new_tokens
        return "captured caption"

    monkeypatch.setattr(svc, "_run_blip", _capture)
    svc.infer({"__file__": sample_jpeg, "max_new_tokens": 128})
    assert captured["max_new_tokens"] == 128


# ── Fingerprint ────────────────────────────────────────────────────


def test_fingerprint_is_deterministic(blip_environment):
    svc = _build_service(blip_environment)
    a = svc.fingerprint()
    b = svc.fingerprint()
    assert a == b
    assert a.startswith("sha256-id:")
