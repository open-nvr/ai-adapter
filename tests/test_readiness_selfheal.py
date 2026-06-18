# Copyright (c) 2026 OpenNVR
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Readiness + self-healing-weights tests.

Covers the fix for "camera offline at startup": the ai-adapter reports the
HTTP server up before its model weights finish their background download.
We add a /ready probe (gates dependents on weights being present) and make
YOLOv8 fetch its weights on demand instead of erroring.
"""
from __future__ import annotations

import sys
import types

import pytest

import download_models as dm


# ── download helpers ──────────────────────────────────────────────────


def test_required_weights_status_reports_missing(tmp_path):
    ok, missing = dm.required_weights_status(str(tmp_path))
    assert ok is False and missing  # enabled adapters' weights aren't there


def test_ensure_adapter_weights_downloads_then_is_idempotent(tmp_path, monkeypatch):
    monkeypatch.setitem(dm.MODEL_REGISTRY, "_t",
                        [{"filename": "sub/w.bin", "url": "http://x", "size_hint": ""}])
    calls = {"n": 0}

    def fake_dl(url, dest):
        calls["n"] += 1
        dest.write_bytes(b"x")
        return True

    monkeypatch.setattr(dm, "_download_file", fake_dl)
    assert dm.ensure_adapter_weights("_t", str(tmp_path)) is True
    assert (tmp_path / "sub" / "w.bin").exists()
    # second call: file present → no re-download
    assert dm.ensure_adapter_weights("_t", str(tmp_path)) is True
    assert calls["n"] == 1


# ── YOLOv8 self-heal ──────────────────────────────────────────────────


def test_yolov8_load_model_self_heals_when_weights_missing(tmp_path, monkeypatch):
    import app.adapters.vision.yolov8_adapter as y

    fake_ort = types.ModuleType("onnxruntime")
    fake_ort.InferenceSession = lambda *a, **k: object()
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)

    adapter = y.YOLOv8Adapter()
    model_path = tmp_path / "yolov8n.onnx"
    adapter._model_path = str(model_path)  # missing on disk

    called = {}

    def fake_ensure(name, weights_dir=None):
        called["name"] = name
        model_path.write_bytes(b"onnx")  # simulate the download landing
        return True

    monkeypatch.setattr(dm, "ensure_adapter_weights", fake_ensure)
    adapter.load_model()

    assert called["name"] == "yolov8_adapter"
    assert adapter.session is not None


def test_yolov8_load_model_raises_if_still_missing(tmp_path, monkeypatch):
    import app.adapters.vision.yolov8_adapter as y

    fake_ort = types.ModuleType("onnxruntime")
    fake_ort.InferenceSession = lambda *a, **k: object()
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)

    adapter = y.YOLOv8Adapter()
    adapter._model_path = str(tmp_path / "missing.onnx")
    monkeypatch.setattr(dm, "ensure_adapter_weights", lambda *a, **k: False)
    with pytest.raises(FileNotFoundError):
        adapter.load_model()


# ── /ready endpoint ───────────────────────────────────────────────────


@pytest.fixture
def ready_client():
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from app.api.endpoints import public_router, set_global_router

    app = FastAPI()
    app.include_router(public_router)
    return TestClient(app), set_global_router


def test_ready_503_before_router_injected(ready_client):
    client, set_global_router = ready_client
    set_global_router(None, None)
    assert client.get("/ready").status_code == 503


def test_ready_503_when_weights_missing(ready_client, monkeypatch):
    client, set_global_router = ready_client
    set_global_router(object(), object())
    monkeypatch.setattr(dm, "required_weights_status", lambda *a, **k: (False, ["yolov8n.onnx"]))
    r = client.get("/ready")
    assert r.status_code == 503
    assert r.json()["missing_weights"] == ["yolov8n.onnx"]


def test_ready_200_when_weights_present(ready_client, monkeypatch):
    client, set_global_router = ready_client
    set_global_router(object(), object())
    monkeypatch.setattr(dm, "required_weights_status", lambda *a, **k: (True, []))
    r = client.get("/ready")
    assert r.status_code == 200
    assert r.json()["status"] == "ready"
