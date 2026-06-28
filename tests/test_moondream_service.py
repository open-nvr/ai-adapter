# Copyright (c) 2026 OpenNVR
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Unit tests for MoondreamService routing (VQA vs caption). The moondream
runtime is stubbed (query/caption); the real model load + a docker build still
need verifying on hardware."""
from __future__ import annotations

import io

import pytest

from opennvr_adapter_sdk import ErrorCategory, HealthStatus, ServiceError

from adapters.moondream.service import MoondreamService


def _jpeg() -> bytes:
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", (32, 32), (10, 20, 30)).save(buf, format="JPEG")
    return buf.getvalue()


class _FakeModel:
    """Mimics moondream 0.0.6: encode_image() first, then caption/query on the
    ENCODED image; caption() takes no ``length`` kwarg (so the service's
    try/except fallback is exercised)."""
    def encode_image(self, image):
        return ("ENC", image)
    def query(self, enc, question):
        assert enc[0] == "ENC", "query must receive the encoded image"
        return {"answer": f"answer to: {question}"}
    def caption(self, enc):
        assert enc[0] == "ENC", "caption must receive the encoded image"
        return {"caption": "a small test scene"}


def _ready(model=None) -> MoondreamService:
    svc = MoondreamService()
    svc._model = model or _FakeModel()
    svc._load_state = HealthStatus.OK
    return svc


def test_vqa_question_returns_answer():
    svc = _ready()
    resp = svc.infer({"__file__": _jpeg(), "question": "what is he wearing?"})
    assert resp.result["task"] == "visual_qa"
    assert "what is he wearing?" in resp.result["answer"]
    assert "caption" not in resp.result


def test_no_question_returns_caption():
    svc = _ready()
    resp = svc.infer({"__file__": _jpeg()})
    assert resp.result["task"] == "scene_caption"
    assert resp.result["caption"] == "a small test scene"
    assert "answer" not in resp.result


def test_scene_caption_task_forces_caption_even_with_question():
    svc = _ready()
    resp = svc.infer({"__file__": _jpeg(), "question": "x", "task": "scene_caption"})
    assert resp.result["task"] == "scene_caption"
    assert "caption" in resp.result


def test_missing_image_raises_400():
    svc = _ready()
    with pytest.raises(ServiceError) as ei:
        svc.infer({})
    assert ei.value.http_status == 400


def test_not_ready_raises_503():
    svc = MoondreamService()  # never loaded
    with pytest.raises(ServiceError) as ei:
        svc.infer({"__file__": _jpeg()})
    assert ei.value.http_status == 503


def test_ensure_model_downloads_when_missing_and_url_set(monkeypatch, tmp_path):
    import urllib.request
    target = tmp_path / "m.mf.gz"
    svc = MoondreamService(model_path=str(target))
    monkeypatch.setenv("OPENNVR_MOONDREAM_MODEL_URL", "http://example/m.mf.gz")
    seen = {}

    def fake(url, dest):
        seen["url"] = url
        open(dest, "wb").write(b"MODEL")
    monkeypatch.setattr(urllib.request, "urlretrieve", fake)
    svc._ensure_model()
    assert seen["url"] == "http://example/m.mf.gz"
    assert target.exists()


def test_ensure_model_skips_when_present(monkeypatch, tmp_path):
    import urllib.request
    target = tmp_path / "m.mf.gz"
    target.write_bytes(b"already here")
    svc = MoondreamService(model_path=str(target))
    monkeypatch.setattr(urllib.request, "urlretrieve",
                        lambda *a: (_ for _ in ()).throw(AssertionError("no dl")))
    svc._ensure_model()      # present → must not download


def test_ensure_model_no_url_no_download(monkeypatch, tmp_path):
    target = tmp_path / "missing.mf.gz"
    monkeypatch.delenv("OPENNVR_MOONDREAM_MODEL_URL", raising=False)
    monkeypatch.delenv("MOONDREAM_MODEL_URL", raising=False)
    svc = MoondreamService(model_path=str(target))
    svc._ensure_model()      # no url → no error, no file (md.vl fails later)
    assert not target.exists()


def test_adapter_declares_no_egress_apache_and_tasks():
    # Assert on the configured AdapterApp directly (no model load / lifespan).
    from adapters.moondream.main import _adapter_app
    assert _adapter_app._permissions.network_egress == []   # local_only-safe
    assert _adapter_app._license == "Apache-2.0"
    assert set(_adapter_app._tasks_advertised) == {"visual_qa", "scene_caption"}
