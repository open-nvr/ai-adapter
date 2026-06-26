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
    """The moondream package API: query(image, q) → {answer}; caption(image,
    length) → {caption}."""
    def query(self, image, question):
        return {"answer": f"answer to: {question}"}
    def caption(self, image, length="short"):
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


def test_adapter_declares_no_egress_apache_and_tasks():
    # Assert on the configured AdapterApp directly (no model load / lifespan).
    from adapters.moondream.main import _adapter_app
    assert _adapter_app._permissions.network_egress == []   # local_only-safe
    assert _adapter_app._license == "Apache-2.0"
    assert set(_adapter_app._tasks_advertised) == {"visual_qa", "scene_caption"}
