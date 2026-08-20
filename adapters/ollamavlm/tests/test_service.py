# Copyright (c) 2026 OpenNVR
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Unit tests for the Ollama-proxy VLM service — no network, no Ollama:
httpx.MockTransport plays the endpoint, covering the states the adapter
must ride out (up+model, up+missing-model, down) and the contract mapping."""
from __future__ import annotations

import json

import httpx
import pytest

from adapters.ollamavlm.service import OllamaVlmService
from opennvr_adapter_sdk import HardwareVerdict, ServiceError

JPEG = b"\xff\xd8\xff" + b"x" * 32
URL = "http://fake-ollama:11434"


def _svc(handler, model="moondream", autopull="false", monkeypatch=None):
    if monkeypatch is not None:
        monkeypatch.setenv("OPENNVR_OLLAMA_VLM_AUTOPULL", autopull)
    return OllamaVlmService(url=URL, model=model,
                            transport=httpx.MockTransport(handler))


def _ollama(models=("moondream:latest",), response="a cat on a wall",
            generate_status=200):
    """A fake Ollama: /api/tags lists models, /api/generate answers."""
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={
                "models": [{"name": m} for m in models]})
        if request.url.path == "/api/generate":
            if generate_status != 200:
                return httpx.Response(generate_status, text="nope")
            body = json.loads(request.content)
            assert body["stream"] is False and body["images"], "contract: b64 image required"
            return httpx.Response(200, json={"response": response})
        if request.url.path == "/api/pull":
            return httpx.Response(200, json={"status": "success"})
        return httpx.Response(404)

    handler.calls = calls
    return handler


def test_lazy_ready_even_when_endpoint_down(monkeypatch):
    def down(request):
        raise httpx.ConnectError("refused")
    svc = _svc(down, monkeypatch=monkeypatch)
    svc.load()
    assert svc.is_ready(), "endpoint availability must not gate readiness"
    hw = svc.hardware_evaluation()
    assert hw.verdict == HardwareVerdict.WARN
    assert "not answering" in hw.reasoning


def test_caption_maps_to_caption_key(monkeypatch):
    svc = _svc(_ollama(), monkeypatch=monkeypatch)
    svc.load()
    out = svc.infer({"__file__": JPEG})
    assert out.result["task"] == "scene_caption"
    assert out.result["caption"] == "a cat on a wall"
    assert out.model_version == "ollama/moondream"


def test_vqa_maps_to_answer_key(monkeypatch):
    handler = _ollama(response="a red jacket")
    svc = _svc(handler, monkeypatch=monkeypatch)
    svc.load()
    out = svc.infer({"__file__": JPEG, "question": "what is he wearing?"})
    assert out.result["task"] == "visual_qa"
    assert out.result["answer"] == "a red jacket"
    assert out.result["question"] == "what is he wearing?"
    sent = json.loads([c for c in handler.calls
                       if c.url.path == "/api/generate"][-1].content)
    assert sent["prompt"] == "what is he wearing?"


def test_missing_model_is_transient_with_pull_hint(monkeypatch):
    svc = _svc(_ollama(generate_status=404), monkeypatch=monkeypatch)
    svc.load()
    with pytest.raises(ServiceError) as e:
        svc.infer({"__file__": JPEG})
    assert e.value.transient and e.value.http_status == 503
    assert "ollama pull moondream" in e.value.message


def test_endpoint_down_is_transient_503(monkeypatch):
    def down(request):
        if request.url.path == "/api/tags":
            raise httpx.ConnectError("refused")
        raise httpx.ConnectError("refused")
    svc = _svc(down, monkeypatch=monkeypatch)
    svc.load()
    with pytest.raises(ServiceError) as e:
        svc.infer({"__file__": JPEG})
    assert e.value.transient and e.value.http_status == 503
    assert e.value.code == "endpoint_unreachable"


def test_non_image_rejected_400(monkeypatch):
    svc = _svc(_ollama(), monkeypatch=monkeypatch)
    svc.load()
    with pytest.raises(ServiceError) as e:
        svc.infer({"__file__": b"definitely not an image"})
    assert e.value.http_status == 400 and not e.value.transient


def test_probe_distinguishes_model_missing(monkeypatch):
    svc = _svc(_ollama(models=("llava:latest",)), monkeypatch=monkeypatch)
    svc.load()
    hw = svc.hardware_evaluation()
    assert hw.verdict == HardwareVerdict.WARN
    assert hw.details["endpoint_state"] == "model_missing"


def test_tagged_model_matches_exactly(monkeypatch):
    svc = _svc(_ollama(models=("moondream:1.8b",)), model="moondream:1.8b",
               monkeypatch=monkeypatch)
    svc.load()
    assert svc.hardware_evaluation().details["endpoint_state"] == "ok"
