# Copyright (c) 2026 OpenNVR
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Unit tests for OllamaAdapter.

httpx is a core dep so it's always available — we use monkeypatch to swap its
``Client`` for a fake that captures calls and returns canned responses. This
keeps tests offline and deterministic; no real Ollama daemon required.
"""
from typing import Any, Dict, List

import pytest

import app.adapters.llm.ollama_adapter as ollama_module
from app.adapters.llm.ollama_adapter import OllamaAdapter


class _FakeResponse:
    def __init__(self, status_code: int = 200, data: Dict[str, Any] | None = None, text: str = ""):
        self.status_code = status_code
        self._data = data if data is not None else {}
        self.text = text or str(self._data)

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self._data


class _FakeClient:
    """Minimal stand-in for ``httpx.Client`` used by OllamaAdapter."""

    def __init__(self, *, tags_response: _FakeResponse, chat_response: _FakeResponse):
        self._tags_response = tags_response
        self._chat_response = chat_response
        self.calls: List[Dict[str, Any]] = []

    def get(self, path: str):
        self.calls.append({"method": "GET", "path": path})
        return self._tags_response

    def post(self, path: str, json: Dict[str, Any]):
        self.calls.append({"method": "POST", "path": path, "json": json})
        return self._chat_response


@pytest.fixture
def fake_httpx(monkeypatch):
    """Patch the httpx module imported lazily inside OllamaAdapter.load_model."""
    created: Dict[str, Any] = {}

    def _factory(tags_response=None, chat_response=None):
        tags = tags_response or _FakeResponse(200, {"models": []})
        chat = chat_response or _FakeResponse(200, {
            "model": "llama3.2:3b",
            "message": {"role": "assistant", "content": "Hi there!"},
            "done": True,
            "prompt_eval_count": 12,
            "eval_count": 5,
        })

        class _FakeHttpxModule:
            @staticmethod
            def Client(**kwargs):
                client = _FakeClient(tags_response=tags, chat_response=chat)
                client.init_kwargs = kwargs
                created["client"] = client
                return client

        import sys
        sys.modules["httpx"] = _FakeHttpxModule
        return created

    return _factory


def test_load_model_probes_ollama_and_stores_client(fake_httpx):
    fake_httpx()
    adapter = OllamaAdapter({"enabled": True, "base_url": "http://test:11434"})
    adapter.ensure_model_loaded()

    assert adapter.model is not None
    assert adapter.model.calls[0] == {"method": "GET", "path": "/api/tags"}


def test_load_model_raises_when_ollama_unreachable(fake_httpx):
    ctx = fake_httpx(tags_response=_FakeResponse(503, {}, "service unavailable"))
    adapter = OllamaAdapter({"enabled": True})

    with pytest.raises(RuntimeError, match="Cannot reach Ollama"):
        adapter.ensure_model_loaded()


def test_infer_with_prompt_shortcut_builds_user_message(fake_httpx):
    ctx = fake_httpx()
    adapter = OllamaAdapter({"enabled": True})
    result = adapter.infer({"task": "chat_completion", "prompt": "Hello"})

    client = ctx["client"]
    chat_call = next(c for c in client.calls if c["path"] == "/api/chat")
    messages = chat_call["json"]["messages"]
    assert messages == [{"role": "user", "content": "Hello"}]
    assert result["message"]["content"] == "Hi there!"
    assert result["prompt_tokens"] == 12
    assert result["completion_tokens"] == 5
    assert result["total_tokens"] == 17


def test_system_prompt_is_prepended(fake_httpx):
    ctx = fake_httpx()
    adapter = OllamaAdapter({"enabled": True})
    adapter.infer({
        "task": "chat_completion",
        "messages": [{"role": "user", "content": "Tell me a joke"}],
        "system": "You are a stoic.",
    })

    client = ctx["client"]
    messages = next(c for c in client.calls if c["path"] == "/api/chat")["json"]["messages"]
    assert messages[0] == {"role": "system", "content": "You are a stoic."}
    assert messages[1] == {"role": "user", "content": "Tell me a joke"}


def test_options_are_passed_through(fake_httpx):
    ctx = fake_httpx()
    adapter = OllamaAdapter({"enabled": True})
    adapter.infer({
        "task": "chat_completion",
        "prompt": "Hi",
        "temperature": 0.2,
        "max_tokens": 128,
        "stop": ["</end>"],
    })
    options = next(c for c in ctx["client"].calls if c["path"] == "/api/chat")["json"]["options"]
    assert options["temperature"] == pytest.approx(0.2)
    assert options["num_predict"] == 128
    assert options["stop"] == ["</end>"]


def test_rejects_unknown_task(fake_httpx):
    fake_httpx()
    adapter = OllamaAdapter({"enabled": True})
    with pytest.raises(ValueError, match="OllamaAdapter supports"):
        adapter.infer({"task": "audio_transcription", "prompt": "Hi"})


def test_rejects_empty_input(fake_httpx):
    fake_httpx()
    adapter = OllamaAdapter({"enabled": True})
    with pytest.raises(ValueError, match="messages"):
        adapter.infer({"task": "chat_completion"})


def test_env_var_overrides_config_base_url(monkeypatch, fake_httpx):
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://env-host:11434")
    fake_httpx()
    adapter = OllamaAdapter({"enabled": True, "base_url": "http://config-host:11434"})
    assert adapter._base_url == "http://env-host:11434"
