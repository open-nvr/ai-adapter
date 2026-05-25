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


# ── Tool calling ───────────────────────────────────────────────────


_TOOL_DEF = {
    "type": "function",
    "function": {
        "name": "describe_camera",
        "description": "Return a one-sentence description of what the camera sees.",
        "parameters": {
            "type": "object",
            "properties": {
                "camera_id": {"type": "string"},
            },
            "required": ["camera_id"],
        },
    },
}


def test_tools_field_is_forwarded_to_ollama(fake_httpx):
    """When the caller passes tools, the adapter relays them into
    /api/chat untouched and the Ollama model decides whether to
    invoke."""
    ctx = fake_httpx()
    adapter = OllamaAdapter({"enabled": True})
    adapter.infer({
        "task": "chat_completion",
        "prompt": "What's at the front door?",
        "tools": [_TOOL_DEF],
        "tool_choice": "auto",
    })
    chat_call = next(c for c in ctx["client"].calls if c["path"] == "/api/chat")
    body = chat_call["json"]
    assert body["tools"] == [_TOOL_DEF]
    assert body["tool_choice"] == "auto"


def test_tools_field_is_not_forwarded_when_absent(fake_httpx):
    """A vanilla chat_completion without tools must NOT include the
    tools key in the upstream request — keeps tool-incapable models
    on their fast path and the /api/chat body free of empty arrays."""
    ctx = fake_httpx()
    adapter = OllamaAdapter({"enabled": True})
    adapter.infer({"task": "chat_completion", "prompt": "Hello"})
    chat_call = next(c for c in ctx["client"].calls if c["path"] == "/api/chat")
    assert "tools" not in chat_call["json"]
    assert "tool_choice" not in chat_call["json"]


def test_empty_tools_list_is_not_forwarded(fake_httpx):
    """An empty tools list is treated the same as no tools at all."""
    ctx = fake_httpx()
    adapter = OllamaAdapter({"enabled": True})
    adapter.infer({"task": "chat_completion", "prompt": "Hello", "tools": []})
    chat_call = next(c for c in ctx["client"].calls if c["path"] == "/api/chat")
    assert "tools" not in chat_call["json"]


def test_tool_calls_response_is_normalised_to_openai_shape(fake_httpx):
    """When Ollama returns a tool_calls list (no id, dict args), the
    adapter surfaces an OpenAI-shaped list with synthesised ids and
    JSON-stringified args. finish_reason becomes 'tool_calls'."""
    chat_response = _FakeResponse(200, {
        "model": "llama3.1:8b",
        "message": {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"function": {
                    "name": "describe_camera",
                    "arguments": {"camera_id": "front-porch"},
                }},
            ],
        },
        "done": True,
        "done_reason": "stop",
        "prompt_eval_count": 30,
        "eval_count": 8,
    })
    ctx = fake_httpx(chat_response=chat_response)
    adapter = OllamaAdapter({"enabled": True})
    result = adapter.infer({
        "task": "chat_completion",
        "prompt": "Look at the porch",
        "tools": [_TOOL_DEF],
    })

    assert result["finish_reason"] == "tool_calls"
    tool_calls = result["message"]["tool_calls"]
    assert len(tool_calls) == 1
    call = tool_calls[0]
    assert call["type"] == "function"
    assert call["function"]["name"] == "describe_camera"
    # Arguments are JSON-stringified for OpenAI / Pipecat compatibility.
    import json as _json
    assert _json.loads(call["function"]["arguments"]) == {"camera_id": "front-porch"}
    # A synthetic id was injected.
    assert call["id"].startswith("call_")


def test_tool_calls_preserve_model_supplied_id(fake_httpx):
    """If a model fork provides its own id, the adapter keeps it
    instead of overwriting with a synthetic one."""
    chat_response = _FakeResponse(200, {
        "model": "llama3.1:8b",
        "message": {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "modelsuppliedid_abc",
                    "function": {
                        "name": "describe_camera",
                        "arguments": {"camera_id": "x"},
                    },
                },
            ],
        },
        "done": True,
    })
    ctx = fake_httpx(chat_response=chat_response)
    adapter = OllamaAdapter({"enabled": True})
    result = adapter.infer({
        "task": "chat_completion",
        "prompt": "x",
        "tools": [_TOOL_DEF],
    })
    assert result["message"]["tool_calls"][0]["id"] == "modelsuppliedid_abc"


def test_string_arguments_pass_through(fake_httpx):
    """Some models emit ``arguments`` as a pre-stringified JSON blob
    already. We must not double-stringify."""
    chat_response = _FakeResponse(200, {
        "model": "llama3.1:8b",
        "message": {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"function": {
                    "name": "describe_camera",
                    "arguments": '{"camera_id":"x"}',
                }},
            ],
        },
        "done": True,
    })
    ctx = fake_httpx(chat_response=chat_response)
    adapter = OllamaAdapter({"enabled": True})
    result = adapter.infer({
        "task": "chat_completion",
        "prompt": "x",
        "tools": [_TOOL_DEF],
    })
    args = result["message"]["tool_calls"][0]["function"]["arguments"]
    assert args == '{"camera_id":"x"}'


def test_plain_text_response_omits_tool_calls(fake_httpx):
    """Even when tools were provided in the request, a plain-text
    response (no tool_calls in message) must NOT carry tool_calls in
    the result, and finish_reason stays 'stop'."""
    ctx = fake_httpx()  # default chat_response is plain text
    adapter = OllamaAdapter({"enabled": True})
    result = adapter.infer({
        "task": "chat_completion",
        "prompt": "Hi",
        "tools": [_TOOL_DEF],
    })
    assert "tool_calls" not in result["message"]
    assert result["finish_reason"] == "stop"


def test_length_finish_reason_passes_through(fake_httpx):
    """``done_reason=length`` must surface as ``finish_reason=length``
    when no tool_calls are present. This is the truncation signal
    callers use to decide whether to ask for a continuation."""
    chat_response = _FakeResponse(200, {
        "model": "llama3.1:8b",
        "message": {"role": "assistant", "content": "An incomplete sente"},
        "done": True,
        "done_reason": "length",
    })
    fake_httpx(chat_response=chat_response)
    adapter = OllamaAdapter({"enabled": True})
    result = adapter.infer({"task": "chat_completion", "prompt": "tell a long story"})
    assert result["finish_reason"] == "length"


def test_tool_calls_override_length_finish_reason(fake_httpx):
    """If a model emits both tool_calls AND done_reason=length, the
    tool_calls signal wins — callers branch on tool_calls regardless
    of vendor-specific truncation flags."""
    chat_response = _FakeResponse(200, {
        "model": "llama3.1:8b",
        "message": {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"function": {"name": "describe_camera", "arguments": {"camera_id": "x"}}},
            ],
        },
        "done": True,
        "done_reason": "length",
    })
    fake_httpx(chat_response=chat_response)
    adapter = OllamaAdapter({"enabled": True})
    result = adapter.infer({
        "task": "chat_completion", "prompt": "x", "tools": [_TOOL_DEF],
    })
    assert result["finish_reason"] == "tool_calls"


def test_null_arguments_become_empty_string(fake_httpx):
    """A null arguments field (some 1-shot tool models) must
    normalise to '' rather than crash json.dumps or downstream
    JSON parsers."""
    chat_response = _FakeResponse(200, {
        "model": "llama3.1:8b",
        "message": {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"function": {"name": "describe_camera", "arguments": None}},
            ],
        },
        "done": True,
    })
    fake_httpx(chat_response=chat_response)
    adapter = OllamaAdapter({"enabled": True})
    result = adapter.infer({
        "task": "chat_completion", "prompt": "x", "tools": [_TOOL_DEF],
    })
    assert result["message"]["tool_calls"][0]["function"]["arguments"] == ""


def test_list_arguments_are_json_stringified(fake_httpx):
    """Some models emit a list (not an object) for tool args — e.g.
    positional rather than named. The adapter coerces to a JSON
    string so OpenAI-style downstream code can parse it back."""
    chat_response = _FakeResponse(200, {
        "model": "llama3.1:8b",
        "message": {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"function": {"name": "describe_camera", "arguments": ["front-porch"]}},
            ],
        },
        "done": True,
    })
    fake_httpx(chat_response=chat_response)
    adapter = OllamaAdapter({"enabled": True})
    result = adapter.infer({
        "task": "chat_completion", "prompt": "x", "tools": [_TOOL_DEF],
    })
    import json as _json
    args = result["message"]["tool_calls"][0]["function"]["arguments"]
    assert _json.loads(args) == ["front-porch"]


def test_tool_role_messages_are_passed_through(fake_httpx):
    """Multi-turn flow: the caller appends tool-result messages with
    role=tool. The adapter must not coerce or drop them."""
    ctx = fake_httpx()
    adapter = OllamaAdapter({"enabled": True})
    adapter.infer({
        "task": "chat_completion",
        "messages": [
            {"role": "user", "content": "What's at the porch?"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "id": "call_x",
                    "type": "function",
                    "function": {"name": "describe_camera", "arguments": '{"camera_id":"porch"}'},
                }],
            },
            {
                "role": "tool",
                "tool_call_id": "call_x",
                "content": "A brown cardboard box on the doormat.",
            },
        ],
        "tools": [_TOOL_DEF],
    })
    chat_call = next(c for c in ctx["client"].calls if c["path"] == "/api/chat")
    roles = [m["role"] for m in chat_call["json"]["messages"]]
    assert roles == ["user", "assistant", "tool"]
