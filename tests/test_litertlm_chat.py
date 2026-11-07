# Copyright (c) 2026 OpenNVR
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The OpenAI-compatible chat endpoint on the NPU adapter: message
splitting, the response shape the camera agent reads, tool calls through a
fake engine, and the context-window refusal."""
from __future__ import annotations

import json
import types
from pathlib import Path

import pytest

from adapters.litertlm.chat import split_messages, to_openai_message
from adapters.litertlm.service import LitertLmService, agent_tool_class
from opennvr_adapter_sdk import HealthStatus


def test_messages_split_into_system_history_and_the_last_user_turn():
    system, history, user = split_messages([
        {"role": "system", "content": "You are the agent."},
        {"role": "user", "content": "is anyone there?"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "call_1", "type": "function", "function": {"name": "detect_objects", "arguments": '{"camera_id": "cam23"}'}}]},
        {"role": "tool", "tool_call_id": "call_1", "content": '{"persons": 2}'},
        {"role": "user", "content": [{"type": "text", "text": "and now?"}]},
    ])
    assert system == "You are the agent." and user == "and now?"
    assert history[0] == {"role": "user", "content": "is anyone there?"}
    assert history[1]["tool_calls"] == [{"name": "detect_objects", "arguments": {"camera_id": "cam23"}, "id": "call_1"}]
    assert history[2] == {"role": "tool", "content": '{"persons": 2}', "tool_call_id": "call_1", "name": None}


def test_the_reply_is_shaped_for_an_openai_client():
    m = to_openai_message("", [{"name": "create_alarm", "arguments": {"camera_id": "cam1", "after": "18:00"}, "id": None}])
    assert m["content"] is None and m["tool_calls"][0]["type"] == "function"
    assert json.loads(m["tool_calls"][0]["function"]["arguments"]) == {"camera_id": "cam1", "after": "18:00"}
    assert m["tool_calls"][0]["id"].startswith("call_")
    assert to_openai_message("hello", []) == {"role": "assistant", "content": "hello"}


class _Msg(dict):
    def __init__(self, text, calls=()):
        super().__init__(content=[{"type": "text", "text": text}] if text else [])
        self.tool_calls = list(calls)


class _Conv:
    def __init__(self, engine, kwargs):
        self.engine, self.kwargs = engine, kwargs
    def __enter__(self):
        return self
    def __exit__(self, *a):
        return False
    def send_message(self, text):
        self.engine.seen.append((self.kwargs, text))
        if self.engine.overflow_at and len(self.kwargs.get("tools") or []) >= self.engine.overflow_at:
            raise RuntimeError("litert_lm_conversation_send_message failed")
        if self.kwargs.get("tools") and "alarm" in text:
            return _Msg("", [types.SimpleNamespace(name="create_alarm", arguments={"camera_id": "cam1"}, id=None)])
        return _Msg(f"reply to: {text}")


class _Engine:
    overflow_at = 0
    def __init__(self):
        self.seen = []
    def create_conversation(self, **kwargs):
        return _Conv(self, kwargs)


class _FakeLm:
    class Tool:
        pass
    class ThinkingConfig:
        def __init__(self, **kw):
            self.kw = kw
    class SamplerConfig:
        def __init__(self, **kw):
            self.kw = kw


def _ready(tmp_path):
    engine = _Engine()
    model = tmp_path / "gemma.litertlm"; model.write_bytes(b"m")
    s = LitertLmService(model_path=str(model), model_url="", fetch_libraries=False, engine_factory=lambda: engine)
    s._engine, s._lm, s._state = engine, _FakeLm, HealthStatus.OK
    return s, engine


def test_chat_passes_system_history_tools_and_returns_calls(tmp_path):
    s, engine = _ready(tmp_path)
    tools = [{"type": "function", "function": {"name": "create_alarm", "description": "arm", "parameters": {"type": "object"}}}]
    text, calls, _ = s.chat(system="sys", history=[{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}],
                            user="set an alarm after 6pm", tools=tools, max_tokens=64, temperature=0.2)
    assert text == "" and calls == [{"name": "create_alarm", "arguments": {"camera_id": "cam1"}, "id": None}]
    kwargs, sent = engine.seen[-1]
    assert sent == "set an alarm after 6pm" and kwargs["system_message"] == "sys" and len(kwargs["messages"]) == 2
    assert kwargs["automatic_tool_calling"] is False and kwargs["max_output_tokens"] == 64
    assert kwargs["tools"][0].get_tool_description() == tools[0]
    assert isinstance(kwargs["tools"][0], _FakeLm.Tool)
    assert kwargs["thinking_config"].kw == {"enable_thinking": False, "thinking_token_budget": 0}
    assert kwargs["sampler_config"].kw == {"temperature": 0.2}
    text, calls, _ = s.chat(system=None, history=[], user="hello there", tools=[], max_tokens=32)
    assert text == "reply to: hello there" and calls == []


def test_a_prompt_past_the_window_is_a_prompt_too_long_error(tmp_path):
    s, engine = _ready(tmp_path)
    engine.overflow_at = 3
    tools = [{"type": "function", "function": {"name": f"t{i}", "parameters": {}}} for i in range(3)]
    with pytest.raises(LitertLmService.PromptTooLong):
        s.chat(system="s", history=[], user="q", tools=tools, max_tokens=8)
    s._state = HealthStatus.LOADING
    with pytest.raises(LitertLmService.NotReady):
        s.chat(system=None, history=[], user="q", tools=[], max_tokens=8)


def test_the_http_endpoint_round_trip(tmp_path, monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from adapters.litertlm.chat import router
    s, engine = _ready(tmp_path)
    app = FastAPI(); app.include_router(router); app.state.litertlm_service = s
    c = TestClient(app)
    r = c.post("/v1/chat/completions", json={"model": "x", "messages": [
        {"role": "system", "content": "sys"}, {"role": "user", "content": "set an alarm after 6pm"}],
        "tools": [{"type": "function", "function": {"name": "create_alarm", "parameters": {}}}], "tool_choice": "auto", "max_tokens": 50})
    assert r.status_code == 200, r.text
    body = r.json(); msg = body["choices"][0]["message"]
    assert body["choices"][0]["finish_reason"] == "tool_calls" and msg["content"] is None
    assert msg["tool_calls"][0]["function"]["name"] == "create_alarm"
    assert json.loads(msg["tool_calls"][0]["function"]["arguments"]) == {"camera_id": "cam1"}
    assert body["usage"]["estimated"] is True and body["opennvr"]["tools_attached"] == 1
    r = c.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "hello"}]})
    assert r.json()["choices"][0]["message"]["content"] == "reply to: hello" and r.json()["choices"][0]["finish_reason"] == "stop"
    engine.overflow_at = 1
    r = c.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "q"}],
                                             "tools": [{"type": "function", "function": {"name": "t", "parameters": {}}}]})
    assert r.status_code == 400 and r.json()["error"]["code"] == "context_length_exceeded"
    assert c.post("/v1/chat/completions", json={"messages": []}).status_code == 400
    assert c.get("/v1/models").json()["data"][0]["object"] == "model"
