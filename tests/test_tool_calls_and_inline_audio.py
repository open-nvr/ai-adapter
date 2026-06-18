# Copyright (c) 2026 OpenNVR
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for the camera-agent contract additions (PR #26):

* tool_calls preserved end-to-end (ChatMessage schema + chat_completion task),
* assistant tool_call ``arguments`` converted from JSON string back to object
  before re-submission to Ollama (the request-direction _build_messages fix),
* inline base64 audio surfaced for HTTP-only callers (SpeechSynthesisResponse
  schema + speech_synthesis task + Piper _wants_inline).

All offline; no model weights or Ollama daemon required.
"""
from __future__ import annotations

from typing import Any, Dict

import pytest

from app.adapters.llm.ollama_adapter import OllamaAdapter
from app.pipelines.chat_completion.task import ChatCompletionTask
from app.pipelines.speech_synthesis.task import SpeechSynthesisTask
from app.schemas.responses import ChatMessage, SpeechSynthesisResponse


class _FakeAdapter:
    """Duck-typed BaseAdapter: returns a canned dict from predict()."""

    def __init__(self, payload: Dict[str, Any]) -> None:
        self._payload = payload
        self.seen: Dict[str, Any] | None = None

    def predict(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        self.seen = payload
        return self._payload


_TOOL_CALL = {
    "id": "call_1",
    "type": "function",
    "function": {"name": "detect_objects", "arguments": '{"camera_id": "cam1"}'},
}


# ── ChatMessage schema: tool_calls serialization ──────────────────────


def test_chat_message_drops_tool_calls_when_absent():
    msg = ChatMessage(role="assistant", content="hello")
    assert "tool_calls" not in msg.model_dump()


def test_chat_message_keeps_tool_calls_when_present():
    msg = ChatMessage(role="assistant", content="", tool_calls=[_TOOL_CALL])
    dumped = msg.model_dump()
    assert dumped["tool_calls"] == [_TOOL_CALL]


# ── chat_completion task: preserve tool_calls from the adapter ─────────


def test_chat_completion_task_preserves_tool_calls():
    adapter = _FakeAdapter({
        "message": {"role": "assistant", "content": "", "tool_calls": [_TOOL_CALL]},
        "model": "qwen2.5:1.5b",
        "finish_reason": "tool_calls",
    })
    resp = ChatCompletionTask().process({"messages": []}, adapter)
    assert resp.message.tool_calls == [_TOOL_CALL]
    assert resp.finish_reason == "tool_calls"


def test_chat_completion_task_omits_empty_tool_calls():
    adapter = _FakeAdapter({
        "message": {"role": "assistant", "content": "hi", "tool_calls": []},
        "model": "qwen2.5:1.5b",
    })
    resp = ChatCompletionTask().process({"messages": []}, adapter)
    assert resp.message.tool_calls is None
    assert "tool_calls" not in resp.message.model_dump()


# ── _build_messages: string args → object on re-submission ────────────


def _assistant_turn(arguments: Any) -> Dict[str, Any]:
    return {
        "messages": [
            {"role": "user", "content": "what's on cam1?"},
            {"role": "assistant", "content": "",
             "tool_calls": [{"id": "c1", "type": "function",
                             "function": {"name": "detect_objects", "arguments": arguments}}]},
            {"role": "tool", "tool_call_id": "c1", "name": "detect_objects",
             "content": "1 person"},
        ]
    }


def test_build_messages_converts_string_args_to_object():
    out = OllamaAdapter._build_messages(_assistant_turn('{"camera_id": "cam1"}'))
    args = out[1]["tool_calls"][0]["function"]["arguments"]
    assert args == {"camera_id": "cam1"}


def test_build_messages_leaves_dict_args_untouched():
    out = OllamaAdapter._build_messages(_assistant_turn({"camera_id": "cam1"}))
    assert out[1]["tool_calls"][0]["function"]["arguments"] == {"camera_id": "cam1"}


def test_build_messages_empty_string_args_become_empty_object():
    out = OllamaAdapter._build_messages(_assistant_turn(""))
    assert out[1]["tool_calls"][0]["function"]["arguments"] == {}


def test_build_messages_invalid_json_args_become_empty_object():
    out = OllamaAdapter._build_messages(_assistant_turn("{not valid json"))
    assert out[1]["tool_calls"][0]["function"]["arguments"] == {}


# ── SpeechSynthesisResponse schema: audio_b64 serialization ───────────


def _speech_kwargs(**extra):
    base = dict(audio_uri="opennvr://audio/clip.wav", duration_seconds=1.0,
                sample_rate=22050, voice="libritts", text_length=5,
                executed_at=1, latency_ms=10)
    base.update(extra)
    return base


def test_speech_response_drops_audio_b64_when_absent():
    resp = SpeechSynthesisResponse(**_speech_kwargs())
    assert "audio_b64" not in resp.model_dump()


def test_speech_response_keeps_audio_b64_when_present():
    resp = SpeechSynthesisResponse(**_speech_kwargs(audio_b64="QUJD"))
    assert resp.model_dump()["audio_b64"] == "QUJD"


# ── speech_synthesis task: surface inline audio from the adapter ──────


def test_speech_task_surfaces_audio_b64():
    adapter = _FakeAdapter({
        "audio_uri": "opennvr://audio/clip.wav", "audio_b64": "QUJD",
        "duration_seconds": 1.0, "sample_rate": 22050, "voice": "libritts",
        "text_length": 5,
    })
    resp = SpeechSynthesisTask().process({"text": "hi", "inline": True}, adapter)
    assert resp.audio_b64 == "QUJD"


def test_speech_task_audio_b64_none_when_adapter_omits_it():
    adapter = _FakeAdapter({
        "audio_uri": "opennvr://audio/clip.wav",
        "duration_seconds": 1.0, "sample_rate": 22050, "voice": "libritts",
        "text_length": 5,
    })
    resp = SpeechSynthesisTask().process({"text": "hi"}, adapter)
    assert resp.audio_b64 is None
    assert "audio_b64" not in resp.model_dump()


# ── Piper _wants_inline helper ────────────────────────────────────────


def test_piper_wants_inline_accepts_synonyms():
    from app.adapters.audio.piper_adapter import PiperAdapter
    assert PiperAdapter._wants_inline({"inline": True})
    assert PiperAdapter._wants_inline({"return_audio_inline": True})
    assert PiperAdapter._wants_inline({"audio_inline": True})
    assert not PiperAdapter._wants_inline({})
    assert not PiperAdapter._wants_inline({"inline": False})
