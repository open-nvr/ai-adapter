# Copyright (c) 2026 OpenNVR
# SPDX-License-Identifier: AGPL-3.0-or-later
"""An OpenAI-compatible chat endpoint on the same NPU engine as the captioner.

``POST /v1/chat/completions`` takes what the camera agent's OpenAI client
sends — ``messages`` (system / user / assistant / tool), ``tools`` as OpenAI
function envelopes, ``tool_choice``, ``temperature``, ``max_tokens`` — and
answers in the shape it reads: ``choices[0].message`` with ``content`` and
``tool_calls[].function.{name, arguments}`` (arguments a JSON string),
``finish_reason`` ``tool_calls`` or ``stop``. So the agent switches to the NPU
with ``llm_provider: openai`` and this adapter's URL, nothing else.

What is measured behind this (bench/ventuno-q-npu-agent-llm-2026-11-07.md in
open-nvr): the agent's real prompt and seven of its tool schemas, 10 of 10
right tool choices on its evaluation questions at 0.8–2.0 s a turn on a
Ventuno Q, against 20–25 s for the CPU model. The engine's window is fixed at
compile time (4,096 tokens on that bundle): a prompt that does not fit is
answered ``400 context_length_exceeded`` so the caller can shorten it, never
silently truncated.

One engine, one request at a time: captions and chat share the service's
lock, so a long answer delays a caption and never corrupts it. The output
cap (``max_tokens``, bounded by OPENNVR_LITERTLM_CHAT_MAX_TOKENS) is the
deadline: the runtime cannot be interrupted mid-answer.
"""
from __future__ import annotations

import json
import logging
import os
import time
import uuid
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)

router = APIRouter()

MODEL_ID = os.getenv("OPENNVR_LITERTLM_CHAT_MODEL_ID", "gemma-4-e2b-npu")
CHAT_MAX_TOKENS = int(os.getenv("OPENNVR_LITERTLM_CHAT_MAX_TOKENS", "512"))
CHAT_DEFAULT_TOKENS = int(os.getenv("OPENNVR_LITERTLM_CHAT_DEFAULT_TOKENS", "256"))


def _error(status: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={
        "error": {"message": message, "type": "invalid_request_error" if status < 500 else "server_error",
                  "code": code}})


def _text_of(content: Any) -> str:
    """OpenAI content is a string or a list of parts; only text is carried."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(str(p.get("text", "")) for p in content if isinstance(p, dict) and p.get("type") == "text")
    return str(content)


def _tool_response(*results: dict[str, Any]) -> dict[str, Any]:
    """Tool results in the runtime's own shape (what its automatic tool
    calling sends back itself): ONE ``tool`` turn whose content is a
    ``tool_response`` part per result, with the tool's name and what it
    returned. A plain text ``tool`` turn is refused by the native side,
    and the Python entry point takes one message, not a list — both
    measured on the Ventuno Q, 2026-11-07."""
    return {"role": "tool", "content": [
        {"type": "tool_response", "name": m.get("name") or "", "response": _text_of(m.get("content"))}
        for m in results]}


def _engine_message(m: dict[str, Any]) -> dict[str, Any]:
    """One OpenAI message as the engine's history takes it."""
    role = m.get("role")
    if role == "assistant":
        entry: dict[str, Any] = {"role": "assistant", "content": _text_of(m.get("content"))}
        calls = []
        for tc in m.get("tool_calls") or []:
            fn = tc.get("function") or {}
            args = fn.get("arguments")
            if isinstance(args, str):
                try:
                    args = json.loads(args) if args.strip() else {}
                except json.JSONDecodeError:
                    args = {"_raw": args}
            # The runtime's chat template reads the OpenAI envelope — a
            # flat {name, arguments} fails it with "undefined value"
            # (measured on the Ventuno Q, 2026-11-07); arguments stay a
            # mapping, not the JSON string the client sent.
            calls.append({"id": tc.get("id") or f"call_{len(calls) + 1}", "type": "function",
                          "function": {"name": fn.get("name") or tc.get("name") or "",
                                       "arguments": args or {}}})
        if calls:
            entry["tool_calls"] = calls
        return entry
    if role == "tool":
        return _tool_response(m)
    return {"role": "user", "content": _text_of(m.get("content"))}


def split_messages(messages: list[dict[str, Any]]
                   ) -> tuple[str | None, list[dict[str, Any]], str | dict[str, Any]]:
    """(system text, history for the engine, what to send).

    System messages are joined into one system prompt. What gets sent is
    the LAST user message's text, with everything else as history in the
    engine's own message shape (assistant turns keep their tool calls) —
    or, when the conversation ENDS with tool results (the agent's loop:
    the model called a tool, here is the result, now answer), those
    results as one ``tool`` turn in the runtime's ``tool_response`` shape,
    with the user's question and the model's call before them as history.
    Sending the question again after the result would put it in the wrong
    order."""
    system_parts = [_text_of(m.get("content")) for m in messages if m.get("role") == "system"]
    system = "\n\n".join(p for p in system_parts if p) or None
    rest = [m for m in messages if m.get("role") != "system"]
    tail: list[dict[str, Any]] = []
    while rest and rest[-1].get("role") == "tool":
        tail.insert(0, rest.pop())
    if tail:
        return system, [_engine_message(m) for m in rest], _tool_response(*tail)
    last_user_index = max((i for i, m in enumerate(rest) if m.get("role") == "user"), default=-1)
    if last_user_index < 0:
        return system, [_engine_message(m) for m in rest], ""
    history = [_engine_message(m) for i, m in enumerate(rest) if i != last_user_index]
    return system, history, _text_of(rest[last_user_index].get("content"))


def to_openai_message(text: str, tool_calls: list[dict[str, Any]]) -> dict[str, Any]:
    msg: dict[str, Any] = {"role": "assistant", "content": text or None}
    if tool_calls:
        msg["tool_calls"] = [{
            "id": c.get("id") or f"call_{uuid.uuid4().hex[:12]}",
            "type": "function",
            "function": {"name": c.get("name") or "", "arguments": json.dumps(c.get("arguments") or {})},
        } for c in tool_calls]
    return msg


def _service_of(request: Request) -> Any:
    """The adapter's service: set directly (tests), or through the getter
    main.py registers, which the SDK only answers once its lifespan has
    built the service — before that the engine is not loaded anyway."""
    state = request.app.state
    service = getattr(state, "litertlm_service", None)
    if service is not None:
        return service
    getter = getattr(state, "litertlm_service_getter", None)
    if getter is None:
        return None
    try:
        return getter()
    except AssertionError:
        return None


@router.get("/v1/models")
async def models() -> dict[str, Any]:
    return {"object": "list", "data": [{"id": MODEL_ID, "object": "model", "owned_by": "open-nvr"}]}


@router.post("/v1/chat/completions")
async def chat_completions(request: Request) -> Any:
    service = _service_of(request)
    if service is None:
        return _error(503, "model_not_loaded", "the chat service is not attached")
    try:
        body = await request.json()
    except Exception:  # noqa: BLE001
        return _error(400, "invalid_json", "the body is not JSON")
    if not isinstance(body, dict) or not isinstance(body.get("messages"), list) or not body["messages"]:
        return _error(400, "invalid_request", "'messages' (a non-empty list) is required")
    if body.get("stream"):
        return _error(400, "stream_unsupported", "streaming is not supported; send stream=false")
    system, history, user = split_messages(body["messages"])
    if not user:
        return _error(400, "invalid_request", "the last message must be from the user, or a tool result")
    tools = [t for t in (body.get("tools") or []) if isinstance(t, dict)]
    if body.get("tool_choice") == "none":
        tools = []
    try:
        max_tokens = int(body.get("max_tokens") or CHAT_DEFAULT_TOKENS)
    except (TypeError, ValueError):
        max_tokens = CHAT_DEFAULT_TOKENS
    max_tokens = max(1, min(max_tokens, CHAT_MAX_TOKENS))
    temperature = body.get("temperature")
    started = time.monotonic()
    try:
        text, calls, ready = service.chat(system=system, history=history, user=user, tools=tools,
                                          max_tokens=max_tokens, temperature=temperature)
    except service.NotReady as exc:
        return _error(503, "model_not_loaded", str(exc))
    except service.PromptTooLong as exc:
        return _error(400, "context_length_exceeded", str(exc))
    except Exception as exc:  # noqa: BLE001
        logger.exception("litertlm chat: inference failed")
        return _error(500, "inference_failed", f"{type(exc).__name__}: {exc}")
    elapsed_ms = int((time.monotonic() - started) * 1000)
    message = to_openai_message(text, calls)
    sent_chars = len(user) if isinstance(user, str) else len(json.dumps(user))
    prompt_chars = len(system or "") + sum(len(json.dumps(m)) for m in history) + sent_chars + len(json.dumps(tools))
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:16]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": MODEL_ID,
        "choices": [{"index": 0, "message": message,
                     "finish_reason": "tool_calls" if calls else "stop"}],
        # Estimated from characters: the runtime's token counter is not
        # available on this build. Marked so nobody bills by it.
        "usage": {"prompt_tokens": prompt_chars // 4, "completion_tokens": len(text) // 4,
                  "total_tokens": (prompt_chars + len(text)) // 4, "estimated": True},
        "opennvr": {"inference_ms": elapsed_ms, "accelerator": "qualcomm-hexagon-npu", "tools_attached": len(tools)},
    }
