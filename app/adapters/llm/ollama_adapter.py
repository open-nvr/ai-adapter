# Copyright (c) 2026 OpenNVR
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Ollama adapter — local open-source LLMs via the Ollama HTTP API.

Ollama (https://ollama.ai) runs as a side-car service that hosts models like
Llama 3, Qwen 2.5, Gemma 2, Phi-3, Mistral, etc. The user swaps models by
pulling them with ``ollama pull <name>`` and selecting via adapter config.

The adapter has **no Python dependencies beyond core** — it only speaks HTTP
to Ollama's ``/api/chat`` endpoint. This keeps deployments lean; users who
don't need an LLM pay zero install cost.

Input shape (either ``messages`` or ``prompt`` is required):
    {
        "task": "chat_completion",
        "messages": [{"role": "system"|"user"|"assistant", "content": "..."}],
        "prompt": "...",                 # shortcut → [{"role":"user","content":prompt}]
        "system": "You are...",          # optional system prompt prepended
        "model": "llama3.2:3b",          # optional override of adapter default
        "temperature": 0.7,              # optional
        "max_tokens": 512,               # optional
        "stop": ["</end>"],              # optional list of stop sequences
    }

Streaming is intentionally NOT supported at this layer — the pipeline engine
and OpenNVR clients expect a single response blob. Streaming is a separate
architectural change at the API boundary (see README streaming follow-up).
"""
import logging
import os
import time
from typing import Any, Dict, List, Optional

from app.adapters.base import BaseAdapter

logger = logging.getLogger(__name__)

_SUPPORTED_TASKS = {"chat_completion"}
_DEFAULT_BASE_URL = "http://localhost:11434"
_DEFAULT_MODEL = "llama3.2:3b"


class OllamaAdapter(BaseAdapter):
    name = "ollama_adapter"
    type = "llm"

    SUPPORTED_TASKS = sorted(_SUPPORTED_TASKS)

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        # Environment variable wins so operators can redirect to a shared Ollama
        # host in Docker/K8s without rebuilding the image.
        self._base_url = (
            os.environ.get("OLLAMA_BASE_URL")
            or self.config.get("base_url")
            or _DEFAULT_BASE_URL
        ).rstrip("/")
        self._default_model = self.config.get("model", _DEFAULT_MODEL)
        self._timeout_s = float(self.config.get("timeout_s", 60.0))
        self._client = None

    def load_model(self) -> None:
        # There is no "loading" step for Ollama — the model lives inside the
        # Ollama daemon. We do a health probe so failures surface on first use
        # rather than mid-pipeline, and construct a reusable httpx.Client so
        # every inference reuses the keep-alive connection.
        import httpx  # core dep (see pyproject.toml)

        self._client = httpx.Client(base_url=self._base_url, timeout=self._timeout_s)

        try:
            resp = self._client.get("/api/tags")
            resp.raise_for_status()
        except Exception as exc:
            raise RuntimeError(
                f"Cannot reach Ollama at {self._base_url}: {exc}. "
                "Is the Ollama daemon running? See https://ollama.ai"
            ) from exc

        self.model = self._client
        logger.info("Ollama adapter connected to %s (default model=%s)", self._base_url, self._default_model)

    def infer_local(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        task = input_data.get("task", "chat_completion")
        if task not in _SUPPORTED_TASKS:
            raise ValueError(f"OllamaAdapter supports {sorted(_SUPPORTED_TASKS)}, got: {task}")

        messages = self._build_messages(input_data)
        if not messages:
            raise ValueError("OllamaAdapter requires 'messages' or 'prompt' in input_data")

        model = input_data.get("model", self._default_model)
        options: Dict[str, Any] = {}
        if "temperature" in input_data:
            options["temperature"] = float(input_data["temperature"])
        if "max_tokens" in input_data:
            # Ollama calls it num_predict; -1 means unlimited.
            options["num_predict"] = int(input_data["max_tokens"])
        if "stop" in input_data:
            stop_val = input_data["stop"]
            options["stop"] = [stop_val] if isinstance(stop_val, str) else list(stop_val)

        body: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": False,
        }
        if options:
            body["options"] = options

        start_time = time.time()
        resp = self._client.post("/api/chat", json=body)
        if resp.status_code >= 400:
            raise RuntimeError(f"Ollama /api/chat returned {resp.status_code}: {resp.text}")
        payload = resp.json()

        message = payload.get("message") or {}
        content = message.get("content", "")
        role = message.get("role", "assistant")

        prompt_tokens = payload.get("prompt_eval_count")
        completion_tokens = payload.get("eval_count")
        total_tokens = None
        if prompt_tokens is not None and completion_tokens is not None:
            total_tokens = int(prompt_tokens) + int(completion_tokens)

        finish_reason = "length" if payload.get("done_reason") == "length" else "stop"

        return {
            "task": "chat_completion",
            "message": {"role": role, "content": content},
            "model": model,
            "finish_reason": finish_reason,
            "prompt_tokens": int(prompt_tokens) if prompt_tokens is not None else None,
            "completion_tokens": int(completion_tokens) if completion_tokens is not None else None,
            "total_tokens": total_tokens,
            "executed_at": int(time.time() * 1000),
            "latency_ms": int((time.time() - start_time) * 1000),
        }

    @staticmethod
    def _build_messages(input_data: Dict[str, Any]) -> List[Dict[str, str]]:
        raw_messages = input_data.get("messages")
        if isinstance(raw_messages, list) and raw_messages:
            messages = [dict(m) for m in raw_messages]
        elif isinstance(input_data.get("prompt"), str) and input_data["prompt"].strip():
            messages = [{"role": "user", "content": input_data["prompt"]}]
        else:
            return []

        system = input_data.get("system")
        if isinstance(system, str) and system.strip():
            # Prepend only if the caller didn't already include one.
            if not messages or messages[0].get("role") != "system":
                messages.insert(0, {"role": "system", "content": system})

        return messages

    @property
    def schema(self) -> Dict[str, Any]:
        return {
            "tasks": sorted(_SUPPORTED_TASKS),
            "description": "Open-source LLM chat completion via a local Ollama daemon.",
            "input_fields": {
                "messages": {"type": "array", "description": "OpenAI-style chat messages"},
                "prompt": {"type": "string", "description": "Shortcut for a single user message"},
                "system": {"type": "string", "description": "System prompt (prepended if not in messages)"},
                "model": {"type": "string", "description": f"Ollama model tag (default: {_DEFAULT_MODEL})"},
                "temperature": {"type": "number"},
                "max_tokens": {"type": "integer"},
                "stop": {"type": "array"},
            },
            "response_fields": {
                "message.role": {"type": "string"},
                "message.content": {"type": "string"},
                "model": {"type": "string"},
                "finish_reason": {"type": "string"},
                "prompt_tokens": {"type": "integer"},
                "completion_tokens": {"type": "integer"},
                "total_tokens": {"type": "integer"},
            },
        }

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "type": self.type,
            "model": self._default_model,
            "framework": "ollama",
            "tasks": sorted(_SUPPORTED_TASKS),
            "base_url": self._base_url,
            "model_loaded": self._client is not None,
        }

    def health_check(self) -> Dict[str, Any]:
        return {
            "status": "healthy",
            "type": self.type,
            "model_loaded": self._client is not None,
            "model_info": self.get_model_info(),
        }
