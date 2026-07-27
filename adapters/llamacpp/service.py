# Copyright (c) 2026 OpenNVR
# SPDX-License-Identifier: Apache-2.0

"""
llama.cpp LLM service — Qwen2.5-3B-Instruct (GGUF Q4_K_M) via a persistent
``llama-server`` child process. Torch-free, CPU-first, GPU-optional (-ngl).

This is the lightweight, governed replacement for the raw ``ollama`` container:
same local llama.cpp engine, but wrapped in the OpenNVR adapter contract so
KAI-C registers, health-polls, and audits it — and it carries TOOL CALLS
(the reference SmolLM2 adapter does not), which the camera-agent's tool loop
needs.

Design: ``load()`` spawns ``llama-server`` bound to loopback INSIDE the
container and polls its ``/health``; ``infer()`` proxies to that child's
OpenAI-compatible ``/v1/chat/completions`` (the wire format is served locally
on 127.0.0.1 — no cloud). The GGUF lives in the mounted weights volume; if
missing it is downloaded ONCE on first boot from OPENNVR_LLM_MODEL_URL (same
first-boot pattern as the whisper adapter). Pre-populate the volume (or set
the URL empty) for fully-offline deployments.

Input:  {"messages": [{"role","content"}...], "tools"?: [...],
         "max_tokens"?: int, "temperature"?: float, "top_p"?: float}
Output: InferResponse result={"text", "tool_calls"?, "finish_reason", "usage"}
"""
from __future__ import annotations

import atexit
import hashlib
import logging
import os
import platform
import subprocess
import threading
import time
from datetime import datetime, timezone
from typing import Any

import httpx

from opennvr_adapter_sdk import (
    AdapterService,
    ErrorCategory,
    HardwareEvaluationResponse,
    HardwareVerdict,
    HealthStatus,
    InferResponse,
    ModelInfo,
    ServiceError,
)
from opennvr_adapter_sdk.model_fetch import ensure_model_file

logger = logging.getLogger(__name__)

MODEL_PATH: str = os.getenv(
    "OPENNVR_LLM_MODEL_PATH", "/app/models/Qwen2.5-3B-Instruct-Q4_K_M.gguf"
)
# Downloaded to MODEL_PATH on first boot when the file is absent. Set to an
# empty string to forbid any fetch (offline / sovereignty-strict installs).
MODEL_URL: str = os.getenv(
    "OPENNVR_LLM_MODEL_URL",
    "https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/"
    "qwen2.5-3b-instruct-q4_k_m.gguf",
)
SERVER_BIN: str = os.getenv("LLAMACPP_SERVER_BIN", "llama-server")
CTX_SIZE: int = int(os.getenv("LLAMACPP_CTX_SIZE", "4096"))
THREADS: int = int(os.getenv("LLAMACPP_THREADS", str(os.cpu_count() or 4)))
BATCH_SIZE: int = int(os.getenv("LLAMACPP_BATCH_SIZE", "256"))
GPU_LAYERS: int = int(os.getenv("LLAMACPP_GPU_LAYERS", "0"))  # -ngl; 0 = CPU-only
INTERNAL_HOST: str = "127.0.0.1"
INTERNAL_PORT: int = int(os.getenv("LLAMACPP_INTERNAL_PORT", "8080"))
STARTUP_TIMEOUT_S: float = float(os.getenv("LLAMACPP_STARTUP_TIMEOUT_S", "180"))
MAX_TOKENS_DEFAULT: int = int(os.getenv("LLAMACPP_MAX_TOKENS", "256"))
MAX_TOKENS_CAP: int = 1024


class LlamaCppService(AdapterService):
    def __init__(self, model_path: str | None = None) -> None:
        self._model_path = model_path or MODEL_PATH
        self._base_url = f"http://{INTERNAL_HOST}:{INTERNAL_PORT}"
        self._proc: subprocess.Popen | None = None
        self._http: httpx.Client | None = None
        self._load_state: HealthStatus = HealthStatus.LOADING
        self._load_error: str | None = None
        self._fingerprint_cache: str | None = None
        self._lock = threading.Lock()
        self._started_at = datetime.now(timezone.utc)

    # ── AdapterService ─────────────────────────────────────────────
    def load(self) -> None:
        with self._lock:
            if self._load_state == HealthStatus.OK:
                return
            try:
                ensure_model_file(
                    self._model_path, MODEL_URL, label="Qwen GGUF", logger=logger
                )
                self._fingerprint_cache = _cheap_fingerprint(self._model_path)
                self._spawn_server()
                self._http = httpx.Client(base_url=self._base_url, timeout=120.0)
                self._wait_healthy()
                self._load_state = HealthStatus.OK
                self._load_error = None
                logger.info("LlamaCppService ready at %s", self._base_url)
            except Exception as exc:
                self._load_state = HealthStatus.ERROR
                self._load_error = str(exc)
                logger.exception("LlamaCppService failed to load")

    def _spawn_server(self) -> None:
        args = [
            SERVER_BIN, "-m", self._model_path,
            "-c", str(CTX_SIZE), "-t", str(THREADS), "-b", str(BATCH_SIZE),
            "--host", INTERNAL_HOST, "--port", str(INTERNAL_PORT),
            "--jinja",  # enable chat-template tool-call parsing
        ]
        if GPU_LAYERS > 0:
            args += ["-ngl", str(GPU_LAYERS)]
        logger.info("spawning: %s", " ".join(args[:3]) + " …")
        self._proc = subprocess.Popen(
            args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        atexit.register(self._terminate)

    def _wait_healthy(self) -> None:
        deadline = time.monotonic() + STARTUP_TIMEOUT_S
        probe = httpx.Client(base_url=self._base_url, timeout=2.0)
        try:
            while time.monotonic() < deadline:
                if self._proc is not None and self._proc.poll() is not None:
                    raise RuntimeError(
                        f"llama-server exited during startup (code {self._proc.returncode})"
                    )
                try:
                    if probe.get("/health").status_code == 200:
                        return
                except httpx.HTTPError:
                    pass
                time.sleep(0.5)
            raise TimeoutError(f"llama-server not healthy within {STARTUP_TIMEOUT_S}s")
        finally:
            probe.close()

    def _terminate(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except Exception:
                self._proc.kill()

    def is_ready(self) -> bool:
        return (
            self._load_state == HealthStatus.OK
            and self._proc is not None
            and self._proc.poll() is None
        )

    def fingerprint(self) -> str | None:
        return self._fingerprint_cache

    def model_info(self) -> ModelInfo:
        return ModelInfo(
            name=os.path.basename(self._model_path),
            version="q4_k_m",
            framework="llama.cpp",
            modalities_in=["text"],
            modalities_out=["text"],
            fingerprint=self.fingerprint(),
        )

    def hardware_evaluation(self) -> HardwareEvaluationResponse:
        if self._load_state == HealthStatus.OK:
            verdict, reasoning = HardwareVerdict.OK, "Qwen2.5-3B GGUF served by llama.cpp."
        elif self._load_state == HealthStatus.LOADING:
            verdict, reasoning = HardwareVerdict.WARN, "Model still loading."
        else:
            verdict, reasoning = HardwareVerdict.BLOCKED, f"Load failed: {self._load_error}"
        uptime = int((datetime.now(timezone.utc) - self._started_at).total_seconds())
        return HardwareEvaluationResponse(
            verdict=verdict, reasoning=reasoning,
            checked_at=datetime.now(timezone.utc),
            details={
                "gpu_required": False, "gpu_layers": GPU_LAYERS,
                "runtime": "llama.cpp", "model_path": self._model_path,
                "ctx_size": CTX_SIZE, "threads": THREADS,
                "cpu_count": os.cpu_count() or 0, "platform": platform.platform(),
                "adapter_uptime_seconds": uptime,
            },
        )

    def infer(self, payload: dict[str, Any]) -> InferResponse:
        if not self.is_ready():
            transient = self._load_state == HealthStatus.LOADING
            raise ServiceError(
                ErrorCategory.MODEL_ERROR, code="model_not_loaded",
                message=self._load_error or "LLM not yet loaded",
                transient=transient, http_status=503,
                retry_after_ms=2000 if transient else None,
            )
        messages = payload.get("messages")
        if not messages or not isinstance(messages, list):
            raise ServiceError(
                ErrorCategory.TRANSPORT_ERROR, code="missing_messages",
                message="'messages' must be a non-empty list of {role, content}",
                transient=False, http_status=400,
            )
        body: dict[str, Any] = {
            "messages": messages,
            "max_tokens": min(int(payload.get("max_tokens") or MAX_TOKENS_DEFAULT), MAX_TOKENS_CAP),
            "temperature": float(payload.get("temperature", 0.3)),
            "top_p": float(payload.get("top_p", 0.9)),
            "stream": False,
        }
        tools = payload.get("tools")
        if tools:
            body["tools"] = tools
            body["tool_choice"] = payload.get("tool_choice", "auto")

        started = time.monotonic()
        try:
            resp = self._http.post("/v1/chat/completions", json=body)  # type: ignore[union-attr]
        except httpx.HTTPError as exc:
            raise ServiceError(
                ErrorCategory.MODEL_ERROR, code="inference_runtime_crash",
                message=f"llama-server unreachable: {exc}",
                transient=True, http_status=502, retry_after_ms=1000,
            ) from exc
        elapsed_ms = int((time.monotonic() - started) * 1000)
        if resp.status_code != 200:
            raise ServiceError(
                ErrorCategory.MODEL_ERROR, code="inference_failed",
                message=f"llama-server HTTP {resp.status_code}: {resp.text[:200]}",
                transient=True, http_status=502,
            )
        data = resp.json()
        choice = (data.get("choices") or [{}])[0]
        msg = choice.get("message", {})
        result: dict[str, Any] = {
            "text": (msg.get("content") or "").strip(),
            "finish_reason": choice.get("finish_reason"),
            "usage": data.get("usage", {}),
        }
        if msg.get("tool_calls"):
            result["tool_calls"] = msg["tool_calls"]
        return InferResponse(
            model_name=os.path.basename(self._model_path),
            model_version="q4_k_m",
            inference_ms=elapsed_ms,
            result=result,
        )


def _cheap_fingerprint(path: str) -> str:
    """Size + first/last 4 MiB hash — fast enough to run at load, stable enough
    to flag a weights swap (full-file hash of a 2 GB GGUF is too slow to be
    worth it here)."""
    size = os.path.getsize(path)
    h = hashlib.sha256()
    h.update(str(size).encode())
    chunk = 4 * 1024 * 1024
    with open(path, "rb") as f:
        h.update(f.read(chunk))
        if size > chunk:
            f.seek(max(0, size - chunk))
            h.update(f.read(chunk))
    return f"sha256:{h.hexdigest()[:16]}"
