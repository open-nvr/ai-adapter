# Copyright (c) 2026 OpenNVR
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Ollama-backed VLM adapter — contract-compliant ``AdapterService``.

Same two tasks and result keys as the moondream adapter (``visual_qa`` →
``answer``, ``scene_caption`` → ``caption``) so it is a drop-in for the
camera-agent's ``CAPTION_ADAPTER`` slot — but instead of running weights
in-process, it forwards each request to an **Ollama endpoint serving a
multimodal model** (moondream, llava, qwen-VL, …).

Why this exists: on macOS/Windows the Docker VM has no GPU access, so
in-container VQA is CPU-only. An Ollama running ON THE HOST uses the real
GPU (Metal on Apple Silicon). This adapter keeps such calls inside the
audited Adapter Contract — the sovereignty seam is preserved, the runtime
moves. Model management collapses to ``ollama pull <model>``.

Design decisions (deliberate, keep them):

* **Lazy-ready.** ``is_ready()`` is True as soon as configuration parses.
  An unreachable endpoint is a *transient per-infer* error (503,
  retry_after), not a boot failure: the endpoint is an independent
  process that may start after us, restart under us, or briefly drop —
  the adapter owns the session and rides it out. This also means the
  compose healthcheck and CI smoke pass without a live Ollama; the
  live probe result is surfaced through ``hardware_evaluation()``.
* **Auto-pull.** If the configured model is missing at startup, a
  background thread asks Ollama to pull it (its own API, its own store,
  on the host) — first boot is self-provisioning, matching the LLM
  flow. Disable with OPENNVR_OLLAMA_VLM_AUTOPULL=false.
* **No image decode.** Frames pass through as base64; the model runtime
  does the decoding. A magic-byte sniff rejects obvious non-images so
  a bad client gets a 400 instead of a confusing model error.
"""
from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import platform
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
    InferResponse,
    ModelInfo,
    ServiceError,
)

logger = logging.getLogger(__name__)

DEFAULT_URL = "http://host.docker.internal:11434"
DEFAULT_MODEL = "moondream"
MAX_IMAGE_BYTES: int = 8 * 1024 * 1024
_SUPPORTED_TASKS: tuple[str, ...] = ("visual_qa", "scene_caption")
_CAPTION_PROMPT = "Describe this image in one short sentence."

# JPEG / PNG / WebP(RIFF) / BMP magic bytes — cheap sanity gate only.
_IMAGE_MAGICS: tuple[bytes, ...] = (b"\xff\xd8\xff", b"\x89PNG", b"RIFF", b"BM")


class OllamaVlmService(AdapterService):
    """Contract adapter that proxies VQA/captioning to an Ollama endpoint."""

    def __init__(
        self,
        url: str | None = None,
        model: str | None = None,
        *,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._url = (url or os.getenv("OPENNVR_OLLAMA_VLM_URL", DEFAULT_URL)).rstrip("/")
        self._model = model or os.getenv("OPENNVR_OLLAMA_VLM_MODEL", DEFAULT_MODEL)
        self._timeout = float(os.getenv("OPENNVR_OLLAMA_VLM_TIMEOUT_S", "120"))
        self._autopull = (os.getenv("OPENNVR_OLLAMA_VLM_AUTOPULL", "true").strip().lower()
                          not in ("false", "0", "no"))
        # One client, connection-pooled. ``transport`` is injectable for tests.
        self._client = httpx.Client(timeout=self._timeout, transport=transport)
        self._ready = False
        self._probe: dict[str, Any] = {"state": "unprobed"}
        self._started_at = datetime.now(timezone.utc)
        self._pull_thread: threading.Thread | None = None

    # ── AdapterService impl ────────────────────────────────────────

    def load(self) -> None:
        """Validate config, probe once, kick auto-pull if needed.

        Never raises and never blocks on the network beyond one short
        probe — see the lazy-ready rationale in the module docstring."""
        self._ready = True
        try:
            self._refresh_probe(timeout=3.0)
        except Exception:  # pragma: no cover - defensive; probe already guards
            logger.exception("ollamavlm: startup probe raised unexpectedly")
        if (self._autopull and self._probe.get("state") == "model_missing"
                and self._pull_thread is None):
            self._pull_thread = threading.Thread(
                target=self._pull_model, name="ollamavlm-pull", daemon=True)
            self._pull_thread.start()

    def is_ready(self) -> bool:
        return self._ready

    def fingerprint(self) -> str | None:
        digest = hashlib.sha256(f"{self._url}::{self._model}".encode()).hexdigest()
        return f"sha256-id:{digest}"

    def model_info(self) -> ModelInfo:
        return ModelInfo(
            name=self._model,
            version=f"ollama/{self._model}",
            framework="ollama",
            modalities_in=["image", "text"],
            modalities_out=["text"],
            fingerprint=self.fingerprint(),
        )

    def hardware_evaluation(self) -> HardwareEvaluationResponse:
        self._refresh_probe(timeout=3.0)
        state = self._probe.get("state")
        if state == "ok":
            verdict = HardwareVerdict.OK
            reasoning = (f"Ollama at {self._url} serves {self._model!r}. Inference "
                         "runs wherever that endpoint runs — host GPU when the "
                         "endpoint is the host machine (Metal on Apple Silicon).")
        elif state == "model_missing":
            verdict = HardwareVerdict.WARN
            reasoning = (f"Ollama at {self._url} is up but {self._model!r} is not "
                         "pulled yet" + (" (auto-pull in progress)." if self._autopull
                                         else f" — run: ollama pull {self._model}"))
        else:
            verdict = HardwareVerdict.WARN
            reasoning = (f"Ollama endpoint {self._url} is not answering: "
                         f"{self._probe.get('error', 'unknown')} — inference will "
                         "return transient errors until it is reachable.")
        uptime_s = int((datetime.now(timezone.utc) - self._started_at).total_seconds())
        return HardwareEvaluationResponse(
            verdict=verdict, reasoning=reasoning,
            checked_at=datetime.now(timezone.utc),
            details={
                "gpu_required": False,
                "runtime": "ollama-proxy",
                "endpoint": self._url,
                "model": self._model,
                "endpoint_state": state,
                "cpu_count": os.cpu_count() or 0,
                "platform": platform.platform(),
                "python_version": platform.python_version(),
                "adapter_uptime_seconds": uptime_s,
            },
        )

    def infer(self, payload: dict[str, Any]) -> InferResponse:
        image_bytes = payload.get("__file__")
        if not image_bytes:
            raise ServiceError(
                ErrorCategory.TRANSPORT_ERROR, code="missing_image",
                message=("ollamavlm expects an image in the request body "
                         "(multipart 'frame' field or JSON 'frame_b64')."),
                transient=False, http_status=400,
            )
        if not image_bytes.startswith(_IMAGE_MAGICS):
            raise ServiceError(
                ErrorCategory.TRANSPORT_ERROR, code="invalid_image",
                message="request body does not look like JPEG/PNG/WebP/BMP",
                transient=False, http_status=400,
            )

        task = (payload.get("task") or "").strip()
        question = (payload.get("question") or payload.get("prompt") or "").strip()
        if task and task not in _SUPPORTED_TASKS:
            raise ServiceError(
                ErrorCategory.NOT_SUPPORTED, code="unsupported_task",
                message=f"task {task!r} not in {list(_SUPPORTED_TASKS)}",
                transient=False, http_status=400,
            )
        is_vqa = bool(question) and task != "scene_caption"
        prompt = question if is_vqa else _CAPTION_PROMPT

        started = time.monotonic()
        text = self._generate(prompt, image_bytes)
        elapsed_ms = int((time.monotonic() - started) * 1000)

        result: dict[str, Any] = {
            "task": "visual_qa" if is_vqa else "scene_caption",
            "endpoint": self._url,
        }
        if is_vqa:
            result["answer"] = text
            result["question"] = question
        else:
            result["caption"] = text
        return InferResponse(
            model_name=self._model,
            model_version=f"ollama/{self._model}",
            inference_ms=elapsed_ms,
            result=result,
        )

    # ── Ollama plumbing ────────────────────────────────────────────

    def _generate(self, prompt: str, image_bytes: bytes) -> str:
        body = {
            "model": self._model,
            "prompt": prompt,
            "images": [base64.b64encode(image_bytes).decode("ascii")],
            "stream": False,
        }
        try:
            resp = self._client.post(f"{self._url}/api/generate", json=body)
        except httpx.HTTPError as exc:
            raise ServiceError(
                ErrorCategory.MODEL_ERROR, code="endpoint_unreachable",
                message=f"Ollama endpoint {self._url} not reachable: {exc}",
                transient=True, http_status=503, retry_after_ms=2000,
            ) from exc
        if resp.status_code == 404:
            # Model not pulled (Ollama 404s /api/generate for unknown models).
            hint = ("auto-pull is running — retry shortly" if self._autopull
                    else f"run: ollama pull {self._model}")
            raise ServiceError(
                ErrorCategory.MODEL_ERROR, code="model_not_pulled",
                message=f"Ollama has no model {self._model!r} ({hint})",
                transient=True, http_status=503, retry_after_ms=5000,
            )
        if resp.status_code != 200:
            raise ServiceError(
                ErrorCategory.MODEL_ERROR, code="endpoint_error",
                message=f"Ollama returned HTTP {resp.status_code}: {resp.text[:300]}",
                transient=True, http_status=502, retry_after_ms=2000,
            )
        try:
            text = str(resp.json().get("response", "")).strip()
        except json.JSONDecodeError as exc:
            raise ServiceError(
                ErrorCategory.MODEL_ERROR, code="bad_response",
                message="Ollama returned non-JSON to /api/generate",
                transient=True, http_status=502, retry_after_ms=2000,
            ) from exc
        if not text:
            raise ServiceError(
                ErrorCategory.MODEL_ERROR, code="empty_response",
                message=f"Ollama model {self._model!r} returned an empty response",
                transient=True, http_status=502, retry_after_ms=2000,
            )
        return text

    def _refresh_probe(self, timeout: float) -> None:
        """Best-effort endpoint+model check; never raises. Feeds
        hardware_evaluation() and the auto-pull decision — NOT readiness."""
        try:
            resp = self._client.get(f"{self._url}/api/tags", timeout=timeout)
            resp.raise_for_status()
            models = {
                str(m.get("name", "")) for m in resp.json().get("models", [])
            }
            # "moondream" matches "moondream:latest"; a tagged configure
            # value must match exactly.
            want = self._model if ":" in self._model else f"{self._model}:latest"
            present = want in models or self._model in models
            self._probe = {"state": "ok" if present else "model_missing",
                           "models": sorted(models)[:20]}
        except Exception as exc:
            self._probe = {"state": "unreachable", "error": str(exc)}

    def _pull_model(self) -> None:
        """Ask Ollama to pull the configured model (blocking, background
        thread). Failure is logged, never fatal — infer() keeps returning
        transient errors with the pull hint until the model exists."""
        logger.info("ollamavlm: model %r missing at %s — requesting pull",
                    self._model, self._url)
        try:
            resp = self._client.post(
                f"{self._url}/api/pull",
                json={"model": self._model, "stream": False},
                timeout=1800.0,
            )
            if resp.status_code == 200:
                logger.info("ollamavlm: pull of %r finished", self._model)
            else:
                logger.warning("ollamavlm: pull of %r failed: HTTP %s %s",
                               self._model, resp.status_code, resp.text[:200])
        except Exception:
            logger.exception("ollamavlm: pull of %r failed", self._model)
        finally:
            self._refresh_probe(timeout=3.0)
