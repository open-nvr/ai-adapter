# Copyright (c) 2026 OpenNVR
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
PiperService — Piper-specific implementation of ``AdapterService``.

Migrated to ``opennvr-adapter-sdk`` in A2.3a. All boilerplate (auth,
metrics, FastAPI routes, body parsing, error-envelope translation) is
now in the SDK; this service holds only the Piper-specific:

* Load lifecycle around the legacy ``PiperAdapter``
* Live ``model.fingerprint`` from the ONNX file
* §3.3 ``HardwareEvaluationResponse`` (CPU verdict)
* Input validation + delegation to ``PiperAdapter.infer``

The legacy ``PiperAdapter`` (in ``app/adapters/audio/piper_adapter.py``)
stays untouched — it's the underlying model wrapper, not the contract
shim.
"""
from __future__ import annotations

import hashlib
import logging
import os
import platform
import threading
import time
from datetime import datetime, timezone
from typing import Any

from app.adapters.audio.piper_adapter import PiperAdapter
from app.config import MODEL_WEIGHTS_DIR
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

logger = logging.getLogger(__name__)

DEFAULT_VOICE: str = "en_US-libritts-high"
MAX_TEXT_CHARS: int = 10_000


class PiperService(AdapterService):
    """Stateful façade around ``PiperAdapter``."""

    def __init__(
        self,
        default_voice: str = DEFAULT_VOICE,
        voice_dir: str | None = None,
    ) -> None:
        self._default_voice = default_voice
        self._voice_dir = voice_dir or os.path.join(MODEL_WEIGHTS_DIR, "piper")
        self._adapter: PiperAdapter = PiperAdapter(
            config={
                "enabled": True,
                "voice": default_voice,
                "voice_dir": self._voice_dir,
            }
        )
        self._load_state: HealthStatus = HealthStatus.LOADING
        self._load_error: str | None = None
        self._fingerprint_cache: str | None = None
        self._lock = threading.Lock()

    # ── AdapterService impl ────────────────────────────────────────

    def load(self) -> None:
        with self._lock:
            if self._load_state == HealthStatus.OK:
                return
            try:
                self._adapter.ensure_model_loaded()
                self._fingerprint_cache = self._compute_fingerprint(self._default_voice)
                self._load_state = HealthStatus.OK
                self._load_error = None
                logger.info(
                    "PiperService ready: voice=%s fingerprint=%s",
                    self._default_voice, self._fingerprint_cache,
                )
            except Exception as exc:
                self._load_state = HealthStatus.ERROR
                self._load_error = str(exc)
                logger.exception("PiperService failed to load voice %s", self._default_voice)

    def is_ready(self) -> bool:
        return self._load_state == HealthStatus.OK

    def fingerprint(self) -> str | None:
        try:
            return self._compute_fingerprint(self._default_voice)
        except OSError:
            return self._fingerprint_cache

    def model_info(self) -> ModelInfo:
        return ModelInfo(
            name=self._default_voice,
            version=f"piper-tts/{self._default_voice}",
            framework="piper-tts",
            modalities_in=["text"],
            modalities_out=["audio"],
            fingerprint=self.fingerprint(),
        )

    def hardware_evaluation(self) -> HardwareEvaluationResponse:
        if self._load_state == HealthStatus.OK:
            verdict = HardwareVerdict.OK
            reasoning = "Piper runs on CPU; default voice loaded."
        elif self._load_state == HealthStatus.LOADING:
            verdict = HardwareVerdict.WARN
            reasoning = "Voice still loading."
        else:
            verdict = HardwareVerdict.BLOCKED
            reasoning = f"Voice failed to load: {self._load_error}"
        return HardwareEvaluationResponse(
            verdict=verdict,
            reasoning=reasoning,
            checked_at=datetime.now(timezone.utc),
            details={
                "gpu_required": False,
                "cpu_count": os.cpu_count() or 0,
                "platform": platform.platform(),
                "python_version": platform.python_version(),
                "default_voice": self._default_voice,
                "voice_dir": self._voice_dir,
            },
        )

    def infer(self, payload: dict[str, Any]) -> InferResponse:
        if self._load_state != HealthStatus.OK:
            raise ServiceError(
                ErrorCategory.MODEL_ERROR,
                code="weights_missing" if self._load_state == HealthStatus.ERROR else "piper.model_loading",
                message=self._load_error or "Model still loading.",
                transient=(self._load_state == HealthStatus.LOADING),
                http_status=503,
                retry_after_ms=2000 if self._load_state == HealthStatus.LOADING else None,
            )

        # SDK strips the ``__file__`` key when body_shape=TEXT, so for
        # Piper we never see it. Just regular dict-of-params.
        text = payload.get("text")
        if not isinstance(text, str) or not text.strip():
            raise ServiceError(
                ErrorCategory.TRANSPORT_ERROR,
                code="malformed_input",
                message="Field 'text' is required and must be a non-empty string.",
                transient=False,
                http_status=400,
            )
        if len(text) > MAX_TEXT_CHARS:
            raise ServiceError(
                ErrorCategory.TRANSPORT_ERROR,
                code="malformed_input",
                message=f"'text' exceeds {MAX_TEXT_CHARS}-char limit.",
                transient=False,
                http_status=400,
            )

        adapter_input: dict[str, Any] = {"task": "speech_synthesis", "text": text}
        for key in ("voice", "length_scale", "noise_scale", "noise_w"):
            if key in payload:
                adapter_input[key] = payload[key]

        try:
            raw = self._adapter.infer(adapter_input)
        except (FileNotFoundError, ValueError) as exc:
            raise ServiceError(
                ErrorCategory.MODEL_ERROR,
                code="weights_missing" if isinstance(exc, FileNotFoundError) else "malformed_input",
                message=str(exc),
                transient=False,
                http_status=400 if isinstance(exc, ValueError) else 500,
            ) from exc
        except Exception as exc:
            logger.exception("PiperAdapter.infer raised unexpectedly")
            raise ServiceError(
                ErrorCategory.MODEL_ERROR,
                code="inference_runtime_crash",
                message="Inference failed.",
                transient=False,
                http_status=500,
            ) from exc

        if isinstance(raw, dict):
            result = raw
        else:
            result = raw.model_dump() if hasattr(raw, "model_dump") else dict(raw)

        return InferResponse(
            model_name=self._default_voice,
            model_version=f"piper-tts/{self._default_voice}",
            inference_ms=int(result.get("latency_ms", 0)),
            result=result,
        )

    # ── Helpers ────────────────────────────────────────────────────

    def list_voices(self) -> dict[str, Any]:
        """Optional /voices endpoint (mounted in main.py)."""
        voices = []
        if os.path.isdir(self._voice_dir):
            for entry in sorted(os.listdir(self._voice_dir)):
                if entry.endswith(".onnx"):
                    voices.append({
                        "name": entry[: -len(".onnx")],
                        "size_bytes": os.path.getsize(os.path.join(self._voice_dir, entry)),
                    })
        return {"voices": voices, "default": self._default_voice}

    def _compute_fingerprint(self, voice_name: str) -> str:
        onnx_path = os.path.join(self._voice_dir, f"{voice_name}.onnx")
        if not os.path.exists(onnx_path):
            return "sha256:unavailable"
        digest = hashlib.sha256()
        with open(onnx_path, "rb") as fh:
            for chunk in iter(lambda: fh.read(65536), b""):
                digest.update(chunk)
        return f"sha256:{digest.hexdigest()}"
