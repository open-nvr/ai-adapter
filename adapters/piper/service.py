# Copyright (c) 2026 OpenNVR
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
PiperService — Piper-specific implementation of ``AdapterService``.

Migrated to ``opennvr-adapter-sdk`` in the SDK migration. All boilerplate (auth,
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
            # Domain metrics (registration is idempotent in the SDK).
            self.metrics.register_counter(
                "adapter_audio_seconds_total",
                "Seconds of speech audio synthesized.")
            self.metrics.register_histogram(
                "adapter_realtime_factor",
                "Compute-seconds per audio-second (RTF); >1 = slower than playback.",
                buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0))
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

        # Optional inline-audio mode. The default response shape returns
        # ``audio_uri`` (the internal ``opennvr://audio/...`` scheme) so
        # operators with shared-volume deployments can stream the file
        # directly without ballooning the response payload. Callers that
        # don't have a shared volume — the camera-agent example, which
        # uses Pipecat's TTS service over the network — set
        # ``inline: true`` (or ``return_audio_inline: true``) in the
        # request body. The adapter then base64-encodes the generated
        # WAV alongside the URI so both consumer styles work from one
        # endpoint.
        #
        # Payload-size posture: Piper at 22050 Hz × 16-bit mono produces
        # ~44 KB/s of audio. The ``MAX_TEXT_CHARS`` input cap (10000
        # chars) bounds the worst case at roughly 5-8 MB of base64
        # alongside the JSON envelope. That's under FastAPI/Starlette's
        # default response limits but above some reverse-proxy
        # defaults — operators behind nginx / Traefik with tight
        # ``client_max_body_size`` may need to raise it for the
        # camera-agent path. Documented in the example README.
        if _wants_inline_audio(payload):
            audio_b64 = _read_audio_inline(result, self._adapter)
            if audio_b64 is not None:
                result["audio_b64"] = audio_b64

        # Domain metrics: synthesized audio volume + realtime factor. RTF is
        # the TTS efficiency number — compute-seconds per audio-second; >1
        # means speech generation is slower than playback, so voice replies
        # lag behind the conversation on this hardware.
        try:
            duration_s = float(result.get("duration_seconds") or 0.0)
            latency_s = float(result.get("latency_ms", 0)) / 1000.0
            if duration_s > 0:
                self.metrics.inc_counter("adapter_audio_seconds_total", duration_s)
                if latency_s > 0:
                    self.metrics.observe(
                        "adapter_realtime_factor", latency_s / duration_s)
        except Exception:  # pragma: no cover - metrics must never break infer
            logger.debug("piper domain metrics recording failed", exc_info=True)

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


# ── Inline-audio helpers ───────────────────────────────────────────


def _wants_inline_audio(payload: dict[str, Any]) -> bool:
    """Truthy if the caller asked for ``audio_b64`` inline.

    Accepts both ``inline`` and ``return_audio_inline`` because the
    canonical name moved during review and we don't want to break
    pre-existing operator clients that picked up the early field
    name from the README draft. Standard Python truthy semantics
    on the parsed JSON value.
    """
    for key in ("inline", "return_audio_inline", "audio_inline"):
        if key in payload and bool(payload[key]):
            return True
    return False


def _read_audio_inline(
    result: dict[str, Any], adapter: Any,
) -> str | None:
    """Base64-encode the WAV the legacy adapter just wrote.

    Resolve the file path from the adapter's response. The adapter
    surfaces it under ``audio_uri`` (the ``opennvr://audio/...``
    scheme) and we resolve via ``resolve_audio_uri`` to a real
    filesystem path. Returns ``None`` if the file isn't there
    (rare — the adapter just wrote it; mostly a defensive guard
    against test fixtures or odd race conditions).
    """
    import base64
    audio_uri = result.get("audio_uri")
    if not isinstance(audio_uri, str) or not audio_uri:
        return None
    # Late import — keeps the legacy ``app/`` dependency localised
    # so a future de-coupling cleanup only has to touch this helper.
    try:
        from app.utils.audio_utils import resolve_audio_uri
    except Exception:
        logger.exception("piper inline: could not import resolve_audio_uri")
        return None
    try:
        path = resolve_audio_uri(audio_uri)
    except Exception:
        logger.exception(
            "piper inline: could not resolve audio_uri %r", audio_uri
        )
        return None
    if not os.path.isfile(path):
        logger.warning(
            "piper inline: resolved path %r does not exist (adapter "
            "may have written it elsewhere)", path,
        )
        return None
    try:
        with open(path, "rb") as fh:
            return base64.b64encode(fh.read()).decode("ascii")
    except Exception:
        logger.exception(
            "piper inline: could not read audio file %r", path
        )
        return None
