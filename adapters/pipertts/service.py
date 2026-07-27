# Copyright (c) 2026 OpenNVR
# SPDX-License-Identifier: Apache-2.0

"""
Piper TTS service — the ``piper-tts`` package (onnxruntime) in-process. Torch-free.

Input:  {"text": "...", "length_scale"?, "noise_scale"?, "inline": true}
Output: InferResponse result = {audio_b64 (WAV), sample_rate, duration_seconds}
"""
from __future__ import annotations

import base64
import hashlib
import io
import logging
import os
import platform
import threading
import time
import wave
from datetime import datetime, timezone
from typing import Any

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

VOICE_PATH: str = os.getenv("OPENNVR_TTS_VOICE_PATH", "/app/models/en_US-amy-medium.onnx")
# Voice (.onnx) + its config (.onnx.json) download to VOICE_PATH on first boot
# when absent. Set the URL to an empty string to forbid any fetch (offline /
# sovereignty-strict installs).
_VOICE_BASE_URL = (
    "https://huggingface.co/rhasspy/piper-voices/resolve/main/"
    "en/en_US/amy/medium/en_US-amy-medium.onnx"
)
VOICE_URL: str = os.getenv("OPENNVR_TTS_VOICE_URL", _VOICE_BASE_URL)
VOICE_CONFIG_URL: str = os.getenv("OPENNVR_TTS_VOICE_CONFIG_URL",
                                  f"{_VOICE_BASE_URL}.json" if _VOICE_BASE_URL else "")
LENGTH_SCALE: float = float(os.getenv("PIPER_LENGTH_SCALE", "1.0"))
NOISE_SCALE: float = float(os.getenv("PIPER_NOISE_SCALE", "0.667"))
MAX_TEXT_CHARS: int = int(os.getenv("PIPER_MAX_TEXT_CHARS", "2000"))


class PiperTtsService(AdapterService):
    def __init__(self, voice_path: str | None = None) -> None:
        self._voice_path = voice_path or VOICE_PATH
        self._voice: Any | None = None
        self._sample_rate: int = 22050
        self._load_state: HealthStatus = HealthStatus.LOADING
        self._load_error: str | None = None
        self._fingerprint_cache: str | None = None
        self._lock = threading.Lock()
        self._started_at = datetime.now(timezone.utc)

    def load(self) -> None:
        with self._lock:
            if self._load_state == HealthStatus.OK:
                return
            try:
                from piper import PiperVoice

                ensure_model_file(
                    self._voice_path, VOICE_URL, label="Piper voice", logger=logger
                )
                # PiperVoice.load also needs the sibling .onnx.json config.
                # Only auto-fetch it for the default voice — for a custom
                # mounted voice a missing config is PiperVoice's error to
                # report, not ours to paper over with the wrong file.
                config_path = f"{self._voice_path}.json"
                if (
                    VOICE_CONFIG_URL
                    and self._voice_path == VOICE_PATH
                    and not os.path.exists(config_path)
                ):
                    ensure_model_file(
                        config_path, VOICE_CONFIG_URL,
                        label="Piper voice config", logger=logger,
                    )
                self._fingerprint_cache = _hash_file(self._voice_path)
                self._voice = PiperVoice.load(self._voice_path)
                rate = getattr(getattr(self._voice, "config", None), "sample_rate", None)
                self._sample_rate = int(rate) if rate else 22050
                self._load_state = HealthStatus.OK
                self._load_error = None
                logger.info("PiperTtsService ready (%s, %d Hz)", self._voice_path, self._sample_rate)
            except Exception as exc:
                self._load_state = HealthStatus.ERROR
                self._load_error = str(exc)
                logger.exception("PiperTtsService failed to load")

    def is_ready(self) -> bool:
        return self._load_state == HealthStatus.OK

    def fingerprint(self) -> str | None:
        return self._fingerprint_cache

    def model_info(self) -> ModelInfo:
        return ModelInfo(
            name=os.path.basename(self._voice_path),
            version="piper",
            framework="piper-tts",
            modalities_in=["text"],
            modalities_out=["audio"],
            fingerprint=self.fingerprint(),
        )

    def hardware_evaluation(self) -> HardwareEvaluationResponse:
        if self._load_state == HealthStatus.OK:
            verdict, reasoning = HardwareVerdict.OK, "Piper voice loaded on CPU (onnxruntime)."
        elif self._load_state == HealthStatus.LOADING:
            verdict, reasoning = HardwareVerdict.WARN, "Voice still loading."
        else:
            verdict, reasoning = HardwareVerdict.BLOCKED, f"Load failed: {self._load_error}"
        uptime = int((datetime.now(timezone.utc) - self._started_at).total_seconds())
        return HardwareEvaluationResponse(
            verdict=verdict, reasoning=reasoning,
            checked_at=datetime.now(timezone.utc),
            details={
                "gpu_required": False, "runtime": "piper-tts/onnxruntime",
                "voice_path": self._voice_path, "sample_rate": self._sample_rate,
                "cpu_count": os.cpu_count() or 0, "platform": platform.platform(),
                "adapter_uptime_seconds": uptime,
            },
        )

    def infer(self, payload: dict[str, Any]) -> InferResponse:
        if not self.is_ready():
            transient = self._load_state == HealthStatus.LOADING
            raise ServiceError(
                ErrorCategory.MODEL_ERROR, code="model_not_loaded",
                message=self._load_error or "TTS not yet loaded",
                transient=transient, http_status=503,
                retry_after_ms=2000 if transient else None,
            )
        text = str(payload.get("text") or "").strip()
        if not text:
            raise ServiceError(
                ErrorCategory.TRANSPORT_ERROR, code="missing_text",
                message="'text' must be a non-empty string",
                transient=False, http_status=400,
            )
        text = text[:MAX_TEXT_CHARS]
        started = time.monotonic()
        try:
            pcm = self._synthesize(text)
        except Exception as exc:
            logger.exception("Piper synthesis failed")
            raise ServiceError(
                ErrorCategory.MODEL_ERROR, code="inference_failed",
                message=f"synthesis raised: {exc}",
                transient=True, http_status=500, retry_after_ms=1000,
            ) from exc
        elapsed_ms = int((time.monotonic() - started) * 1000)

        wav = _pcm_to_wav(pcm, self._sample_rate)
        duration = round(len(pcm) / 2 / self._sample_rate, 3)  # 2 bytes/sample
        return InferResponse(
            model_name=os.path.basename(self._voice_path),
            model_version="piper",
            inference_ms=elapsed_ms,
            result={
                "audio_b64": base64.b64encode(wav).decode("ascii"),
                "audio_format": "wav",
                "sample_rate": self._sample_rate,
                "duration_seconds": duration,
                "voice": os.path.basename(self._voice_path),
            },
        )

    def _synthesize(self, text: str) -> bytes:
        chunks: list[bytes] = []
        try:  # newer piper: synthesize() -> AudioChunk objects
            for ch in self._voice.synthesize(text):  # type: ignore[union-attr]
                data = getattr(ch, "audio_int16_bytes", None)
                if data is None and hasattr(ch, "audio_int16_array"):
                    data = ch.audio_int16_array.tobytes()
                if data:
                    chunks.append(data)
            if chunks:
                return b"".join(chunks)
        except (TypeError, AttributeError):
            pass
        # older piper: synthesize_stream_raw() -> raw int16 byte chunks
        for raw in self._voice.synthesize_stream_raw(  # type: ignore[union-attr]
            text, length_scale=LENGTH_SCALE, noise_scale=NOISE_SCALE
        ):
            chunks.append(raw)
        return b"".join(chunks)


def _pcm_to_wav(pcm: bytes, sample_rate: int) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm)
    return buf.getvalue()


def _hash_file(path: str) -> str:
    size = os.path.getsize(path)
    h = hashlib.sha256()
    h.update(str(size).encode())
    with open(path, "rb") as f:
        h.update(f.read(4 * 1024 * 1024))
    return f"sha256:{h.hexdigest()[:16]}"
