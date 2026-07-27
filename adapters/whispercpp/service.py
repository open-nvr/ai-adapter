# Copyright (c) 2026 OpenNVR
# SPDX-License-Identifier: Apache-2.0

"""
whisper.cpp STT service — ggml Whisper via the in-process ``pywhispercpp``
binding (the native whisper.cpp/ggml engine, no PyTorch, no CTranslate2).

Input:  audio bytes at payload["__file__"] (16 kHz mono 16-bit WAV) +
        optional {"language","task"}.
Output: InferResponse result = AsrResult {transcript, language, segments}.
"""
from __future__ import annotations

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

import numpy as np

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

MODEL_PATH: str = os.getenv("OPENNVR_STT_MODEL_PATH", "/app/models/ggml-base.en.bin")
# Downloaded to MODEL_PATH on first boot when the file is absent. Set to an
# empty string to forbid any fetch (offline / sovereignty-strict installs).
MODEL_URL: str = os.getenv(
    "OPENNVR_STT_MODEL_URL",
    "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en.bin",
)
STT_THREADS: int = int(os.getenv("WHISPERCPP_THREADS", str(os.cpu_count() or 4)))
STT_LANGUAGE: str = os.getenv("WHISPERCPP_LANGUAGE", "en")
MAX_AUDIO_BYTES: int = int(os.getenv("WHISPERCPP_MAX_AUDIO_BYTES", str(25 * 1024 * 1024)))
TARGET_RATE: int = 16000


class WhisperCppService(AdapterService):
    def __init__(self, model_path: str | None = None) -> None:
        self._model_path = model_path or MODEL_PATH
        self._model: Any | None = None
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
                from pywhispercpp.model import Model

                ensure_model_file(
                    self._model_path, MODEL_URL, label="whisper ggml model",
                    logger=logger,
                )
                self._fingerprint_cache = _hash_file(self._model_path)
                self._model = Model(model=self._model_path, n_threads=STT_THREADS)
                self._load_state = HealthStatus.OK
                self._load_error = None
                logger.info("WhisperCppService ready (%s)", self._model_path)
            except Exception as exc:
                self._load_state = HealthStatus.ERROR
                self._load_error = str(exc)
                logger.exception("WhisperCppService failed to load")

    def is_ready(self) -> bool:
        return self._load_state == HealthStatus.OK

    def fingerprint(self) -> str | None:
        return self._fingerprint_cache

    def model_info(self) -> ModelInfo:
        return ModelInfo(
            name=os.path.basename(self._model_path),
            version="ggml",
            framework="whisper.cpp",
            modalities_in=["audio"],
            modalities_out=["text"],
            fingerprint=self.fingerprint(),
        )

    def hardware_evaluation(self) -> HardwareEvaluationResponse:
        if self._load_state == HealthStatus.OK:
            verdict, reasoning = HardwareVerdict.OK, "whisper.cpp ggml model loaded on CPU."
        elif self._load_state == HealthStatus.LOADING:
            verdict, reasoning = HardwareVerdict.WARN, "Model still loading."
        else:
            verdict, reasoning = HardwareVerdict.BLOCKED, f"Load failed: {self._load_error}"
        uptime = int((datetime.now(timezone.utc) - self._started_at).total_seconds())
        return HardwareEvaluationResponse(
            verdict=verdict, reasoning=reasoning,
            checked_at=datetime.now(timezone.utc),
            details={
                "gpu_required": False, "runtime": "whisper.cpp",
                "model_path": self._model_path, "threads": STT_THREADS,
                "cpu_count": os.cpu_count() or 0, "platform": platform.platform(),
                "adapter_uptime_seconds": uptime,
            },
        )

    def infer(self, payload: dict[str, Any]) -> InferResponse:
        if not self.is_ready():
            transient = self._load_state == HealthStatus.LOADING
            raise ServiceError(
                ErrorCategory.MODEL_ERROR, code="model_not_loaded",
                message=self._load_error or "STT not yet loaded",
                transient=transient, http_status=503,
                retry_after_ms=2000 if transient else None,
            )
        audio_bytes = payload.get("__file__")
        if not audio_bytes:
            raise ServiceError(
                ErrorCategory.TRANSPORT_ERROR, code="missing_audio",
                message="no audio supplied (multipart 'audio' or JSON 'audio_b64')",
                transient=False, http_status=400,
            )
        try:
            pcm = _wav_to_float32_16k(audio_bytes)
        except Exception as exc:
            raise ServiceError(
                ErrorCategory.TRANSPORT_ERROR, code="malformed_input",
                message=f"could not decode audio as 16-bit WAV: {exc}",
                transient=False, http_status=400,
            ) from exc

        language = str(payload.get("language") or STT_LANGUAGE)
        started = time.monotonic()
        try:
            segments = self._model.transcribe(pcm, language=language)  # type: ignore[union-attr]
        except Exception as exc:
            logger.exception("whisper.cpp transcription failed")
            raise ServiceError(
                ErrorCategory.MODEL_ERROR, code="inference_failed",
                message=f"transcription raised: {exc}",
                transient=True, http_status=500, retry_after_ms=1000,
            ) from exc
        elapsed_ms = int((time.monotonic() - started) * 1000)

        seg_out = []
        parts = []
        for s in segments:
            text = (getattr(s, "text", "") or "").strip()
            if not text:
                continue
            parts.append(text)
            seg_out.append({
                "start_ms": int(getattr(s, "t0", 0) * 10),   # centiseconds -> ms
                "end_ms": int(getattr(s, "t1", 0) * 10),
                "text": text,
            })
        transcript = " ".join(parts).strip()
        return InferResponse(
            model_name=os.path.basename(self._model_path),
            model_version="ggml",
            inference_ms=elapsed_ms,
            result={
                "transcript": transcript,
                "language": language,
                "segments": seg_out,
                "duration_seconds": round(len(pcm) / TARGET_RATE, 3),
            },
        )


def _wav_to_float32_16k(audio_bytes: bytes) -> np.ndarray:
    with wave.open(io.BytesIO(audio_bytes), "rb") as wf:
        n_ch = wf.getnchannels()
        rate = wf.getframerate()
        width = wf.getsampwidth()
        frames = wf.readframes(wf.getnframes())
    if width != 2:
        raise ValueError(f"expected 16-bit PCM, got {width * 8}-bit")
    audio = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    if n_ch > 1:
        audio = audio.reshape(-1, n_ch).mean(axis=1)
    if rate != TARGET_RATE and len(audio):
        n_out = int(round(len(audio) * TARGET_RATE / rate))
        audio = np.interp(
            np.linspace(0, len(audio) - 1, n_out), np.arange(len(audio)), audio
        ).astype(np.float32)
    return np.ascontiguousarray(audio, dtype=np.float32)


def _hash_file(path: str) -> str:
    size = os.path.getsize(path)
    h = hashlib.sha256()
    h.update(str(size).encode())
    with open(path, "rb") as f:
        h.update(f.read(4 * 1024 * 1024))
    return f"sha256:{h.hexdigest()[:16]}"
