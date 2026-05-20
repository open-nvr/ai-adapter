# Copyright (c) 2026 OpenNVR
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
WhisperService — Whisper-specific implementation of ``AdapterService``.

Migrated to ``opennvr-adapter-sdk`` in A2.3c. All cross-adapter
boilerplate (auth, metrics, FastAPI routes, request body parsing,
error-envelope translation, lifespan) is now in the SDK; this module
holds only the Whisper-specific pieces:

* Load lifecycle around the legacy ``WhisperAdapter``
* Live ``model.fingerprint`` from the model's ``config.json``
* §3.3 ``HardwareEvaluationResponse`` (CUDA vs CPU verdict)
* §3.5 ``infer(payload)`` — driven by the SDK's AUDIO-shape body
  parser

The legacy ``WhisperAdapter`` (in ``app/adapters/audio/whisper_adapter.py``)
stays untouched — it's the underlying model wrapper, not the contract
shim.

Interesting translations:

* Audio bytes come in via the SDK at ``payload[BODY_BYTES_KEY]`` (set
  by the SDK's AUDIO body parser, which accepts both multipart
  ``audio`` and JSON ``audio_b64``). We write them to a tmp file and
  pass that path to the legacy adapter — ``faster-whisper`` takes a
  path because CTranslate2 decodes audio via libav internally.
* Whisper returns segments with ``start`` / ``end`` in float seconds;
  §5.3 ASR convention uses ``start_ms`` / ``end_ms`` integers in
  milliseconds. The service rounds and translates.
* ``language_confidence`` and ``duration_seconds`` are
  Whisper-specific extras — §5 explicitly allows extra fields in the
  ``result`` body, so they land alongside the §5.3 canonical keys
  without breaking schema-strict consumers.
"""
from __future__ import annotations

import hashlib
import logging
import os
import platform
import shutil
import tempfile
import time
import uuid
from datetime import datetime, timezone
from typing import Any

from app.adapters.audio.whisper_adapter import WhisperAdapter
from app.config import MODEL_WEIGHTS_DIR
from opennvr_adapter_sdk.contract import (
    AsrResult,
    AsrSegment,
    ErrorCategory,
    HardwareEvaluationResponse,
    HardwareVerdict,
    HealthStatus,
    InferResponse,
    ModelInfo,
)
from opennvr_adapter_sdk import (
    AdapterService,
    BODY_BYTES_KEY,
    ServiceError,
)

logger = logging.getLogger(__name__)

MODEL_FRAMEWORK: str = "faster-whisper"

# Default audio body cap. 25 MiB comfortably holds 25 minutes of
# 16-bit 16 kHz mono WAV. Longer clips should be split client-side.
MAX_AUDIO_BYTES: int = 25 * 1024 * 1024

DEFAULT_MODEL_SIZE: str = "base"
DEFAULT_BEAM_SIZE: int = 5

# Contract task name → faster-whisper "task" param. WhisperAdapter
# has the same map internally; duplicated here so the validation
# error message lists the contract-shaped names.
_TASK_TO_WHISPER_MODE: dict[str, str] = {
    "audio_transcription": "transcribe",
    "audio_translation": "translate",
}


class WhisperService(AdapterService):
    """Stateful façade around WhisperAdapter."""

    def __init__(
        self,
        model_size: str | None = None,
        *,
        download_root: str | None = None,
        device: str = "auto",
        compute_type: str = "auto",
    ) -> None:
        self._model_size = model_size or DEFAULT_MODEL_SIZE
        self._download_root = download_root or os.path.join(MODEL_WEIGHTS_DIR, "whisper")
        self._device_setting = device
        self._compute_setting = compute_type

        self._adapter: WhisperAdapter = WhisperAdapter(
            config={
                "enabled": True,
                "model_size": self._model_size,
                "device": device,
                "compute_type": compute_type,
            }
        )

        self._load_state: HealthStatus = HealthStatus.LOADING
        self._load_error: str | None = None
        self._fingerprint_cache: str | None = None

    # ── AdapterService impl ────────────────────────────────────────

    def load(self) -> None:
        """Eagerly load the Whisper model. Idempotent."""
        if self._load_state == HealthStatus.OK:
            return
        try:
            self._adapter.ensure_model_loaded()
            self._fingerprint_cache = self._compute_fingerprint()
            self._load_state = HealthStatus.OK
            self._load_error = None
            logger.info(
                "WhisperService ready: model=%s device=%s compute=%s fingerprint=%s",
                self._model_size,
                self._adapter._device,
                self._adapter._compute_type,
                self._fingerprint_cache,
            )
        except Exception as exc:
            self._load_state = HealthStatus.ERROR
            self._load_error = str(exc)
            logger.exception("WhisperService failed to load model %s", self._model_size)

    def is_ready(self) -> bool:
        return self._load_state == HealthStatus.OK

    def fingerprint(self) -> str | None:
        """Recompute live on each call so §11.3 drift detection sees
        weight rotation. Cached value used as fallback if the on-disk
        config.json becomes unreadable mid-flight."""
        try:
            return self._compute_fingerprint()
        except OSError:
            return self._fingerprint_cache

    def model_info(self) -> ModelInfo:
        return ModelInfo(
            name=f"whisper-{self._model_size}",
            version=self._adapter_model_version(),
            framework=MODEL_FRAMEWORK,
            modalities_in=["audio"],
            modalities_out=["text"],
            fingerprint=self.fingerprint(),
        )

    def hardware_evaluation(self) -> HardwareEvaluationResponse:
        device = self._adapter._device or "unknown"
        compute = self._adapter._compute_type or "unknown"
        if self._load_state == HealthStatus.OK:
            if device == "cuda":
                verdict = HardwareVerdict.OK
                reasoning = f"Whisper loaded on CUDA with compute_type={compute}."
            else:
                verdict = HardwareVerdict.WARN
                reasoning = (
                    f"Whisper loaded on CPU (compute_type={compute}). "
                    f"Expect 5-30x slower than CUDA depending on model size."
                )
        elif self._load_state == HealthStatus.LOADING:
            verdict = HardwareVerdict.WARN
            reasoning = "Model still loading."
        else:
            verdict = HardwareVerdict.BLOCKED
            reasoning = f"Whisper failed to load: {self._load_error}"

        return HardwareEvaluationResponse(
            verdict=verdict,
            reasoning=reasoning,
            checked_at=datetime.now(timezone.utc),
            details={
                "gpu_required": False,
                "device": device,
                "compute_type": compute,
                "model_size": self._model_size,
                "cpu_count": os.cpu_count() or 0,
                "platform": platform.platform(),
                "python_version": platform.python_version(),
                "download_root": self._download_root,
            },
        )

    def infer(self, payload: dict[str, Any]) -> InferResponse:
        """SDK /infer entry point. The audio bytes live at
        ``payload[BODY_BYTES_KEY]`` (set by the SDK's AUDIO-shape body
        parser); the rest of the dict is request params
        (task, language, beam_size, vad_filter)."""
        audio_bytes = payload.get(BODY_BYTES_KEY)
        if not isinstance(audio_bytes, (bytes, bytearray)) or not audio_bytes:
            raise ServiceError(
                ErrorCategory.TRANSPORT_ERROR,
                code="malformed_input",
                message="Audio bytes are required.",
                transient=False,
                http_status=400,
            )
        params = {k: v for k, v in payload.items() if k != BODY_BYTES_KEY}
        return self._infer_audio_bytes(bytes(audio_bytes), params)

    # ── Inference core ─────────────────────────────────────────────

    def _infer_audio_bytes(
        self,
        audio_bytes: bytes,
        params: dict[str, Any],
    ) -> InferResponse:
        """Transcribe (or translate) audio bytes. Returns an
        InferResponse whose ``result`` is shaped to §5.3 ASR
        convention with adapter-specific extras."""
        if self._load_state != HealthStatus.OK:
            raise ServiceError(
                ErrorCategory.MODEL_ERROR,
                code=(
                    "weights_missing"
                    if self._load_state == HealthStatus.ERROR
                    else "whisper.model_loading"
                ),
                message=self._load_error or "Model still loading.",
                transient=(self._load_state == HealthStatus.LOADING),
                http_status=503,
                retry_after_ms=2000 if self._load_state == HealthStatus.LOADING else None,
            )

        # The SDK enforces ``max_body_bytes`` before calling us, but
        # we keep a defense-in-depth check that mirrors the original
        # adapter behavior.
        if len(audio_bytes) > MAX_AUDIO_BYTES:
            raise ServiceError(
                ErrorCategory.TRANSPORT_ERROR,
                code="malformed_input",
                message=(
                    f"Audio exceeds {MAX_AUDIO_BYTES}-byte limit "
                    f"({len(audio_bytes)} received)."
                ),
                transient=False,
                http_status=413,
            )

        task = params.get("task", "audio_transcription")
        if task not in _TASK_TO_WHISPER_MODE:
            raise ServiceError(
                ErrorCategory.TRANSPORT_ERROR,
                code="malformed_input",
                message=(
                    f"Unknown task {task!r}; expected one of "
                    f"{sorted(_TASK_TO_WHISPER_MODE.keys())}."
                ),
                transient=False,
                http_status=400,
            )

        language = params.get("language")  # may be None for auto-detect
        try:
            beam_size = int(params.get("beam_size", DEFAULT_BEAM_SIZE))
        except (TypeError, ValueError) as exc:
            raise ServiceError(
                ErrorCategory.TRANSPORT_ERROR,
                code="malformed_input",
                message="beam_size must be an integer.",
                transient=False,
                http_status=400,
            ) from exc
        if not 1 <= beam_size <= 32:
            raise ServiceError(
                ErrorCategory.TRANSPORT_ERROR,
                code="malformed_input",
                message="beam_size must be between 1 and 32.",
                transient=False,
                http_status=400,
            )
        vad_filter = bool(params.get("vad_filter", False))

        # Write bytes to a tmp file — faster-whisper takes a path.
        # Cleanup is unconditional so a model crash doesn't leak.
        tmp_dir = tempfile.mkdtemp(prefix="whisper-svc-")
        tmp_path = os.path.join(tmp_dir, f"audio-{uuid.uuid4().hex}.bin")
        try:
            with open(tmp_path, "wb") as fh:
                fh.write(audio_bytes)

            start = time.monotonic()
            try:
                raw = self._call_adapter(task, tmp_path, language, beam_size, vad_filter)
            except (FileNotFoundError, ValueError) as exc:
                raise ServiceError(
                    ErrorCategory.TRANSPORT_ERROR,
                    code="malformed_input",
                    message=str(exc),
                    transient=False,
                    http_status=400,
                ) from exc
            except Exception as exc:
                logger.exception("Whisper inference raised unexpectedly")
                raise ServiceError(
                    ErrorCategory.MODEL_ERROR,
                    code="inference_runtime_crash",
                    message="Inference failed.",
                    transient=False,
                    http_status=500,
                ) from exc
            inference_ms = int((time.monotonic() - start) * 1000)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

        result_body = self._shape_asr_result(raw)
        return InferResponse(
            model_name=f"whisper-{self._model_size}",
            model_version=self._adapter_model_version(),
            inference_ms=inference_ms,
            result=result_body,
        )

    def _call_adapter(
        self,
        task: str,
        audio_path: str,
        language: str | None,
        beam_size: int,
        vad_filter: bool,
    ) -> dict[str, Any]:
        """Invoke the legacy WhisperAdapter via its loaded model.

        We bypass ``infer_local`` because that expects an
        ``opennvr://audio/...`` URI; we already have an absolute
        filesystem path. Reach down to ``model.transcribe()``
        directly — same pattern YoloV8Service uses for its
        in-memory bytes path.
        """
        whisper_mode = _TASK_TO_WHISPER_MODE[task]
        segments_iter, info = self._adapter.model.transcribe(
            audio_path,
            task=whisper_mode,
            language=language,
            beam_size=beam_size,
            vad_filter=vad_filter,
        )
        segments: list[dict[str, Any]] = []
        for seg in segments_iter:
            segments.append({
                "start": float(seg.start),
                "end": float(seg.end),
                "text": (seg.text or "").strip(),
                "avg_logprob": (
                    float(seg.avg_logprob) if seg.avg_logprob is not None else None
                ),
                "no_speech_prob": (
                    float(seg.no_speech_prob) if seg.no_speech_prob is not None else None
                ),
            })
        return {
            "task": task,
            "whisper_mode": whisper_mode,
            "segments": segments,
            "info": {
                "language": info.language,
                "language_probability": (
                    float(info.language_probability)
                    if info.language_probability is not None
                    else None
                ),
                "duration": float(info.duration),
            },
        }

    def _shape_asr_result(self, raw: dict[str, Any]) -> dict[str, Any]:
        """Translate the legacy float-seconds shape into §5.3
        ``AsrResult`` (transcript + language + segments with
        ``start_ms`` / ``end_ms`` integers).

        Adapter-specific extras (``language_confidence``,
        ``duration_seconds``, ``translated_to_english``, ``model``)
        ride alongside the canonical keys — §5 allows extras in the
        ``result`` body.

        Note: segments with empty text (after stripping) are dropped.
        §5.3 says segments are transcribed *speech*; whitespace-only
        segments are silence markers and not part of the transcript.
        This is a deliberate divergence from the legacy
        ``WhisperAdapter`` (which kept all segments) — the legacy
        shape wasn't contract-compliant, so the translation layer
        tightens the semantics.
        """
        contract_segments: list[AsrSegment] = []
        for seg in raw["segments"]:
            text = seg["text"]
            if not text:
                continue
            contract_segments.append(
                AsrSegment(
                    start_ms=int(round(seg["start"] * 1000)),
                    end_ms=int(round(seg["end"] * 1000)),
                    text=text,
                )
            )
        # Stitch the full transcript from segment texts (preserves
        # spacing across silence regions better than the legacy
        # behaviour of " ".join).
        transcript = " ".join(s.text for s in contract_segments).strip()

        info = raw["info"]
        whisper_mode = raw["whisper_mode"]
        # When task=translate, language is always English regardless
        # of source — mirror the legacy adapter's behaviour.
        language = (
            "en" if whisper_mode == "translate"
            else (info["language"] or "unknown")
        )

        asr = AsrResult(
            transcript=transcript,
            language=language,
            segments=contract_segments,
        )
        body = asr.model_dump(mode="json")
        body["language_confidence"] = info["language_probability"]
        body["duration_seconds"] = info["duration"]
        body["translated_to_english"] = (whisper_mode == "translate")
        body["model"] = f"whisper-{self._model_size}"
        return body

    # ── Helpers ────────────────────────────────────────────────────

    def _adapter_model_version(self) -> str:
        return f"{MODEL_FRAMEWORK}/{self._model_size}"

    def _compute_fingerprint(self) -> str:
        """sha256 of the model's ``config.json`` (small, stable file
        that uniquely identifies the model variant). Falls back to a
        hash of ``model_size::compute_type`` when the config file
        isn't on disk yet (e.g., before the first download
        completes).

        faster-whisper uses the HuggingFace cache layout
        (``models--<org>--<repo>/snapshots/<hash>/``) which is awkward
        to pin a path for — the snapshot hash changes per upload. We
        walk ``download_root`` for the first ``config.json`` we find;
        good enough since each model_size lives under a separate
        subtree and only one model is loaded per service.
        """
        for root, _dirs, files in os.walk(self._download_root):
            if "config.json" in files:
                candidate = os.path.join(root, "config.json")
                try:
                    digest = hashlib.sha256()
                    with open(candidate, "rb") as fh:
                        digest.update(fh.read())
                    return f"sha256:{digest.hexdigest()}"
                except OSError:
                    continue
        material = (
            f"{self._model_size}::{self._device_setting}::{self._compute_setting}"
        )
        return f"sha256:{hashlib.sha256(material.encode()).hexdigest()}"
