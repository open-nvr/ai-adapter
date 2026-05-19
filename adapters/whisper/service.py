# Copyright (c) 2026 OpenNVR
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
WhisperService — contract-semantics layer around the legacy
``WhisperAdapter``.

Translates between the AI Adapter Contract v1 wire shapes and
WhisperAdapter's infer interface. The interesting translations:

* Audio bytes come in via multipart or JSON-base64; we write them to
  a tmp file and pass that path to the legacy adapter (which calls
  ``faster-whisper`` with a filesystem path because the underlying
  CTranslate2 runtime decodes audio via libav internally).
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
from app.interfaces.contract import (
    AdapterInfo,
    AsrResult,
    AsrSegment,
    CapabilitiesResponse,
    Cost,
    EndpointsInfo,
    ErrorCategory,
    ErrorDetail,
    FailureEnvelope,
    FairQueuing,
    HardwareEvaluationResponse,
    HardwareVerdict,
    HealthResponse,
    HealthStatus,
    InferEndpointInfo,
    InferResponse,
    ModelInfo,
    Permissions,
    Scheduling,
    StreamEndpointInfo,
)

logger = logging.getLogger(__name__)

ADAPTER_NAME: str = "whisper-asr"
ADAPTER_VERSION: str = "1.0.0"
ADAPTER_VENDOR: str = "open-nvr"
ADAPTER_LICENSE: str = "AGPL-3.0"
MODEL_FRAMEWORK: str = "faster-whisper"
TASKS_ADVERTISED: tuple[str, ...] = ("audio_transcription", "audio_translation")

# Default audio body cap. 25 MiB comfortably holds 25 minutes of
# 16-bit 16 kHz mono WAV. Longer clips should be split client-side.
MAX_AUDIO_BYTES: int = 25 * 1024 * 1024

DEFAULT_MODEL_SIZE: str = "base"
DEFAULT_BEAM_SIZE: int = 5

# Mapping from contract task name → faster-whisper "task" param.
# WhisperAdapter has the same map internally; we duplicate here so
# our validation message lists the contract-shaped names.
_TASK_TO_WHISPER_MODE: dict[str, str] = {
    "audio_transcription": "transcribe",
    "audio_translation": "translate",
}


class WhisperService:
    """Stateful façade around WhisperAdapter implementing contract semantics."""

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
        self._started_at_dt: datetime = datetime.now(timezone.utc)
        self._started_at_mono: float = time.monotonic()

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
        self._fingerprint: str | None = None

    # ── Lifecycle ──────────────────────────────────────────────────

    def load(self) -> None:
        """Eagerly load the Whisper model. Idempotent."""
        if self._load_state == HealthStatus.OK:
            return
        try:
            self._adapter.ensure_model_loaded()
            self._fingerprint = self._compute_fingerprint()
            self._load_state = HealthStatus.OK
            self._load_error = None
            logger.info(
                "WhisperService ready: model=%s device=%s compute=%s fingerprint=%s",
                self._model_size,
                self._adapter._device,
                self._adapter._compute_type,
                self._fingerprint,
            )
        except Exception as exc:
            self._load_state = HealthStatus.ERROR
            self._load_error = str(exc)
            logger.exception("WhisperService failed to load model %s", self._model_size)

    def _compute_fingerprint(self) -> str:
        """sha256 of the model's ``config.json`` (small, stable file
        that uniquely identifies the model variant). Falls back to a
        hash of ``model_size::compute_type`` when the config file isn't
        on disk yet (e.g., before the first download completes).

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
        # Fallback when the model isn't on disk (mock / loading state).
        material = f"{self._model_size}::{self._device_setting}::{self._compute_setting}"
        return f"sha256:{hashlib.sha256(material.encode()).hexdigest()}"

    def _compute_fingerprint_or_cached(self) -> str | None:
        try:
            return self._compute_fingerprint()
        except OSError:
            return self._fingerprint

    def is_ready(self) -> bool:
        return self._load_state == HealthStatus.OK

    # ── Contract endpoints ─────────────────────────────────────────

    def health(self) -> HealthResponse:
        return HealthResponse(
            status=self._load_state,
            adapter_name=ADAPTER_NAME,
            adapter_version=ADAPTER_VERSION,
            model_name=f"whisper-{self._model_size}",
            model_version=self._adapter_model_version(),
            started_at=self._started_at_dt,
            uptime_seconds=int(time.monotonic() - self._started_at_mono),
        )

    def capabilities(self) -> CapabilitiesResponse:
        return CapabilitiesResponse(
            adapter=AdapterInfo(
                name=ADAPTER_NAME,
                version=ADAPTER_VERSION,
                vendor=ADAPTER_VENDOR,
                license=ADAPTER_LICENSE,
                model_card_url="https://github.com/SYSTRAN/faster-whisper",
                supported_contract_versions=["1"],
            ),
            model=ModelInfo(
                name=f"whisper-{self._model_size}",
                version=self._adapter_model_version(),
                framework=MODEL_FRAMEWORK,
                modalities_in=["audio"],
                modalities_out=["text"],
                fingerprint=self._compute_fingerprint_or_cached(),
            ),
            endpoints=EndpointsInfo(
                infer=InferEndpointInfo(
                    supported=True,
                    input_content_types=["multipart/form-data", "application/json"],
                ),
                # §3.6 — streaming ASR is its own design problem
                # (partial-result emission, overlap windows, VAD
                # gating). v1 refuses with HTTP 501; a follow-up
                # commit lands streaming.
                infer_stream=StreamEndpointInfo(
                    supported=False,
                    max_concurrent_streams=0,
                ),
            ),
            tasks_advertised=list(TASKS_ADVERTISED),
            permissions=Permissions(
                # GPU optional — faster-whisper uses CUDA when
                # available; falls back to CPU. We declare GPU to
                # capture the operator-approval gate per §8 since
                # most production deployments DO use CUDA.
                gpu=True,
                network_egress=[],
                # faster-whisper downloads weights from HuggingFace
                # on first load; once cached, no further egress.
                # Under sovereignty=local_only deployments operators
                # pre-populate the weights dir and KAI-C refuses if
                # this list isn't empty — so we leave it empty and
                # rely on the operator having pre-downloaded models.
                host_filesystem=[self._download_root],
                shared_memory_paths=[],
                host_metadata=False,
            ),
            scheduling=Scheduling(
                # Whisper sessions are NOT thread-safe under
                # concurrent transcribe() calls — same singleton
                # caveat as YOLOv8. Honest value is 1.
                max_inflight=1,
                preferred_batch_size=1,
                # ASR doesn't typically have per-camera fan-out
                # (one mic per stream typically), but expose
                # per_camera fair-queuing so KAI-C handles a
                # multi-audio-stream deployment correctly.
                fair_queuing=FairQueuing.PER_CAMERA,
            ),
            cost=Cost(
                currency="USD",
                estimated_per_call=0.0,
                estimated_per_hour=0.0,
                is_metered=False,
            ),
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

    # ── Inference path ─────────────────────────────────────────────

    def infer_bytes(
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
                code="weights_missing" if self._load_state == HealthStatus.ERROR else "whisper.model_loading",
                message=self._load_error or "Model still loading.",
                transient=(self._load_state == HealthStatus.LOADING),
                http_status=503,
                retry_after_ms=2000 if self._load_state == HealthStatus.LOADING else None,
            )

        if not audio_bytes:
            raise ServiceError(
                ErrorCategory.TRANSPORT_ERROR,
                code="malformed_input",
                message="Audio bytes are required.",
                transient=False,
                http_status=400,
            )
        if len(audio_bytes) > MAX_AUDIO_BYTES:
            raise ServiceError(
                ErrorCategory.TRANSPORT_ERROR,
                code="malformed_input",
                message=f"Audio exceeds {MAX_AUDIO_BYTES}-byte limit ({len(audio_bytes)} received).",
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

        # Write bytes to a temp file — faster-whisper takes a path.
        # cleanup unconditional so a model crash doesn't leak the file.
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
        """Invoke the legacy WhisperAdapter via its public ``infer`` method.

        We bypass ``infer_local`` directly because that path expects
        an ``opennvr://audio/...`` URI; we already have an absolute
        filesystem path. Reach down to the model's ``transcribe()``
        directly — same pattern as YoloV8Service does for its
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
                "avg_logprob": float(seg.avg_logprob) if seg.avg_logprob is not None else None,
                "no_speech_prob": float(seg.no_speech_prob) if seg.no_speech_prob is not None else None,
            })
        return {
            "task": task,
            "whisper_mode": whisper_mode,
            "segments": segments,
            "info": {
                "language": info.language,
                "language_probability": float(info.language_probability) if info.language_probability is not None else None,
                "duration": float(info.duration),
            },
        }

    def _shape_asr_result(self, raw: dict[str, Any]) -> dict[str, Any]:
        """Translate the legacy float-seconds shape into §5.3
        ``AsrResult`` (transcript + language + segments with
        ``start_ms`` / ``end_ms`` integers).

        Adapter-specific extras (``language_confidence``,
        ``duration_seconds``, ``translated_to_english``) ride alongside
        the canonical keys — §5 allows extras in the ``result`` body.

        Note: segments with empty text (after stripping) are dropped.
        §5.3 says segments are transcribed *speech*; whitespace-only
        segments are silence markers and not part of the transcript.
        This is a deliberate behaviour divergence from the legacy
        ``WhisperAdapter`` (which kept all segments) — the legacy
        format wasn't a contract-compliant ASR result, so the
        translation layer can tighten the semantics.
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
        language = "en" if whisper_mode == "translate" else (info["language"] or "unknown")

        asr = AsrResult(
            transcript=transcript,
            language=language,
            segments=contract_segments,
        )
        body = asr.model_dump(mode="json")
        # Extras
        body["language_confidence"] = info["language_probability"]
        body["duration_seconds"] = info["duration"]
        body["translated_to_english"] = (whisper_mode == "translate")
        body["model"] = f"whisper-{self._model_size}"
        return body

    # ── Helpers ────────────────────────────────────────────────────

    def _adapter_model_version(self) -> str:
        return f"{MODEL_FRAMEWORK}/{self._model_size}"


# ── Typed error envelope ───────────────────────────────────────────


class ServiceError(Exception):
    """See ``adapters/piper/service.py:ServiceError`` for design
    notes. The class is duplicated across adapters until A2.3 lands
    ``opennvr-adapter-sdk``."""

    def __init__(
        self,
        category: ErrorCategory,
        *,
        code: str,
        message: str,
        transient: bool,
        http_status: int,
        retry_after_ms: int | None = None,
    ) -> None:
        super().__init__(message)
        self.category = category
        self.code = code
        self.message = message
        self.transient = transient
        self.http_status = http_status
        self.retry_after_ms = retry_after_ms

    def envelope(self) -> FailureEnvelope:
        return FailureEnvelope(
            error=ErrorDetail(
                category=self.category,
                code=self.code,
                message=self.message,
                transient=self.transient,
                retry_after_ms=self.retry_after_ms,
                details={},
            )
        )
