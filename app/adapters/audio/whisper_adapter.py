# Copyright (c) 2026 OpenNVR
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Whisper speech-to-text adapter using ``faster-whisper`` (CTranslate2).

Supports two tasks:
  * ``audio_transcription`` — transcribe speech in the source language.
  * ``audio_translation``   — transcribe and translate to English.

Both map to the same adapter; the ``task`` key on the input payload selects
the Whisper "task" parameter ("transcribe" or "translate").

Input shape:
    {
        "task": "audio_transcription" | "audio_translation",
        "audio": {"uri": "opennvr://audio/<path>"},
        "language": "en"      # optional ISO-639-1 code; auto-detect if omitted
        "beam_size": 5,       # optional decoder beam width
        "vad_filter": true,   # optional Silero VAD to strip silence/noise
    }

Output shape: see :class:`AudioTranscriptionResponse`.
"""
import logging
import os
import time
from typing import Any, Dict, List, Optional

from app.adapters.base import BaseAdapter
from app.config import MODEL_WEIGHTS_DIR
from app.utils.audio_utils import resolve_audio_uri

logger = logging.getLogger(__name__)

_SUPPORTED_TASKS = {"audio_transcription", "audio_translation"}
_WHISPER_MODE_BY_TASK = {
    "audio_transcription": "transcribe",
    "audio_translation": "translate",
}


class WhisperAdapter(BaseAdapter):
    name = "whisper_adapter"
    type = "audio"

    SUPPORTED_TASKS = sorted(_SUPPORTED_TASKS)

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self._model_size = self.config.get("model_size", "base")
        self._requested_device = self.config.get("device", "auto")
        self._requested_compute_type = self.config.get("compute_type", "auto")
        self._device: Optional[str] = None
        self._compute_type: Optional[str] = None
        self._download_root = os.path.join(MODEL_WEIGHTS_DIR, "whisper")

    def _resolve_device(self) -> str:
        if self._requested_device != "auto":
            return self._requested_device
        try:
            import torch  # faster-whisper vendors ctranslate2; torch is optional
            if torch.cuda.is_available():
                return "cuda"
        except Exception:
            pass
        return "cpu"

    def _resolve_compute_type(self, device: str) -> str:
        if self._requested_compute_type != "auto":
            return self._requested_compute_type
        return "float16" if device == "cuda" else "int8"

    def load_model(self) -> None:
        from faster_whisper import WhisperModel  # optional dep: uv sync --extra stt

        self._device = self._resolve_device()
        self._compute_type = self._resolve_compute_type(self._device)

        os.makedirs(self._download_root, exist_ok=True)

        logger.info(
            "Loading Whisper model size=%s device=%s compute_type=%s",
            self._model_size,
            self._device,
            self._compute_type,
        )
        self.model = WhisperModel(
            self._model_size,
            device=self._device,
            compute_type=self._compute_type,
            download_root=self._download_root,
        )
        logger.info("Whisper model loaded")

    def infer_local(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        task = input_data.get("task", "audio_transcription")
        if task not in _SUPPORTED_TASKS:
            raise ValueError(
                f"WhisperAdapter supports {sorted(_SUPPORTED_TASKS)}, got: {task}"
            )

        audio_block = input_data.get("audio") or {}
        uri = audio_block.get("uri") if isinstance(audio_block, dict) else None
        if not uri:
            raise ValueError("WhisperAdapter requires 'audio.uri' in input_data")

        audio_path = resolve_audio_uri(uri)

        whisper_task = _WHISPER_MODE_BY_TASK[task]
        language = input_data.get("language")
        beam_size = int(input_data.get("beam_size", 5))
        vad_filter = bool(input_data.get("vad_filter", False))

        start_time = time.time()
        # faster-whisper returns (segments_generator, info); iterating segments
        # is what actually runs inference — it is a lazy generator.
        segments_iter, info = self.model.transcribe(
            str(audio_path),
            task=whisper_task,
            language=language,
            beam_size=beam_size,
            vad_filter=vad_filter,
        )

        segments: List[Dict[str, Any]] = []
        text_parts: List[str] = []
        for seg in segments_iter:
            seg_text = (seg.text or "").strip()
            text_parts.append(seg_text)
            segments.append(
                {
                    "start": float(seg.start),
                    "end": float(seg.end),
                    "text": seg_text,
                    "avg_logprob": float(seg.avg_logprob) if seg.avg_logprob is not None else None,
                    "no_speech_prob": float(seg.no_speech_prob) if seg.no_speech_prob is not None else None,
                }
            )

        full_text = " ".join(part for part in text_parts if part).strip()

        return {
            "task": task,
            "text": full_text,
            "segments": segments,
            "language": "en" if whisper_task == "translate" else info.language,
            "language_confidence": float(info.language_probability)
            if info.language_probability is not None
            else None,
            "duration_seconds": float(info.duration),
            "model": self._model_size,
            "translated_to_english": whisper_task == "translate",
            "executed_at": int(time.time() * 1000),
            "latency_ms": int((time.time() - start_time) * 1000),
        }

    @property
    def schema(self) -> Dict[str, Any]:
        return {
            "tasks": sorted(_SUPPORTED_TASKS),
            "description": (
                "Whisper speech-to-text. 'audio_transcription' preserves the "
                "source language; 'audio_translation' emits English."
            ),
            "input_fields": {
                "audio.uri": {"type": "string", "description": "opennvr://audio/<path>"},
                "language": {"type": "string", "description": "ISO-639-1 code; auto-detected if omitted"},
                "beam_size": {"type": "integer", "description": "Decoder beam width (default 5)"},
                "vad_filter": {"type": "boolean", "description": "Silero VAD to skip silence (default false)"},
            },
            "response_fields": {
                "task": {"type": "string"},
                "text": {"type": "string"},
                "segments": {"type": "array"},
                "language": {"type": "string"},
                "language_confidence": {"type": "number"},
                "duration_seconds": {"type": "number"},
                "model": {"type": "string"},
                "translated_to_english": {"type": "boolean"},
                "executed_at": {"type": "integer"},
                "latency_ms": {"type": "integer"},
            },
        }

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "type": self.type,
            "model": f"whisper-{self._model_size}",
            "framework": "faster-whisper",
            "tasks": sorted(_SUPPORTED_TASKS),
            "device": self._device or "not_loaded",
            "compute_type": self._compute_type or "not_loaded",
            "model_loaded": self.model is not None,
        }

    def health_check(self) -> Dict[str, Any]:
        return {
            "status": "healthy",
            "type": self.type,
            "model_loaded": self.model is not None,
            "model_info": self.get_model_info(),
        }
