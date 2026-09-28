# Copyright (c) 2026 OpenNVR
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Piper TTS adapter — fast neural text-to-speech via ONNX.

Piper (https://github.com/rhasspy/piper) ships as small ONNX voices (~25 MB
each) and runs on CPU fast enough for real-time phone audio. Because we
already depend on ``onnxruntime`` for vision, Piper adds only a thin Python
wrapper as a dependency.

A "voice" is two files that live together:
    <voice_name>.onnx            ← the model
    <voice_name>.onnx.json       ← phoneme / prosody config

Default layout (configurable via ``voice_dir``):
    <MODEL_WEIGHTS_DIR>/piper/
        en_US-libritts-high.onnx
        en_US-libritts-high.onnx.json

Input shape:
    {
        "task": "speech_synthesis",
        "text": "Hello, this is an automated security alert.",
        "voice": "en_US-libritts-high",   # optional override of adapter default
        "length_scale": 1.0,              # optional; >1 = slower speech
        "noise_scale": 0.667,             # optional; voice variability
    }

Output:
    Writes a WAV under ``opennvr://audio/tts/<uuid>.wav`` and returns the URI.
    Downstream tasks (or an outbound phone bridge) consume that URI.
"""
import logging
import os
import time
import wave
from typing import Any, Dict, Optional

from app.adapters.base import BaseAdapter
from app.config import MODEL_WEIGHTS_DIR
from app.utils.audio_utils import mint_audio_path

logger = logging.getLogger(__name__)

_SUPPORTED_TASKS = {"speech_synthesis"}
_MAX_TEXT_CHARS = 10_000  # basic guardrail against pathological prompts


class PiperAdapter(BaseAdapter):
    name = "piper_adapter"
    type = "audio"

    SUPPORTED_TASKS = sorted(_SUPPORTED_TASKS)

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self._voice_dir = self.config.get("voice_dir") or os.path.join(MODEL_WEIGHTS_DIR, "piper")
        self._default_voice = self.config.get("voice", "en_US-libritts-high")
        self._output_subdir = self.config.get("output_subdir", "tts")
        # ONNX Runtime intra-op threads per synthesis. Unset, ORT fans one
        # utterance across every core — measured at ~390% of a 4-core box
        # — and fights the detector and recorder the box exists for. Two
        # threads keeps a medium voice well under real time. 0/None = ORT's
        # own default (all cores).
        self._threads = self._int_or_none(self.config.get("threads"))
        self._voice_cache: Dict[str, Any] = {}
        self._session_fallback_logged = False
        self._missing_logged: set = set()

    def _fallback_voice(self, missing: str) -> Optional[str]:
        """The default voice when it is on disk, else the first voice that
        is; None when the directory holds nothing to speak with."""
        default_onnx, _ = self._voice_paths(self._default_voice)
        if self._default_voice != missing and os.path.exists(default_onnx):
            return self._default_voice
        try:
            names = sorted(e[: -len(".onnx")] for e in os.listdir(self._voice_dir)
                           if e.endswith(".onnx"))
        except OSError:
            return None
        names = [n for n in names if n != missing]
        return names[0] if names else None

    @staticmethod
    def _int_or_none(value: Any) -> Optional[int]:
        try:
            n = int(value) if value is not None and str(value).strip() != "" else 0
        except (TypeError, ValueError):
            return None
        return n if n > 0 else None

    def _load_voice(self, onnx_path: str, config_path: str):
        """A ``PiperVoice`` whose ONNX session honours ``threads``.

        ``PiperVoice.load`` builds its session with default options, which
        means every core. piper-tts ≥ 1.3 exposes the constructor
        ``PiperVoice(session=..., config=...)`` and ``PiperConfig.from_dict``,
        so the session can be ours. An older piper falls back to ``load``
        — with a warning, because then the cap is not in effect.
        """
        from piper.voice import PiperVoice  # optional dep: uv sync --extra tts

        if self._threads is None:
            return PiperVoice.load(onnx_path, config_path=config_path)
        try:
            import json

            import onnxruntime as ort
            from piper.config import PiperConfig

            opts = ort.SessionOptions()
            opts.intra_op_num_threads = int(self._threads)
            opts.inter_op_num_threads = 1
            session = ort.InferenceSession(
                onnx_path, sess_options=opts, providers=["CPUExecutionProvider"])
            with open(config_path, "r", encoding="utf-8") as fh:
                config = PiperConfig.from_dict(json.load(fh))
            return PiperVoice(session=session, config=config)
        except Exception as exc:  # noqa: BLE001 — an old piper, not a broken voice
            if not self._session_fallback_logged:
                logger.warning(
                    "Piper: cannot build a thread-capped session (%s: %s); loading "
                    "with piper's defaults — the threads=%s cap is NOT in effect",
                    type(exc).__name__, exc, self._threads)
                self._session_fallback_logged = True
            return PiperVoice.load(onnx_path, config_path=config_path)

    def load_model(self) -> None:
        # We don't pre-load any specific voice here — voices are tiny and loaded
        # on demand into the cache. What we DO verify is that the default voice
        # exists on disk, so misconfiguration surfaces at warmup rather than
        # mid-inference.
        onnx_path, config_path = self._voice_paths(self._default_voice)
        if not os.path.exists(onnx_path):
            raise FileNotFoundError(
                f"Piper voice '{self._default_voice}' not found at {onnx_path}. "
                "Download voices from https://github.com/rhasspy/piper/blob/master/VOICES.md "
                f"into {self._voice_dir}"
            )

        self._voice_cache[self._default_voice] = self._load_voice(onnx_path, config_path)
        self.model = self._voice_cache
        logger.info("Piper adapter loaded default voice '%s' from %s (threads=%s)",
                    self._default_voice, onnx_path, self._threads or "all")

    def _voice_paths(self, voice_name: str) -> tuple[str, str]:
        if not voice_name or "/" in voice_name or "\\" in voice_name or ".." in voice_name:
            raise ValueError(f"Invalid voice name: {voice_name!r}")
        onnx_path = os.path.join(self._voice_dir, f"{voice_name}.onnx")
        config_path = os.path.join(self._voice_dir, f"{voice_name}.onnx.json")
        return onnx_path, config_path

    def _get_voice(self, voice_name: str):
        if voice_name in self._voice_cache:
            return self._voice_cache[voice_name]

        onnx_path, config_path = self._voice_paths(voice_name)
        if not os.path.exists(onnx_path):
            # A requested voice that is not on disk falls back to what IS
            # — the default voice, else any voice present — rather than
            # muting the box. The case: an upgraded, offline deployment
            # whose agent now names a newer default voice the init could
            # not download, while the old one sits right there. Said
            # once per missing name, not per sentence.
            fallback = self._fallback_voice(voice_name)
            if fallback is None:
                raise FileNotFoundError(f"Piper voice '{voice_name}' not found at {onnx_path}")
            if voice_name not in self._missing_logged:
                self._missing_logged.add(voice_name)
                logger.warning(
                    "Piper voice '%s' not found at %s; speaking with '%s' instead "
                    "(place the requested voice's .onnx + .onnx.json in %s to use it)",
                    voice_name, onnx_path, fallback, self._voice_dir)
            voice = self._get_voice(fallback)
            self._voice_cache[voice_name] = voice     # the alias, so it is decided once
            return voice

        logger.info("Loading Piper voice on demand: %s", voice_name)
        voice = self._load_voice(onnx_path, config_path)
        self._voice_cache[voice_name] = voice
        return voice

    def infer_local(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        task = input_data.get("task", "speech_synthesis")
        if task not in _SUPPORTED_TASKS:
            raise ValueError(f"PiperAdapter supports {sorted(_SUPPORTED_TASKS)}, got: {task}")

        text = input_data.get("text")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("PiperAdapter requires non-empty 'text' in input_data")
        if len(text) > _MAX_TEXT_CHARS:
            raise ValueError(f"'text' exceeds {_MAX_TEXT_CHARS}-char limit; split into chunks")

        voice_name = input_data.get("voice", self._default_voice)
        voice = self._get_voice(voice_name)

        synth_kwargs: Dict[str, Any] = {}
        if "length_scale" in input_data:
            synth_kwargs["length_scale"] = float(input_data["length_scale"])
        if "noise_scale" in input_data:
            synth_kwargs["noise_scale"] = float(input_data["noise_scale"])
        if "noise_w" in input_data:
            synth_kwargs["noise_w"] = float(input_data["noise_w"])

        audio_path, audio_uri = mint_audio_path(self._output_subdir, extension="wav")

        start_time = time.time()
        with wave.open(str(audio_path), "wb") as wav_file:
            voice.synthesize(text, wav_file, **synth_kwargs)

        sample_rate, duration_seconds = self._probe_wav(audio_path)

        return {
            "task": "speech_synthesis",
            "audio_uri": audio_uri,
            "duration_seconds": duration_seconds,
            "sample_rate": sample_rate,
            "voice": voice_name,
            "text_length": len(text),
            "executed_at": int(time.time() * 1000),
            "latency_ms": int((time.time() - start_time) * 1000),
        }

    @staticmethod
    def _probe_wav(path) -> tuple[int, float]:
        with wave.open(str(path), "rb") as wav_file:
            sample_rate = wav_file.getframerate()
            frames = wav_file.getnframes()
            duration = frames / float(sample_rate) if sample_rate else 0.0
        return sample_rate, round(duration, 3)

    @property
    def schema(self) -> Dict[str, Any]:
        return {
            "tasks": sorted(_SUPPORTED_TASKS),
            "description": "Neural text-to-speech via Piper (ONNX voices, CPU-friendly).",
            "input_fields": {
                "text": {"type": "string", "description": "Text to synthesize"},
                "voice": {"type": "string", "description": f"Voice name (default: {self._default_voice})"},
                "length_scale": {"type": "number", "description": ">1.0 = slower speech"},
                "noise_scale": {"type": "number"},
                "noise_w": {"type": "number"},
            },
            "response_fields": {
                "audio_uri": {"type": "string", "description": "opennvr://audio/... WAV"},
                "duration_seconds": {"type": "number"},
                "sample_rate": {"type": "integer"},
                "voice": {"type": "string"},
                "text_length": {"type": "integer"},
            },
        }

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "type": self.type,
            "model": f"piper:{self._default_voice}",
            "framework": "piper-tts",
            "tasks": sorted(_SUPPORTED_TASKS),
            "voice_dir": self._voice_dir,
            "cached_voices": sorted(self._voice_cache.keys()),
            "model_loaded": bool(self._voice_cache),
        }

    def health_check(self) -> Dict[str, Any]:
        return {
            "status": "healthy",
            "type": self.type,
            "model_loaded": bool(self._voice_cache),
            "model_info": self.get_model_info(),
        }
