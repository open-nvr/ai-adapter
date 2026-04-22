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
        self._voice_cache: Dict[str, Any] = {}

    def load_model(self) -> None:
        # We don't pre-load any specific voice here — voices are tiny and loaded
        # on demand into the cache. What we DO verify is that the default voice
        # exists on disk, so misconfiguration surfaces at warmup rather than
        # mid-inference.
        from piper.voice import PiperVoice  # optional dep: uv sync --extra tts

        onnx_path, config_path = self._voice_paths(self._default_voice)
        if not os.path.exists(onnx_path):
            raise FileNotFoundError(
                f"Piper voice '{self._default_voice}' not found at {onnx_path}. "
                "Download voices from https://github.com/rhasspy/piper/blob/master/VOICES.md "
                f"into {self._voice_dir}"
            )

        self._voice_cache[self._default_voice] = PiperVoice.load(onnx_path, config_path=config_path)
        self.model = self._voice_cache
        logger.info("Piper adapter loaded default voice '%s' from %s", self._default_voice, onnx_path)

    def _voice_paths(self, voice_name: str) -> tuple[str, str]:
        if not voice_name or "/" in voice_name or "\\" in voice_name or ".." in voice_name:
            raise ValueError(f"Invalid voice name: {voice_name!r}")
        onnx_path = os.path.join(self._voice_dir, f"{voice_name}.onnx")
        config_path = os.path.join(self._voice_dir, f"{voice_name}.onnx.json")
        return onnx_path, config_path

    def _get_voice(self, voice_name: str):
        if voice_name in self._voice_cache:
            return self._voice_cache[voice_name]

        from piper.voice import PiperVoice

        onnx_path, config_path = self._voice_paths(voice_name)
        if not os.path.exists(onnx_path):
            raise FileNotFoundError(f"Piper voice '{voice_name}' not found at {onnx_path}")

        logger.info("Loading Piper voice on demand: %s", voice_name)
        voice = PiperVoice.load(onnx_path, config_path=config_path)
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
