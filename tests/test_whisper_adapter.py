# Copyright (c) 2026 OpenNVR
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Unit tests for WhisperAdapter.

faster_whisper is an optional dependency (``uv sync --extra stt``) and we
cannot assume it is installed on every CI runner. So we inject a fake
``faster_whisper`` module into ``sys.modules`` before the adapter tries to
import it. This also gives us full control over what ``WhisperModel`` returns,
keeping the tests deterministic and free of network/model-weight downloads.
"""
import sys
import types
from types import SimpleNamespace
from unittest.mock import patch

import pytest


def _install_fake_faster_whisper(segments, info):
    """Inject a stub ``faster_whisper`` module returning canned segments + info."""

    class _FakeWhisperModel:
        def __init__(self, *args, **kwargs):
            self.init_args = args
            self.init_kwargs = kwargs

        def transcribe(self, path, **kwargs):
            self.last_call = {"path": path, **kwargs}
            return iter(segments), info

    module = types.ModuleType("faster_whisper")
    module.WhisperModel = _FakeWhisperModel
    sys.modules["faster_whisper"] = module
    return module


@pytest.fixture
def fake_audio_file(tmp_path):
    audio_file = tmp_path / "clip.wav"
    audio_file.write_bytes(b"\x00\x00")  # content does not matter — adapter delegates to WhisperModel
    return tmp_path, audio_file


@pytest.fixture
def whisper_adapter(fake_audio_file, monkeypatch):
    base_dir, _ = fake_audio_file
    # BASE_AUDIO_DIR must be patched BEFORE importing the adapter so that
    # resolve_audio_uri resolves against tmp_path, not the real project dir.
    import app.config.config as config_module
    monkeypatch.setattr(config_module, "BASE_AUDIO_DIR", str(base_dir))

    # Reimport audio_utils so it picks up the patched BASE_AUDIO_DIR
    import importlib
    import app.utils.audio_utils as audio_utils
    importlib.reload(audio_utils)

    from app.adapters.audio.whisper_adapter import WhisperAdapter
    adapter = WhisperAdapter({"enabled": True, "model_size": "tiny", "device": "cpu", "compute_type": "int8"})
    return adapter


def test_transcription_happy_path(whisper_adapter, fake_audio_file):
    segments = [
        SimpleNamespace(start=0.0, end=1.5, text=" Hello", avg_logprob=-0.2, no_speech_prob=0.01),
        SimpleNamespace(start=1.5, end=3.0, text=" world", avg_logprob=-0.3, no_speech_prob=0.02),
    ]
    info = SimpleNamespace(language="en", language_probability=0.99, duration=3.0)
    _install_fake_faster_whisper(segments, info)

    result = whisper_adapter.infer({
        "task": "audio_transcription",
        "audio": {"uri": "opennvr://audio/clip.wav"},
    })

    assert result["task"] == "audio_transcription"
    assert result["text"] == "Hello world"
    assert len(result["segments"]) == 2
    assert result["segments"][0]["text"] == "Hello"
    assert result["language"] == "en"
    assert result["language_confidence"] == pytest.approx(0.99)
    assert result["duration_seconds"] == pytest.approx(3.0)
    assert result["translated_to_english"] is False
    assert result["model"] == "tiny"


def test_translation_forces_english_language(whisper_adapter):
    segments = [SimpleNamespace(start=0.0, end=1.0, text=" Hola", avg_logprob=-0.4, no_speech_prob=0.0)]
    info = SimpleNamespace(language="es", language_probability=0.95, duration=1.0)
    _install_fake_faster_whisper(segments, info)

    result = whisper_adapter.infer({
        "task": "audio_translation",
        "audio": {"uri": "opennvr://audio/clip.wav"},
    })

    # Whisper's "translate" task always emits English text regardless of source.
    # The adapter reflects that by overriding `language` to "en" for translation.
    assert result["task"] == "audio_translation"
    assert result["language"] == "en"
    assert result["translated_to_english"] is True


def test_rejects_unknown_task(whisper_adapter):
    _install_fake_faster_whisper([], SimpleNamespace(language="en", language_probability=1.0, duration=0.0))
    with pytest.raises(ValueError, match="WhisperAdapter supports"):
        whisper_adapter.infer({
            "task": "scene_description",
            "audio": {"uri": "opennvr://audio/clip.wav"},
        })


def test_rejects_missing_audio_uri(whisper_adapter):
    _install_fake_faster_whisper([], SimpleNamespace(language="en", language_probability=1.0, duration=0.0))
    with pytest.raises(ValueError, match="audio.uri"):
        whisper_adapter.infer({"task": "audio_transcription"})


def test_rejects_path_traversal(whisper_adapter):
    _install_fake_faster_whisper([], SimpleNamespace(language="en", language_probability=1.0, duration=0.0))
    with pytest.raises(Exception) as exc_info:
        whisper_adapter.infer({
            "task": "audio_transcription",
            "audio": {"uri": "opennvr://audio/../../etc/passwd"},
        })
    # HTTPException or similar — we just care that traversal is blocked.
    assert "path traversal" in str(exc_info.value).lower() or "not found" in str(exc_info.value).lower() or "invalid" in str(exc_info.value).lower()


def test_health_check_before_model_load(whisper_adapter):
    info = whisper_adapter.health_check()
    assert info["status"] == "healthy"
    assert info["type"] == "audio"
    assert info["model_loaded"] is False
    assert "audio_transcription" in info["model_info"]["tasks"]
    assert "audio_translation" in info["model_info"]["tasks"]
