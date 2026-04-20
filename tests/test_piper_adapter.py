# Copyright (c) 2026 OpenNVR
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Unit tests for PiperAdapter.

piper-tts is an optional dep (``uv sync --extra tts``). We inject a fake
``piper.voice`` module so tests run without real voice ONNX files on disk.
The fake ``PiperVoice.synthesize`` writes a valid (but tiny) WAV to the
supplied wave handle, which is exactly what the real Piper library does.
"""
import importlib
import sys
import types

import pytest


def _install_fake_piper():
    """Inject a stub ``piper.voice`` module providing ``PiperVoice``."""

    class _FakePiperVoice:
        def __init__(self, onnx_path, config_path=None):
            self.onnx_path = onnx_path
            self.config_path = config_path

        @classmethod
        def load(cls, onnx_path, config_path=None):
            return cls(onnx_path, config_path=config_path)

        def synthesize(self, text, wav_file, **kwargs):
            # Real Piper writes int16 PCM frames. We write 100ms of silence at
            # 22050 Hz so wave.open can parse duration/sample_rate afterwards.
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(22050)
            wav_file.writeframes(b"\x00\x00" * 2205)
            self.last_call = {"text": text, **kwargs}

    voice_module = types.ModuleType("piper.voice")
    voice_module.PiperVoice = _FakePiperVoice

    package = types.ModuleType("piper")
    package.voice = voice_module

    sys.modules["piper"] = package
    sys.modules["piper.voice"] = voice_module
    return _FakePiperVoice


@pytest.fixture
def piper_adapter(tmp_path, monkeypatch):
    # Redirect BASE_AUDIO_DIR + MODEL_WEIGHTS_DIR into tmp so the adapter is
    # fully sandboxed.
    audio_dir = tmp_path / "audio"
    weights_dir = tmp_path / "model_weights"
    voice_dir = weights_dir / "piper"
    voice_dir.mkdir(parents=True)

    # The adapter probes for <voice>.onnx at load time; create an empty file so
    # existence check passes — our fake PiperVoice.load ignores contents.
    voice_name = "test-voice"
    (voice_dir / f"{voice_name}.onnx").write_bytes(b"")
    (voice_dir / f"{voice_name}.onnx.json").write_text("{}")

    import app.config.config as config_module
    monkeypatch.setattr(config_module, "BASE_AUDIO_DIR", str(audio_dir))
    monkeypatch.setattr(config_module, "MODEL_WEIGHTS_DIR", str(weights_dir))

    # Reimport modules that captured the old values at import time.
    import app.utils.audio_utils as audio_utils
    importlib.reload(audio_utils)
    import app.adapters.audio.piper_adapter as piper_module
    importlib.reload(piper_module)

    _install_fake_piper()

    adapter = piper_module.PiperAdapter({
        "enabled": True,
        "voice": voice_name,
        "voice_dir": str(voice_dir),
    })
    adapter.ensure_model_loaded()
    return adapter, audio_dir


def test_synthesize_writes_wav_and_returns_uri(piper_adapter):
    adapter, audio_dir = piper_adapter
    result = adapter.infer({"task": "speech_synthesis", "text": "Hello world"})

    assert result["task"] == "speech_synthesis"
    assert result["audio_uri"].startswith("opennvr://audio/tts/")
    assert result["audio_uri"].endswith(".wav")
    assert result["voice"] == "test-voice"
    assert result["sample_rate"] == 22050
    assert result["duration_seconds"] > 0
    assert result["text_length"] == len("Hello world")

    # Physical file should exist under BASE_AUDIO_DIR/tts/
    relative = result["audio_uri"][len("opennvr://audio/"):]
    assert (audio_dir / relative).exists()


def test_synthesize_forwards_optional_kwargs(piper_adapter):
    adapter, _ = piper_adapter
    result = adapter.infer({
        "task": "speech_synthesis",
        "text": "Hi",
        "length_scale": 1.2,
        "noise_scale": 0.5,
    })
    voice = adapter._voice_cache["test-voice"]
    assert voice.last_call["length_scale"] == pytest.approx(1.2)
    assert voice.last_call["noise_scale"] == pytest.approx(0.5)
    assert result["audio_uri"].endswith(".wav")


def test_rejects_empty_text(piper_adapter):
    adapter, _ = piper_adapter
    with pytest.raises(ValueError, match="text"):
        adapter.infer({"task": "speech_synthesis", "text": ""})


def test_rejects_unknown_task(piper_adapter):
    adapter, _ = piper_adapter
    with pytest.raises(ValueError, match="PiperAdapter supports"):
        adapter.infer({"task": "audio_transcription", "text": "Hi"})


def test_rejects_invalid_voice_name(piper_adapter):
    adapter, _ = piper_adapter
    with pytest.raises(ValueError, match="Invalid voice name"):
        adapter.infer({"task": "speech_synthesis", "text": "Hi", "voice": "../escape"})


def test_load_fails_when_default_voice_missing(tmp_path, monkeypatch):
    audio_dir = tmp_path / "audio"
    weights_dir = tmp_path / "model_weights"
    voice_dir = weights_dir / "piper"
    voice_dir.mkdir(parents=True)  # directory exists but no voice file

    import app.config.config as config_module
    monkeypatch.setattr(config_module, "BASE_AUDIO_DIR", str(audio_dir))
    monkeypatch.setattr(config_module, "MODEL_WEIGHTS_DIR", str(weights_dir))

    import app.utils.audio_utils as audio_utils
    importlib.reload(audio_utils)
    import app.adapters.audio.piper_adapter as piper_module
    importlib.reload(piper_module)

    _install_fake_piper()

    adapter = piper_module.PiperAdapter({
        "enabled": True,
        "voice": "nonexistent-voice",
        "voice_dir": str(voice_dir),
    })
    with pytest.raises(RuntimeError, match="Piper voice"):
        adapter.ensure_model_loaded()


def test_health_check_before_load(tmp_path, monkeypatch):
    import app.adapters.audio.piper_adapter as piper_module
    adapter = piper_module.PiperAdapter({"enabled": True})
    info = adapter.health_check()
    assert info["status"] == "healthy"
    assert info["type"] == "audio"
    assert info["model_loaded"] is False
    assert "speech_synthesis" in info["model_info"]["tasks"]
