# Copyright (c) 2026 OpenNVR
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Shared fixtures for the new Piper contract-service tests.

Kept as a regular module (leading underscore so pytest doesn't try to
collect it as a test file) rather than a conftest.py because a nested
``tests/adapters/`` dir would namespace-shadow the top-level
``adapters/`` package and break ``import adapters.piper.*``. Flat
fixture module + explicit import in the test files dodges the
collision.
"""
from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

import pytest


def install_fake_piper() -> type:
    """Inject a stub ``piper.voice`` module providing ``PiperVoice``."""

    class _FakePiperVoice:
        def __init__(self, onnx_path, config_path=None):
            self.onnx_path = onnx_path
            self.config_path = config_path

        @classmethod
        def load(cls, onnx_path, config_path=None):
            return cls(onnx_path, config_path=config_path)

        def synthesize(self, text, wav_file, **kwargs):
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
def piper_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    audio_dir = tmp_path / "audio"
    weights_dir = tmp_path / "model_weights"
    voice_dir = weights_dir / "piper"
    voice_dir.mkdir(parents=True)

    voice_name = "test-voice"
    (voice_dir / f"{voice_name}.onnx").write_bytes(b"PIPER_TEST_VOICE_PAYLOAD")
    (voice_dir / f"{voice_name}.onnx.json").write_text("{}")

    import app.config.config as config_module
    monkeypatch.setattr(config_module, "BASE_AUDIO_DIR", str(audio_dir))
    monkeypatch.setattr(config_module, "MODEL_WEIGHTS_DIR", str(weights_dir))
    import app.config as config_pkg
    monkeypatch.setattr(config_pkg, "MODEL_WEIGHTS_DIR", str(weights_dir), raising=False)

    import app.utils.audio_utils as audio_utils
    importlib.reload(audio_utils)
    import app.adapters.audio.piper_adapter as piper_module
    importlib.reload(piper_module)

    install_fake_piper()

    return {
        "audio_dir": audio_dir,
        "voice_dir": voice_dir,
        "voice_name": voice_name,
    }


def _boot_app(piper_environment_dict, monkeypatch: pytest.MonkeyPatch):
    """Reload service modules + boot TestClient against sandboxed env."""
    # Reload everything that captured config at import time so the new
    # patched config takes effect.
    for mod_name in ("adapters.piper.service", "adapters.piper.auth", "adapters.piper.main"):
        if mod_name in sys.modules:
            importlib.reload(sys.modules[mod_name])

    import adapters.piper.service as service_module
    original_init = service_module.PiperService.__init__

    def patched_init(self, default_voice="test-voice", voice_dir=None):
        original_init(
            self,
            default_voice=default_voice,
            voice_dir=str(piper_environment_dict["voice_dir"]),
        )

    monkeypatch.setattr(service_module.PiperService, "__init__", patched_init)

    from fastapi.testclient import TestClient
    import adapters.piper.main as main_module
    return TestClient(main_module.app), main_module


@pytest.fixture
def piper_app(piper_environment, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("OPENNVR_ADAPTER_TOKEN", raising=False)
    client, _ = _boot_app(piper_environment, monkeypatch)
    with client:
        yield client


@pytest.fixture
def piper_app_with_auth(piper_environment, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OPENNVR_ADAPTER_TOKEN", "test-token")
    client, _ = _boot_app(piper_environment, monkeypatch)
    with client:
        yield client, "test-token"
