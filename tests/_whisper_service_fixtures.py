# Copyright (c) 2026 OpenNVR
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Shared fixtures for the Whisper contract-service tests.

Stubs out ``faster_whisper`` so tests run without the real model
weights. The fake WhisperModel returns deterministic segments so
the §5.3 translation logic has something concrete to validate.

Same flat-module pattern as the Piper and YOLOv8 fixtures
(``tests/_piper_service_fixtures.py``, ``tests/_yolov8_service_fixtures.py``)
to sidestep the ``tests/adapters/`` namespace-collision.
"""
from __future__ import annotations

import importlib
import io
import sys
import types
import wave
from pathlib import Path
from types import SimpleNamespace

import pytest


def install_fake_faster_whisper():
    """Inject a stub ``faster_whisper`` module providing ``WhisperModel``.

    The stub's ``transcribe`` returns three deterministic segments
    plus an ``info`` namespace — same shape that real faster-whisper
    returns (segments_iter, info).
    """

    segments = [
        SimpleNamespace(start=0.0, end=1.5, text=" Hello", avg_logprob=-0.2, no_speech_prob=0.01),
        SimpleNamespace(start=1.5, end=3.0, text=" world", avg_logprob=-0.3, no_speech_prob=0.02),
        SimpleNamespace(start=3.0, end=4.2, text="  ", avg_logprob=-0.5, no_speech_prob=0.99),
    ]
    info = SimpleNamespace(language="en", language_probability=0.97, duration=4.2)

    class _FakeWhisperModel:
        last_call: dict | None = None

        def __init__(self, *args, **kwargs):
            self.init_args = args
            self.init_kwargs = kwargs

        def transcribe(self, path, **kwargs):
            _FakeWhisperModel.last_call = {"path": path, **kwargs}
            return iter(segments), info

    module = types.ModuleType("faster_whisper")
    module.WhisperModel = _FakeWhisperModel
    sys.modules["faster_whisper"] = module
    return _FakeWhisperModel


def _silent_wav_bytes() -> bytes:
    """Generate a small valid WAV file for tests. The fake
    WhisperModel ignores the file contents, but FileFrameSource
    style code paths may probe / validate."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(8000)
        w.writeframes(b"\x00\x00" * 800)
    return buf.getvalue()


@pytest.fixture
def whisper_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Sandboxed env: tmp download_root + fake faster_whisper + a tmp
    ``config.json`` so the fingerprint computation has something
    deterministic to hash."""
    weights_dir = tmp_path / "model_weights"
    download_root = weights_dir / "whisper"
    download_root.mkdir(parents=True)
    # The service's fingerprint walks download_root looking for a
    # config.json — plant a small one.
    (download_root / "config.json").write_text('{"model_size": "test"}')

    import app.config.config as config_module
    monkeypatch.setattr(config_module, "MODEL_WEIGHTS_DIR", str(weights_dir))
    import app.config as config_pkg
    monkeypatch.setattr(config_pkg, "MODEL_WEIGHTS_DIR", str(weights_dir), raising=False)

    install_fake_faster_whisper()

    import app.adapters.audio.whisper_adapter as whisper_module
    importlib.reload(whisper_module)

    return {
        "weights_dir": weights_dir,
        "download_root": download_root,
        "sample_wav": _silent_wav_bytes(),
    }


def _boot_app(env, monkeypatch: pytest.MonkeyPatch):
    for mod_name in ("adapters.whisper.service", "adapters.whisper.auth", "adapters.whisper.main"):
        if mod_name in sys.modules:
            importlib.reload(sys.modules[mod_name])

    import adapters.whisper.service as service_module
    original_init = service_module.WhisperService.__init__

    def patched_init(self, model_size=None, *, download_root=None, device="cpu", compute_type="int8"):
        original_init(
            self,
            model_size=model_size or "tiny",
            download_root=str(env["download_root"]),
            device=device,
            compute_type=compute_type,
        )

    monkeypatch.setattr(service_module.WhisperService, "__init__", patched_init)

    from fastapi.testclient import TestClient
    import adapters.whisper.main as main_module
    return TestClient(main_module.app), main_module


@pytest.fixture
def whisper_app(whisper_environment, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("OPENNVR_ADAPTER_TOKEN", raising=False)
    client, _ = _boot_app(whisper_environment, monkeypatch)
    with client:
        yield client


@pytest.fixture
def whisper_app_with_auth(whisper_environment, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OPENNVR_ADAPTER_TOKEN", "test-token")
    client, _ = _boot_app(whisper_environment, monkeypatch)
    with client:
        yield client, "test-token"


@pytest.fixture
def sample_wav(whisper_environment) -> bytes:
    """The tmp silent WAV the service will read in tests."""
    return whisper_environment["sample_wav"]
