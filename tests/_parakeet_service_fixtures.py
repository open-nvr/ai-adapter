"""Shared fixtures for the Parakeet adapter tests: a fake ``onnx_asr`` whose
model returns a deterministic timestamped result, so the service, the
SDK plumbing and the conformance kit run without the 650 MB weights."""
from __future__ import annotations

import importlib
import io
import sys
import types
import wave
from pathlib import Path

import pytest


class _FakeResult:
    def __init__(self, text, tokens, timestamps):
        self.text, self.tokens, self.timestamps = text, tokens, timestamps


class _FakeModel:
    last_call: dict = {}
    # "Hello world. Thanks" — a sentence end mid-clip, so two segments.
    result = _FakeResult("Hello world. Thanks",
                         [" Hello", " world", ".", " Thanks"], [0.1, 0.5, 0.9, 1.4])

    def with_timestamps(self):
        return self

    def recognize(self, path, **kw):
        _FakeModel.last_call = {"path": path, **kw}
        return self.result


def install_fake_onnx_asr(monkeypatch):
    module = types.ModuleType("onnx_asr")

    def load_model(name, path=None, *, quantization=None, providers=None, **kw):
        module.last_load = {"name": name, "path": path, "quantization": quantization, "providers": providers}
        return _FakeModel()

    module.load_model = load_model
    module.last_load = None
    monkeypatch.setitem(sys.modules, "onnx_asr", module)
    ort = types.ModuleType("onnxruntime")
    ort.get_available_providers = lambda: ["CPUExecutionProvider"]
    monkeypatch.setitem(sys.modules, "onnxruntime", ort)
    return module


def silent_wav16k(seconds: float = 2.0) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000)
        w.writeframes(b"\x00\x00" * int(16000 * seconds))
    return buf.getvalue()


@pytest.fixture
def parakeet_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    weights = tmp_path / "parakeet"
    weights.mkdir()
    (weights / "encoder-model.int8.onnx").write_bytes(b"\x00" * 64)
    (weights / "config.json").write_text('{"model": "fake"}')
    monkeypatch.setenv("PARAKEET_WEIGHTS_DIR", str(weights))
    monkeypatch.delenv("PARAKEET_MODEL", raising=False)
    fake = install_fake_onnx_asr(monkeypatch)
    for mod in ("adapters.parakeet.service", "adapters.parakeet.main"):
        if mod in sys.modules:
            importlib.reload(sys.modules[mod])
    return {"weights": weights, "onnx_asr": fake}


@pytest.fixture
def parakeet_app(parakeet_environment, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("OPENNVR_ADAPTER_TOKEN", raising=False)
    from fastapi.testclient import TestClient
    import adapters.parakeet.main as main_module
    importlib.reload(main_module)
    with TestClient(main_module.app) as client:
        yield client


@pytest.fixture
def sample_wav() -> bytes:
    return silent_wav16k()
