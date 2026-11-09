"""Shared fixtures for the Whisper-QNN adapter tests: an assets directory
with the real export's metadata shape (12 layers, the fixed-point specs),
fake encoder/decoder sessions whose decoder follows a token script, and a
fake tokenizer — so the engine, the service, the SDK plumbing and the
conformance kit run without the Hexagon or the 340 MB binaries."""
from __future__ import annotations

import importlib
import io
import json
import sys
import wave
from pathlib import Path

import numpy as np
import pytest

N_LAYERS = 12
HELLO_WORLD = [2425, 1002, 13]          # " Hello", " world", "." in Whisper's vocabulary
EOT = 50257

ENC_SPEC = {"shape": [1, 80, 3000], "dtype": "uint16",
            "quantization_parameters": {"scale": 4.677007018472068e-05, "zero_point": 32072}}
MASK_SPEC = {"shape": [1, 1, 1, 200], "dtype": "uint16",
             "quantization_parameters": {"scale": 0.0015259021893143654, "zero_point": 65535}}
LOGITS_SPEC = {"shape": [1, 51865, 1, 1], "dtype": "uint16",
               "quantization_parameters": {"scale": 0.0012925398768857121, "zero_point": 17867}}


def _cache(shape, scale):
    return {"shape": shape, "dtype": "uint8", "quantization_parameters": {"scale": scale, "zero_point": 128}}


def make_metadata() -> dict:
    enc_out, dec_in, dec_out = {}, {"input_ids": {"shape": [1, 1], "dtype": "int32"},
                                    "position_ids": {"shape": [1], "dtype": "int32"},
                                    "attention_mask": MASK_SPEC}, {}
    for i in range(N_LAYERS):
        enc_out[f"k_cache_cross_{i}"] = _cache([12, 1, 64, 1500], 0.07 + i / 1000)
        enc_out[f"v_cache_cross_{i}"] = _cache([12, 1, 1500, 64], 0.04 + i / 1000)
        dec_in[f"k_cache_self_{i}_in"] = _cache([12, 1, 64, 199], 0.06 + i / 1000)
        dec_in[f"v_cache_self_{i}_in"] = _cache([12, 1, 199, 64], 0.03 + i / 1000)
        dec_in[f"k_cache_cross_{i}"] = enc_out[f"k_cache_cross_{i}"]
        dec_in[f"v_cache_cross_{i}"] = enc_out[f"v_cache_cross_{i}"]
        dec_out[f"k_cache_self_{i}_out"] = dec_in[f"k_cache_self_{i}_in"]
        dec_out[f"v_cache_self_{i}_out"] = dec_in[f"v_cache_self_{i}_in"]
    dec_out["logits"] = LOGITS_SPEC
    return {
        "model_id": "whisper_small_quantized", "model_name": "Whisper-Small-Quantized",
        "runtime": "precompiled_qnn_onnx", "precision": "w8a16",
        "tool_versions": {"qairt": "2.50.0.260828221209", "onnx_runtime": "1.30.0"},
        "model_files": {"encoder.onnx": {"inputs": {"input_features": ENC_SPEC}, "outputs": enc_out},
                        "decoder.onnx": {"inputs": dec_in, "outputs": dec_out}},
        "chipset_attributes": {"name": "qualcomm-qcs8275", "htp_version": 75,
                               "reference_device": "Arduino VENTUNO Q"},
        "use_case": "Speech Recognition",
    }


class _Out:
    def __init__(self, name):
        self.name = name


class FakeEncoder:
    def __init__(self, metadata):
        self.spec = metadata["model_files"]["encoder.onnx"]
        self.calls: list[dict] = []

    def get_outputs(self):
        return [_Out(n) for n in self.spec["outputs"]]

    def run(self, _names, feeds):
        self.calls.append({k: (v.dtype, v.shape) for k, v in feeds.items()})
        assert feeds["input_features"].dtype == np.uint16 and feeds["input_features"].shape == (1, 80, 3000)
        return [np.full(tuple(s["shape"]), 128, dtype=np.uint8) for s in self.spec["outputs"].values()]


class FakeDecoder:
    """Emits ``script`` tokens after the forced prefix, then EOT. Records
    every feed so tests can check the mask, the positions and the cache
    feedback. ``no_speech_logit`` sets how loud <|nocaptions|> is at step 0."""

    def __init__(self, metadata, script=None, *, no_speech_logit: float = 0.0):
        self.spec = metadata["model_files"]["decoder.onnx"]
        self.script = list(script if script is not None else HELLO_WORLD)
        self.no_speech_logit = no_speech_logit
        self.calls: list[dict] = []
        self.n = 0

    def get_outputs(self):
        return [_Out(n) for n in self.spec["outputs"]]

    def _logits(self, favourite: int, step: int) -> np.ndarray:
        q = LOGITS_SPEC["quantization_parameters"]
        logits = np.zeros(51865, dtype=np.float32)
        logits[favourite] = 20.0
        if step == 0:
            logits[50362] = self.no_speech_logit
        raw = np.rint(logits / q["scale"]) + q["zero_point"]
        return np.clip(raw, 0, 65535).astype(np.uint16).reshape(1, 51865, 1, 1)

    def run(self, _names, feeds):
        step = self.n
        self.calls.append({k: np.array(v, copy=True) for k, v in feeds.items()})
        self.n += 1
        # Steps 0..2 are the forced prefix (the engine ignores our pick);
        # from step 3 on the script decides, then EOT.
        gen = step - 3
        favourite = self.script[gen] if 0 <= gen < len(self.script) else EOT
        outs = []
        for name, spec in self.spec["outputs"].items():
            if name == "logits":
                outs.append(self._logits(favourite, step))
            else:
                src = feeds[name.replace("_out", "_in")]
                # shift by one slot, newest at the end — as the real decoder does
                shifted = np.roll(src, -1, axis=-1 if "k_cache" in name else -2)
                outs.append(shifted.astype(np.uint8))
        return outs


class FakeTokenizer:
    words = {2425: " Hello", 1002: " world", 13: ".", 50259: "<|en|>", 50276: "<|hi|>"}

    def decode(self, ids, skip_special_tokens=True):
        return "".join(self.words.get(int(i), f"<{int(i)}>") for i in ids if not (skip_special_tokens and int(i) >= 50257))

    def token_to_id(self, token):
        return {v: k for k, v in self.words.items()}.get(token)


def silent_wav16k(seconds: float = 2.0) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000)
        w.writeframes(b"\x00\x00" * int(16000 * seconds))
    return buf.getvalue()


def write_assets(dir_: Path) -> dict:
    dir_.mkdir(parents=True, exist_ok=True)
    meta = make_metadata()
    (dir_ / "metadata.json").write_text(json.dumps(meta))
    for name in ("encoder.onnx", "encoder.bin", "decoder.onnx", "decoder.bin", "tokenizer.json"):
        (dir_ / name).write_bytes(b"\x00" * 64)
    rng = np.random.default_rng(0)
    (dir_ / "preprocessor_config.json").write_text(json.dumps({"mel_filters": rng.random((80, 201)).tolist()}))
    return meta


@pytest.fixture
def whisper_qnn_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    assets = tmp_path / "whisper-qnn"
    meta = write_assets(assets)
    monkeypatch.setenv("WHISPER_QNN_ASSETS_DIR", str(assets))
    monkeypatch.delenv("ADAPTER_ONNX_PROVIDERS", raising=False)
    import adapters.whisper_qnn.service as svc

    made: dict[str, object] = {}

    def fake_session(path, requested):
        made["requested"] = requested
        if path.endswith("encoder.onnx"):
            made["encoder"] = FakeEncoder(meta)
            return made["encoder"]
        made["decoder"] = FakeDecoder(meta)
        return made["decoder"]

    monkeypatch.setattr(svc, "build_session", fake_session)
    monkeypatch.setattr(svc, "load_tokenizer", lambda path: FakeTokenizer())
    for mod in ("adapters.whisper_qnn.main",):
        if mod in sys.modules:
            importlib.reload(sys.modules[mod])
    return {"assets": assets, "metadata": meta, "made": made}


@pytest.fixture
def whisper_qnn_app(whisper_qnn_environment, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("OPENNVR_ADAPTER_TOKEN", raising=False)
    from fastapi.testclient import TestClient
    import adapters.whisper_qnn.main as main_module
    importlib.reload(main_module)
    with TestClient(main_module.app) as client:
        yield client


@pytest.fixture
def sample_wav() -> bytes:
    return silent_wav16k()
