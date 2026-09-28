"""CPU tuning for the two voice adapters (open-nvr #583).

A 71 s voice turn on an 8-core box spent 14 s in Whisper and 25 s in
Piper. The models were not the cost; their defaults were: beam 5 on a
three-second clip, and an ONNX session fanning one utterance across
every core the detector and recorder also needed.
"""
from __future__ import annotations

import importlib
import json
import sys
import types

import pytest


# ── Piper: a thread-capped ONNX session ───────────────────────────────

def _install_fake_piper_and_ort(monkeypatch, *, with_config: bool = True):
    """piper.voice / piper.config / onnxruntime stubs that record how the
    session was built. ``with_config=False`` mimics an older piper with
    no ``piper.config`` module, which must fall back to ``load``."""
    built: dict = {}

    class _Opts:
        def __init__(self):
            self.intra_op_num_threads = None
            self.inter_op_num_threads = None

    class _Session:
        def __init__(self, path, sess_options=None, providers=None):
            built.update(path=path, opts=sess_options, providers=providers)

    ort = types.ModuleType("onnxruntime")
    ort.SessionOptions = _Opts
    ort.InferenceSession = _Session
    monkeypatch.setitem(sys.modules, "onnxruntime", ort)

    class _Config:
        def __init__(self, d):
            self.d = d

        @classmethod
        def from_dict(cls, d):
            return cls(d)

    class _Voice:
        def __init__(self, session=None, config=None, onnx_path=None, config_path=None):
            self.session, self.config = session, config
            self.loaded_via = "ctor" if session is not None else "load"

        @classmethod
        def load(cls, onnx_path, config_path=None):
            return cls(onnx_path=onnx_path, config_path=config_path)

        def synthesize(self, text, wav_file, **kwargs):
            wav_file.setnchannels(1); wav_file.setsampwidth(2); wav_file.setframerate(22050)
            wav_file.writeframes(b"\x00\x00" * 2205)

    voice_mod = types.ModuleType("piper.voice"); voice_mod.PiperVoice = _Voice
    pkg = types.ModuleType("piper"); pkg.voice = voice_mod
    monkeypatch.setitem(sys.modules, "piper", pkg)
    monkeypatch.setitem(sys.modules, "piper.voice", voice_mod)
    if with_config:
        cfg_mod = types.ModuleType("piper.config"); cfg_mod.PiperConfig = _Config
        pkg.config = cfg_mod
        monkeypatch.setitem(sys.modules, "piper.config", cfg_mod)
    else:
        monkeypatch.delitem(sys.modules, "piper.config", raising=False)
    return built


@pytest.fixture()
def voice_dir(tmp_path, monkeypatch):
    weights = tmp_path / "model_weights"
    vdir = weights / "piper"
    vdir.mkdir(parents=True)
    (vdir / "v.onnx").write_bytes(b"")
    (vdir / "v.onnx.json").write_text(json.dumps({"audio": {"sample_rate": 22050}}))
    import app.config.config as config_module
    monkeypatch.setattr(config_module, "BASE_AUDIO_DIR", str(tmp_path / "audio"))
    monkeypatch.setattr(config_module, "MODEL_WEIGHTS_DIR", str(weights))
    import app.utils.audio_utils as audio_utils
    importlib.reload(audio_utils)
    return vdir


def _piper(voice_dir, threads):
    import app.adapters.audio.piper_adapter as mod
    importlib.reload(mod)
    a = mod.PiperAdapter({"enabled": True, "voice": "v", "voice_dir": str(voice_dir), "threads": threads})
    a.ensure_model_loaded()
    return a


def test_piper_builds_its_own_session_with_the_thread_cap(voice_dir, monkeypatch):
    built = _install_fake_piper_and_ort(monkeypatch)
    a = _piper(voice_dir, threads=2)
    voice = a._voice_cache["v"]
    assert voice.loaded_via == "ctor", "the session must be ours, not PiperVoice.load's"
    assert built["opts"].intra_op_num_threads == 2
    assert built["opts"].inter_op_num_threads == 1
    assert built["providers"] == ["CPUExecutionProvider"]
    assert voice.config.d == {"audio": {"sample_rate": 22050}}


def test_piper_without_a_cap_uses_pipers_own_loader(voice_dir, monkeypatch):
    built = _install_fake_piper_and_ort(monkeypatch)
    for threads in (None, 0, "", "x"):
        a = _piper(voice_dir, threads=threads)
        assert a._voice_cache["v"].loaded_via == "load", f"threads={threads!r}"
    assert built == {}, "no session of ours was built"


def test_an_older_piper_falls_back_and_says_the_cap_is_off(voice_dir, monkeypatch, caplog):
    _install_fake_piper_and_ort(monkeypatch, with_config=False)
    import logging
    with caplog.at_level(logging.WARNING):
        a = _piper(voice_dir, threads=2)
    assert a._voice_cache["v"].loaded_via == "load"
    assert "cap is NOT in effect" in caplog.text
    # ...said once, not per voice
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        a._get_voice("v")
    assert "cap is NOT in effect" not in caplog.text


def test_piper_service_reads_its_knobs_from_the_environment(monkeypatch):
    from adapters.piper import service as svc
    monkeypatch.setenv("OPENNVR_PIPER_THREADS", "3")
    monkeypatch.setenv("OPENNVR_PIPER_VOICE", "en_US-lessac-medium")
    s = svc.PiperService()
    assert s._threads == 3 and s._default_voice == "en_US-lessac-medium"
    assert s._adapter.config["threads"] == 3
    monkeypatch.delenv("OPENNVR_PIPER_THREADS"); monkeypatch.delenv("OPENNVR_PIPER_VOICE")
    s = svc.PiperService()
    assert s._threads == svc.DEFAULT_THREADS == 2, "two by default: a medium voice several times real time, the rest of the box left alone"
    assert s._default_voice == svc.DEFAULT_VOICE
    monkeypatch.setenv("OPENNVR_PIPER_THREADS", "lots")
    assert svc.PiperService()._threads == 2, "garbage falls back, never crashes the adapter"


# ── Whisper: greedy by default, threads capped on CPU ─────────────────

def _install_fake_faster_whisper(monkeypatch):
    class _Model:
        instances = []

        def __init__(self, *args, **kwargs):
            self.args, self.kwargs = args, kwargs
            _Model.instances.append(self)

        def transcribe(self, path, **kwargs):
            self.last = kwargs
            return iter([]), types.SimpleNamespace(language="en", language_probability=1.0, duration=1.0)

    mod = types.ModuleType("faster_whisper"); mod.WhisperModel = _Model
    monkeypatch.setitem(sys.modules, "faster_whisper", mod)
    return _Model


def _whisper(tmp_path, monkeypatch, **cfg):
    import app.config.config as config_module
    monkeypatch.setattr(config_module, "BASE_AUDIO_DIR", str(tmp_path))
    import app.utils.audio_utils as audio_utils
    importlib.reload(audio_utils)
    (tmp_path / "a.wav").write_bytes(b"\x00\x00")
    from app.adapters.audio.whisper_adapter import WhisperAdapter
    a = WhisperAdapter({"enabled": True, "model_size": "tiny", "device": "cpu",
                        "compute_type": "int8", **cfg})
    a.ensure_model_loaded()
    return a


def test_whisper_decodes_greedily_unless_asked_otherwise(tmp_path, monkeypatch):
    model = _install_fake_faster_whisper(monkeypatch)
    a = _whisper(tmp_path, monkeypatch)
    a.infer({"task": "audio_transcription", "audio": {"uri": "opennvr://audio/a.wav"}})
    assert model.instances[-1].last["beam_size"] == 1
    a.infer({"task": "audio_transcription", "audio": {"uri": "opennvr://audio/a.wav"}, "beam_size": 5})
    assert model.instances[-1].last["beam_size"] == 5, "offline accuracy is still one request away"


def test_whisper_caps_ctranslate2_threads_on_cpu_only(tmp_path, monkeypatch):
    model = _install_fake_faster_whisper(monkeypatch)
    _whisper(tmp_path, monkeypatch, cpu_threads=2)
    assert model.instances[-1].kwargs["cpu_threads"] == 2
    _whisper(tmp_path, monkeypatch)
    assert "cpu_threads" not in model.instances[-1].kwargs, "0/unset = CTranslate2's own default"
    _whisper(tmp_path, monkeypatch, cpu_threads=2, device="cuda")
    assert "cpu_threads" not in model.instances[-1].kwargs, "a GPU box is not what the cap is for"


def test_whisper_service_finally_honours_the_documented_env(monkeypatch):
    """WHISPER_MODEL_SIZE has been in the README, the Dockerfile and the
    compose stack since the adapter shipped — and read by nothing."""
    from adapters.whisper import service as svc
    monkeypatch.setenv("WHISPER_MODEL_SIZE", "small.en")
    monkeypatch.setenv("OPENNVR_WHISPER_CPU_THREADS", "4")
    s = svc.WhisperService()
    assert s._model_size == "small.en"
    assert s._adapter.config["cpu_threads"] == 4
    monkeypatch.delenv("WHISPER_MODEL_SIZE"); monkeypatch.delenv("OPENNVR_WHISPER_CPU_THREADS")
    s = svc.WhisperService()
    assert s._model_size == svc.DEFAULT_MODEL_SIZE and s._adapter.config["cpu_threads"] == 0
    assert svc.WhisperService(model_size="tiny")._model_size == "tiny", "an explicit argument still wins"
    assert svc.DEFAULT_BEAM_SIZE == 1


# ── a requested voice that is not on disk does not mute the box ───────

def test_a_missing_requested_voice_falls_back_to_what_is_on_disk(voice_dir, monkeypatch, caplog):
    """An upgraded, offline deployment: the agent names the new default
    voice, the init could not download it, the old voice is right there.
    Speak with it — and say so once, not per sentence."""
    import logging
    _install_fake_piper_and_ort(monkeypatch)
    a = _piper(voice_dir, threads=None)            # default voice "v" is on disk
    with caplog.at_level(logging.WARNING):
        got = a._get_voice("en_US-lessac-medium")
        again = a._get_voice("en_US-lessac-medium")
    assert got is a._voice_cache["v"] and again is got
    assert caplog.text.count("not found") == 1, "said once per missing name"
    assert "speaking with 'v' instead" in caplog.text


def test_with_no_voice_at_all_the_error_still_names_the_path(voice_dir, monkeypatch):
    _install_fake_piper_and_ort(monkeypatch)
    a = _piper(voice_dir, threads=None)
    (voice_dir / "v.onnx").unlink()
    a._voice_cache.clear()
    with pytest.raises(FileNotFoundError, match="not found"):
        a._get_voice("en_US-lessac-medium")


def test_the_fallback_prefers_the_default_then_any_voice(voice_dir, monkeypatch):
    _install_fake_piper_and_ort(monkeypatch)
    a = _piper(voice_dir, threads=None)
    (voice_dir / "other.onnx").write_bytes(b""); (voice_dir / "other.onnx.json").write_text("{}")
    assert a._fallback_voice("missing") == "v", "the default first"
    (voice_dir / "v.onnx").unlink()
    assert a._fallback_voice("missing") == "other", "then whatever is there"
    assert a._fallback_voice("other") is None, "never itself"
