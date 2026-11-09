"""Whisper-QNN adapter: the front end, the decode loop against the export's
interface, and the contract surface, with the Hexagon faked
(tests/_whisper_qnn_fixtures.py)."""
from __future__ import annotations

import base64
import json

import numpy as np
import pytest

from opennvr_adapter_sdk.contract import AsrResult, FailureEnvelope, InferResponse

from tests._whisper_qnn_fixtures import (  # noqa: F401
    EOT, HELLO_WORLD, FakeDecoder, FakeEncoder, FakeTokenizer, make_metadata,
    sample_wav, silent_wav16k, whisper_qnn_app, whisper_qnn_environment,
)


# ── features ────────────────────────────────────────────────────────

def test_log_mel_shape_padding_and_silence_floor():
    from adapters.whisper_qnn.features import N_FRAMES, N_MELS, log_mel

    filters = np.random.default_rng(1).random((80, 201)).astype(np.float32)
    short = log_mel(np.zeros(16000, np.float32), filters)          # 1 s is padded to 30
    assert short.shape == (N_MELS, N_FRAMES) and short.dtype == np.float32
    # Silence: every bin at the 1e-10 floor, log10 = -10, (x + 4) / 4 = -1.5 — as the reference extractor gives.
    assert np.allclose(short, -1.5)
    long = log_mel(np.ones(16000 * 45, np.float32), filters)       # 45 s is cut to 30
    assert long.shape == (N_MELS, N_FRAMES)
    tone = log_mel(np.sin(np.arange(48000) / 16000 * 2 * np.pi * 440).astype(np.float32), filters)
    # the floor is eight decades under the loudest bin: (log + 4) / 4 spans at most 2.0
    assert tone.max() > tone.min() and float(tone.max() - tone.min()) <= 2.0 + 1e-5


def test_mel_filters_of_the_wrong_shape_are_refused():
    from adapters.whisper_qnn.features import log_mel

    with pytest.raises(ValueError):
        log_mel(np.zeros(16000, np.float32), np.zeros((201, 80), np.float32))


def test_quantize_round_trips_and_clips():
    from adapters.whisper_qnn.features import dequantize, quantize, zero_tensor

    spec = {"shape": [3], "dtype": "uint16", "quantization_parameters": {"scale": 0.5, "zero_point": 100}}
    q = quantize(np.array([0.0, 1.0, -60.0, 1e9], np.float32), spec)
    assert q.dtype == np.uint16 and q.tolist() == [100, 102, 0, 65535]
    assert dequantize(np.array([100, 102], np.uint16), spec).tolist() == [0.0, 1.0]
    z = zero_tensor({"shape": [2, 2], "dtype": "uint8", "quantization_parameters": {"scale": 1, "zero_point": 128}})
    assert z.dtype == np.uint8 and (z == 128).all()


# ── the decode loop against the export's interface ──────────────────

def _engine(script=None, **kw):
    from adapters.whisper_qnn.engine import WhisperQnnEngine

    meta = make_metadata()
    enc, dec = FakeEncoder(meta), FakeDecoder(meta, script, **kw)
    filters = np.random.default_rng(0).random((80, 201)).astype(np.float32)
    return WhisperQnnEngine(enc, dec, meta, FakeTokenizer(), filters), enc, dec


def test_decoder_is_driven_the_way_the_export_expects():
    from adapters.whisper_qnn.engine import LANG_EN, NO_TIMESTAMPS, SOT, TRANSCRIBE

    engine, enc, dec = _engine()
    out = engine.transcribe(np.zeros(16000, np.float32))
    assert out.text == "Hello world."
    assert out.tokens == HELLO_WORLD
    # prefix forced (4 steps) + 3 generated + the step that produced EOT
    assert out.steps == len(dec.calls) == 4 + 3
    fed = [int(c["input_ids"][0, 0]) for c in dec.calls]
    assert fed == [SOT, LANG_EN, TRANSCRIBE, NO_TIMESTAMPS] + HELLO_WORLD
    assert [int(c["position_ids"][0]) for c in dec.calls] == list(range(7))
    # the mask opens from the end: slot 199 at step 0, 198 at step 1, …; 65535 = open, 0 = masked (-100)
    for n, c in enumerate(dec.calls):
        mask = c["attention_mask"][0, 0, 0]
        assert mask.dtype == np.uint16
        assert (mask[200 - n - 1:] == 65535).all() and (mask[:200 - n - 1] == 0).all()
    # the self cache returned by step n is fed to step n+1
    for prev, nxt in zip(dec.calls, dec.calls[1:]):
        shifted = np.roll(prev["k_cache_self_0_in"], -1, axis=-1)
        assert (nxt["k_cache_self_0_in"] == shifted).all()
    # the first step's cache is the fixed-point zero (zero point 128), not 0
    assert (dec.calls[0]["v_cache_self_3_in"] == 128).all() and dec.calls[0]["v_cache_self_3_in"].dtype == np.uint8
    # cross caches straight from the encoder, every layer
    assert all(f"k_cache_cross_{i}" in dec.calls[0] and f"v_cache_cross_{i}" in dec.calls[0] for i in range(12))
    assert len(enc.calls) == 1


def test_no_speech_prob_comes_from_the_first_step():
    engine, _enc, _dec = _engine(no_speech_logit=25.0)   # louder than the fake's favourite (20)
    out = engine.transcribe(np.zeros(16000, np.float32))
    assert out.no_speech_prob > 0.99
    quiet, _e, _d = _engine(no_speech_logit=0.0)
    assert quiet.transcribe(np.zeros(16000, np.float32)).no_speech_prob < 0.01


def test_empty_script_is_an_empty_transcript_not_a_crash():
    engine, _enc, dec = _engine(script=[])
    out = engine.transcribe(np.zeros(8000, np.float32))
    assert out.text == "" and out.tokens == [] and out.steps == 4


def test_generation_stops_at_max_tokens_without_eot():
    from adapters.whisper_qnn.engine import WhisperQnnEngine

    meta = make_metadata()
    filters = np.random.default_rng(0).random((80, 201)).astype(np.float32)
    engine = WhisperQnnEngine(FakeEncoder(meta), FakeDecoder(meta, [1002] * 500), meta, FakeTokenizer(),
                              filters, max_tokens=5)
    out = engine.transcribe(np.zeros(16000, np.float32))
    assert out.tokens == [1002] * 5


def test_language_token_falls_back_to_english_for_unknown_codes():
    from adapters.whisper_qnn.engine import LANG_EN

    engine, _e, _d = _engine()
    assert engine.language_token(None) == LANG_EN
    assert engine.language_token("en") == LANG_EN
    assert engine.language_token("hi") == 50276
    assert engine.language_token("xx") == LANG_EN


def test_longer_than_thirty_seconds_is_chunked():
    engine, enc, _dec = _engine()
    out = engine.transcribe(np.zeros(16000 * 61, np.float32))
    assert out.chunks == 3 and len(enc.calls) == 3


# ── the contract surface ────────────────────────────────────────────

def test_health_capabilities_and_hardware(whisper_qnn_app, whisper_qnn_environment):
    assert whisper_qnn_app.get("/health").status_code == 200
    cap = whisper_qnn_app.get("/capabilities").json()
    assert cap["tasks_advertised"] == ["audio_transcription"]
    assert cap["permissions"]["network_egress"] == [] and cap["permissions"]["gpu"] is False
    assert cap["model"]["fingerprint"].startswith("sha256:")
    hw = whisper_qnn_app.get("/hardware/evaluation").json()
    assert hw["verdict"] == "ok" and "qcs8275" in hw["reasoning"]
    assert hw["details"]["requested_providers"] == "QNNExecutionProvider:htp"
    assert whisper_qnn_environment["made"]["requested"] == "QNNExecutionProvider:htp"


def test_infer_json_base64_is_what_the_agent_sends(whisper_qnn_app, sample_wav):
    body = {"task": "audio_transcription", "audio_b64": base64.b64encode(sample_wav).decode(),
            "language": "en", "vad_filter": False, "beam_size": 1}
    r = whisper_qnn_app.post("/infer", json=body)
    assert r.status_code == 200, r.text
    resp = InferResponse.model_validate(r.json())
    asr = AsrResult.model_validate(resp.result)
    assert asr.transcript == "Hello world." and asr.language == "en"
    assert [s.text for s in asr.segments] == ["Hello world."]
    assert asr.segments[0].start_ms == 0 and asr.segments[0].end_ms == 2000
    res = resp.result
    assert 0.0 <= res["no_speech_prob"] <= 1.0 and res["duration_seconds"] == 2.0
    assert res["model"] == "whisper_small_quantized-w8a16" and res["decode_steps"] == 7
    assert "encoder_ms" in res and "decoder_ms" in res


def test_infer_multipart_wav(whisper_qnn_app, sample_wav):
    r = whisper_qnn_app.post("/infer", files={"audio": ("clip.wav", sample_wav, "audio/wav")})
    assert r.status_code == 200, r.text
    assert r.json()["result"]["transcript"] == "Hello world."


def test_translation_is_refused_and_names_whisper(whisper_qnn_app, sample_wav):
    r = whisper_qnn_app.post("/infer", files={"audio": ("c.wav", sample_wav, "audio/wav")},
                             data={"params": json.dumps({"task": "audio_translation"})})
    assert r.status_code == 400
    env = FailureEnvelope.model_validate(r.json())
    assert env.error.code == "unsupported_task" and "whisper" in env.error.message


def test_missing_audio_is_malformed_input(whisper_qnn_app):
    r = whisper_qnn_app.post("/infer", data={"params": "{}"}, files={"_marker": ("", b"", "text/plain")})
    assert r.status_code == 400
    assert FailureEnvelope.model_validate(r.json()).error.code == "malformed_input"


def test_missing_assets_make_the_adapter_say_so(tmp_path, monkeypatch):
    import adapters.whisper_qnn.service as svc

    s = svc.WhisperQnnService(str(tmp_path / "nowhere"))
    s.load()
    assert not s.is_ready()
    hw = s.hardware_evaluation()
    assert hw.verdict.value == "blocked" and "encoder.onnx" in hw.reasoning


def test_a_host_without_the_hexagon_is_blocked_not_cpu(tmp_path, monkeypatch):
    from tests._whisper_qnn_fixtures import write_assets
    import adapters.whisper_qnn.service as svc

    write_assets(tmp_path / "a")

    def refuse(path, requested):
        raise RuntimeError("QNN HTP needs /dev/fastrpc-cdsp in the container")

    s = svc.WhisperQnnService(str(tmp_path / "a"), session_factory=refuse, tokenizer_factory=lambda p: FakeTokenizer())
    s.load()
    assert not s.is_ready()
    assert "fastrpc" in s.hardware_evaluation().reasoning


def test_a_pcm_wav_at_another_rate_needs_no_ffmpeg(whisper_qnn_app, monkeypatch):
    """The conformance kit probes with an 8 kHz WAV and CI has no ffmpeg:
    a PCM WAV at any rate is resampled in numpy."""
    import shutil

    import adapters.whisper_qnn.audio as audio

    monkeypatch.setattr(shutil, "which", lambda name: None)
    import io
    import wave

    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(8000)
        w.writeframes(b"\x00\x00" * 8000)
    r = whisper_qnn_app.post("/infer", files={"audio": ("c.wav", buf.getvalue(), "audio/wav")})
    assert r.status_code == 200, r.text
    assert r.json()["result"]["duration_seconds"] == 1.0
    x = audio.resample_linear(np.arange(8, dtype=np.float32), 8000, 16000)
    assert x.shape == (16,) and abs(float(x[2]) - 1.0) < 1e-6
