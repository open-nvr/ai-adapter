"""Parakeet adapter: the contract surface and the §5.3 result, with the
model faked (tests/_parakeet_service_fixtures.py)."""
from __future__ import annotations

import base64
import json

from opennvr_adapter_sdk.contract import AsrResult, FailureEnvelope, InferResponse

from tests._parakeet_service_fixtures import (  # noqa: F401
    parakeet_app, parakeet_environment, sample_wav, silent_wav16k,
)


def test_health_and_capabilities(parakeet_app, parakeet_environment):
    assert parakeet_app.get("/health").status_code == 200
    cap = parakeet_app.get("/capabilities").json()
    assert cap["tasks_advertised"] == ["audio_transcription"]
    assert cap["permissions"]["network_egress"] == []
    assert cap["permissions"]["gpu"] is False
    assert cap["model"]["fingerprint"].startswith("sha256:")
    # Loaded from the baked directory, by path, int8, CPU.
    load = parakeet_environment["onnx_asr"].last_load
    assert load["path"] == str(parakeet_environment["weights"])
    assert load["quantization"] == "int8" and load["providers"] == ["CPUExecutionProvider"]


def test_infer_multipart_wav_returns_asr_result(parakeet_app, sample_wav):
    r = parakeet_app.post("/infer", files={"audio": ("clip.wav", sample_wav, "audio/wav")})
    assert r.status_code == 200, r.text
    result = InferResponse.model_validate(r.json()).result
    asr = AsrResult.model_validate(result)
    assert asr.transcript == "Hello world. Thanks"
    assert asr.language == "unknown"
    assert [s.text for s in asr.segments] == ["Hello world.", "Thanks"]
    assert asr.segments[0].start_ms == 100 and asr.segments[0].end_ms == 1400
    assert asr.segments[1].start_ms == 1400 and asr.segments[1].end_ms == 2000
    assert result["duration_seconds"] == 2.0 and result["model"] == "nemo-parakeet-tdt-0.6b-v3"


def test_infer_json_base64_is_what_the_agent_sends(parakeet_app, sample_wav):
    """camera-agent's transcribe(): JSON, audio_b64, task, language, the
    whisper params. Parakeet ignores beam_size/vad_filter and echoes the
    language."""
    body = {"task": "audio_transcription", "audio_b64": base64.b64encode(sample_wav).decode(),
            "language": "en", "vad_filter": False, "beam_size": 1}
    r = parakeet_app.post("/infer", json=body)
    assert r.status_code == 200, r.text
    result = r.json()["result"]
    assert result["transcript"] == "Hello world. Thanks" and result["language"] == "en"
    # The 16 kHz mono PCM WAV went to the model as samples, no ffmpeg.
    from tests._parakeet_service_fixtures import _FakeModel
    assert _FakeModel.last_call["sample_rate"] == 16000
    assert _FakeModel.last_call["samples"] == 32000


def test_translation_is_refused_and_names_whisper(parakeet_app, sample_wav):
    r = parakeet_app.post("/infer", files={"audio": ("c.wav", sample_wav, "audio/wav")},
                          data={"params": json.dumps({"task": "audio_translation"})})
    assert r.status_code == 400
    env = FailureEnvelope.model_validate(r.json())
    assert env.error.code == "unsupported_task" and "whisper" in env.error.message


def test_missing_audio_is_malformed_input(parakeet_app):
    r = parakeet_app.post("/infer", data={"params": "{}"}, files={"_marker": ("", b"", "text/plain")})
    assert r.status_code == 400
    assert FailureEnvelope.model_validate(r.json()).error.code == "malformed_input"


def test_segments_split_on_long_gaps_and_a_bare_string_is_one_segment():
    from adapters.parakeet.service import _segments
    from tests._parakeet_service_fixtures import _FakeResult
    res = _FakeResult("one two three", [" one", " two", " three"], [0.0, 0.3, 2.0])
    segs = _segments(res, 2.5)
    assert [s.text for s in segs] == ["one two", "three"]
    assert segs[0].end_ms == 300 and segs[1].start_ms == 2000 and segs[1].end_ms == 2500
    assert [s.text for s in _segments("just text", 1.0)] == ["just text"]
    assert _segments(_FakeResult("", [], []), 1.0) == []


def test_an_abbreviation_dot_does_not_split_a_segment():
    from adapters.parakeet.service import _segments
    from tests._parakeet_service_fixtures import _FakeResult
    res = _FakeResult("after 6 p. m. Thanks", [" after", " 6", " p", ".", " m", ".", " Thanks"],
                      [0.0, 0.2, 0.4, 0.5, 0.6, 0.7, 1.2])
    assert [s.text for s in _segments(res, 1.5)] == ["after 6 p. m.", "Thanks"]


def test_a_pcm_wav_at_another_rate_needs_no_ffmpeg(parakeet_app, monkeypatch):
    """The conformance kit probes with an 8 kHz WAV and CI has no ffmpeg;
    onnx-asr resamples a PCM waveform itself, so no conversion is run."""
    import shutil
    import wave, io
    monkeypatch.setattr(shutil, "which", lambda name: None)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(2); w.setsampwidth(2); w.setframerate(8000)
        w.writeframes(b"\x00\x00\x00\x00" * 4000)
    r = parakeet_app.post("/infer", files={"audio": ("c.wav", buf.getvalue(), "audio/wav")})
    assert r.status_code == 200, r.text
    from tests._parakeet_service_fixtures import _FakeModel
    assert _FakeModel.last_call["sample_rate"] == 8000 and _FakeModel.last_call["samples"] == 4000
    assert r.json()["result"]["duration_seconds"] == 0.5
