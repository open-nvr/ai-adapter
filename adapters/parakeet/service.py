"""
Parakeet ASR service — the model behind ``adapters/parakeet/main.py``.

Speech in (a PCM WAV straight to the model, which resamples; anything
else through ffmpeg first), §5.3 ``AsrResult`` out: ``transcript``, ``language``,
``segments`` with ``start_ms`` / ``end_ms``, plus the extras the whisper
adapter also carries (``duration_seconds``, ``model``).

Where it differs from whisper, honestly:

* One task, ``audio_transcription``. Parakeet does not translate;
  ``audio_translation`` is refused with a 400 that names the whisper
  adapter.
* No ``no_speech_prob`` on segments: a transducer has no such number.
  The agent logs it when present and never depends on it.
* Segments come from the model's token timestamps, split at sentence
  punctuation; a clip with no punctuation is one segment.
* v3 detects the language itself (25 European languages, no Hindi); a
  ``language`` param is echoed back, not enforced.
"""
from __future__ import annotations

import hashlib
import io
import logging
import os
import platform
import shutil
import subprocess
import tempfile
import threading
import time
import wave
from datetime import datetime, timezone
from typing import Any

from app.config import MODEL_WEIGHTS_DIR
from opennvr_adapter_sdk import (
    BODY_BYTES_KEY,
    AdapterService,
    ErrorCategory,
    HardwareEvaluationResponse,
    HardwareVerdict,
    HealthStatus,
    InferResponse,
    ModelInfo,
    ServiceError,
)
from opennvr_adapter_sdk.contract import AsrResult, AsrSegment

logger = logging.getLogger(__name__)

MODEL_FRAMEWORK: str = "onnx-asr"
#: The onnx-asr model name; v3 is multilingual, v2 English-only and a
#: hair more accurate on English (1.9 % vs 1.4 % WER measured — noise).
DEFAULT_MODEL_NAME: str = "nemo-parakeet-tdt-0.6b-v3"
MODEL_NAME_ENV: str = "PARAKEET_MODEL"
#: Where the baked weights live (the Dockerfile fills it). Empty or
#: missing, onnx-asr downloads from HuggingFace — a development
#: convenience, not the shipped posture.
WEIGHTS_DIR_ENV: str = "PARAKEET_WEIGHTS_DIR"
MAX_AUDIO_BYTES: int = 25 * 1024 * 1024
SAMPLE_RATE: int = 16000  # what ffmpeg converts to
#: A gap between tokens longer than this also ends a segment, so a
#: pause reads as one in the transcript's timing.
SEGMENT_GAP_S: float = 0.8
_SENTENCE_END = (".", "?", "!")


class ParakeetService(AdapterService):
    def __init__(self, model_name: str | None = None, weights_dir: str | None = None) -> None:
        self._model_name = model_name or os.environ.get(MODEL_NAME_ENV, "").strip() or DEFAULT_MODEL_NAME
        self._weights_dir = weights_dir or os.environ.get(WEIGHTS_DIR_ENV, "").strip() \
            or os.path.join(MODEL_WEIGHTS_DIR, "parakeet")
        self._model: Any = None
        self._load_state = HealthStatus.LOADING
        self._load_error: str | None = None
        self._fingerprint_cache: str | None = None
        self._lock = threading.Lock()
        self._started_at = datetime.now(timezone.utc)
        self._providers: list[str] = []

    # ── AdapterService impl ────────────────────────────────────────

    def load(self) -> None:
        with self._lock:
            if self._load_state == HealthStatus.OK:
                return
            try:
                import onnx_asr
                import onnxruntime as ort

                path = self._weights_dir if _has_weights(self._weights_dir) else None
                # Providers from ADAPTER_ONNX_PROVIDERS resolved against this
                # build (OpenNVR Accelerate flavours): unset keeps what it
                # always was, CUDA when the build has it, else CPU.
                from opennvr_adapter_sdk.providers import build_with_fallback, session_providers
                requested = session_providers(
                    None if os.environ.get("ADAPTER_ONNX_PROVIDERS", "").strip() else
                    [p for p in ("CUDAExecutionProvider",) if p in ort.get_available_providers()])
                logger.info("ParakeetService: loading %s from %s providers=%s",
                            self._model_name, path or "huggingface", requested)
                self._model, providers, self._fallback_error = build_with_fallback(
                    lambda provs: onnx_asr.load_model(
                        self._model_name, path, quantization="int8", providers=provs,
                    ).with_timestamps(),
                    requested, adapter="parakeet")
                self._providers = providers
                self._fingerprint_cache = self._compute_fingerprint()
                self._load_state = HealthStatus.OK
                self._load_error = None
                logger.info("ParakeetService ready: model=%s fingerprint=%s",
                            self._model_name, self._fingerprint_cache)
            except Exception as exc:  # noqa: BLE001
                self._load_state = HealthStatus.ERROR
                self._load_error = str(exc)
                logger.exception("ParakeetService failed to load %s", self._model_name)

    def is_ready(self) -> bool:
        return self._load_state == HealthStatus.OK

    def fingerprint(self) -> str | None:
        return self._fingerprint_cache

    def model_info(self) -> ModelInfo:
        return ModelInfo(
            name=self._model_name,
            framework=MODEL_FRAMEWORK,
            version=f"{MODEL_FRAMEWORK}/{self._model_name}",
            modalities_in=["audio"],
            modalities_out=["text"],
            fingerprint=self.fingerprint(),
        )

    def hardware_evaluation(self) -> HardwareEvaluationResponse:
        if self._load_state == HealthStatus.OK:
            from opennvr_adapter_sdk.providers import provider_name
            names = [provider_name(p) for p in self._providers]
            on_gpu = names[:1] not in ([], ["CPUExecutionProvider"])
            verdict = HardwareVerdict.OK
            reasoning = (f"Parakeet loaded on {names[0]}." if on_gpu else
                         "Parakeet loaded on CPU: a few hundred milliseconds an utterance on four cores.")
        elif self._load_state == HealthStatus.LOADING:
            verdict, reasoning = HardwareVerdict.WARN, "Model still loading."
        else:
            verdict, reasoning = HardwareVerdict.BLOCKED, f"Model failed to load: {self._load_error}"
        return HardwareEvaluationResponse(
            verdict=verdict,
            reasoning=reasoning,
            checked_at=datetime.now(timezone.utc),
            details={
                "gpu_required": False,
                "providers": self._providers,
                "cpu_count": os.cpu_count() or 0,
                "platform": platform.platform(),
                "model": self._model_name,
                "weights_dir": self._weights_dir,
                "adapter_uptime_seconds": int((datetime.now(timezone.utc) - self._started_at).total_seconds()),
            },
        )

    def infer(self, payload: dict[str, Any]) -> InferResponse:
        audio = payload.get(BODY_BYTES_KEY)
        if not isinstance(audio, (bytes, bytearray)) or not audio:
            raise _bad("malformed_input", "Audio bytes are required.")
        if len(audio) > MAX_AUDIO_BYTES:
            raise _bad("malformed_input", f"Audio exceeds {MAX_AUDIO_BYTES}-byte limit ({len(audio)} received).", 413)
        if self._load_state != HealthStatus.OK:
            raise ServiceError(
                ErrorCategory.MODEL_ERROR,
                code="weights_missing" if self._load_state == HealthStatus.ERROR else "parakeet.model_loading",
                message=self._load_error or "Model still loading.",
                transient=(self._load_state == HealthStatus.LOADING),
                http_status=503,
                retry_after_ms=2000 if self._load_state == HealthStatus.LOADING else None,
            )
        params = {k: v for k, v in payload.items() if k != BODY_BYTES_KEY}
        task = params.get("task", "audio_transcription")
        if task == "audio_translation":
            raise _bad("unsupported_task",
                       "Parakeet transcribes only; for audio_translation use the whisper adapter.")
        if task != "audio_transcription":
            raise _bad("malformed_input", f"Unknown task {task!r}; expected audio_transcription.")
        language = params.get("language")

        t0 = time.perf_counter()
        waveform, rate, duration_s = _load_audio(bytes(audio))
        try:
            res = self._model.recognize(waveform, sample_rate=rate)
        except Exception as exc:  # noqa: BLE001
            raise ServiceError(
                ErrorCategory.MODEL_ERROR, code="inference_failed",
                message=f"Parakeet raised during recognition: {exc}",
                transient=True, http_status=500, retry_after_ms=1000,
            ) from exc

        segments = _segments(res, duration_s)
        asr = AsrResult(
            transcript=" ".join(s.text for s in segments).strip(),
            language=str(language) if language else "unknown",
            segments=segments,
        )
        body = asr.model_dump(mode="json")
        body["duration_seconds"] = round(duration_s, 3)
        body["model"] = self._model_name
        return InferResponse(
            status="ok",
            model_name="parakeet-asr",
            model_version=f"{MODEL_FRAMEWORK}/{self._model_name}",
            inference_ms=int((time.perf_counter() - t0) * 1000),
            result=body,
        )

    # ── helpers ────────────────────────────────────────────────────

    def _compute_fingerprint(self) -> str:
        """sha256 over the weights' file names and sizes plus config.json
        when baked; over the model name alone when they come from the
        hub. The encoder is 650 MB, so its bytes are not hashed."""
        h = hashlib.sha256()
        if _has_weights(self._weights_dir):
            for name in sorted(os.listdir(self._weights_dir)):
                p = os.path.join(self._weights_dir, name)
                if not os.path.isfile(p):
                    continue
                h.update(f"{name}:{os.path.getsize(p)}".encode())
                if name == "config.json":
                    h.update(open(p, "rb").read())
            return f"sha256:{h.hexdigest()}"
        h.update(self._model_name.encode())
        return f"sha256-id:{h.hexdigest()}"


def _has_weights(d: str) -> bool:
    try:
        return any(n.endswith(".onnx") for n in os.listdir(d))
    except OSError:
        return False


def _bad(code: str, message: str, status: int = 400) -> ServiceError:
    return ServiceError(ErrorCategory.TRANSPORT_ERROR, code=code, message=message,
                        transient=False, http_status=status)


#: The rates onnx-asr resamples from itself; a PCM WAV at one of these
#: needs no ffmpeg.
_NATIVE_RATES = (8000, 11025, 16000, 22050, 24000, 32000, 44100, 48000)


def _load_audio(audio: bytes) -> tuple[Any, int, float]:
    """``(waveform float32 mono, sample rate, duration s)``. A PCM WAV is
    read with the standard library and handed to the model as samples
    (onnx-asr resamples from any common rate), so the agent's 16 kHz
    WAV and the conformance kit's 8 kHz probe both work with no ffmpeg
    in the image; anything else (a browser's webm, an mp3) is converted
    by ffmpeg, which the published image carries."""
    import numpy as np

    if audio[:4] == b"RIFF" and audio[8:12] == b"WAVE":
        try:
            with wave.open(io.BytesIO(audio), "rb") as w:
                ch, width, rate, n = w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()
                pcm = w.readframes(n) if w.getcomptype() == "NONE" else None
        except wave.Error:
            pcm = None
        if pcm is not None and width in (1, 2, 4) and rate in _NATIVE_RATES:
            if width == 1:
                x = (np.frombuffer(pcm, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
            elif width == 2:
                x = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
            else:
                x = np.frombuffer(pcm, dtype="<i4").astype(np.float32) / 2147483648.0
            if ch > 1:
                x = x.reshape(-1, ch).mean(axis=1)
            return np.ascontiguousarray(x), rate, len(x) / float(rate)
    if shutil.which("ffmpeg") is None:
        raise _bad("malformed_input", "Audio must be a PCM WAV (ffmpeg is not installed to convert anything else).")
    tmp_dir = tempfile.mkdtemp(prefix="parakeet-svc-")
    try:
        src = os.path.join(tmp_dir, "in.bin"); out = os.path.join(tmp_dir, "in16k.wav")
        with open(src, "wb") as fh:
            fh.write(audio)
        proc = subprocess.run(
            ["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", src,
             "-ac", "1", "-ar", str(SAMPLE_RATE), "-sample_fmt", "s16", "-f", "wav", out],
            capture_output=True, text=True,
        )
        if proc.returncode != 0 or not os.path.exists(out):
            raise _bad("malformed_input", f"Audio could not be decoded: {proc.stderr.strip()[:200]}")
        with open(out, "rb") as fh:
            converted = fh.read()
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    return _load_audio(converted)


def _segments(res: Any, duration_s: float) -> list[AsrSegment]:
    """Token timestamps → sentence segments. ``res`` is onnx-asr's
    ``TimestampedResult`` (``text``, ``timestamps``, ``tokens``); a bare
    string (a model without timestamps) is one segment."""
    text = (getattr(res, "text", None) if not isinstance(res, str) else res) or ""
    tokens = list(getattr(res, "tokens", None) or [])
    stamps = list(getattr(res, "timestamps", None) or [])
    text = text.strip()
    if not text:
        return []
    if not tokens or len(tokens) != len(stamps):
        return [AsrSegment(start_ms=0, end_ms=int(round(duration_s * 1000)), text=text)]
    out: list[AsrSegment] = []
    buf: list[str] = []
    start = stamps[0]
    for i, (tok, ts) in enumerate(zip(tokens, stamps)):
        if buf and ts - stamps[i - 1] > SEGMENT_GAP_S:
            out.append(AsrSegment(start_ms=int(round(start * 1000)), end_ms=int(round(stamps[i - 1] * 1000)),
                                  text="".join(buf).strip()))
            buf, start = [], ts
        buf.append(tok)
        # A sentence end followed by a capitalised word. Not every dot:
        # "6 p. m." is one sentence, and the model emits " p" "." " m" "."
        # as tokens; the next token after an abbreviation's dot is
        # lowercase, the one after a real sentence end is not.
        nxt = tokens[i + 1].strip() if i + 1 < len(tokens) else ""
        if tok.rstrip().endswith(_SENTENCE_END) and nxt and nxt[0].isupper():
            out.append(AsrSegment(start_ms=int(round(start * 1000)), end_ms=int(round(stamps[i + 1] * 1000)),
                                  text="".join(buf).strip()))
            buf, start = [], stamps[i + 1]
    if buf:
        # The clip's duration, unless the model's own timestamps run past
        # it (a probe shorter than the tokens it was answered with).
        end = max(duration_s, stamps[-1] + 0.1, start)
        out.append(AsrSegment(start_ms=int(round(start * 1000)), end_ms=int(round(end * 1000)),
                              text="".join(buf).strip()))
    return [s for s in out if s.text]
