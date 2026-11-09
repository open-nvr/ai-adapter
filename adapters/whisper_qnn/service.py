"""
Whisper-Small (w8a16) on the Qualcomm Hexagon NPU — the ASR service behind
``adapters/whisper_qnn/main.py``.

Why this adapter exists: on the Ventuno Q (QCS8275) the camera agent's
speech-to-text ran Parakeet on the CPU at 0.7–2.3 s an utterance, with no
confidence signal to gate noise and the odd hallucinated sentence in
another language. Qualcomm publishes Whisper-Small compiled for exactly
this chip (AI Hub, ``whisper_small_quantized``, QNN context binaries for
HTP v75 — the asset's own metadata names the "Arduino VENTUNO Q" as its
reference device): 56 ms encoder + 9 ms a token on their numbers, and a
decoder whose first step yields a no-speech probability.

What it is NOT: a general Whisper. The context binaries run only on the
Hexagon of the chip they were compiled for, through ONNX Runtime's QNN
execution provider. On any other host ``/health`` says ``error`` and
``/hardware/evaluation`` says ``blocked`` with the reason; the CPU
alternatives are the ``whisper`` (faster-whisper) and ``parakeet`` adapters.

Contract: §5.3 ``AsrResult`` — ``transcript``, ``language``, one ``segment``
spanning the clip — plus ``no_speech_prob`` (top level and on the segment),
``duration_seconds``, ``model``, and the engine's ``decode_steps``,
``encoder_ms``, ``decoder_ms`` so a slow turn can be blamed on the right half.
``audio_transcription`` only; ``audio_translation`` is refused and names the
whisper adapter.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import threading
import time
from datetime import datetime, timezone
from typing import Any, Callable

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

from .audio import AudioError, load_audio_16k
from .engine import DEFAULT_MAX_TOKENS, WhisperQnnEngine

logger = logging.getLogger(__name__)

MODEL_FRAMEWORK: str = "onnxruntime-qnn"
MODEL_NAME: str = "whisper-small-quantized"
ASSETS_DIR_ENV: str = "WHISPER_QNN_ASSETS_DIR"
MAX_TOKENS_ENV: str = "WHISPER_QNN_MAX_TOKENS"
PROVIDERS_ENV: str = "ADAPTER_ONNX_PROVIDERS"
#: The only provider these binaries run on. The operator may name another
#: QNN backend (``QNNExecutionProvider:gpu``) for experiments; CPU is refused.
DEFAULT_PROVIDERS: str = "QNNExecutionProvider:htp"
MAX_AUDIO_BYTES: int = 25 * 1024 * 1024
#: What the asset zip unpacks to, plus the two HuggingFace files the
#: Dockerfile adds beside them.
REQUIRED_FILES: tuple[str, ...] = (
    "encoder.onnx", "encoder.bin", "decoder.onnx", "decoder.bin", "metadata.json",
    "tokenizer.json", "preprocessor_config.json",
)

SessionFactory = Callable[[str, str], Any]
TokenizerFactory = Callable[[str], Any]


def build_session(model_path: str, requested: str) -> Any:
    """An ONNX Runtime session with the QNN plugin provider attached by device.

    The SDK's ``plugin_session`` compiles a graph and caches the EP context
    it dumps; these models ARE EP contexts already, so the plugin is attached
    the same way (``add_provider_for_devices``) without the dump. Refuses,
    with the reason, when the HTP is not reachable from this container — a
    context binary that cannot find its device would otherwise be reported as
    loaded and run nothing.
    """
    import onnxruntime as ort
    from opennvr_adapter_sdk.providers import (
        FIXED_SHAPES_KEY,
        QNN_EP,
        plugin_device_missing,
        provider_name,
        provider_options,
        register_plugin_providers,
        session_providers,
    )

    usable = register_plugin_providers(requested)
    entries = [e for e in session_providers(requested) if provider_name(e) == QNN_EP]
    if QNN_EP not in usable or not entries:
        raise RuntimeError(
            "Whisper's QNN context binaries run only on the Qualcomm Hexagon through "
            f"{QNN_EP}; this build has no usable QNN provider (requested {requested!r}). "
            "Use the qualcomm image flavour on a Qualcomm host, or the whisper/parakeet adapters on CPU."
        )
    entry = entries[0]
    missing = plugin_device_missing(QNN_EP, entry)
    if missing:
        raise RuntimeError(missing)
    devices = [d for d in ort.get_ep_devices() if d.ep_name == QNN_EP]
    if not devices:
        raise RuntimeError(f"{QNN_EP} reports no device on this host")
    popts = provider_options(entry)
    popts.pop(FIXED_SHAPES_KEY, None)
    opts = ort.SessionOptions()
    opts.add_provider_for_devices(devices, popts)
    return ort.InferenceSession(model_path, sess_options=opts)


def load_tokenizer(path: str) -> Any:
    from tokenizers import Tokenizer

    return Tokenizer.from_file(path)


class WhisperQnnService(AdapterService):
    def __init__(
        self,
        assets_dir: str | None = None,
        *,
        session_factory: SessionFactory | None = None,
        tokenizer_factory: TokenizerFactory | None = None,
    ) -> None:
        self._assets_dir = (assets_dir or os.environ.get(ASSETS_DIR_ENV, "").strip()
                            or os.path.join(MODEL_WEIGHTS_DIR, "whisper-qnn"))
        self._requested = os.environ.get(PROVIDERS_ENV, "").strip() or DEFAULT_PROVIDERS
        self._max_tokens = int(os.environ.get(MAX_TOKENS_ENV, "") or DEFAULT_MAX_TOKENS)
        self._session_factory = session_factory or build_session
        self._tokenizer_factory = tokenizer_factory or load_tokenizer
        self._engine: WhisperQnnEngine | None = None
        self._metadata: dict[str, Any] = {}
        self._load_state = HealthStatus.LOADING
        self._load_error: str | None = None
        self._fingerprint_cache: str | None = None
        self._lock = threading.Lock()
        self._started_at = datetime.now(timezone.utc)

    # ── AdapterService impl ────────────────────────────────────────

    def load(self) -> None:
        with self._lock:
            if self._load_state == HealthStatus.OK:
                return
            try:
                missing = [f for f in REQUIRED_FILES if not os.path.isfile(os.path.join(self._assets_dir, f))]
                if missing:
                    raise FileNotFoundError(
                        f"Whisper QNN assets missing from {self._assets_dir}: {', '.join(missing)} "
                        f"(set {ASSETS_DIR_ENV}; the image bakes them at build)")
                with open(os.path.join(self._assets_dir, "metadata.json"), encoding="utf-8") as fh:
                    self._metadata = json.load(fh)
                with open(os.path.join(self._assets_dir, "preprocessor_config.json"), encoding="utf-8") as fh:
                    mel_filters = json.load(fh)["mel_filters"]
                logger.info("WhisperQnnService: loading %s from %s providers=%s",
                            self._metadata.get("model_id", MODEL_NAME), self._assets_dir, self._requested)
                encoder = self._session_factory(os.path.join(self._assets_dir, "encoder.onnx"), self._requested)
                decoder = self._session_factory(os.path.join(self._assets_dir, "decoder.onnx"), self._requested)
                tokenizer = self._tokenizer_factory(os.path.join(self._assets_dir, "tokenizer.json"))
                self._engine = WhisperQnnEngine(encoder, decoder, self._metadata, tokenizer, mel_filters,
                                                max_tokens=self._max_tokens)
                self._fingerprint_cache = self._compute_fingerprint()
                self._load_state = HealthStatus.OK
                self._load_error = None
                logger.info("WhisperQnnService ready: %s %s on %s, fingerprint=%s",
                            self._metadata.get("model_id"), self._metadata.get("precision"),
                            self._requested, self._fingerprint_cache)
            except Exception as exc:  # noqa: BLE001
                self._load_state = HealthStatus.ERROR
                self._load_error = str(exc)
                self._engine = None
                logger.exception("WhisperQnnService failed to load from %s", self._assets_dir)

    def is_ready(self) -> bool:
        return self._load_state == HealthStatus.OK

    def health_status(self) -> HealthStatus | None:
        return self._load_state

    def fingerprint(self) -> str | None:
        return self._fingerprint_cache

    def model_info(self) -> ModelInfo:
        version = f"{self._metadata.get('model_id', MODEL_NAME)}/{self._metadata.get('precision', 'w8a16')}"
        return ModelInfo(
            name=MODEL_NAME,
            framework=MODEL_FRAMEWORK,
            version=version,
            modalities_in=["audio"],
            modalities_out=["text"],
            fingerprint=self.fingerprint(),
        )

    def hardware_evaluation(self) -> HardwareEvaluationResponse:
        chip = (self._metadata.get("chipset_attributes") or {}).get("name", "qualcomm")
        if self._load_state == HealthStatus.OK:
            verdict = HardwareVerdict.OK
            reasoning = (f"Whisper-Small {self._metadata.get('precision', 'w8a16')} context binaries "
                         f"for {chip} loaded on {self._requested}.")
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
                "accelerator": "Qualcomm Hexagon NPU (QNN HTP)",
                "requested_providers": self._requested,
                "chipset": chip,
                "tool_versions": self._metadata.get("tool_versions"),
                "cpu_count": os.cpu_count() or 0,
                "platform": platform.platform(),
                "assets_dir": self._assets_dir,
                "adapter_uptime_seconds": int((datetime.now(timezone.utc) - self._started_at).total_seconds()),
            },
        )

    def infer(self, payload: dict[str, Any]) -> InferResponse:
        audio = payload.get(BODY_BYTES_KEY)
        if not isinstance(audio, (bytes, bytearray)) or not audio:
            raise _bad("malformed_input", "Audio bytes are required.")
        if len(audio) > MAX_AUDIO_BYTES:
            raise _bad("malformed_input", f"Audio exceeds {MAX_AUDIO_BYTES}-byte limit ({len(audio)} received).", 413)
        if self._load_state != HealthStatus.OK or self._engine is None:
            raise ServiceError(
                ErrorCategory.MODEL_ERROR,
                code="weights_missing" if self._load_state == HealthStatus.ERROR else "whisper_qnn.model_loading",
                message=self._load_error or "Model still loading.",
                transient=(self._load_state == HealthStatus.LOADING),
                http_status=503,
                retry_after_ms=2000 if self._load_state == HealthStatus.LOADING else None,
            )
        params = {k: v for k, v in payload.items() if k != BODY_BYTES_KEY}
        task = params.get("task", "audio_transcription")
        if task == "audio_translation":
            raise _bad("unsupported_task",
                       "This build transcribes only; for audio_translation use the whisper adapter.")
        if task != "audio_transcription":
            raise _bad("malformed_input", f"Unknown task {task!r}; expected audio_transcription.")
        language = str(params.get("language") or "en")

        t0 = time.perf_counter()
        try:
            wave, duration_s = load_audio_16k(bytes(audio))
        except AudioError as exc:
            raise _bad("malformed_input", str(exc)) from exc
        try:
            with self._lock:                      # one decode at a time on the HTP
                out = self._engine.transcribe(wave, language=language)
        except Exception as exc:  # noqa: BLE001
            raise ServiceError(
                ErrorCategory.MODEL_ERROR, code="inference_failed",
                message=f"Whisper raised during decoding: {exc}",
                transient=True, http_status=500, retry_after_ms=1000,
            ) from exc

        duration_ms = int(round(duration_s * 1000))
        segments = [AsrSegment(start_ms=0, end_ms=duration_ms, text=out.text,
                               no_speech_prob=round(out.no_speech_prob, 4))] if out.text else []
        asr = AsrResult(transcript=out.text, language=language, segments=segments)
        body = asr.model_dump(mode="json")
        body["no_speech_prob"] = round(out.no_speech_prob, 4)
        body["duration_seconds"] = round(duration_s, 3)
        body["model"] = f"{self._metadata.get('model_id', MODEL_NAME)}-{self._metadata.get('precision', 'w8a16')}"
        body["decode_steps"] = out.steps
        body["encoder_ms"] = round(out.encoder_ms, 1)
        body["decoder_ms"] = round(out.decoder_ms, 1)
        if out.chunks > 1:
            body["chunks"] = out.chunks
        return InferResponse(
            status="ok",
            model_name="whisper-qnn-asr",
            model_version=f"{MODEL_FRAMEWORK}/{body['model']}",
            inference_ms=int((time.perf_counter() - t0) * 1000),
            result=body,
        )

    # ── helpers ────────────────────────────────────────────────────

    def _compute_fingerprint(self) -> str:
        """sha256 over the asset file names and sizes plus metadata.json's
        bytes: the context binaries are 340 MB and are not hashed."""
        h = hashlib.sha256()
        for name in sorted(os.listdir(self._assets_dir)):
            p = os.path.join(self._assets_dir, name)
            if os.path.isfile(p):
                h.update(f"{name}:{os.path.getsize(p)}".encode())
        with open(os.path.join(self._assets_dir, "metadata.json"), "rb") as fh:
            h.update(fh.read())
        return f"sha256:{h.hexdigest()}"


def _bad(code: str, message: str, status: int = 400) -> ServiceError:
    return ServiceError(ErrorCategory.MODEL_ERROR, code=code, message=message,
                        transient=False, http_status=status)
