"""
SigLIP2 embeddings on the Qualcomm Hexagon — the ``embed`` task, contract
v1 (the same request and answer as the clip adapter, on the NPU).

The model is Google's ``siglip2-base-patch16-224`` (Apache-2.0) as
Qualcomm AI Hub exports it: two quantized ONNX graphs (w8a16, opset 21),
``image_encoder`` and ``text_encoder``, a 768-dimensional shared space.
They are not precompiled: ONNX Runtime's QNN execution provider compiles
each graph for the HTP at the first session build and the SDK caches the
compiled context beside the model, so every later start loads in
seconds. Measured by Qualcomm on the Dragonwing IQ-8275 (the Ventuno Q's
chip): image 5.1 ms, text 2.1 ms an inference.

Request (JSON, exactly one of)::

    {"task": "embed", "text": "a red truck at the loading bay"}
    {"task": "embed", "frame_b64": "...", "camera_id": "cam3"}

Answer: ``result.embedding`` (768 floats, unit length), ``dim``,
``model``, ``modality``, ``normalized: true`` — what core's
``embed_enrichment`` and search's query vector read.

The text tower carries the 256k-piece Gemma vocabulary (1.1 GB in the
export); ``SIGLIP_QNN_TEXT_PROVIDERS`` can keep it on the CPU
(``CPUExecutionProvider``) where the HTP compile of it is too slow or
fails, with the image tower still on the NPU — the two towers share the
space whichever device computes them.
"""
from __future__ import annotations

import base64
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

from .engine import SiglipQnnEngine
from .features import EMBED_DIM

logger = logging.getLogger(__name__)

MODEL_FRAMEWORK: str = "onnxruntime-qnn"
MODEL_NAME: str = "siglip2-base-patch16-224"
ASSETS_DIR_ENV: str = "SIGLIP_QNN_ASSETS_DIR"
PROVIDERS_ENV: str = "ADAPTER_ONNX_PROVIDERS"
TEXT_PROVIDERS_ENV: str = "SIGLIP_QNN_TEXT_PROVIDERS"
DEFAULT_PROVIDERS: str = "QNNExecutionProvider:htp"
CPU_PROVIDER: str = "CPUExecutionProvider"
MAX_IMAGE_BYTES: int = 12 * 1024 * 1024
MAX_TEXT_CHARS: int = 2000
REQUIRED_FILES: tuple[str, ...] = (
    "image_encoder.onnx", "image_encoder.data", "text_encoder.onnx", "text_encoder.data",
    "metadata.json", "tokenizer.json",
)

SessionFactory = Callable[[str, str], Any]
TokenizerFactory = Callable[[str], Any]


def build_session(model_path: str, requested: str) -> Any:
    """An ONNX Runtime session for one tower.

    ``CPUExecutionProvider`` → a plain CPU session (the quantized graph
    runs there too, slower). Anything else → the SDK's ``plugin_session``:
    the QNN plugin provider attached by device, the graph compiled for the
    HTP at the first build and the EP context cached beside the model
    (``.opennvr-accel/``), a stale cache rebuilt. Refuses with the reason
    when the HTP is not reachable from this container.
    """
    import onnxruntime as ort

    if requested.strip().lower().startswith("cpu"):
        return ort.InferenceSession(model_path, providers=[CPU_PROVIDER])

    from opennvr_adapter_sdk.providers import (
        QNN_EP,
        plugin_device_missing,
        plugin_session,
        provider_name,
        register_plugin_providers,
        session_providers,
    )

    usable = register_plugin_providers(requested)
    entries = [e for e in session_providers(requested) if provider_name(e) == QNN_EP]
    if QNN_EP not in usable or not entries:
        raise RuntimeError(
            f"SigLIP2 on the NPU needs {QNN_EP}; this build has no usable QNN provider "
            f"(requested {requested!r}). Use the qualcomm image flavour on a Qualcomm host, "
            "or the clip adapter on CPU."
        )
    missing = plugin_device_missing(QNN_EP, entries[0])
    if missing:
        raise RuntimeError(missing)
    return plugin_session(model_path, entries, adapter="siglip-qnn")


def load_tokenizer(path: str) -> Any:
    from tokenizers import Tokenizer

    return Tokenizer.from_file(path)


def _bad(code: str, message: str, status: int = 400) -> ServiceError:
    return ServiceError(ErrorCategory.MODEL_ERROR, code=code, message=message,
                        transient=False, http_status=status)


class SiglipQnnService(AdapterService):
    def __init__(
        self,
        assets_dir: str | None = None,
        *,
        session_factory: SessionFactory | None = None,
        tokenizer_factory: TokenizerFactory | None = None,
    ) -> None:
        self._assets_dir = (assets_dir or os.environ.get(ASSETS_DIR_ENV, "").strip()
                            or os.path.join(MODEL_WEIGHTS_DIR, "siglip2-qnn"))
        self._requested = os.environ.get(PROVIDERS_ENV, "").strip() or DEFAULT_PROVIDERS
        self._text_requested = os.environ.get(TEXT_PROVIDERS_ENV, "").strip() or self._requested
        self._session_factory = session_factory or build_session
        self._tokenizer_factory = tokenizer_factory or load_tokenizer
        self._engine: SiglipQnnEngine | None = None
        self._metadata: dict[str, Any] = {}
        self._load_state: HealthStatus = HealthStatus.LOADING
        self._load_error: str | None = None
        self._fingerprint_cache: str | None = None
        self._text_device: str = self._text_requested
        self._lock = threading.Lock()

    # ── lifecycle ────────────────────────────────────────────────────

    def load(self) -> None:
        with self._lock:
            if self._load_state == HealthStatus.OK:
                return
            try:
                missing = [f for f in REQUIRED_FILES if not os.path.isfile(os.path.join(self._assets_dir, f))]
                if missing:
                    raise FileNotFoundError(
                        f"SigLIP2 QNN assets missing from {self._assets_dir}: {', '.join(missing)} "
                        f"(set {ASSETS_DIR_ENV}; the image bakes them at build)")
                with open(os.path.join(self._assets_dir, "metadata.json"), encoding="utf-8") as fh:
                    self._metadata = json.load(fh)
                logger.info("SiglipQnnService: loading %s %s from %s (image on %s, text on %s)",
                            self._metadata.get("model_id", MODEL_NAME), self._metadata.get("precision"),
                            self._assets_dir, self._requested, self._text_requested)
                t0 = time.perf_counter()
                image = self._session_factory(os.path.join(self._assets_dir, "image_encoder.onnx"), self._requested)
                logger.info("SiglipQnnService: image encoder ready in %.1f s", time.perf_counter() - t0)
                t0 = time.perf_counter()
                try:
                    text = self._session_factory(os.path.join(self._assets_dir, "text_encoder.onnx"),
                                                 self._text_requested)
                except Exception as exc:  # noqa: BLE001
                    if self._text_requested.lower().startswith("cpu"):
                        raise
                    # The 1.1 GB text tower may not compile for the HTP in a
                    # reasonable time on every board: the CPU computes the
                    # same vectors, the image tower stays on the NPU.
                    logger.warning("SiglipQnnService: text encoder on %s failed (%s); using the CPU",
                                   self._text_requested, exc)
                    text = self._session_factory(os.path.join(self._assets_dir, "text_encoder.onnx"),
                                                 CPU_PROVIDER)
                    self._text_device = CPU_PROVIDER
                logger.info("SiglipQnnService: text encoder ready in %.1f s on %s",
                            time.perf_counter() - t0, self._text_device)
                tokenizer = self._tokenizer_factory(os.path.join(self._assets_dir, "tokenizer.json"))
                self._engine = SiglipQnnEngine(image, text, self._metadata, tokenizer)
                self._fingerprint_cache = self._compute_fingerprint()
                self._load_state = HealthStatus.OK
                self._load_error = None
                logger.info("SiglipQnnService ready: %s on %s (text on %s), fingerprint=%s",
                            self._metadata.get("model_id"), self._requested, self._text_device,
                            self._fingerprint_cache)
            except Exception as exc:  # noqa: BLE001
                self._load_state = HealthStatus.ERROR
                self._load_error = str(exc)
                self._engine = None
                logger.exception("SiglipQnnService failed to load from %s", self._assets_dir)

    def is_ready(self) -> bool:
        return self._load_state == HealthStatus.OK

    def health_status(self) -> HealthStatus | None:
        return self._load_state

    def fingerprint(self) -> str | None:
        return self._fingerprint_cache

    @property
    def model_label(self) -> str:
        return f"{self._metadata.get('model_id', 'siglip2')}/{MODEL_NAME}/{self._metadata.get('precision', 'w8a16')}-qnn"

    def model_info(self) -> ModelInfo:
        return ModelInfo(
            name=MODEL_NAME,
            framework=MODEL_FRAMEWORK,
            version=f"{self._metadata.get('precision', 'w8a16')}-qnn",
            modalities_in=["image", "text"],
            modalities_out=["embedding"],
            fingerprint=self.fingerprint(),
        )

    def hardware_evaluation(self) -> HardwareEvaluationResponse:
        if self._load_state == HealthStatus.OK:
            verdict = HardwareVerdict.OK
            reasoning = (f"SigLIP2 {self._metadata.get('precision', 'w8a16')} on {self._requested}: "
                         f"image encoder on the NPU, text encoder on {self._text_device}.")
        elif self._load_state == HealthStatus.LOADING:
            verdict, reasoning = HardwareVerdict.WARN, "Model still loading (the first start compiles the graphs for the HTP)."
        else:
            verdict, reasoning = HardwareVerdict.BLOCKED, f"Model failed to load: {self._load_error}"
        return HardwareEvaluationResponse(
            verdict=verdict, reasoning=reasoning, checked_at=datetime.now(timezone.utc),
            details={
                "gpu_required": False, "gpu_in_use": False,
                "providers": self._requested, "text_providers": self._text_device,
                "embedding_dim": (self._engine.dim if self._engine and self._engine.dim else EMBED_DIM),
                "assets_dir": self._assets_dir,
                "model_id": self._metadata.get("model_id"),
                "precision": self._metadata.get("precision"),
                "platform": platform.platform(), "python_version": platform.python_version(),
            },
        )

    # ── inference ────────────────────────────────────────────────────

    def infer(self, payload: dict[str, Any]) -> InferResponse:
        if self._load_state != HealthStatus.OK or self._engine is None:
            raise ServiceError(
                ErrorCategory.MODEL_ERROR,
                code="weights_missing" if self._load_state == HealthStatus.ERROR else "siglip_qnn.model_loading",
                message=self._load_error or "Model still loading.",
                transient=(self._load_state == HealthStatus.LOADING),
                http_status=503,
                retry_after_ms=5000 if self._load_state == HealthStatus.LOADING else None,
            )
        task = str(payload.get("task") or "embed")
        if task not in ("embed", "embedding", "image_embedding", "clip"):
            raise _bad("malformed_input", f"Unknown task {task!r}; this adapter embeds (task 'embed').")
        text = payload.get("text")
        text = text.strip() if isinstance(text, str) else None
        image = self._image_bytes(payload)
        if text and image:
            raise _bad("malformed_input", "Send either 'text' or an image, not both — one embedding per request.")
        if not text and not image:
            raise _bad("malformed_input", "Send JSON with 'text' (a query) or 'frame_b64' (an image).")
        if text and len(text) > MAX_TEXT_CHARS:
            raise _bad("text_too_long", f"'text' exceeds {MAX_TEXT_CHARS} characters.", 413)
        if image and len(image) > MAX_IMAGE_BYTES:
            raise _bad("malformed_input", f"Image exceeds {MAX_IMAGE_BYTES}-byte limit ({len(image)} received).", 413)

        t0 = time.perf_counter()
        try:
            with self._lock:                      # one request at a time on the HTP
                out = self._engine.embed_text(text) if text else self._engine.embed_image(image)
        except ValueError as exc:
            raise _bad("malformed_input", str(exc)) from exc
        except Exception as exc:  # noqa: BLE001
            raise ServiceError(
                ErrorCategory.MODEL_ERROR, code="inference_failed",
                message=f"SigLIP2 raised during inference: {exc}",
                transient=True, http_status=500, retry_after_ms=1000,
            ) from exc
        return InferResponse(
            status="ok",
            model_name="siglip-qnn",
            model_version=f"{MODEL_FRAMEWORK}/{self.model_label}",
            inference_ms=int((time.perf_counter() - t0) * 1000),
            result={
                "embedding": out.vector,
                "dim": len(out.vector),
                "model": self.model_label,
                "modality": out.modality,
                "normalized": True,
                "device": self._text_device if out.modality == "text" else self._requested,
                "encode_ms": round(out.inference_ms, 1),
            },
        )

    # ── helpers ────────────────────────────────────────────────────

    @staticmethod
    def _image_bytes(payload: dict[str, Any]) -> bytes | None:
        raw = payload.get(BODY_BYTES_KEY)
        if isinstance(raw, (bytes, bytearray)) and raw:
            return bytes(raw)
        for key in ("frame_b64", "image_b64", "image"):
            b64 = payload.get(key)
            if isinstance(b64, str) and b64.strip():
                try:
                    return base64.b64decode(b64, validate=True)
                except Exception as exc:  # noqa: BLE001
                    raise _bad("malformed_input", f"'{key}' is not valid base64: {exc}") from exc
        return None

    def _compute_fingerprint(self) -> str:
        """sha256 over the asset names and sizes plus metadata.json's bytes:
        the externals are 1.4 GB and are not hashed."""
        h = hashlib.sha256()
        for name in sorted(os.listdir(self._assets_dir)):
            p = os.path.join(self._assets_dir, name)
            if os.path.isfile(p):
                h.update(f"{name}:{os.path.getsize(p)}".encode())
        with open(os.path.join(self._assets_dir, "metadata.json"), "rb") as fh:
            h.update(fh.read())
        return "sha256:" + h.hexdigest()[:32]
