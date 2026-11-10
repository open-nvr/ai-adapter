"""
Sentence embeddings for search's caption arm — the ``text_embed`` task.

The ``embed`` adapters (clip, siglip-qnn) put a FRAME and a query in one
space, which is what the frame arm of search needs. Their text tower is a
poor judge of text against text: on the Ventuno Q, SigLIP2's text-to-text
cosines for relevant and irrelevant caption/query pairs overlap (0.76–0.87
against 0.59–0.75, 2026-11-10), so no floor separates them. This adapter
is a sentence encoder trained for exactly that — a query against a
caption — and nothing else: text in, a 384-dimensional unit vector out.

Model: ``BAAI/bge-small-en-v1.5`` (MIT), 33 M parameters, 133 MB ONNX,
run on the CPU by ONNX Runtime. A caption costs a few milliseconds a
core, which at tens of captions an hour is nothing; the NPU would buy
nothing here. ``TEXT_EMBED_MODEL_DIR`` can point at any BERT-style export
with ``model.onnx`` + ``tokenizer.json`` (+ optional ``model.json``:
``{"model_id":..., "pooling":"cls"|"mean", "query_prefix":...}``).

Request / answer
----------------

    {"task": "text_embed", "text": "person in a blue shirt", "kind": "query"}
    {"task": "text_embed", "texts": ["a red truck parked...", "..."], "kind": "document"}

``kind`` is ``query`` (a search) or ``document`` (a caption); a model whose
card asks for a query instruction (``query_prefix``) gets it prepended for
queries only, as the BGE card says. One ``embedding`` for ``text``, an
``embeddings`` list for ``texts``, each unit length; ``dim`` and ``model``
so a store can refuse to compare what is not comparable.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import threading
import time
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from app.config import MODEL_WEIGHTS_DIR
from opennvr_adapter_sdk import (
    AdapterService,
    ErrorCategory,
    HardwareEvaluationResponse,
    HardwareVerdict,
    HealthStatus,
    InferResponse,
    ModelInfo,
    ServiceError,
)

from .engine import TextEmbedEngine
from .features import CLS, MAX_TOKENS, MEAN

logger = logging.getLogger(__name__)

MODEL_FRAMEWORK: str = "onnxruntime"
DEFAULT_MODEL_ID: str = "BAAI/bge-small-en-v1.5"
MODEL_DIR_ENV: str = "TEXT_EMBED_MODEL_DIR"
PROVIDERS_ENV: str = "ADAPTER_ONNX_PROVIDERS"
THREADS_ENV: str = "TEXT_EMBED_THREADS"
#: Two intra-op threads: the detector shares this box.
DEFAULT_THREADS: int = 2
CPU_PROVIDER: str = "CPUExecutionProvider"
TASKS: tuple[str, ...] = ("text_embed", "text_embedding", "sentence_embedding")
MAX_TEXT_CHARS: int = 4096
MAX_BATCH: int = 64
REQUIRED_FILES: tuple[str, ...] = ("model.onnx", "tokenizer.json")

SessionFactory = Callable[[str, str, int], Any]
TokenizerFactory = Callable[[str], Any]


def build_session(model_path: str, requested: str, threads: int) -> Any:
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = max(1, threads)
    opts.inter_op_num_threads = 1
    providers = [p.strip() for p in requested.split(",") if p.strip()] or [CPU_PROVIDER]
    available = set(ort.get_available_providers())
    chosen = [p.split(":")[0] for p in providers if p.split(":")[0] in available] or [CPU_PROVIDER]
    return ort.InferenceSession(model_path, sess_options=opts, providers=chosen)


def load_tokenizer(path: str) -> Any:
    from tokenizers import Tokenizer

    return Tokenizer.from_file(path)


def _bad(code: str, message: str, status: int = 400) -> ServiceError:
    return ServiceError(ErrorCategory.MODEL_ERROR, code=code, message=message,
                        transient=False, http_status=status)


class TextEmbedService(AdapterService):
    def __init__(
        self,
        model_dir: str | None = None,
        *,
        session_factory: SessionFactory | None = None,
        tokenizer_factory: TokenizerFactory | None = None,
    ) -> None:
        self._model_dir = (model_dir or os.environ.get(MODEL_DIR_ENV, "").strip()
                           or os.path.join(MODEL_WEIGHTS_DIR, "text-embed"))
        self._requested = os.environ.get(PROVIDERS_ENV, "").strip() or CPU_PROVIDER
        try:
            self._threads = int(os.environ.get(THREADS_ENV, "") or DEFAULT_THREADS)
        except ValueError:
            self._threads = DEFAULT_THREADS
        self._session_factory = session_factory or build_session
        self._tokenizer_factory = tokenizer_factory or load_tokenizer
        self._engine: TextEmbedEngine | None = None
        self._meta: dict[str, Any] = {}
        self._load_state: HealthStatus = HealthStatus.LOADING
        self._load_error: str | None = None
        self._fingerprint_cache: str | None = None
        self._lock = threading.Lock()

    # ── lifecycle ────────────────────────────────────────────────────

    def load(self) -> None:
        with self._lock:
            if self._load_state == HealthStatus.OK:
                return
            try:
                missing = [f for f in REQUIRED_FILES if not os.path.isfile(os.path.join(self._model_dir, f))]
                if missing:
                    raise FileNotFoundError(
                        f"text-embed model files missing from {self._model_dir}: {', '.join(missing)} "
                        f"(set {MODEL_DIR_ENV}; the image bakes them at build)")
                meta_path = os.path.join(self._model_dir, "model.json")
                if os.path.isfile(meta_path):
                    with open(meta_path, encoding="utf-8") as fh:
                        self._meta = json.load(fh)
                pooling = str(self._meta.get("pooling") or CLS).lower()
                if pooling not in (CLS, MEAN):
                    raise ValueError(f"model.json pooling must be 'cls' or 'mean', not {pooling!r}")
                t0 = time.perf_counter()
                session = self._session_factory(os.path.join(self._model_dir, "model.onnx"),
                                                self._requested, self._threads)
                tokenizer = self._tokenizer_factory(os.path.join(self._model_dir, "tokenizer.json"))
                self._engine = TextEmbedEngine(session, tokenizer, pooling=pooling,
                                               max_tokens=int(self._meta.get("max_tokens") or MAX_TOKENS))
                self._fingerprint_cache = self._compute_fingerprint()
                self._load_state = HealthStatus.OK
                self._load_error = None
                logger.info("TextEmbedService ready: %s (%s pooling) on %s, %d threads, %.1f s, fingerprint=%s",
                            self.model_id, pooling, self._requested, self._threads,
                            time.perf_counter() - t0, self._fingerprint_cache)
            except Exception as exc:
                self._load_state = HealthStatus.ERROR
                self._load_error = str(exc)
                self._engine = None
                logger.exception("TextEmbedService failed to load from %s", self._model_dir)

    def is_ready(self) -> bool:
        return self._load_state == HealthStatus.OK

    def health_status(self) -> HealthStatus | None:
        return self._load_state

    def fingerprint(self) -> str | None:
        return self._fingerprint_cache

    @property
    def model_id(self) -> str:
        return str(self._meta.get("model_id") or DEFAULT_MODEL_ID)

    @property
    def model_label(self) -> str:
        """What the store records as the producer: the model and its
        pooling, so a different export of the same name is a different
        producer."""
        return f"{self.model_id}/{self._meta.get('pooling') or CLS}-onnx"

    def model_info(self) -> ModelInfo:
        return ModelInfo(
            name=self.model_id,
            framework=MODEL_FRAMEWORK,
            version=f"{self._meta.get('pooling') or CLS}-onnx",
            modalities_in=["text"],
            modalities_out=["embedding"],
            fingerprint=self.fingerprint(),
        )

    def hardware_evaluation(self) -> HardwareEvaluationResponse:
        if self._load_state == HealthStatus.OK:
            verdict = HardwareVerdict.OK
            reasoning = f"{self.model_id} on {self._requested} with {self._threads} threads."
        elif self._load_state == HealthStatus.LOADING:
            verdict, reasoning = HardwareVerdict.WARN, "Model still loading."
        else:
            verdict, reasoning = HardwareVerdict.BLOCKED, f"Model failed to load: {self._load_error}"
        return HardwareEvaluationResponse(
            verdict=verdict, reasoning=reasoning, checked_at=datetime.now(UTC),
            details={
                "gpu_required": False, "gpu_in_use": False,
                "providers": self._requested, "threads": self._threads,
                "embedding_dim": self._engine.dim if self._engine and self._engine.dim else None,
                "model_dir": self._model_dir, "model_id": self.model_id,
                "pooling": self._meta.get("pooling") or CLS,
                "platform": platform.platform(), "python_version": platform.python_version(),
            },
        )

    # ── inference ────────────────────────────────────────────────────

    def infer(self, payload: dict[str, Any]) -> InferResponse:
        if self._load_state != HealthStatus.OK or self._engine is None:
            raise ServiceError(
                ErrorCategory.MODEL_ERROR,
                code="weights_missing" if self._load_state == HealthStatus.ERROR else "text_embed.model_loading",
                message=self._load_error or "Model still loading.",
                transient=(self._load_state == HealthStatus.LOADING),
                http_status=503,
                retry_after_ms=5000 if self._load_state == HealthStatus.LOADING else None,
            )
        task = str(payload.get("task") or TASKS[0])
        if task not in TASKS:
            raise _bad("malformed_input", f"Unknown task {task!r}; this adapter embeds text (task 'text_embed').")
        kind = str(payload.get("kind") or "document").lower()
        if kind not in ("query", "document"):
            raise _bad("malformed_input", "'kind' must be 'query' or 'document'.")
        one = payload.get("text")
        many = payload.get("texts")
        if one is not None and many is not None:
            raise _bad("malformed_input", "Send 'text' or 'texts', not both.")
        if many is not None:
            if not isinstance(many, list) or not many or not all(isinstance(t, str) for t in many):
                raise _bad("malformed_input", "'texts' must be a non-empty list of strings.")
            if len(many) > MAX_BATCH:
                raise _bad("malformed_input", f"'texts' holds more than {MAX_BATCH} items.", 413)
            texts = [t.strip() for t in many]
        elif isinstance(one, str):
            texts = [one.strip()]
        else:
            raise _bad("malformed_input", "Send JSON with 'text' (a string) or 'texts' (a list of strings).")
        if any(not t for t in texts):
            raise _bad("malformed_input", "An empty string has no embedding.")
        if any(len(t) > MAX_TEXT_CHARS for t in texts):
            raise _bad("text_too_long", f"A text exceeds {MAX_TEXT_CHARS} characters.", 413)
        prefix = str(self._meta.get("query_prefix") or "")
        if kind == "query" and prefix:
            texts = [prefix + t for t in texts]

        t0 = time.perf_counter()
        try:
            with self._lock:
                outs = self._engine.embed(texts)
        except Exception as exc:
            raise ServiceError(
                ErrorCategory.MODEL_ERROR, code="inference_failed",
                message=f"{self.model_id} raised during inference: {exc}",
                transient=True, http_status=500, retry_after_ms=1000,
            ) from exc
        result: dict[str, Any] = {
            "dim": len(outs[0].vector),
            "model": self.model_label,
            "modality": "text",
            "kind": kind,
            "normalized": True,
            "device": self._requested,
            "encode_ms": round(sum(o.inference_ms for o in outs), 1),
        }
        if many is not None:
            result["embeddings"] = [o.vector for o in outs]
        else:
            result["embedding"] = outs[0].vector
        return InferResponse(
            status="ok",
            model_name="text-embed",
            model_version=f"{MODEL_FRAMEWORK}/{self.model_label}",
            inference_ms=int((time.perf_counter() - t0) * 1000),
            result=result,
        )

    # ── helpers ──────────────────────────────────────────────────────

    def _compute_fingerprint(self) -> str:
        h = hashlib.sha256()
        for name in REQUIRED_FILES:
            p = os.path.join(self._model_dir, name)
            h.update(name.encode())
            try:
                with open(p, "rb") as fh:
                    for chunk in iter(lambda: fh.read(1 << 20), b""):
                        h.update(chunk)
            except OSError:
                h.update(b"?")
        h.update(json.dumps(self._meta, sort_keys=True).encode())
        return f"sha256:{h.hexdigest()}"
