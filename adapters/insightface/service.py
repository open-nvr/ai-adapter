# Copyright (c) 2026 OpenNVR
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
InsightFaceService — face detection + recognition adapter on the
``opennvr-adapter-sdk`` contract.

This is the SDK-based replacement for the legacy in-tree adapter at
``app/adapters/vision/insightface_adapter.py``. The legacy code only
accepted ``opennvr://frames/<camera>/<file>`` URIs that required a
shared volume between the adapter and the upstream — fine for the
old bundled-monolith deployment, awkward for the SDK contract's
"any caller, anywhere" promise. This service accepts image bytes
directly via the contract's ``BodyShape.IMAGE`` body parser, so
the caller can be a polling daemon, a webhook, or anything else
that has JPEG/PNG bytes in hand.

Tasks supported on ``/infer``:

* ``face_detection`` (default) — return every face's bbox, confidence,
  optional landmarks/age/gender. Equivalent to ``insightface.app.get()``.
* ``face_recognition`` — detect ONE face (highest confidence), look it
  up in the face DB, return either a match (with person_id / name /
  similarity / category) or "unknown."
* ``face_embedding`` — return the raw 512-d embedding vector for the
  highest-confidence face. Useful when the caller wants to do its
  own matching outside the adapter.

Face DB CRUD (register / list / get / delete) is mounted as extra
routes on the adapter's FastAPI app — see ``main.py``. Those routes
share the same `FaceDB` instance the recognition task uses.
"""
from __future__ import annotations

import hashlib
import io
import logging
import os
import platform
import threading
import time
from datetime import datetime, timezone
from typing import Any

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

from adapters.insightface.face_db import FaceDB

logger = logging.getLogger(__name__)


# Default model pack name. InsightFace ships several — ``buffalo_l``
# is the largest / most accurate; ``buffalo_s`` is smaller / faster.
# Operators override via the OPENNVR_INSIGHTFACE_MODEL env var.
DEFAULT_MODEL_PACK: str = "buffalo_l"
DEFAULT_MIN_FACE_CONFIDENCE: float = 0.5
DEFAULT_RECOGNITION_THRESHOLD: float = 0.5

# Maximum image bytes per call. A 4K JPEG is well under 4 MB; the
# 8 MB cap is the same shape as the YOLOv8 adapter's cap. Above this,
# the SDK returns a 413 envelope before the service is ever invoked.
MAX_IMAGE_BYTES: int = 8 * 1024 * 1024

# Tasks the service knows about. Used to fail fast with a clear
# "unsupported_task" envelope rather than silently fall through.
_SUPPORTED_TASKS: tuple[str, ...] = (
    "face_detection",
    "face_recognition",
    "face_embedding",
)


class InsightFaceService(AdapterService):
    """SDK-based face-recognition service backed by InsightFace."""

    def __init__(
        self,
        model_pack: str | None = None,
        face_db: FaceDB | None = None,
        min_face_confidence: float = DEFAULT_MIN_FACE_CONFIDENCE,
        recognition_threshold: float = DEFAULT_RECOGNITION_THRESHOLD,
    ) -> None:
        self._model_pack: str = model_pack or os.getenv(
            "OPENNVR_INSIGHTFACE_MODEL", DEFAULT_MODEL_PACK
        )
        self._min_face_confidence: float = float(min_face_confidence)
        self._recognition_threshold: float = float(recognition_threshold)
        self._face_app: Any | None = None  # populated in load()
        self._load_state: HealthStatus = HealthStatus.LOADING
        self._load_error: str | None = None
        self._fingerprint_cache: str | None = None
        self._fingerprint_stat_key: tuple[str, int, int] | None = None
        self._lock = threading.Lock()
        self._started_at: datetime = datetime.now(timezone.utc)
        # Allow injection for tests; production constructs its own.
        self._face_db: FaceDB = face_db if face_db is not None else FaceDB()

    # ── AdapterService impl ────────────────────────────────────────

    def load(self) -> None:
        with self._lock:
            if self._load_state == HealthStatus.OK:
                return
            # Domain metrics (registration is idempotent in the SDK).
            self.metrics.register_counter(
                "adapter_faces_total",
                "Faces processed, by pipeline stage.",
                label_key="stage",
                allowed_values=("detected", "recognized", "unrecognized"))
            try:
                # Lazy import per the project's "no heavy imports at
                # discovery time" convention.
                from insightface.app import FaceAnalysis

                self._face_app = FaceAnalysis(name=self._model_pack)
                # ctx_id=-1 keeps it on CPU. GPU users override via
                # the OPENNVR_INSIGHTFACE_CTX env var.
                ctx_id = int(os.getenv("OPENNVR_INSIGHTFACE_CTX", "-1"))
                self._face_app.prepare(ctx_id=ctx_id, det_size=(640, 640))
                self._fingerprint_cache = self._compute_fingerprint()
                self._load_state = HealthStatus.OK
                self._load_error = None
                logger.info(
                    "InsightFaceService ready: pack=%s ctx_id=%d fingerprint=%s",
                    self._model_pack, ctx_id, self._fingerprint_cache,
                )
            except Exception as exc:
                self._load_state = HealthStatus.ERROR
                self._load_error = str(exc)
                logger.exception(
                    "InsightFaceService failed to load pack %s",
                    self._model_pack,
                )

    def is_ready(self) -> bool:
        return self._load_state == HealthStatus.OK

    def fingerprint(self) -> str | None:
        try:
            return self._compute_fingerprint()
        except OSError:
            return self._fingerprint_cache

    def model_info(self) -> ModelInfo:
        return ModelInfo(
            name=self._model_pack,
            version=f"insightface/{self._model_pack}",
            framework="onnx",
            modalities_in=["image"],
            modalities_out=["bbox_classes", "text"],
            fingerprint=self.fingerprint(),
        )

    def hardware_evaluation(self) -> HardwareEvaluationResponse:
        if self._load_state == HealthStatus.OK:
            verdict = HardwareVerdict.OK
            reasoning = (
                f"InsightFace runs on onnxruntime; pack "
                f"{self._model_pack!r} loaded."
            )
        elif self._load_state == HealthStatus.LOADING:
            verdict = HardwareVerdict.WARN
            reasoning = "Model still loading."
        else:
            verdict = HardwareVerdict.BLOCKED
            reasoning = f"Model failed to load: {self._load_error}"

        uptime_s = int(
            (datetime.now(timezone.utc) - self._started_at).total_seconds()
        )
        return HardwareEvaluationResponse(
            verdict=verdict,
            reasoning=reasoning,
            checked_at=datetime.now(timezone.utc),
            details={
                "gpu_required": False,
                "cpu_count": os.cpu_count() or 0,
                "platform": platform.platform(),
                "python_version": platform.python_version(),
                "model_pack": self._model_pack,
                "min_face_confidence": self._min_face_confidence,
                "recognition_threshold": self._recognition_threshold,
                "registered_faces": len(self._face_db),
                "adapter_uptime_seconds": uptime_s,
            },
        )

    def infer(self, payload: dict[str, Any]) -> InferResponse:
        """Run a face task on a single image.

        ``payload['__file__']`` holds the raw image bytes (multipart
        upload or base64-decoded JSON). The optional ``task`` field
        chooses the task; default is ``face_detection``.
        """
        if not self.is_ready():
            transient = self._load_state == HealthStatus.LOADING
            raise ServiceError(
                ErrorCategory.MODEL_ERROR,
                code="model_not_loaded",
                message=self._load_error or "InsightFace model not yet loaded",
                transient=transient,
                http_status=503,
                retry_after_ms=2000 if transient else None,
            )

        image_bytes = payload.get("__file__")
        if not image_bytes:
            raise ServiceError(
                ErrorCategory.TRANSPORT_ERROR,
                code="missing_image",
                message=(
                    "insightface expects an image in the request body "
                    "(multipart 'frame' field or JSON 'frame_b64'). "
                    "No image bytes were provided."
                ),
                transient=False,
                http_status=400,
            )

        task = (payload.get("task") or "face_detection").strip()
        if task not in _SUPPORTED_TASKS:
            raise ServiceError(
                ErrorCategory.NOT_SUPPORTED,
                code="unsupported_task",
                message=(
                    f"task {task!r} is not supported; "
                    f"choose one of {list(_SUPPORTED_TASKS)}"
                ),
                transient=False,
                http_status=400,
            )

        started = time.monotonic()
        try:
            faces = _run_insightface(self._face_app, image_bytes)
        except _ImageDecodeError as exc:
            raise ServiceError(
                ErrorCategory.TRANSPORT_ERROR,
                code="invalid_image",
                message=str(exc),
                transient=False,
                http_status=400,
            ) from exc
        except Exception as exc:
            logger.exception("insightface inference failed")
            raise ServiceError(
                ErrorCategory.MODEL_ERROR,
                code="inference_failed",
                message=f"insightface raised during inference: {exc}",
                transient=True,
                http_status=500,
                retry_after_ms=1000,
            ) from exc

        # Drop sub-threshold faces so the recognition path doesn't
        # try to OCR a wall texture.
        faces = [f for f in faces if f["confidence"] >= self._min_face_confidence]

        elapsed_ms = int((time.monotonic() - started) * 1000)

        if task == "face_detection":
            result = _build_detection_result(faces)
        elif task == "face_embedding":
            result = _build_embedding_result(faces)
        else:  # face_recognition
            threshold = self._resolve_threshold(payload, self._recognition_threshold)
            result = _build_recognition_result(faces, self._face_db, threshold)

        result["model_pack"] = self._model_pack
        try:
            self.metrics.inc_counter(
                "adapter_faces_total", len(faces), label_value="detected")
            if task == "face_recognition" and faces:
                # Recognition answers for the single best face per request.
                if result.get("recognized"):
                    self.metrics.inc_counter(
                        "adapter_faces_total", 1, label_value="recognized")
                else:
                    self.metrics.inc_counter(
                        "adapter_faces_total", 1, label_value="unrecognized")
        except Exception:  # pragma: no cover - metrics must never break infer
            pass

        return InferResponse(
            model_name=self._model_pack,
            model_version=f"insightface/{self._model_pack}",
            inference_ms=elapsed_ms,
            result=result,
        )

    # ── Helpers ────────────────────────────────────────────────────

    def _resolve_threshold(
        self, payload: dict[str, Any], default: float
    ) -> float:
        raw = payload.get("threshold")
        if raw is None:
            return default
        try:
            value = float(raw)
        except (TypeError, ValueError):
            raise ServiceError(
                ErrorCategory.TRANSPORT_ERROR,
                code="invalid_threshold",
                message=f"threshold must be a float; got {raw!r}",
                transient=False,
                http_status=400,
            )
        if not (0.0 < value <= 1.0):
            raise ServiceError(
                ErrorCategory.TRANSPORT_ERROR,
                code="threshold_out_of_range",
                message=f"threshold must be in (0.0, 1.0]; got {value}",
                transient=False,
                http_status=400,
            )
        return value

    @property
    def face_db(self) -> FaceDB:
        """Exposed for ``main.py`` to share the same FaceDB instance
        with the /faces/* CRUD routes."""
        return self._face_db

    def _compute_fingerprint(self) -> str:
        """sha256 over the InsightFace model pack's primary weight
        file, or an identifier-derived synthetic when the path can't
        be discovered. Caches by (path, size, mtime_ns) so KAI-C's
        60s /capabilities polls don't rehash a 200 MB ONNX file
        every time."""
        path = _primary_weight_path(self._face_app)
        if path and os.path.isfile(path):
            stat = os.stat(path)
            stat_key = (path, stat.st_size, stat.st_mtime_ns)
            if (
                self._fingerprint_stat_key == stat_key
                and self._fingerprint_cache
                and self._fingerprint_cache.startswith("sha256:")
            ):
                return self._fingerprint_cache
            h = hashlib.sha256()
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(64 * 1024), b""):
                    h.update(chunk)
            fp = f"sha256:{h.hexdigest()}"
            self._fingerprint_stat_key = stat_key
            self._fingerprint_cache = fp
            return fp
        digest = hashlib.sha256(self._model_pack.encode("utf-8")).hexdigest()
        return f"sha256-id:{digest}"


# ── Inference + result-builder helpers ─────────────────────────────


class _ImageDecodeError(Exception):
    """Raised when the input bytes are not a decodable image."""


def _run_insightface(face_app: Any, image_bytes: bytes) -> list[dict[str, Any]]:
    """Decode JPEG/PNG bytes, run InsightFace, return one dict per
    detected face with bbox / confidence / embedding / optional
    age+gender+landmarks."""
    # Lazy ML imports — keeps the module importable in test
    # environments that don't have cv2 / numpy installed yet.
    import cv2
    import numpy as np

    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise _ImageDecodeError(
            "could not decode image bytes (expected JPEG / PNG / WebP)"
        )
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    raw_faces = face_app.get(img_rgb)

    results: list[dict[str, Any]] = []
    for face in raw_faces:
        entry: dict[str, Any] = {
            "bbox": [int(v) for v in face.bbox.tolist()],
            "confidence": round(float(face.det_score), 4),
        }
        kps = getattr(face, "kps", None)
        if kps is not None:
            entry["landmarks"] = [[int(x), int(y)] for x, y in kps.tolist()]
        age = getattr(face, "age", None)
        if age is not None:
            entry["age"] = int(age)
        gender = getattr(face, "gender", None)
        if gender is not None:
            entry["gender"] = "M" if int(gender) == 1 else "F"
        embedding = getattr(face, "normed_embedding", None)
        if embedding is None:
            embedding = getattr(face, "embedding", None)
        if embedding is not None:
            entry["embedding"] = [float(v) for v in embedding.tolist()]
        results.append(entry)

    return results


def _build_detection_result(faces: list[dict[str, Any]]) -> dict[str, Any]:
    detections = [{k: v for k, v in f.items() if k != "embedding"} for f in faces]
    return {
        "task": "face_detection",
        "faces": detections,
        "face_count": len(detections),
    }


def _build_embedding_result(faces: list[dict[str, Any]]) -> dict[str, Any]:
    if not faces:
        return {
            "task": "face_embedding",
            "embedding": None,
            "face_bbox": None,
            "face_count": 0,
            "message": "no face detected",
        }
    best = max(faces, key=lambda f: f["confidence"])
    return {
        "task": "face_embedding",
        "embedding": best.get("embedding"),
        "face_bbox": best["bbox"],
        "confidence": best["confidence"],
        # face_count lets callers (notably /faces/register) reject
        # multi-face enrollment images without a second /infer call.
        "face_count": len(faces),
    }


def _build_recognition_result(
    faces: list[dict[str, Any]],
    face_db: FaceDB,
    threshold: float,
) -> dict[str, Any]:
    if not faces:
        return {
            "task": "face_recognition",
            "recognized": False,
            "face_bbox": None,
            "message": "no face detected",
            "threshold": threshold,
        }
    best = max(faces, key=lambda f: f["confidence"])
    embedding = best.get("embedding")
    if embedding is None:
        return {
            "task": "face_recognition",
            "recognized": False,
            "face_bbox": best["bbox"],
            "message": "model did not produce an embedding for the best face",
            "threshold": threshold,
        }
    match = face_db.best_match(embedding, threshold=threshold)
    if match is None:
        return {
            "task": "face_recognition",
            "recognized": False,
            "face_bbox": best["bbox"],
            "message": f"no match above threshold {threshold}",
            "threshold": threshold,
            "registered_faces": len(face_db),
        }
    return {
        "task": "face_recognition",
        "recognized": True,
        "person_id": match["person_id"],
        "name": match["name"],
        "category": match["category"],
        "similarity": match["similarity"],
        "face_bbox": best["bbox"],
        "threshold": threshold,
    }


def _primary_weight_path(face_app: Any) -> str | None:
    """Dig into the FaceAnalysis object for one of the underlying ONNX
    file paths so the fingerprint reflects on-disk bytes. InsightFace
    keeps a ``models`` dict of ``{name: model_obj}``; each model_obj
    exposes ``model_file`` (a path). We pick the recognition model
    by preference (that's the one Smart Doorbell cares about), then
    fall back to whatever's first.

    Name matching is by-token rather than substring so a future
    third-party model named e.g. ``face_embedding_quality_estimator``
    can't shadow the real ArcFace recognition model. We split the
    model name on ``[_-]`` and check the tokens for an exact match
    against the preferred role labels.
    """
    if face_app is None:
        return None
    models = getattr(face_app, "models", None)
    if not isinstance(models, dict) or not models:
        return None
    preferred_roles = ("recognition", "embedding", "arcface")
    for preferred in preferred_roles:
        for name, model in models.items():
            tokens = {tok for tok in name.lower().replace("-", "_").split("_") if tok}
            if preferred in tokens:
                path = getattr(model, "model_file", None)
                if isinstance(path, str):
                    return path
    for model in models.values():
        path = getattr(model, "model_file", None)
        if isinstance(path, str):
            return path
    return None
