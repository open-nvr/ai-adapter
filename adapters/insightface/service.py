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
* ``face_identify`` — look up EVERY face in the frame, not just the
  best one: a classroom or a shop floor, where who is in the room is
  the question. One entry per face, matched against the whole DB in a
  single matrix product (``FaceDB.match_many``).

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

# Detector input size (square). The pack's detector sees the frame
# scaled to this, so at 640 a 1080p frame shrinks ~3x and a 50 px face
# at the back of a classroom reaches it at ~17 px — below what it finds.
# OPENNVR_INSIGHTFACE_DET_SIZE raises it (1280 for rooms) at roughly 4x
# the detection cost. A multiple of 32, as the detector's strides need.
DEFAULT_DET_SIZE: int = 640
_DET_SIZE_RANGE: tuple[int, int] = (160, 2048)

# face_identify bounds. Faces past ``max_faces`` (lowest detector
# confidence first) are dropped and counted, so one crowded frame cannot
# hold the adapter's single inference slot for every camera behind it.
DEFAULT_MAX_FACES: int = 100
_MAX_FACES_LIMIT: int = 500

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
    "face_identify",
)


def _det_size_from_env() -> int:
    """OPENNVR_INSIGHTFACE_DET_SIZE, validated. A bad value is a load
    failure with the reason on /health, not a silent fall back to 640."""
    raw = os.getenv("OPENNVR_INSIGHTFACE_DET_SIZE", "").strip()
    if not raw:
        return DEFAULT_DET_SIZE
    try:
        size = int(raw)
    except ValueError:
        raise ValueError(f"OPENNVR_INSIGHTFACE_DET_SIZE must be an integer; got {raw!r}")
    low, high = _DET_SIZE_RANGE
    if not (low <= size <= high) or size % 32:
        raise ValueError(
            f"OPENNVR_INSIGHTFACE_DET_SIZE must be a multiple of 32 in "
            f"[{low}, {high}]; got {size}")
    return size


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
        self._det_size: int = DEFAULT_DET_SIZE  # resolved in load()
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
                from opennvr_adapter_sdk.providers import (
                    accelerator_available, build_with_fallback, session_providers,
                    static_shapes, wants_static_shapes)
                # Providers first, insightface second: registering a plugin
                # provider installs the SDK's InferenceSession shim, and
                # insightface's model_zoo captures the constructor at import
                # time — imported earlier, it would build its sessions past
                # the shim and the NPU would never see them.
                self._det_size = _det_size_from_env()
                det = self._det_size
                self._requested_providers = session_providers()
                if wants_static_shapes(self._requested_providers):
                    # An accelerator that compiles fixed shapes (OpenVINO
                    # device, QNN HTP): the pack's detector is a dynamic
                    # export and prepare() always feeds it det_size, so pin
                    # the first dynamic input to that; the fixed-size
                    # recogniser built from the same list is left alone.
                    self._requested_providers = static_shapes(
                        self._requested_providers, {"*": [1, 3, det, det]})
                from insightface.app import FaceAnalysis

                # Providers from ADAPTER_ONNX_PROVIDERS resolved against this build
                # (OpenNVR Accelerate flavours). InsightFace's own default is
                # CUDA-then-CPU, which on the -intel flavour silently means CPU.
                # Only the two models recognition needs. FaceAnalysis's
                # default loads five (detector, 3D-68 and 2D-106 landmark
                # nets, gender/age, recogniser) and runs four of them per
                # detected face; the detector's own 5 keypoints are what
                # the ArcFace alignment uses, so the landmark and
                # attribute nets were pure waste on every face.
                # FaceAnalysis builds its onnxruntime sessions in __init__;
                # an accelerator that cannot start must not take the
                # adapter down — build_with_fallback rebuilds on CPU and
                # /hardware/evaluation carries the reason.
                self._face_app, self._requested_providers, self._fallback_error = (
                    build_with_fallback(
                        lambda providers: FaceAnalysis(
                            name=self._model_pack, providers=providers,
                            allowed_modules=["detection", "recognition"]),
                        self._requested_providers, adapter="insightface"))
                # ctx_id=-1 keeps it on CPU. GPU users override via
                # the OPENNVR_INSIGHTFACE_CTX env var.
                # ctx_id < 0 makes InsightFace force CPUExecutionProvider on every
                # model, whatever providers were given — so the default follows the
                # build: 0 when an accelerator provider exists, -1 on the plain cpu
                # image. OPENNVR_INSIGHTFACE_CTX overrides either way.
                default_ctx = "0" if accelerator_available() else "-1"
                ctx_id = int(os.getenv("OPENNVR_INSIGHTFACE_CTX", default_ctx))
                self._face_app.prepare(ctx_id=ctx_id, det_size=(det, det))
                self._provider_report = self._insightface_provider_report()
                self._fingerprint_cache = self._compute_fingerprint()
                self._load_state = HealthStatus.OK
                self._load_error = None
                logger.info(
                    "InsightFaceService ready: pack=%s ctx_id=%d det_size=%d fingerprint=%s",
                    self._model_pack, ctx_id, det, self._fingerprint_cache,
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

    def _insightface_provider_report(self) -> dict:
        """What the pack's sessions really run on: InsightFace builds one
        onnxruntime session per model, so ask the first one."""
        from opennvr_adapter_sdk.providers import effective_providers
        models = getattr(self._face_app, "models", None) or {}
        for model in models.values():
            session = getattr(model, "session", None)
            if session is not None and hasattr(session, "get_providers"):
                return effective_providers(
                    session, getattr(self, "_requested_providers", None), adapter="insightface",
                    fallback_error=getattr(self, "_fallback_error", None))
        return {}

    def hardware_evaluation(self) -> HardwareEvaluationResponse:
        report = getattr(self, "_provider_report", None) or {}
        if self._load_state == HealthStatus.OK:
            verdict = HardwareVerdict.OK
            where = (
                f"accelerator in use ({report.get('requested_providers')})"
                if report.get("accelerator_in_use") else "CPU"
            )
            reasoning = (
                f"InsightFace runs on onnxruntime ({where}); pack "
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
                "gpu_in_use": bool(report.get("accelerator_in_use")),
                **report,
                "cpu_count": os.cpu_count() or 0,
                "platform": platform.platform(),
                "python_version": platform.python_version(),
                "model_pack": self._model_pack,
                "det_size": self._det_size,
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

        # face_identify with match=false: where the faces are and how they
        # are turned, nobody looked up. Only the detector runs (no
        # embedding, no age/gender), for a page that checks a webcam frame
        # every second while somebody enrols.
        match = task != "face_identify" or _truthy(payload.get("match", True))
        started = time.monotonic()
        frame_size: dict[str, int] = {}
        try:
            faces = _run_insightface(self._face_app, image_bytes, frame_size,
                                     detect_only=not match)
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
        elif task == "face_identify":
            threshold = self._resolve_threshold(payload, self._recognition_threshold)
            result = _build_identify_result(
                faces, self._face_db if match else None, threshold,
                frame=frame_size,
                min_face_px=self._resolve_int(payload, "min_face_px", 0, 0, 4096),
                max_faces=self._resolve_int(
                    payload, "max_faces", DEFAULT_MAX_FACES, 1, _MAX_FACES_LIMIT),
                category=(str(payload.get("category") or "").strip() or None),
            )
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
            elif task == "face_identify" and result["faces"]:
                hits = sum(1 for f in result["faces"] if f["recognized"])
                if hits:
                    self.metrics.inc_counter(
                        "adapter_faces_total", hits, label_value="recognized")
                if len(result["faces"]) - hits:
                    self.metrics.inc_counter(
                        "adapter_faces_total", len(result["faces"]) - hits,
                        label_value="unrecognized")
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

    def _resolve_int(
        self, payload: dict[str, Any], key: str, default: int, low: int, high: int
    ) -> int:
        raw = payload.get(key)
        if raw is None:
            return default
        try:
            value = int(raw)
        except (TypeError, ValueError):
            value = None
        if value is None or isinstance(raw, bool) or not (low <= value <= high):
            raise ServiceError(
                ErrorCategory.TRANSPORT_ERROR,
                code=f"invalid_{key}",
                message=f"{key} must be an integer in [{low}, {high}]; got {raw!r}",
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


def _truthy(value: Any) -> bool:
    """A JSON flag that may also arrive as a form string."""
    if isinstance(value, str):
        return value.strip().lower() not in ("false", "0", "no", "off", "")
    return bool(value)


def _run_insightface(
    face_app: Any, image_bytes: bytes, frame_size: dict[str, int] | None = None,
    *, detect_only: bool = False,
) -> list[dict[str, Any]]:
    """Decode JPEG/PNG bytes, run InsightFace, return one dict per
    detected face with bbox / confidence / embedding / optional
    age+gender+landmarks. ``frame_size``, when given, is filled with
    the decoded ``w`` and ``h`` (face_identify reports it so callers
    can normalise the pixel boxes)."""
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
    if frame_size is not None:
        frame_size["h"], frame_size["w"] = int(img.shape[0]), int(img.shape[1])

    det_model = getattr(face_app, "det_model", None)
    if detect_only and det_model is not None:
        # FaceAnalysis.get() runs this first, then every other model on
        # each face; stopping here skips the embeddings.
        bboxes, kpss = det_model.detect(img_rgb, max_num=0, metric="default")
        out: list[dict[str, Any]] = []
        for i in range(bboxes.shape[0]):
            entry = {"bbox": [int(v) for v in bboxes[i, :4].tolist()],
                     "confidence": round(float(bboxes[i, 4]), 4)}
            if kpss is not None:
                entry["landmarks"] = [[int(x), int(y)] for x, y in kpss[i].tolist()]
            out.append(entry)
        return out

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
        # A miss carries its evidence. The caller cannot otherwise tell
        # "almost Alice, seen side-on at 0.45" from "nobody we know at
        # 0.10", nor a crisp face from a blurred half of one — and a
        # doorbell deciding whether to call somebody a stranger needs
        # exactly those two numbers. ``similarity`` is the NEAREST
        # enrolled face, below the threshold by definition; who it
        # was is deliberately not said, because a near miss is not an
        # identification. None when nobody is enrolled.
        nearest = face_db.best_match(embedding, threshold=-1.0)
        return {
            "task": "face_recognition",
            "recognized": False,
            "face_bbox": best["bbox"],
            "confidence": best["confidence"],
            "similarity": nearest["similarity"] if nearest else None,
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
        "confidence": best["confidence"],
        "threshold": threshold,
    }


def _build_identify_result(
    faces: list[dict[str, Any]],
    face_db: FaceDB | None,
    threshold: float,
    *,
    frame: dict[str, int],
    min_face_px: int,
    max_faces: int,
    category: str | None,
) -> dict[str, Any]:
    """Every face in the frame, each looked up in the face DB.

    Faces narrower or shorter than ``min_face_px`` are dropped before
    matching: a face that small embeds to noise, and a noisy embedding
    near the threshold is how a stranger becomes somebody. What was
    dropped is counted, never silently lost.

    A miss carries ``similarity`` (the nearest person's score) but not
    who that was, for the same reason as ``face_recognition``: a near
    miss is not an identification. ``second_similarity`` is the
    runner-up's score on a hit or a miss — the caller's margin check.

    ``face_db`` None (``match=false``): the same faces, nobody looked up.
    """
    detected = len(faces)
    sized = [
        f for f in faces
        if min(f["bbox"][2] - f["bbox"][0], f["bbox"][3] - f["bbox"][1]) >= min_face_px
    ]
    too_small = detected - len(sized)
    sized.sort(key=lambda f: f["confidence"], reverse=True)
    kept, over_limit = sized[:max_faces], max(0, len(sized) - max_faces)

    with_emb = [f for f in kept if f.get("embedding") is not None] if face_db is not None else []
    lookups = iter([] if face_db is None else face_db.match_many(
        [f["embedding"] for f in with_emb], threshold=threshold, category=category))
    by_face = {id(f): next(lookups) for f in with_emb}

    entries: list[dict[str, Any]] = []
    for face in kept:
        lookup = by_face.get(id(face))
        match = lookup["match"] if lookup else None
        entries.append({
            "bbox": face["bbox"],
            "confidence": face["confidence"],
            # The detector's five points (eyes, nose, mouth corners): how
            # the head is turned, for a page guiding an enrolment.
            "landmarks": face.get("landmarks"),
            "recognized": match is not None,
            "person_id": match["person_id"] if match else None,
            "name": match["name"] if match else None,
            "category": match["category"] if match else None,
            "similarity": (match["similarity"] if match
                           else (lookup["nearest_similarity"] if lookup else None)),
            "second_similarity": lookup["second_similarity"] if lookup else None,
        })
    return {
        "task": "face_identify",
        "faces": entries,
        "face_count": len(entries),
        "recognized_count": sum(1 for e in entries if e["recognized"]),
        "detected_count": detected,
        "dropped": {"too_small": too_small, "over_limit": over_limit},
        "frame": {"w": frame.get("w"), "h": frame.get("h")},
        "threshold": threshold,
        "registered_faces": len(face_db) if face_db is not None else None,
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
