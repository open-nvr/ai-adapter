# Copyright (c) 2026 OpenNVR
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
RekognitionService — face_identify on Amazon Rekognition.

A drop-in for the attendance InsightFace: the same ``face_identify``
result and the same ``/faces/register`` / ``/faces/{person_id}`` routes
(main.py), so OpenNVR's attendance works unchanged on either. The faces
live in a Rekognition collection, each indexed with ``ExternalImageId`` =
the person's id (OpenNVR's uuid for them).

THIS SENDS FACES OFF THE SITE. Every frame looked at and every photo
enrolled goes to AWS. The adapter declares that egress, so KAI-C refuses
it under ``AI_SOVEREIGNTY=local_only`` until an operator changes that on
purpose. Prefer the local InsightFace; this exists for sites that already
run on AWS and accept that.

face_identify: ``DetectFaces`` on the frame, then ``SearchFacesByImage``
on each face crop — one call per face, so a 40-face classroom is 41 API
calls per look (the cost line to watch). Similarity comes back 0-100 and
is reported 0-1. Rekognition's scores are not cosine similarities: a true
match is ~0.99 and a different person rarely clears 0.8, so the adapter
never searches below ``OPENNVR_REKOGNITION_MIN_SIMILARITY`` (default 90)
whatever threshold the caller asks for.
"""
from __future__ import annotations

import io
import json
import logging
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
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

logger = logging.getLogger(__name__)

MAX_IMAGE_BYTES: int = 5 * 1024 * 1024        # Rekognition's own limit for raw bytes
DEFAULT_COLLECTION = "opennvr-attendance"
DEFAULT_MIN_SIMILARITY = 90.0
DEFAULT_MAX_FACES = 100
_MAX_FACES_LIMIT = 100                         # DetectFaces returns at most 100
_SUPPORTED_TASKS = ("face_identify",)


def _client_from_env():
    import boto3  # lazy: tests inject a fake

    return boto3.client("rekognition", region_name=os.getenv("AWS_REGION") or None)


class RekognitionService(AdapterService):
    """face_identify and a face gallery on one Rekognition collection."""

    def __init__(self, client: Any | None = None, *, collection: str | None = None,
                 index_path: str | None = None) -> None:
        self._client = client
        self._collection = collection or os.getenv("OPENNVR_REKOGNITION_COLLECTION",
                                                   DEFAULT_COLLECTION)
        self._min_sim = float(os.getenv("OPENNVR_REKOGNITION_MIN_SIMILARITY",
                                        DEFAULT_MIN_SIMILARITY))
        self._index_path = index_path or os.getenv("OPENNVR_REKOGNITION_INDEX", "")
        #: person_id -> Rekognition FaceIds: sample counts and deletes
        #: without scanning the collection. Rebuilt from ListFaces on load.
        self._index: dict[str, list[str]] = {}
        self._lock = threading.Lock()
        self._state = HealthStatus.LOADING
        self._error: str | None = None
        self._started = datetime.now(timezone.utc)

    # ── lifecycle ──────────────────────────────────────────────────

    def load(self) -> None:
        try:
            if self._client is None:
                self._client = _client_from_env()
            self._ensure_collection()
            self._rebuild_index()
            self._state, self._error = HealthStatus.OK, None
            logger.info("Rekognition ready: collection=%s people=%d",
                        self._collection, len(self._index))
        except Exception as exc:  # noqa: BLE001 - reported on /health, not raised
            self._state, self._error = HealthStatus.ERROR, str(exc)
            logger.exception("Rekognition failed to start")

    def is_ready(self) -> bool:
        return self._state == HealthStatus.OK

    def _ensure_collection(self) -> None:
        try:
            self._client.describe_collection(CollectionId=self._collection)
        except Exception as exc:  # noqa: BLE001
            if "ResourceNotFound" not in type(exc).__name__ + str(exc):
                raise
            self._client.create_collection(CollectionId=self._collection)

    def _rebuild_index(self) -> None:
        index: dict[str, list[str]] = {}
        token = None
        while True:
            kw = {"CollectionId": self._collection, "MaxResults": 4096}
            if token:
                kw["NextToken"] = token
            page = self._client.list_faces(**kw)
            for f in page.get("Faces") or []:
                pid = f.get("ExternalImageId")
                if pid:
                    index.setdefault(pid, []).append(f["FaceId"])
            token = page.get("NextToken")
            if not token:
                break
        with self._lock:
            self._index = index
        self._save_index()

    def _save_index(self) -> None:
        if not self._index_path:
            return
        path = Path(self._index_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with self._lock:
            tmp.write_text(json.dumps(self._index))
        os.replace(tmp, path)

    # ── contract ───────────────────────────────────────────────────

    def fingerprint(self) -> str | None:
        return f"aws-rekognition:{self._collection}"

    def model_info(self) -> ModelInfo:
        return ModelInfo(name="amazon-rekognition", version="aws/rekognition",
                         framework="cloud", modalities_in=["image"],
                         modalities_out=["bbox_classes", "text"], fingerprint=self.fingerprint())

    def hardware_evaluation(self) -> HardwareEvaluationResponse:
        verdict = {HealthStatus.OK: HardwareVerdict.OK,
                   HealthStatus.LOADING: HardwareVerdict.WARN}.get(self._state, HardwareVerdict.BLOCKED)
        return HardwareEvaluationResponse(
            verdict=verdict,
            reasoning=("Faces are sent to Amazon Rekognition (off-site)."
                       if self._state == HealthStatus.OK else f"Not ready: {self._error}"),
            checked_at=datetime.now(timezone.utc),
            details={"gpu_required": False, "collection": self._collection,
                     "people": len(self._index), "min_similarity": self._min_sim,
                     "region": os.getenv("AWS_REGION"),
                     "adapter_uptime_seconds": int((datetime.now(timezone.utc) - self._started).total_seconds())})

    def infer(self, payload: dict[str, Any]) -> InferResponse:
        if not self.is_ready():
            raise ServiceError(ErrorCategory.MODEL_ERROR, code="not_ready",
                               message=self._error or "Rekognition not ready",
                               transient=self._state == HealthStatus.LOADING, http_status=503)
        image = payload.get("__file__")
        if not image:
            raise ServiceError(ErrorCategory.TRANSPORT_ERROR, code="missing_image",
                               message="no image in the request", transient=False, http_status=400)
        task = (payload.get("task") or "face_identify").strip()
        if task not in _SUPPORTED_TASKS:
            raise ServiceError(ErrorCategory.NOT_SUPPORTED, code="unsupported_task",
                               message=f"task {task!r} is not supported; choose one of {list(_SUPPORTED_TASKS)}",
                               transient=False, http_status=400)
        threshold = _bounded_float(payload.get("threshold"), 0.5, 0.0, 1.0, "threshold")
        min_px = _bounded_int(payload.get("min_face_px"), 0, 0, 4096, "min_face_px")
        max_faces = _bounded_int(payload.get("max_faces"), DEFAULT_MAX_FACES, 1, _MAX_FACES_LIMIT, "max_faces")
        started = time.monotonic()
        try:
            result = self.identify(image, threshold=threshold, min_face_px=min_px, max_faces=max_faces)
        except ServiceError:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.exception("Rekognition face_identify failed")
            raise ServiceError(ErrorCategory.MODEL_ERROR, code="inference_failed",
                               message=f"Rekognition failed: {exc}", transient=True,
                               http_status=502, retry_after_ms=2000) from exc
        return InferResponse(model_name="amazon-rekognition", model_version="aws/rekognition",
                             inference_ms=int((time.monotonic() - started) * 1000), result=result)

    # ── face_identify ──────────────────────────────────────────────

    def identify(self, image: bytes, *, threshold: float, min_face_px: int = 0,
                 max_faces: int = DEFAULT_MAX_FACES) -> dict[str, Any]:
        from PIL import Image

        try:
            pic = Image.open(io.BytesIO(image)).convert("RGB")
        except Exception:  # noqa: BLE001
            raise ServiceError(ErrorCategory.TRANSPORT_ERROR, code="invalid_image",
                               message="could not decode the image", transient=False,
                               http_status=400) from None
        w, h = pic.size
        details = self._client.detect_faces(Image={"Bytes": image}).get("FaceDetails") or []
        boxes = []
        for d in details:
            b = d.get("BoundingBox") or {}
            x1 = max(0, int(round(b.get("Left", 0) * w)))
            y1 = max(0, int(round(b.get("Top", 0) * h)))
            x2 = min(w, int(round((b.get("Left", 0) + b.get("Width", 0)) * w)))
            y2 = min(h, int(round((b.get("Top", 0) + b.get("Height", 0)) * h)))
            if x2 > x1 and y2 > y1:
                boxes.append(([x1, y1, x2, y2], round(float(d.get("Confidence", 0)) / 100, 4)))
        detected = len(boxes)
        sized = [bc for bc in boxes if min(bc[0][2] - bc[0][0], bc[0][3] - bc[0][1]) >= min_face_px]
        sized.sort(key=lambda bc: bc[1], reverse=True)
        kept = sized[:max_faces]
        floor = max(self._min_sim, threshold * 100)

        faces = []
        for bbox, conf in kept:
            crop = io.BytesIO()
            pic.crop(tuple(_pad(bbox, w, h))).save(crop, format="JPEG", quality=92)
            try:
                matches = self._client.search_faces_by_image(
                    CollectionId=self._collection, Image={"Bytes": crop.getvalue()},
                    FaceMatchThreshold=floor, MaxFaces=5).get("FaceMatches") or []
            except Exception as exc:  # noqa: BLE001
                # A crop Rekognition cannot find a face in is a miss, not a failure.
                if "InvalidParameter" in type(exc).__name__ + str(exc):
                    matches = []
                else:
                    raise
            best: dict[str, float] = {}
            for m in matches:
                pid = (m.get("Face") or {}).get("ExternalImageId")
                if pid:
                    best[pid] = max(best.get(pid, 0.0), float(m.get("Similarity", 0)) / 100)
            ranked = sorted(best.items(), key=lambda kv: (-kv[1], kv[0]))
            top = ranked[0] if ranked else None
            faces.append({
                "bbox": bbox, "confidence": conf, "recognized": top is not None,
                "person_id": top[0] if top else None, "name": None,
                "category": "attendance" if top else None,
                "similarity": round(top[1], 4) if top else None,
                "second_similarity": round(ranked[1][1], 4) if len(ranked) > 1 else None,
            })
        return {
            "task": "face_identify", "faces": faces, "face_count": len(faces),
            "recognized_count": sum(1 for f in faces if f["recognized"]),
            "detected_count": detected,
            "dropped": {"too_small": detected - len(sized), "over_limit": len(sized) - len(kept)},
            "frame": {"w": w, "h": h}, "threshold": threshold,
            "registered_faces": len(self._index),
        }

    # ── gallery (main.py routes) ───────────────────────────────────

    def register(self, person_id: str, image: bytes) -> int:
        details = self._client.detect_faces(Image={"Bytes": image}).get("FaceDetails") or []
        if len(details) != 1:
            raise ValueError("no face detected in the supplied image" if not details else
                             f"image contains {len(details)} faces; enrolment requires exactly one face")
        resp = self._client.index_faces(CollectionId=self._collection, Image={"Bytes": image},
                                        ExternalImageId=person_id, MaxFaces=1,
                                        QualityFilter="AUTO", DetectionAttributes=[])
        records = resp.get("FaceRecords") or []
        if not records:
            reasons = [r.get("Reasons") for r in resp.get("UnindexedFaces") or []]
            raise ValueError(f"Rekognition would not index this face (quality): {reasons}")
        with self._lock:
            self._index.setdefault(person_id, []).append(records[0]["Face"]["FaceId"])
            n = len(self._index[person_id])
        self._save_index()
        return n

    def forget(self, person_id: str) -> bool:
        with self._lock:
            ids = list(self._index.get(person_id) or [])
        if not ids:
            return False
        for i in range(0, len(ids), 1000):          # DeleteFaces takes up to 4096
            self._client.delete_faces(CollectionId=self._collection, FaceIds=ids[i:i + 1000])
        with self._lock:
            self._index.pop(person_id, None)
        self._save_index()
        return True

    def people(self) -> dict[str, int]:
        with self._lock:
            return {pid: len(ids) for pid, ids in self._index.items()}


def _pad(bbox: list[int], w: int, h: int, frac: float = 0.15) -> list[int]:
    """A little room around the face: SearchFacesByImage re-detects in
    the crop and misses a face cropped tight to the chin."""
    x1, y1, x2, y2 = bbox
    dx, dy = int((x2 - x1) * frac), int((y2 - y1) * frac)
    return [max(0, x1 - dx), max(0, y1 - dy), min(w, x2 + dx), min(h, y2 + dy)]


def _bounded_float(raw, default, low, high, key):
    if raw is None:
        return default
    try:
        v = float(raw)
    except (TypeError, ValueError):
        v = None
    if v is None or isinstance(raw, bool) or not (low <= v <= high):
        raise ServiceError(ErrorCategory.TRANSPORT_ERROR, code=f"invalid_{key}",
                           message=f"{key} must be in [{low}, {high}]; got {raw!r}",
                           transient=False, http_status=400)
    return v


def _bounded_int(raw, default, low, high, key):
    if raw is None:
        return default
    try:
        v = int(raw)
    except (TypeError, ValueError):
        v = None
    if v is None or isinstance(raw, bool) or not (low <= v <= high):
        raise ServiceError(ErrorCategory.TRANSPORT_ERROR, code=f"invalid_{key}",
                           message=f"{key} must be an integer in [{low}, {high}]; got {raw!r}",
                           transient=False, http_status=400)
    return v
