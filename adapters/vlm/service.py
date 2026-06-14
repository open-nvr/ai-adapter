# Copyright (c) 2026 OpenNVR
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Open-vocabulary detection adapter — contract-compliant ``AdapterService``.

Where YOLOv8 detects a fixed set of COCO classes, this adapter detects
*whatever text you ask for*: "red truck", "person wearing a backpack",
"forklift", "license plate". The caller passes a list of free-text
queries with the image; the model returns a box + score for each query
phrase it finds. Backed by OWL-ViT v2 (``google/owlv2-base-patch16-
ensemble``) from HuggingFace.

Single task: ``open_vocab_detection``.

Why this exists: it gives OpenNVR precise *attribute* detection — colour,
clothing, object types outside COCO — that a fixed-class detector and a
scene captioner can only approximate. The ``footage-search`` example can
point its indexer here for sharper "red truck" / "yellow jacket"
matching; an application can detect site-specific objects without
training a model.

CPU-runnable (slow: ~3-8 s/image on a 4-core box, query-count dependent);
much faster with CUDA passthrough. First inference triggers the
HuggingFace download — pre-bake the model or mount a populated HF cache
to avoid first-request latency.

Output bboxes are normalized (x, y, w, h in [0, 1], top-left origin) to
match the contract's §5.1 ``NormalizedBBox`` — identical shape to the
YOLOv8 adapter, so zones, counting, and footage-search consume them with
no special-casing.
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

logger = logging.getLogger(__name__)

# Default HuggingFace model. owlv2-base is a good accuracy/size balance
# (~600 MB). Swap to ``google/owlvit-base-patch32`` (smaller, faster,
# less accurate) or an owlv2-large variant via OPENNVR_VLM_MODEL.
DEFAULT_MODEL_ID: str = "google/owlv2-base-patch16-ensemble"

MAX_IMAGE_BYTES: int = 8 * 1024 * 1024

# Score floor below which detections are dropped. Open-vocab models are
# noisier than fixed-class ones, so this is conservative; callers can
# raise it per-request via ``threshold``.
DEFAULT_THRESHOLD: float = 0.10

# Cap on the number of query phrases per call — each query is a forward
# pass cost, and an unbounded list is a DoS vector.
MAX_QUERIES: int = 32

_SUPPORTED_TASKS: tuple[str, ...] = ("open_vocab_detection",)


class VlmService(AdapterService):
    """SDK-based open-vocabulary detection service backed by OWL-ViT v2."""

    def __init__(self, model_id: str | None = None) -> None:
        self._model_id: str = model_id or os.getenv("OPENNVR_VLM_MODEL", DEFAULT_MODEL_ID)
        self._processor: Any | None = None
        self._model: Any | None = None
        self._device: str | None = None
        self._load_state: HealthStatus = HealthStatus.LOADING
        self._load_error: str | None = None
        self._fingerprint_cache: str | None = None
        self._lock = threading.Lock()
        self._started_at: datetime = datetime.now(timezone.utc)

    # ── AdapterService impl ────────────────────────────────────────

    def load(self) -> None:
        with self._lock:
            if self._load_state == HealthStatus.OK:
                return
            try:
                import torch
                from transformers import Owlv2ForObjectDetection, Owlv2Processor

                logger.info("VlmService: loading %s", self._model_id)
                self._processor = Owlv2Processor.from_pretrained(self._model_id)
                self._model = Owlv2ForObjectDetection.from_pretrained(self._model_id)
                self._device = "cuda" if torch.cuda.is_available() else "cpu"
                self._model.to(self._device)
                self._model.eval()
                self._fingerprint_cache = self._compute_fingerprint()
                self._load_state = HealthStatus.OK
                self._load_error = None
                logger.info(
                    "VlmService ready: model=%s device=%s fingerprint=%s",
                    self._model_id, self._device, self._fingerprint_cache,
                )
            except Exception as exc:
                self._load_state = HealthStatus.ERROR
                self._load_error = str(exc)
                logger.exception("VlmService failed to load model %s", self._model_id)

    def is_ready(self) -> bool:
        return self._load_state == HealthStatus.OK

    def fingerprint(self) -> str | None:
        try:
            return self._compute_fingerprint()
        except OSError:
            return self._fingerprint_cache

    def model_info(self) -> ModelInfo:
        return ModelInfo(
            name=self._model_id,
            version=f"owlv2/{self._model_id}",
            framework="transformers",
            modalities_in=["image", "text"],
            modalities_out=["detections"],
            fingerprint=self.fingerprint(),
        )

    def hardware_evaluation(self) -> HardwareEvaluationResponse:
        if self._load_state == HealthStatus.OK:
            verdict = HardwareVerdict.OK
            reasoning = (
                f"OWL-ViT loaded on device {self._device!r}. CPU inference "
                "works but is slow; GPU strongly recommended for video rates."
            )
        elif self._load_state == HealthStatus.LOADING:
            verdict = HardwareVerdict.WARN
            reasoning = "Model still loading."
        else:
            verdict = HardwareVerdict.BLOCKED
            reasoning = f"Model failed to load: {self._load_error}"
        uptime_s = int((datetime.now(timezone.utc) - self._started_at).total_seconds())
        return HardwareEvaluationResponse(
            verdict=verdict,
            reasoning=reasoning,
            checked_at=datetime.now(timezone.utc),
            details={
                "gpu_required": False,
                "gpu_recommended": True,
                "device": self._device or "not_loaded",
                "cpu_count": os.cpu_count() or 0,
                "platform": platform.platform(),
                "model_id": self._model_id,
                "adapter_uptime_seconds": uptime_s,
            },
        )

    def infer(self, payload: dict[str, Any]) -> InferResponse:
        """Run open-vocabulary detection.

        ``payload['__file__']`` holds the image bytes. ``payload['queries']``
        is a list of text phrases to detect (e.g. ``["red truck",
        "person"]``); ``payload['threshold']`` optionally overrides the
        score floor.
        """
        if not self.is_ready():
            transient = self._load_state == HealthStatus.LOADING
            raise ServiceError(
                ErrorCategory.MODEL_ERROR,
                code="model_not_loaded",
                message=self._load_error or "OWL-ViT model not yet loaded",
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
                    "vlm expects an image in the request body (multipart "
                    "'frame' field or JSON 'frame_b64')."
                ),
                transient=False,
                http_status=400,
            )

        task = (payload.get("task") or "open_vocab_detection").strip()
        if task not in _SUPPORTED_TASKS:
            raise ServiceError(
                ErrorCategory.NOT_SUPPORTED,
                code="unsupported_task",
                message=f"task {task!r} is not supported; choose one of {list(_SUPPORTED_TASKS)}",
                transient=False,
                http_status=400,
            )

        queries = self._resolve_queries(payload)
        threshold = self._resolve_threshold(payload)

        try:
            image = _decode_image(image_bytes)
        except _ImageDecodeError as exc:
            raise ServiceError(
                ErrorCategory.TRANSPORT_ERROR,
                code="invalid_image",
                message=str(exc),
                transient=False,
                http_status=400,
            ) from exc

        started = time.monotonic()
        try:
            detections = self._run_detect(image, queries=queries, threshold=threshold)
        except Exception as exc:
            logger.exception("OWL-ViT inference failed")
            raise ServiceError(
                ErrorCategory.MODEL_ERROR,
                code="inference_failed",
                message=f"OWL-ViT raised during inference: {exc}",
                transient=True,
                http_status=500,
                retry_after_ms=1000,
            ) from exc
        elapsed_ms = int((time.monotonic() - started) * 1000)

        return InferResponse(
            model_name=self._model_id,
            model_version=f"owlv2/{self._model_id}",
            inference_ms=elapsed_ms,
            result={
                "task": "open_vocab_detection",
                "queries": queries,
                "threshold": threshold,
                "detections": detections,
                "model_id": self._model_id,
                "device": self._device,
            },
        )

    # ── Helpers ────────────────────────────────────────────────────

    def _resolve_queries(self, payload: dict[str, Any]) -> list[str]:
        raw = payload.get("queries")
        # Accept a comma-separated string too (multipart form fields are
        # strings), e.g. queries="red truck, person".
        if isinstance(raw, str):
            raw = [s.strip() for s in raw.split(",")]
        if not isinstance(raw, list) or not raw:
            raise ServiceError(
                ErrorCategory.TRANSPORT_ERROR,
                code="missing_queries",
                message=(
                    "vlm requires a non-empty 'queries' list of text phrases "
                    "to detect, e.g. queries=['red truck', 'person']."
                ),
                transient=False,
                http_status=400,
            )
        queries = [str(q).strip() for q in raw if str(q).strip()]
        if not queries:
            raise ServiceError(
                ErrorCategory.TRANSPORT_ERROR,
                code="empty_queries",
                message="'queries' contained no non-empty phrases.",
                transient=False,
                http_status=400,
            )
        if len(queries) > MAX_QUERIES:
            raise ServiceError(
                ErrorCategory.TRANSPORT_ERROR,
                code="too_many_queries",
                message=f"at most {MAX_QUERIES} queries per call; got {len(queries)}.",
                transient=False,
                http_status=400,
            )
        return queries

    def _resolve_threshold(self, payload: dict[str, Any]) -> float:
        raw = payload.get("threshold")
        if raw is None:
            return DEFAULT_THRESHOLD
        try:
            value = float(raw)
        except (TypeError, ValueError):
            raise ServiceError(
                ErrorCategory.TRANSPORT_ERROR,
                code="invalid_threshold",
                message=f"threshold must be a number in [0, 1]; got {raw!r}",
                transient=False,
                http_status=400,
            )
        if not 0.0 <= value <= 1.0:
            raise ServiceError(
                ErrorCategory.TRANSPORT_ERROR,
                code="threshold_out_of_range",
                message=f"threshold must be in [0, 1]; got {value}",
                transient=False,
                http_status=400,
            )
        return value

    def _run_detect(
        self, image: Any, *, queries: list[str], threshold: float
    ) -> list[dict[str, Any]]:
        """Run OWL-ViT and return normalized detections. Pulled into a
        helper so tests can monkeypatch a stub without real transformers."""
        import torch

        assert self._processor is not None
        assert self._model is not None
        width, height = image.size
        inputs = self._processor(
            text=[queries], images=image, return_tensors="pt"
        ).to(self._device)
        with torch.no_grad():
            outputs = self._model(**inputs)
        target_sizes = torch.tensor([[height, width]]).to(self._device)
        results = self._processor.post_process_object_detection(
            outputs=outputs, target_sizes=target_sizes, threshold=threshold
        )[0]

        detections: list[dict[str, Any]] = []
        for score, label_idx, box in zip(
            results["scores"], results["labels"], results["boxes"]
        ):
            xmin, ymin, xmax, ymax = [float(v) for v in box.tolist()]
            detections.append({
                "label": queries[int(label_idx)],
                "confidence": round(float(score), 4),
                "bbox": _to_normalized(xmin, ymin, xmax, ymax, width, height),
            })
        # Highest-confidence first for readable downstream summaries.
        detections.sort(key=lambda d: d["confidence"], reverse=True)
        return detections

    def _compute_fingerprint(self) -> str:
        digest = hashlib.sha256(self._model_id.encode("utf-8")).hexdigest()
        return f"sha256-id:{digest}"


# ── Inference helpers ──────────────────────────────────────────────


def _to_normalized(
    xmin: float, ymin: float, xmax: float, ymax: float,
    width: int, height: int,
) -> dict[str, float]:
    """Convert pixel [xmin,ymin,xmax,ymax] to the §5.1 normalized
    {x, y, w, h} (top-left origin, all in [0, 1]), clamped to frame."""
    w = max(1, width)
    h = max(1, height)
    x0 = min(max(xmin / w, 0.0), 1.0)
    y0 = min(max(ymin / h, 0.0), 1.0)
    x1 = min(max(xmax / w, 0.0), 1.0)
    y1 = min(max(ymax / h, 0.0), 1.0)
    return {
        "x": round(x0, 5),
        "y": round(y0, 5),
        "w": round(max(0.0, x1 - x0), 5),
        "h": round(max(0.0, y1 - y0), 5),
    }


class _ImageDecodeError(Exception):
    """Raised when the input bytes don't decode as an image."""


def _decode_image(image_bytes: bytes) -> Any:
    from PIL import Image, UnidentifiedImageError

    try:
        image = Image.open(io.BytesIO(image_bytes))
        image.load()
    except (UnidentifiedImageError, OSError) as exc:
        raise _ImageDecodeError(
            "could not decode image bytes (expected JPEG / PNG / WebP / BMP)"
        ) from exc
    return image.convert("RGB")
