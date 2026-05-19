# Copyright (c) 2026 OpenNVR
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
YoloV8Service — contract-semantics layer around ``YOLOv8Adapter``.

Translates between the AI Adapter Contract v1 wire shapes and the
legacy ``YOLOv8Adapter`` infer interface. Two paths:

* ``infer_bytes(image_bytes, params)`` — runs inference on raw image
  bytes (the multipart /infer path). Returns a typed ``InferResponse``.
* ``infer_frame_for_stream(image_bytes, seq, ts_ms)`` — same inference
  but shaped as a §6.3 ``result`` message for the /infer/stream
  WebSocket loop. Error paths embed a §7 ``FailureEnvelope`` in the
  ``result`` body so HTTP and WS consumers parse failures uniformly.

Both translate the legacy pixel-bbox output to §5.1's normalized
``DetectionResult`` so downstream consumers don't need adapter-aware
parsers.
"""
from __future__ import annotations

import hashlib
import logging
import os
import platform
import time
from datetime import datetime, timezone
from typing import Any

from app.adapters.vision.yolov8_adapter import YOLOv8Adapter
from app.config import INPUT_SIZE, MODEL_WEIGHTS_DIR
from app.interfaces.contract import (
    AdapterInfo,
    CapabilitiesResponse,
    Cost,
    DetectionItem,
    DetectionResult,
    EndpointsInfo,
    ErrorCategory,
    ErrorDetail,
    FailureEnvelope,
    FairQueuing,
    FrameDimensions,
    HardwareEvaluationResponse,
    HardwareVerdict,
    HealthResponse,
    HealthStatus,
    InferEndpointInfo,
    InferResponse,
    ModelInfo,
    NormalizedBBox,
    Permissions,
    ResultMessage,
    Scheduling,
    StreamEndpointInfo,
)
from adapters.yolov8.coco_classes import class_id_to_label

logger = logging.getLogger(__name__)

ADAPTER_NAME: str = "yolov8-object-detection"
ADAPTER_VERSION: str = "1.0.0"
ADAPTER_VENDOR: str = "open-nvr"
ADAPTER_LICENSE: str = "AGPL-3.0"
MODEL_FRAMEWORK: str = "onnxruntime"
TASKS_ADVERTISED: tuple[str, ...] = ("object_detection",)

# Default request body cap for /infer (per §3.8 — adapters MAY
# advertise lower limits via capabilities). 8 MiB comfortably holds
# a 4K JPEG; oversize bodies are rejected with malformed_input.
MAX_IMAGE_BYTES: int = 8 * 1024 * 1024

# Confidence threshold default if the caller doesn't supply one.
# Mirrors app.config.CONFIDENCE_THRESHOLD but kept local so this
# service has a single source of truth.
DEFAULT_CONFIDENCE_THRESHOLD: float = 0.25


class YoloV8Service:
    """Stateful façade around YOLOv8Adapter implementing contract semantics."""

    def __init__(self, weights_path: str | None = None) -> None:
        self._weights_path = weights_path or os.path.join(MODEL_WEIGHTS_DIR, "yolov8n.onnx")
        self._started_at_dt: datetime = datetime.now(timezone.utc)
        self._started_at_mono: float = time.monotonic()

        self._adapter: YOLOv8Adapter = YOLOv8Adapter(
            config={"enabled": True, "weights_path": self._weights_path}
        )

        self._load_state: HealthStatus = HealthStatus.LOADING
        self._load_error: str | None = None
        self._fingerprint: str | None = None
        self._gpu_in_use: bool = False

    # ── Lifecycle ──────────────────────────────────────────────────

    def load(self) -> None:
        """Eagerly load the ONNX weights. Idempotent."""
        if self._load_state == HealthStatus.OK:
            return
        try:
            self._adapter.ensure_model_loaded()
            self._fingerprint = self._compute_fingerprint()
            self._gpu_in_use = self._detect_gpu_in_use()
            self._load_state = HealthStatus.OK
            self._load_error = None
            logger.info(
                "YoloV8Service ready: weights=%s fingerprint=%s gpu=%s",
                self._weights_path,
                self._fingerprint,
                self._gpu_in_use,
            )
        except Exception as exc:
            self._load_state = HealthStatus.ERROR
            self._load_error = str(exc)
            logger.exception("YoloV8Service failed to load weights %s", self._weights_path)

    def _compute_fingerprint(self) -> str:
        if not os.path.exists(self._weights_path):
            return "sha256:unavailable"
        digest = hashlib.sha256()
        with open(self._weights_path, "rb") as fh:
            for chunk in iter(lambda: fh.read(65536), b""):
                digest.update(chunk)
        return f"sha256:{digest.hexdigest()}"

    def _compute_fingerprint_or_cached(self) -> str | None:
        """Recompute the fingerprint live on each /capabilities call so
        weight-rotation and tampering surface as KAI-C drift events.
        Falls back to the cached value if the file became unreadable
        between calls — never crashes /capabilities."""
        try:
            return self._compute_fingerprint()
        except OSError:
            return self._fingerprint

    def _detect_gpu_in_use(self) -> bool:
        """True if onnxruntime picked CUDAExecutionProvider over CPU."""
        try:
            providers = self._adapter.session.get_providers()
            return "CUDAExecutionProvider" in providers
        except Exception:
            return False

    def is_ready(self) -> bool:
        return self._load_state == HealthStatus.OK

    # ── Contract endpoints ─────────────────────────────────────────

    def health(self) -> HealthResponse:
        return HealthResponse(
            status=self._load_state,
            adapter_name=ADAPTER_NAME,
            adapter_version=ADAPTER_VERSION,
            model_name="yolov8n",
            model_version=self._adapter_model_version(),
            started_at=self._started_at_dt,
            uptime_seconds=int(time.monotonic() - self._started_at_mono),
        )

    def capabilities(self) -> CapabilitiesResponse:
        return CapabilitiesResponse(
            adapter=AdapterInfo(
                name=ADAPTER_NAME,
                version=ADAPTER_VERSION,
                vendor=ADAPTER_VENDOR,
                license=ADAPTER_LICENSE,
                model_card_url="https://github.com/ultralytics/ultralytics",
                supported_contract_versions=["1"],
            ),
            model=ModelInfo(
                name="yolov8n",
                version=self._adapter_model_version(),
                framework=MODEL_FRAMEWORK,
                size_mb=self._weights_size_mb(),
                modalities_in=["image"],
                modalities_out=["bbox_classes"],
                # Recompute on every call so KAI-C's §11.3 drift
                # detection catches operator weight rotations and
                # tamper attempts. ~10ms for a 6 MB ONNX — cheap.
                # Falls back to the cached value on read errors
                # so /capabilities never crashes mid-flight.
                fingerprint=self._compute_fingerprint_or_cached(),
            ),
            endpoints=EndpointsInfo(
                infer=InferEndpointInfo(
                    supported=True,
                    # §3.5 — both content types accepted. Real image
                    # bytes via multipart is the preferred path; JSON
                    # with base64 is the fallback for clients that
                    # can't easily build multipart.
                    input_content_types=["multipart/form-data", "application/json"],
                ),
                infer_stream=StreamEndpointInfo(
                    supported=True,
                    max_concurrent_streams=16,
                    # Shared-memory fast path is documented in §6.2 but
                    # not yet implemented in this adapter. Advertise
                    # false so KAI-C never sends frame_ref. A2.2b will
                    # land shm support; bump shared_memory_protocol_version
                    # to 1 then.
                    supports_shared_memory=False,
                ),
            ),
            tasks_advertised=list(TASKS_ADVERTISED),
            permissions=Permissions(
                # §8 — GPU permission requires operator approval at
                # KAI-C registration time.
                gpu=True,
                network_egress=[],
                host_filesystem=[os.path.dirname(self._weights_path)],
                shared_memory_paths=[],
                host_metadata=False,
            ),
            scheduling=Scheduling(
                # max_inflight=1 is the honest value for v1: the
                # underlying onnxruntime session is a shared singleton
                # and we do not serialize inference calls across
                # WebSocket streams. KAI-C uses this as its global cap
                # per §9. Concurrent inference within a single
                # adapter (multiple ONNX sessions, or a thread-safe
                # batch path) lands in a follow-up; bump this then.
                max_inflight=1,
                preferred_batch_size=1,
                # §9 — opt in to KAI-C's per-camera fair queuing so
                # one chatty camera can't starve the rest.
                fair_queuing=FairQueuing.PER_CAMERA,
            ),
            cost=Cost(
                currency="USD",
                estimated_per_call=0.0,
                estimated_per_hour=0.0,
                is_metered=False,
            ),
        )

    def hardware_evaluation(self) -> HardwareEvaluationResponse:
        if self._load_state == HealthStatus.OK:
            if self._gpu_in_use:
                verdict = HardwareVerdict.OK
                reasoning = "GPU detected and in use; weights loaded."
            else:
                verdict = HardwareVerdict.WARN
                reasoning = (
                    "Weights loaded but no CUDA device — running on CPU "
                    "(expect 5-20x slower inference)."
                )
        elif self._load_state == HealthStatus.LOADING:
            verdict = HardwareVerdict.WARN
            reasoning = "Model still loading."
        else:
            verdict = HardwareVerdict.BLOCKED
            reasoning = f"Weights failed to load: {self._load_error}"

        # Probe available providers without forcing a session re-creation.
        providers: list[str] = []
        try:
            import onnxruntime as ort
            providers = list(ort.get_available_providers())
        except Exception:
            pass

        return HardwareEvaluationResponse(
            verdict=verdict,
            reasoning=reasoning,
            checked_at=datetime.now(timezone.utc),
            details={
                "gpu_required": False,
                "gpu_in_use": self._gpu_in_use,
                "onnxruntime_providers": providers,
                "cpu_count": os.cpu_count() or 0,
                "platform": platform.platform(),
                "python_version": platform.python_version(),
                "weights_path": self._weights_path,
            },
        )

    # ── Inference path ─────────────────────────────────────────────

    def infer_bytes(
        self,
        image_bytes: bytes,
        params: dict[str, Any],
    ) -> InferResponse:
        """Run YOLOv8 against a frame's raw bytes. Returns an
        InferResponse whose ``result`` is a §5.1 DetectionResult.

        Raises ``ServiceError`` on every failure path."""
        if self._load_state != HealthStatus.OK:
            raise ServiceError(
                ErrorCategory.MODEL_ERROR,
                code="weights_missing" if self._load_state == HealthStatus.ERROR else "yolov8.model_loading",
                message=self._load_error or "Model still loading.",
                transient=(self._load_state == HealthStatus.LOADING),
                http_status=503,
                retry_after_ms=2000 if self._load_state == HealthStatus.LOADING else None,
            )

        if not image_bytes:
            raise ServiceError(
                ErrorCategory.TRANSPORT_ERROR,
                code="malformed_input",
                message="Frame bytes are required.",
                transient=False,
                http_status=400,
            )
        if len(image_bytes) > MAX_IMAGE_BYTES:
            raise ServiceError(
                ErrorCategory.TRANSPORT_ERROR,
                code="malformed_input",
                message=f"Frame exceeds {MAX_IMAGE_BYTES}-byte limit ({len(image_bytes)} received).",
                transient=False,
                http_status=413,
            )

        raw_threshold = params.get("confidence_threshold", DEFAULT_CONFIDENCE_THRESHOLD)
        try:
            confidence_threshold = float(raw_threshold)
        except (TypeError, ValueError) as exc:
            raise ServiceError(
                ErrorCategory.TRANSPORT_ERROR,
                code="malformed_input",
                message=f"confidence_threshold must be a number, got {raw_threshold!r}.",
                transient=False,
                http_status=400,
            ) from exc
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ServiceError(
                ErrorCategory.TRANSPORT_ERROR,
                code="malformed_input",
                message="confidence_threshold must be between 0.0 and 1.0.",
                transient=False,
                http_status=400,
            )

        start = time.monotonic()
        try:
            img, width, height = _decode_image(image_bytes)
            raw_predictions, raw_count = self._run_inference(img, confidence_threshold)
        except DecodeError as exc:
            raise ServiceError(
                ErrorCategory.TRANSPORT_ERROR,
                code="malformed_input",
                message=str(exc),
                transient=False,
                http_status=400,
            ) from exc
        except Exception as exc:
            logger.exception("YOLOv8 inference raised unexpectedly")
            raise ServiceError(
                ErrorCategory.MODEL_ERROR,
                code="inference_runtime_crash",
                message="Inference failed.",
                transient=False,
                http_status=500,
            ) from exc

        inference_ms = int((time.monotonic() - start) * 1000)
        detection_result = self._shape_detections(
            raw_predictions, width, height, classes_filter=params.get("classes")
        )
        return InferResponse(
            model_name="yolov8n",
            model_version=self._adapter_model_version(),
            inference_ms=inference_ms,
            result={
                **detection_result.model_dump(mode="json"),
                "raw_prediction_count": raw_count,
            },
        )

    def _run_inference(
        self,
        img: Any,
        confidence_threshold: float,
    ) -> tuple[list[dict[str, Any]], int]:
        """
        Bridge to the legacy adapter. We can't reuse ``infer_local``
        directly because it loads from a URI; instead we drive the
        underlying ``_preprocess`` / ``_run_inference`` directly so the
        bytes path stays in-memory (no temp-file dance).
        """
        import numpy as np

        blob = self._adapter._preprocess(img)
        raw = self._adapter._run_inference(blob)
        if raw.ndim == 1:
            raw = np.expand_dims(raw, axis=0)

        detections: list[dict[str, Any]] = []
        for pred in raw:
            if len(pred) < 5:
                continue
            cx, cy, w, h = pred[:4]
            class_scores = pred[4:]
            class_id = int(np.argmax(class_scores))
            confidence = float(class_scores[class_id])
            if confidence < confidence_threshold:
                continue
            detections.append(
                {
                    "cx": float(cx),
                    "cy": float(cy),
                    "w": float(w),
                    "h": float(h),
                    "class_id": class_id,
                    "confidence": confidence,
                }
            )
        return detections, int(raw.shape[0])

    def _shape_detections(
        self,
        detections: list[dict[str, Any]],
        width: int,
        height: int,
        *,
        classes_filter: list[str] | None = None,
    ) -> DetectionResult:
        """Translate pixel/raw detections into a §5.1 DetectionResult
        with normalized [0,1] bboxes and human-readable labels."""
        items: list[DetectionItem] = []
        allowed_labels: set[str] | None = (
            {label.lower() for label in classes_filter}
            if isinstance(classes_filter, list) and classes_filter
            else None
        )

        for det in detections:
            label = class_id_to_label(det["class_id"])
            if allowed_labels is not None and label.lower() not in allowed_labels:
                continue
            # YOLOv8 raw output coordinates are in the model's input
            # resolution (640x640 by default). The legacy adapter
            # handles two cases: outputs already in [0,1] (small w<1)
            # or pixel coordinates relative to INPUT_SIZE. We mirror
            # the same logic but normalize to [0,1] for the contract.
            cx, cy, w, h = det["cx"], det["cy"], det["w"], det["h"]
            if w < 1.0:
                # Already in [0,1] — clamp + center-to-corner conversion.
                x = max(0.0, min(1.0, cx - w / 2.0))
                y = max(0.0, min(1.0, cy - h / 2.0))
                nw = max(0.0, min(1.0 - x, w))
                nh = max(0.0, min(1.0 - y, h))
            else:
                # Pixel coords relative to INPUT_SIZE (640). Normalize.
                x = max(0.0, min(1.0, (cx - w / 2.0) / INPUT_SIZE))
                y = max(0.0, min(1.0, (cy - h / 2.0) / INPUT_SIZE))
                nw = max(0.0, min(1.0 - x, w / INPUT_SIZE))
                nh = max(0.0, min(1.0 - y, h / INPUT_SIZE))

            items.append(
                DetectionItem(
                    label=label,
                    confidence=round(det["confidence"], 4),
                    bbox=NormalizedBBox(x=x, y=y, w=nw, h=nh),
                    track_id=None,
                    attributes={"class_id": det["class_id"]},
                )
            )

        return DetectionResult(
            detections=items,
            frame_dimensions=FrameDimensions(w=width, h=height),
        )

    # ── Streaming-specific inference ────────────────────────────────

    def infer_frame_for_stream(
        self,
        image_bytes: bytes,
        seq: int,
        ts_ms: int,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Run inference and shape the result for the WS ``result``
        message. Returns the full JSON dict that the route will pass
        to ``ResultMessage`` — kept here so the route stays thin.

        Error paths produce a §6.3 ``result`` message whose ``result``
        body is a §7 ``FailureEnvelope`` — same wire shape as the HTTP
        /infer error response so downstream parsers handle both
        identically. (The §6.3 contract allows the adapter to embed
        error-shaped results without closing the socket; the caller
        decides whether to keep going.)
        """
        params = params or {}
        try:
            infer = self.infer_bytes(image_bytes, params)
        except ServiceError as exc:
            envelope = exc.envelope().model_dump(mode="json")
            return ResultMessage(
                seq=seq,
                ts_ms=ts_ms,
                inference_ms=0,
                result=envelope,
            ).model_dump(mode="json")

        return ResultMessage(
            seq=seq,
            ts_ms=ts_ms,
            inference_ms=infer.inference_ms,
            result=infer.result,
        ).model_dump(mode="json")

    # ── Helpers ────────────────────────────────────────────────────

    def _adapter_model_version(self) -> str:
        return f"{MODEL_FRAMEWORK}/yolov8n"

    def _weights_size_mb(self) -> float | None:
        try:
            return round(os.path.getsize(self._weights_path) / (1024 * 1024), 2)
        except OSError:
            return None


# ── Image decode helper ─────────────────────────────────────────────


class DecodeError(Exception):
    """Raised when we can't turn the request bytes into a numpy image."""


def _decode_image(image_bytes: bytes) -> tuple[Any, int, int]:
    """Decode JPEG/PNG bytes into an OpenCV-style BGR numpy array.

    Returns (image, width, height). Raises ``DecodeError`` on any
    decode failure — keeps the caller's exception handling lean.
    """
    import cv2
    import numpy as np

    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise DecodeError("Could not decode frame as JPEG/PNG.")
    height, width = img.shape[:2]
    return img, width, height


# ── Typed error envelope ───────────────────────────────────────────


class ServiceError(Exception):
    """
    Carries enough information to construct a FailureEnvelope without
    re-parsing exception strings in the FastAPI route.
    """

    def __init__(
        self,
        category: ErrorCategory,
        *,
        code: str,
        message: str,
        transient: bool,
        http_status: int,
        retry_after_ms: int | None = None,
    ) -> None:
        super().__init__(message)
        self.category = category
        self.code = code
        self.message = message
        self.transient = transient
        self.http_status = http_status
        self.retry_after_ms = retry_after_ms

    def envelope(self) -> FailureEnvelope:
        return FailureEnvelope(
            error=ErrorDetail(
                category=self.category,
                code=self.code,
                message=self.message,
                transient=self.transient,
                retry_after_ms=self.retry_after_ms,
                details={},
            )
        )
