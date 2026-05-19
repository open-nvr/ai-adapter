# Copyright (c) 2026 OpenNVR
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
YoloV8Service — Yolo-specific implementation of ``AdapterService``.

Migrated to ``opennvr-adapter-sdk`` in A2.3b. All cross-adapter
boilerplate (auth, metrics, FastAPI routes, request body parsing,
error-envelope translation, lifespan) is now in the SDK; this module
holds only the YOLOv8-specific pieces:

* Load lifecycle around the legacy ``YOLOv8Adapter``
* Live ``model.fingerprint`` from the ONNX weights
* §3.3 ``HardwareEvaluationResponse`` (CPU vs CUDA verdict)
* §3.5 ``infer(payload)`` — driven by the SDK body parser
* §6 ``handle_stream(websocket)`` — the full WS protocol loop

The legacy ``YOLOv8Adapter`` (in ``app/adapters/vision/yolov8_adapter.py``)
stays untouched — it's the underlying model wrapper, not the contract
shim.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import time
import uuid
from datetime import datetime, timezone
from typing import Any

from fastapi import WebSocket, WebSocketDisconnect

from adapters.yolov8.coco_classes import class_id_to_label
from app.adapters.vision.yolov8_adapter import YOLOv8Adapter
from app.config import INPUT_SIZE, MODEL_WEIGHTS_DIR
from app.interfaces.contract import (
    DetectionItem,
    DetectionResult,
    ErrorCategory,
    FrameDimensions,
    FrameMessage,
    FrameTransport,
    HandshakeAckMessage,
    HandshakeMessage,
    HardwareEvaluationResponse,
    HardwareVerdict,
    HealthStatus,
    InferResponse,
    ModelInfo,
    NormalizedBBox,
    ResultMessage,
    StreamCloseCode,
)
from opennvr_adapter_sdk import AdapterService, BODY_BYTES_KEY, ServiceError

logger = logging.getLogger(__name__)

MODEL_FRAMEWORK: str = "onnxruntime"

# Default request body cap for /infer (per §3.8 — adapters MAY
# advertise lower limits via capabilities). 8 MiB comfortably holds
# a 4K JPEG; oversize bodies are rejected with malformed_input by the
# SDK before reaching ``infer()``.
MAX_IMAGE_BYTES: int = 8 * 1024 * 1024

# Confidence threshold default if the caller doesn't supply one.
DEFAULT_CONFIDENCE_THRESHOLD: float = 0.25


class YoloV8Service(AdapterService):
    """Stateful façade around YOLOv8Adapter."""

    def __init__(self, weights_path: str | None = None) -> None:
        self._weights_path = weights_path or os.path.join(MODEL_WEIGHTS_DIR, "yolov8n.onnx")
        self._adapter: YOLOv8Adapter = YOLOv8Adapter(
            config={"enabled": True, "weights_path": self._weights_path}
        )
        self._load_state: HealthStatus = HealthStatus.LOADING
        self._load_error: str | None = None
        self._fingerprint_cache: str | None = None
        self._gpu_in_use: bool = False

    # ── AdapterService impl ────────────────────────────────────────

    def load(self) -> None:
        """Eagerly load the ONNX weights. Idempotent."""
        if self._load_state == HealthStatus.OK:
            return
        try:
            self._adapter.ensure_model_loaded()
            self._fingerprint_cache = self._compute_fingerprint()
            self._gpu_in_use = self._detect_gpu_in_use()
            self._load_state = HealthStatus.OK
            self._load_error = None
            logger.info(
                "YoloV8Service ready: weights=%s fingerprint=%s gpu=%s",
                self._weights_path,
                self._fingerprint_cache,
                self._gpu_in_use,
            )
        except Exception as exc:
            self._load_state = HealthStatus.ERROR
            self._load_error = str(exc)
            logger.exception("YoloV8Service failed to load weights %s", self._weights_path)

    def is_ready(self) -> bool:
        return self._load_state == HealthStatus.OK

    def fingerprint(self) -> str | None:
        """Recompute live on each call so §11.3 drift detection sees
        weight rotation. ~10ms for a 6 MB ONNX — cheap."""
        try:
            return self._compute_fingerprint()
        except OSError:
            return self._fingerprint_cache

    def model_info(self) -> ModelInfo:
        return ModelInfo(
            name="yolov8n",
            version=self._adapter_model_version(),
            framework=MODEL_FRAMEWORK,
            size_mb=self._weights_size_mb(),
            modalities_in=["image"],
            modalities_out=["bbox_classes"],
            fingerprint=self.fingerprint(),
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

    def infer(self, payload: dict[str, Any]) -> InferResponse:
        """SDK /infer entry point. The image bytes live at
        ``payload[BODY_BYTES_KEY]`` (set by the SDK's IMAGE-shape body
        parser); the rest of the dict is request params
        (confidence_threshold, classes, etc.)."""
        image_bytes = payload.get(BODY_BYTES_KEY)
        if not isinstance(image_bytes, (bytes, bytearray)) or not image_bytes:
            raise ServiceError(
                ErrorCategory.TRANSPORT_ERROR,
                code="malformed_input",
                message="Frame bytes are required.",
                transient=False,
                http_status=400,
            )
        params = {k: v for k, v in payload.items() if k != BODY_BYTES_KEY}
        return self._infer_image_bytes(bytes(image_bytes), params)

    # ── §6 WebSocket streaming protocol ────────────────────────────

    async def handle_stream(self, websocket: WebSocket) -> None:
        """Implements the full §6 WS protocol. The SDK has already
        verified the bearer token and wrapped this call with
        ``inc/dec_stream_connection``.

        Flow:
          1. Accept the upgrade.
          2. Receive the first message; must be a ``handshake``. Reply
             with ``handshake_ack`` echoing the negotiated transport
             (downgrades shared_memory → websocket since A2.2b hasn't
             landed yet).
          3. Loop: receive either a control message (pause/resume/
             close/stats) or a ``frame`` JSON message followed by a
             binary message carrying the frame bytes. Send back a
             ``result`` message after inference.
          4. On protocol violation, close with the §6.5 code.
        """
        await websocket.accept()
        session_id = uuid.uuid4().hex

        # ── Handshake ──────────────────────────────────────────────
        try:
            first_raw = await websocket.receive_text()
            handshake = HandshakeMessage.model_validate(json.loads(first_raw))
        except (WebSocketDisconnect, json.JSONDecodeError):
            await websocket.close(
                code=StreamCloseCode.POLICY_REFUSED.value,
                reason="bad handshake",
            )
            return
        except Exception as exc:
            logger.info("handshake validation failed: %s", exc)
            await websocket.close(
                code=StreamCloseCode.POLICY_REFUSED.value,
                reason="bad handshake",
            )
            return

        # Reject shared-memory offers with a websocket fallback —
        # §6.1 allows the adapter to downgrade the transport in the
        # ack. A2.2b will land shm support; bump
        # shared_memory_protocol_version to 1 then.
        ack = HandshakeAckMessage(
            frame_transport=FrameTransport.WEBSOCKET,
            result_sink="websocket",  # NATS sink lands with B1
            max_inflight=1,            # serial inference for v1
            session_id=session_id,
        )
        await websocket.send_text(json.dumps(ack.model_dump(mode="json")))

        # Service-readiness check after the handshake — clients should
        # see a typed close rather than hangs if weights never loaded.
        if not self.is_ready():
            await websocket.close(
                code=StreamCloseCode.MODEL_ERROR.value,
                reason="model not loaded",
            )
            return

        logger.info(
            "stream open session_id=%s client_id=%s camera_id=%s",
            session_id,
            handshake.client_id,
            handshake.camera_id or "-",
        )

        # ── Message loop ───────────────────────────────────────────
        paused = False
        while True:
            try:
                msg = await websocket.receive()
            except WebSocketDisconnect:
                logger.info("stream closed (client disconnect) session_id=%s", session_id)
                return

            if msg.get("type") == "websocket.disconnect":
                return

            text = msg.get("text")
            if text is not None:
                try:
                    payload = json.loads(text)
                except json.JSONDecodeError:
                    await websocket.close(
                        code=StreamCloseCode.POLICY_REFUSED.value,
                        reason="non-JSON control message",
                    )
                    return
                msg_type = payload.get("type")

                if msg_type == "close":
                    return
                if msg_type == "pause":
                    paused = True
                    continue
                if msg_type == "resume":
                    paused = False
                    continue
                if msg_type == "stats":
                    await websocket.send_text(json.dumps({
                        "type": "stats",
                        "inflight": 0,
                        "queue_depth": 0,
                        "fps": 0.0,
                    }))
                    continue
                if msg_type == "frame":
                    try:
                        frame_meta = FrameMessage.model_validate(payload)
                    except Exception:
                        await websocket.close(
                            code=StreamCloseCode.POLICY_REFUSED.value,
                            reason="bad frame metadata",
                        )
                        return
                    try:
                        binary_msg = await websocket.receive()
                    except WebSocketDisconnect:
                        return
                    frame_bytes = binary_msg.get("bytes")
                    if not isinstance(frame_bytes, (bytes, bytearray)) or not frame_bytes:
                        await websocket.close(
                            code=StreamCloseCode.POLICY_REFUSED.value,
                            reason="frame must be followed by binary message",
                        )
                        return

                    if paused:
                        # Per §6.4 — once paused, drop frames until resume.
                        continue

                    metrics = self.metrics
                    metrics.inc_inflight()
                    try:
                        result_dict = self._infer_frame_for_stream(
                            bytes(frame_bytes),
                            seq=frame_meta.seq,
                            ts_ms=frame_meta.ts_ms,
                        )
                        await websocket.send_text(json.dumps(result_dict))
                        latency_seconds = result_dict.get("inference_ms", 0) / 1000.0
                        result_payload = result_dict.get("result") or {}
                        if (
                            isinstance(result_payload, dict)
                            and result_payload.get("status") == "error"
                        ):
                            category_value = (result_payload.get("error") or {}).get(
                                "category", ""
                            )
                            outcome = _outcome_for_category_value(category_value)
                        else:
                            outcome = "ok"
                        metrics.record_infer(outcome, latency_seconds)
                    finally:
                        metrics.dec_inflight()
                    continue

                await websocket.close(
                    code=StreamCloseCode.POLICY_REFUSED.value,
                    reason=f"unknown message type: {msg_type}",
                )
                return

            if msg.get("bytes") is not None:
                await websocket.close(
                    code=StreamCloseCode.POLICY_REFUSED.value,
                    reason="binary frame without preceding frame metadata",
                )
                return

    # ── Inference core ─────────────────────────────────────────────

    def _infer_image_bytes(
        self,
        image_bytes: bytes,
        params: dict[str, Any],
    ) -> InferResponse:
        """Run YOLOv8 against raw image bytes. Shared by the HTTP and
        WS paths so they produce identical InferResponse shapes."""
        if self._load_state != HealthStatus.OK:
            raise ServiceError(
                ErrorCategory.MODEL_ERROR,
                code=(
                    "weights_missing"
                    if self._load_state == HealthStatus.ERROR
                    else "yolov8.model_loading"
                ),
                message=self._load_error or "Model still loading.",
                transient=(self._load_state == HealthStatus.LOADING),
                http_status=503,
                retry_after_ms=2000 if self._load_state == HealthStatus.LOADING else None,
            )

        # The SDK already enforces ``max_body_bytes`` before calling
        # us, but we keep a defense-in-depth check for the WS path
        # (the SDK doesn't see those bytes).
        if len(image_bytes) > MAX_IMAGE_BYTES:
            raise ServiceError(
                ErrorCategory.TRANSPORT_ERROR,
                code="malformed_input",
                message=(
                    f"Frame exceeds {MAX_IMAGE_BYTES}-byte limit "
                    f"({len(image_bytes)} received)."
                ),
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

    def _infer_frame_for_stream(
        self,
        image_bytes: bytes,
        seq: int,
        ts_ms: int,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Run inference and shape the result as a §6.3 ``result``
        message. Error paths embed a §7 FailureEnvelope in the result
        body — same wire shape as the HTTP /infer error response so
        downstream parsers handle both identically."""
        params = params or {}
        try:
            infer = self._infer_image_bytes(image_bytes, params)
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

    def _run_inference(
        self,
        img: Any,
        confidence_threshold: float,
    ) -> tuple[list[dict[str, Any]], int]:
        """Bridge to the legacy adapter. We can't reuse ``infer_local``
        directly because it loads from a URI; instead we drive the
        underlying ``_preprocess`` / ``_run_inference`` directly so the
        bytes path stays in-memory (no temp-file dance)."""
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
            cx, cy, w, h = det["cx"], det["cy"], det["w"], det["h"]
            if w < 1.0:
                x = max(0.0, min(1.0, cx - w / 2.0))
                y = max(0.0, min(1.0, cy - h / 2.0))
                nw = max(0.0, min(1.0 - x, w))
                nh = max(0.0, min(1.0 - y, h))
            else:
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

    # ── Helpers ────────────────────────────────────────────────────

    def _adapter_model_version(self) -> str:
        return f"{MODEL_FRAMEWORK}/yolov8n"

    def _weights_size_mb(self) -> float | None:
        try:
            return round(os.path.getsize(self._weights_path) / (1024 * 1024), 2)
        except OSError:
            return None

    def _compute_fingerprint(self) -> str:
        if not os.path.exists(self._weights_path):
            return "sha256:unavailable"
        digest = hashlib.sha256()
        with open(self._weights_path, "rb") as fh:
            for chunk in iter(lambda: fh.read(65536), b""):
                digest.update(chunk)
        return f"sha256:{digest.hexdigest()}"

    def _detect_gpu_in_use(self) -> bool:
        """True if onnxruntime picked CUDAExecutionProvider over CPU."""
        try:
            providers = self._adapter.session.get_providers()
            return "CUDAExecutionProvider" in providers
        except Exception:
            return False


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


# ── Outcome category mapping (shared with the SDK route layer) ─────


_CATEGORY_TO_OUTCOME: dict[str, str] = {
    "model_error": "model_error",
    "transport_error": "transport_error",
    "provider_error": "provider_error",
    "permission_denied": "refused",
    "overloaded": "refused",
    "not_supported": "refused",
}


def _outcome_for_category_value(value: str) -> str:
    """Map the §7 category wire-string to the Prometheus outcome label."""
    return _CATEGORY_TO_OUTCOME.get(value, "model_error")
