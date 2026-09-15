# Copyright (c) 2026 OpenNVR
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
YoloPoseService — human-pose implementation of ``AdapterService``.

Wraps a YOLO11n-pose ONNX model on ``onnxruntime`` and returns, per
person in the frame, a box plus the 17 COCO body keypoints. The
adapter is the geometry supplier for apps that have to reason about
what a body is DOING rather than where it is: the wand-compliance app
this was built for (did the guard sweep the detector along the
visitor's arms?) reads wrist and elbow tracks; a fall-detection app
reads torso angle; a queue app reads shoulder orientation.

Design choices worth pinning:

* **CPU is the target, not the fallback.** The calling app runs ~10
  fps per camera on a 4-core box with no GPU, so the defaults are
  chosen for that budget: the nano model, 448 px input (not 640), and
  a single serial ``onnxruntime`` session. GPU is supported — the
  provider list is the same CUDA-first list ``adapters/yolov8/`` uses
  — but nothing about the adapter assumes it.

* **Pixel coordinates, not normalized ones.** §5.1's
  ``DetectionResult`` normalizes boxes to [0, 1] because a detection
  is consumed as a region. A pose is consumed as GEOMETRY — angles
  between joints, wrist travel in the image plane — and normalized
  coordinates silently distort every angle on a non-square frame.
  So this adapter returns pixel coordinates in the input image's own
  frame and ships ``frame_dimensions`` alongside, which is lossless:
  a consumer that wants [0, 1] divides, and one that wants degrees
  doesn't have to un-distort first. The output convention is
  ``persons`` / ``keypoints``, not §5.1 ``detections`` — pretending a
  pose is a detection would smuggle the keypoints into ``attributes``
  where no schema describes them.

* **Letterbox preprocessing.** ``adapters/yolov8/`` stretches the
  frame to a square via ``cv2.dnn.blobFromImage``, which is
  acceptable when the output is a box you re-scale per axis. It is
  NOT acceptable here: a stretched 16:9 frame rotates every limb
  angle the app is trying to measure. We pad to square instead (the
  preprocessing Ultralytics itself trains with) and unmap the padding
  when converting back to pixels.

* **The model file is acquired the way the SDK prescribes** —
  ``ensure_model_file``: a file already present at the weights path
  always wins and no network is touched; a missing file is fetched
  once from ``YOLO_POSE_MODEL_URL`` into the mounted weights volume;
  an empty URL (the default) means "the operator pre-populates,
  never download" and a missing file is then a typed load failure,
  exactly as ``adapters/yolov8/`` fails on missing weights. Every
  later boot is offline either way.

The adapter is self-contained: unlike ``adapters/yolov8/`` it does
not delegate to a legacy ``app/adapters/vision/*`` class, so the image
copies only the SDK and this package.
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

from adapters.yolo_pose.coco_keypoints import COCO_KEYPOINTS, KEYPOINT_STRIDE
from opennvr_adapter_sdk import AdapterService, BODY_BYTES_KEY, ServiceError
from opennvr_adapter_sdk.contract import (
    ErrorCategory,
    FrameMessage,
    FrameTransport,
    HandshakeAckMessage,
    HandshakeMessage,
    HardwareEvaluationResponse,
    HardwareVerdict,
    HealthStatus,
    InferResponse,
    ModelInfo,
    ResultMessage,
    StreamCloseCode,
)
from opennvr_adapter_sdk.model_fetch import ensure_model_file

logger = logging.getLogger(__name__)

MODEL_FRAMEWORK: str = "onnxruntime"

#: Model identity reported on /capabilities. Swapping the weights file
#: without changing this string is exactly the drift §11.3 catches —
#: the fingerprint moves and the name doesn't.
MODEL_NAME: str = "yolo11n-pose"

# Default request body cap for /infer (per §3.8 — adapters MAY
# advertise lower limits via capabilities). 8 MiB comfortably holds
# a 4K JPEG; oversize bodies are rejected with malformed_input by the
# SDK before reaching ``infer()``. Same cap as adapters/yolov8/.
MAX_IMAGE_BYTES: int = 8 * 1024 * 1024

#: Where the ONNX file lives. The Docker image sets
#: ``YOLO_POSE_WEIGHTS_DIR=/weights`` and mounts a volume there; a
#: source checkout falls back to the repo's ``model_weights/``, which
#: is what ``download_models.py`` populates.
DEFAULT_WEIGHTS_DIR: str = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "model_weights")
)
WEIGHTS_FILENAME: str = "yolo11n-pose.onnx"

#: First-boot download source, read at construction time. Ultralytics
#: publishes the ``.pt`` checkpoint but no pre-built ONNX, so there is
#: no upstream URL we could honestly default to — the ONNX is derived
#: locally (``python download_models.py``, or the export recipe in the
#: README) and mounted. An operator who hosts the exported file on
#: their own artifact store sets this and gets the whisper-style
#: first-boot fetch instead. The empty default therefore means "never
#: download", which is also the ``sovereignty=local_only`` posture.
MODEL_URL_ENV: str = "YOLO_POSE_MODEL_URL"

# ── Inference defaults ─────────────────────────────────────────────
# Tuned for the CPU budget described above. All four are overridable
# per call via the /infer params block.

#: Person-confidence floor. 0.4 rather than YOLO's usual 0.25: a pose
#: app acts on skeletons, and a half-confident detection produces a
#: plausible-looking but wrong skeleton, which is worse than no
#: skeleton at all.
DEFAULT_CONF: float = 0.4
#: IoU threshold for the class-agnostic NMS over person boxes.
DEFAULT_IOU: float = 0.45
#: Model input side in pixels. 448 (not 640) is the CPU-first choice:
#: roughly twice the throughput of 640 for a keypoint error that stays
#: well inside what limb-angle logic tolerates at entrance-camera
#: framing. Must be a multiple of the model's stride (32).
DEFAULT_IMGSZ: int = 448
IMGSZ_STRIDE: int = 32
MIN_IMGSZ: int = 160
MAX_IMGSZ: int = 1280
#: Hard cap on persons per frame — a payload/latency guard, not a
#: scene assumption. Keypoints are ~50 numbers per person, so an
#: uncapped pathological frame is a megabyte of JSON at 10 fps.
DEFAULT_MAX_PERSONS: int = 20
ABSOLUTE_MAX_PERSONS: int = 100
#: Per-keypoint confidence at or above which a joint counts as
#: "visible" for the domain metric. Does NOT filter the response —
#: all 17 slots are always returned, each with its own confidence, so
#: consumers apply their own floor.
KEYPOINT_VISIBLE_CONF: float = 0.5

#: Feature count of one raw prediction row: cx, cy, w, h, person
#: score, then 17 × (x, y, conf).
EXPECTED_FEATURES: int = 5 + len(COCO_KEYPOINTS) * KEYPOINT_STRIDE


class YoloPoseService(AdapterService):
    """Stateful façade around the YOLO11n-pose ONNX session."""

    def __init__(self, weights_path: str | None = None) -> None:
        self._weights_path = weights_path or os.path.join(
            os.getenv("YOLO_POSE_WEIGHTS_DIR", DEFAULT_WEIGHTS_DIR),
            WEIGHTS_FILENAME,
        )
        self._model_url: str = os.getenv(MODEL_URL_ENV, "")
        self._session: Any | None = None
        self._input_name: str = "images"
        # Set at load() from the ONNX input signature: an int when the
        # export has a FIXED input size, None when it was exported with
        # dynamic axes. See ``_static_input_size``.
        self._static_imgsz: int | None = None
        self._load_state: HealthStatus = HealthStatus.LOADING
        self._load_error: str | None = None
        self._fingerprint_cache: str | None = None
        self._gpu_in_use: bool = False

    # ── AdapterService impl ────────────────────────────────────────

    def load(self) -> None:
        """Eagerly load the ONNX weights. Idempotent."""
        if self._load_state == HealthStatus.OK:
            return
        # Domain metrics: per-joint visibility is THE diagnostic for
        # this adapter. A camera re-aimed slightly high stops seeing
        # wrists long before anyone notices the wand-compliance app
        # has gone quiet — and "left_wrist went to zero on cam-3" is
        # visible in a single scrape. Label set = the model's own 17
        # joints, so cardinality is bounded by the weights, not by the
        # input.
        self.metrics.register_counter(
            "adapter_pose_keypoints_visible_total",
            "Keypoints returned at or above the visibility floor, by joint.",
            label_key="keypoint", allowed_values=COCO_KEYPOINTS)
        self.metrics.register_counter(
            "adapter_pose_persons_total",
            "Persons returned with a pose.")
        try:
            import onnxruntime as ort  # optional dep: uv sync --extra pose

            ensure_model_file(
                self._weights_path,
                self._model_url,
                label=f"{MODEL_NAME} weights",
                logger=logger,
            )
            # Provider list and session construction mirror
            # adapters/yolov8/ exactly: CUDA first, CPU fallback, and
            # no hand-tuned SessionOptions. onnxruntime's own defaults
            # (intra-op threads = the core count) are what the README's
            # throughput expectations assume; pinning thread counts
            # here would make this adapter behave differently from its
            # sibling on the same host, for no measured gain.
            self._session = ort.InferenceSession(
                self._weights_path,
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            )
            input_meta = self._session.get_inputs()[0]
            self._input_name = input_meta.name
            self._static_imgsz = _static_input_size(
                getattr(input_meta, "shape", None)
            )
            self._fingerprint_cache = self._compute_fingerprint()
            self._gpu_in_use = self._detect_gpu_in_use()
            self._warm_up()
            self._load_state = HealthStatus.OK
            self._load_error = None
            logger.info(
                "YoloPoseService ready: weights=%s fingerprint=%s gpu=%s imgsz=%s",
                self._weights_path,
                self._fingerprint_cache,
                self._gpu_in_use,
                self._static_imgsz or f"dynamic (default {DEFAULT_IMGSZ})",
            )
        except Exception as exc:
            self._load_state = HealthStatus.ERROR
            self._load_error = str(exc)
            logger.exception(
                "YoloPoseService failed to load weights %s", self._weights_path
            )

    def is_ready(self) -> bool:
        return self._load_state == HealthStatus.OK

    def fingerprint(self) -> str | None:
        """Recompute live on each call so §11.3 drift detection sees
        weight rotation. ~10ms for a 12 MB ONNX — cheap."""
        try:
            return self._compute_fingerprint()
        except OSError:
            return self._fingerprint_cache

    def model_info(self) -> ModelInfo:
        return ModelInfo(
            name=MODEL_NAME,
            version=self._adapter_model_version(),
            framework=MODEL_FRAMEWORK,
            size_mb=self._weights_size_mb(),
            modalities_in=["image"],
            # "keypoints", not "pose_keypoints": this string is the
            # platform's vocabulary, and server/config/adapters_index.yml
            # lists this adapter with modalities_out: [keypoints]. Two
            # spellings of one concept is how a registry lookup starts
            # missing an adapter that is right there.
            modalities_out=["keypoints"],
            fingerprint=self.fingerprint(),
        )

    def hardware_evaluation(self) -> HardwareEvaluationResponse:
        cpu_count = os.cpu_count() or 0
        if self._load_state == HealthStatus.OK:
            if self._gpu_in_use:
                verdict = HardwareVerdict.OK
                reasoning = "GPU detected and in use; weights loaded."
            elif cpu_count >= 4:
                # Unlike adapters/yolov8/, CPU is not a degraded mode
                # here — the model and the default 448 px input were
                # picked for it, so the verdict flips on core count
                # rather than on "no CUDA". Four cores is the measured
                # floor for the ~10 fps/camera the streaming path
                # targets.
                verdict = HardwareVerdict.OK
                reasoning = (
                    f"Running on CPU with {cpu_count} cores — the design "
                    f"target for this adapter. ~10 fps/camera at "
                    f"imgsz={DEFAULT_IMGSZ} needs ~8 cores; on 4, run at "
                    f"imgsz=320 or give it one camera."
                )
            else:
                verdict = HardwareVerdict.WARN
                reasoning = (
                    f"Weights loaded but only {cpu_count} CPU cores and no "
                    f"CUDA device — expect well under 10 fps per camera. "
                    f"Lower imgsz, or point fewer cameras at this adapter."
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
                "cpu_count": cpu_count,
                "platform": platform.platform(),
                "python_version": platform.python_version(),
                "weights_path": self._weights_path,
                "default_imgsz": DEFAULT_IMGSZ,
            },
        )

    def infer(self, payload: dict[str, Any]) -> InferResponse:
        """SDK /infer entry point. The image bytes live at
        ``payload[BODY_BYTES_KEY]`` (set by the SDK's IMAGE-shape body
        parser); the rest of the dict is request params (conf, iou,
        imgsz, max_persons)."""
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
        """Implements the §6 WS protocol — the path the wand-compliance
        app uses at 10 fps, where one HTTP request per frame would spend
        more time on connection setup than on inference. The SDK has
        already verified the bearer token and wrapped this call with
        ``inc/dec_stream_connection``.

        Two §6 options are NOT implemented, and both are answered by
        downgrading in the ack rather than by refusing the session:
        §6.2's shared-memory ``frame_ref`` transport (capabilities
        advertise ``supports_shared_memory: false``) and §6.3's NATS
        ``result_sink`` — results always come back over this socket. A
        client that offers either gets ``websocket`` in the ack and
        should read it rather than assume its offer was taken.

        Frames inferred over a stream use the adapter's default
        params: §6's ``frame`` message carries metadata only, and
        ``handshake`` forbids extra fields, so there is nowhere for a
        per-frame ``conf`` to ride. Same limitation (and the same
        defaults-are-the-contract consequence) as
        ``adapters/yolov8/``; callers that need custom thresholds use
        HTTP /infer.

        Flow, identical to ``adapters/yolov8/`` — deliberately, so a
        client that can drive one streaming adapter can drive both:
          1. Accept the upgrade.
          2. Receive the first message; must be a ``handshake``. Reply
             with ``handshake_ack`` echoing the negotiated transport
             (downgrades shared_memory → websocket since shm isn't
             implemented).
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
        # ack. Flip ``stream_supports_shared_memory`` in main.py if
        # shm ever lands here.
        ack = HandshakeAckMessage(
            frame_transport=FrameTransport.WEBSOCKET,
            result_sink="websocket",
            max_inflight=1,            # serial inference, see main.py
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
        # Frames inferred on THIS session, for the §6.4 stats reply.
        # Session-scoped rather than adapter-wide: a client asking for
        # stats is asking what its own camera is getting.
        frames_done = 0
        session_start = time.monotonic()
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
                    # §6.4 says this reply carries the real numbers. The
                    # SDK is already maintaining the two gauges; fps is
                    # this session's own average since the handshake,
                    # which is what a client tuning its send rate wants.
                    gauges = self.metrics.gauges()
                    elapsed = max(time.monotonic() - session_start, 1e-6)
                    await websocket.send_text(json.dumps({
                        "type": "stats",
                        "inflight": gauges["inflight"],
                        "queue_depth": gauges["queue_depth"],
                        "fps": round(frames_done / elapsed, 2),
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
                        frames_done += 1
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
        """Run pose estimation against raw image bytes. Shared by the
        HTTP and WS paths so they produce identical InferResponse
        shapes."""
        if self._load_state != HealthStatus.OK:
            raise ServiceError(
                ErrorCategory.MODEL_ERROR,
                code=(
                    "weights_missing"
                    if self._load_state == HealthStatus.ERROR
                    else "yolo_pose.model_loading"
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

        conf = _float_param(params, "conf", DEFAULT_CONF, 0.0, 1.0,
                            aliases=("confidence_threshold",))
        iou = _float_param(params, "iou", DEFAULT_IOU, 0.0, 1.0,
                           aliases=("iou_threshold", "nms_threshold"))
        imgsz = self._resolve_imgsz(params)
        max_persons = _int_param(params, "max_persons", DEFAULT_MAX_PERSONS,
                                 1, ABSOLUTE_MAX_PERSONS)

        start = time.monotonic()
        # Decode, inference AND post-processing all live inside this
        # guard. Post-processing used to sit outside it, which meant an
        # unexpected numpy error while shaping the response escaped as a
        # bare exception: the SDK route only catches ServiceError, so the
        # caller got a 500 with no §7 envelope and the failure was never
        # counted by record_infer. Every exit from this method is now a
        # typed ServiceError.
        try:
            img, width, height = _decode_image(image_bytes)
            raw, scale, pad_x, pad_y = self._run_inference(img, imgsz)
            persons = self._shape_persons(
                raw,
                scale=scale,
                pad_x=pad_x,
                pad_y=pad_y,
                width=width,
                height=height,
                conf=conf,
                iou=iou,
                max_persons=max_persons,
            )
        except DecodeError as exc:
            raise ServiceError(
                ErrorCategory.TRANSPORT_ERROR,
                code="malformed_input",
                message=str(exc),
                transient=False,
                http_status=400,
            ) from exc
        except ServiceError:
            raise
        except Exception as exc:
            logger.exception("YOLO-pose inference raised unexpectedly")
            raise ServiceError(
                ErrorCategory.MODEL_ERROR,
                code="inference_runtime_crash",
                message="Inference failed.",
                transient=False,
                http_status=500,
            ) from exc

        inference_ms = int((time.monotonic() - start) * 1000)

        try:
            self.metrics.inc_counter("adapter_pose_persons_total", len(persons))
            for person in persons:
                for index, keypoint in enumerate(person["keypoints"]):
                    if keypoint[2] >= KEYPOINT_VISIBLE_CONF:
                        self.metrics.inc_counter(
                            "adapter_pose_keypoints_visible_total",
                            label_value=COCO_KEYPOINTS[index],
                        )
        except Exception:  # pragma: no cover - metrics must never break infer
            logger.debug("yolo-pose domain metrics recording failed", exc_info=True)

        return InferResponse(
            model_name=MODEL_NAME,
            model_version=self._adapter_model_version(),
            inference_ms=inference_ms,
            result={
                "persons": persons,
                # Echoed so a consumer never has to hard-code the COCO
                # order to know which slot is a wrist, and so a future
                # model with a different skeleton cannot silently
                # change the meaning of index 9.
                "keypoint_names": list(COCO_KEYPOINTS),
                # The pixel coordinates above are only interpretable
                # against the frame they were measured in.
                "frame_dimensions": {"w": width, "h": height},
            },
        )

    def _resolve_imgsz(self, params: dict[str, Any]) -> int:
        """Pick the input size for this call.

        An Ultralytics ONNX export is fixed-size unless it was
        exported with ``dynamic=True``, and feeding a fixed-size graph
        anything else is an opaque onnxruntime shape error. So: a
        fixed-size export dictates the size when the caller didn't ask
        for one (a 640 export just works, with no configuration), and
        a caller who DID ask for something else gets told why it can't
        be honoured instead of a 500 from deep inside the session.
        """
        requested = _imgsz_param(params)
        if self._static_imgsz is None:
            return requested
        if "imgsz" in params and requested != self._static_imgsz:
            raise ServiceError(
                ErrorCategory.TRANSPORT_ERROR,
                code="malformed_input",
                message=(
                    f"This model was exported with a fixed "
                    f"{self._static_imgsz}px input, so imgsz={requested} "
                    f"cannot be served. Use imgsz={self._static_imgsz}, or "
                    f"re-export with dynamic=True to make imgsz tunable."
                ),
                transient=False,
                http_status=400,
            )
        return self._static_imgsz

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
        except Exception:
            # Belt and braces. _infer_image_bytes types every failure it
            # knows about, but an exception that escapes here would
            # propagate out of handle_stream and tear the socket down
            # mid-session: the client sees an abrupt close instead of a
            # §6.5 code or an error result, and the frame never reaches
            # record_infer. One bad frame must not end a camera's stream.
            logger.exception("YOLO-pose stream frame failed unexpectedly seq=%s", seq)
            envelope = ServiceError(
                ErrorCategory.MODEL_ERROR,
                code="inference_runtime_crash",
                message="Inference failed.",
                transient=False,
                http_status=500,
            ).envelope().model_dump(mode="json")
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
        imgsz: int,
    ) -> tuple[Any, float, float, float]:
        """Preprocess, run the session, and return the raw predictions
        as an ``(N, 56)`` array plus the letterbox geometry needed to
        map model coordinates back to source pixels.

        The output transpose matches ``adapters/yolov8/``: Ultralytics
        ONNX exports emit ``(1, features, anchors)`` and every consumer
        wants ``(anchors, features)``.
        """
        import numpy as np

        blob, scale, pad_x, pad_y = _letterbox_blob(img, imgsz)
        outputs = self._session.run(None, {self._input_name: blob})
        raw = np.transpose(outputs[0], (0, 2, 1)).squeeze()
        if raw.ndim == 1:
            raw = np.expand_dims(raw, axis=0)

        # A detection-only export (84 features) pointed at this
        # adapter would otherwise produce silently truncated
        # keypoints. Fail typed instead — this is a misconfiguration,
        # not a bad frame.
        if raw.shape[-1] != EXPECTED_FEATURES:
            raise ServiceError(
                ErrorCategory.MODEL_ERROR,
                code="yolo_pose.unexpected_model_output",
                message=(
                    f"Model emitted {raw.shape[-1]} features per prediction, "
                    f"expected {EXPECTED_FEATURES} (4 box + 1 score + "
                    f"{len(COCO_KEYPOINTS)}x{KEYPOINT_STRIDE} keypoints). "
                    f"Is {self._weights_path} a pose export?"
                ),
                transient=False,
                http_status=500,
            )
        return raw, scale, pad_x, pad_y

    def _shape_persons(
        self,
        raw: Any,
        *,
        scale: float,
        pad_x: float,
        pad_y: float,
        width: int,
        height: int,
        conf: float,
        iou: float,
        max_persons: int,
    ) -> list[dict[str, Any]]:
        """Turn raw model rows into the documented ``persons`` list:
        pixel-space ``bbox`` / ``keypoints`` in the SOURCE image's
        frame, NMS-ed, sorted by descending score and capped."""
        import numpy as np

        scores = raw[:, 4].astype(float)
        keep_mask = scores >= conf
        if not bool(keep_mask.any()):
            return []
        rows = raw[keep_mask]
        scores = scores[keep_mask]

        # Model-space xywh (centre form) → letterboxed xyxy, then
        # un-letterbox into source pixels. NMS runs in source pixels so
        # the IoU a caller tunes means what they think it means.
        cx, cy, bw, bh = rows[:, 0], rows[:, 1], rows[:, 2], rows[:, 3]
        x1 = (cx - bw / 2.0 - pad_x) / scale
        y1 = (cy - bh / 2.0 - pad_y) / scale
        x2 = (cx + bw / 2.0 - pad_x) / scale
        y2 = (cy + bh / 2.0 - pad_y) / scale
        boxes = np.stack([x1, y1, x2, y2], axis=1).astype(float)
        np.clip(boxes[:, 0::2], 0.0, float(width), out=boxes[:, 0::2])
        np.clip(boxes[:, 1::2], 0.0, float(height), out=boxes[:, 1::2])

        order = _nms(boxes, scores, iou)[:max_persons]

        persons: list[dict[str, Any]] = []
        for index in order:
            row = rows[index]
            keypoints: list[list[float]] = []
            for slot in range(len(COCO_KEYPOINTS)):
                base = 5 + slot * KEYPOINT_STRIDE
                kx = (float(row[base]) - pad_x) / scale
                ky = (float(row[base + 1]) - pad_y) / scale
                kconf = float(row[base + 2])
                # Clamp into the frame: a joint predicted just outside
                # the border is a real joint the model placed slightly
                # wrong, and consumers index pixels with these.
                kx = min(max(kx, 0.0), float(width))
                ky = min(max(ky, 0.0), float(height))
                keypoints.append([
                    round(kx, 1),
                    round(ky, 1),
                    round(min(max(kconf, 0.0), 1.0), 4),
                ])
            box = boxes[index]
            persons.append({
                "bbox": [
                    round(float(box[0]), 1),
                    round(float(box[1]), 1),
                    round(float(box[2]), 1),
                    round(float(box[3]), 1),
                ],
                "score": round(float(scores[index]), 4),
                "keypoints": keypoints,
            })
        return persons

    # ── Helpers ────────────────────────────────────────────────────

    def _warm_up(self) -> None:
        """Run one throwaway inference on a black frame.

        ``adapters/yolov8/`` has no warm-up because its callers are
        event-driven and one slow first frame is invisible. This
        adapter is opened as a 10 fps stream: without a warm-up the
        first real frame pays onnxruntime's arena allocation and
        thread-pool spin-up (several times the steady-state latency)
        at exactly the moment a camera connects. Failures here are
        logged and swallowed — a warm-up is an optimisation, never a
        reason for /health to go red.
        """
        try:
            import numpy as np

            imgsz = self._static_imgsz or DEFAULT_IMGSZ
            blank = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
            blob, _, _, _ = _letterbox_blob(blank, imgsz)
            self._session.run(None, {self._input_name: blob})
        except Exception:  # pragma: no cover - never fatal
            logger.debug("yolo-pose warm-up inference failed", exc_info=True)

    def _adapter_model_version(self) -> str:
        return f"{MODEL_FRAMEWORK}/{MODEL_NAME}"

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
            providers = self._session.get_providers()
            return "CUDAExecutionProvider" in providers
        except Exception:
            return False


# ── Request-parameter parsing ───────────────────────────────────────
# Every caller-supplied number is validated here rather than at the
# numpy call site: a string ``conf`` has to come back as a typed 400,
# not as a ValueError the SDK can only translate into a 500.


def _float_param(
    params: dict[str, Any],
    name: str,
    default: float,
    low: float,
    high: float,
    *,
    aliases: tuple[str, ...] = (),
) -> float:
    """Read a float param. ``aliases`` exist because the sibling
    adapters spell the same knob differently (yolov8 takes
    ``confidence_threshold``); accepting both keeps a caller from
    having to special-case which vision adapter it is talking to."""
    raw = params.get(name)
    for alias in aliases:
        if raw is None:
            raw = params.get(alias)
    if raw is None:
        return default
    if isinstance(raw, bool):  # bool is an int subclass — reject explicitly,
        raw = repr(raw)        # same as _int_param. conf=true is a typo, not 1.0.
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise ServiceError(
            ErrorCategory.TRANSPORT_ERROR,
            code="malformed_input",
            message=f"{name} must be a number, got {raw!r}.",
            transient=False,
            http_status=400,
        ) from exc
    if not low <= value <= high:
        raise ServiceError(
            ErrorCategory.TRANSPORT_ERROR,
            code="malformed_input",
            message=f"{name} must be between {low} and {high}.",
            transient=False,
            http_status=400,
        )
    return value


def _int_param(
    params: dict[str, Any],
    name: str,
    default: int,
    low: int,
    high: int,
) -> int:
    raw = params.get(name)
    if raw is None:
        return default
    if isinstance(raw, bool):  # bool is an int subclass — reject explicitly
        raw = repr(raw)
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise ServiceError(
            ErrorCategory.TRANSPORT_ERROR,
            code="malformed_input",
            message=f"{name} must be an integer, got {raw!r}.",
            transient=False,
            http_status=400,
        ) from exc
    if not low <= value <= high:
        raise ServiceError(
            ErrorCategory.TRANSPORT_ERROR,
            code="malformed_input",
            message=f"{name} must be between {low} and {high}.",
            transient=False,
            http_status=400,
        )
    return value


def _imgsz_param(params: dict[str, Any]) -> int:
    """``imgsz`` additionally has to be a multiple of the model's
    stride — an off-stride value doesn't fail loudly in onnxruntime,
    it changes the anchor grid and the keypoints quietly land in the
    wrong place."""
    value = _int_param(params, "imgsz", DEFAULT_IMGSZ, MIN_IMGSZ, MAX_IMGSZ)
    if value % IMGSZ_STRIDE != 0:
        raise ServiceError(
            ErrorCategory.TRANSPORT_ERROR,
            code="malformed_input",
            message=(
                f"imgsz must be a multiple of {IMGSZ_STRIDE} "
                f"(got {value}; try {value - value % IMGSZ_STRIDE})."
            ),
            transient=False,
            http_status=400,
        )
    return value


# ── Image helpers ───────────────────────────────────────────────────


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


def _static_input_size(shape: Any) -> int | None:
    """Return the fixed square input size an ONNX graph demands, or
    None when the export has dynamic spatial axes.

    onnxruntime reports the input shape as e.g. ``[1, 3, 640, 640]``
    for a fixed export and ``[1, 3, 'height', 'width']`` (strings or
    None for the symbolic dims) for a dynamic one. Anything we can't
    read confidently is treated as dynamic — the adapter then uses its
    own default and a genuine mismatch still surfaces as an
    inference-time error, which is no worse than not looking at all.
    """
    if not isinstance(shape, (list, tuple)) or len(shape) < 2:
        return None
    height, width = shape[-2], shape[-1]
    if isinstance(height, int) and isinstance(width, int) and height == width > 0:
        return int(height)
    return None


def _letterbox_blob(img: Any, imgsz: int) -> tuple[Any, float, float, float]:
    """Resize ``img`` into an ``imgsz × imgsz`` letterbox and return
    the NCHW float32 blob plus ``(scale, pad_x, pad_y)``.

    Aspect ratio is preserved and the remainder padded with grey
    (114, 114, 114) — the exact preprocessing Ultralytics trains and
    validates with. Mapping back is ``source = (model - pad) / scale``.
    """
    import cv2
    import numpy as np

    height, width = img.shape[:2]
    scale = min(imgsz / float(width), imgsz / float(height))
    new_w = max(1, int(round(width * scale)))
    new_h = max(1, int(round(height * scale)))
    left = int(round((imgsz - new_w) / 2.0))
    top = int(round((imgsz - new_h) / 2.0))

    interpolation = cv2.INTER_LINEAR if scale > 1 else cv2.INTER_AREA
    resized = cv2.resize(img, (new_w, new_h), interpolation=interpolation)
    canvas = np.full((imgsz, imgsz, 3), 114, dtype=np.uint8)
    canvas[top:top + new_h, left:left + new_w] = resized

    # BGR → RGB, [0, 255] → [0, 1], HWC → NCHW. The same normalisation
    # ``cv2.dnn.blobFromImage(1/255, swapRB=True)`` applies in
    # adapters/yolov8/ — written out here because we did the resize
    # ourselves to keep the aspect ratio.
    blob = canvas[:, :, ::-1].astype(np.float32) / 255.0
    blob = np.ascontiguousarray(blob.transpose(2, 0, 1)[None, ...])
    return blob, scale, float(left), float(top)


def _nms(boxes: Any, scores: Any, iou_threshold: float) -> list[int]:
    """Greedy class-agnostic NMS over xyxy boxes. Returns the kept
    indices in descending-score order.

    Written out in numpy rather than pulled from ``cv2.dnn.NMSBoxes``
    so the whole post-processing path is readable in one file and
    testable without depending on OpenCV's dnn module.
    """
    import numpy as np

    areas = (boxes[:, 2] - boxes[:, 0]).clip(min=0) * (
        boxes[:, 3] - boxes[:, 1]).clip(min=0)
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size > 0:
        current = int(order[0])
        keep.append(current)
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(boxes[current, 0], boxes[rest, 0])
        yy1 = np.maximum(boxes[current, 1], boxes[rest, 1])
        xx2 = np.minimum(boxes[current, 2], boxes[rest, 2])
        yy2 = np.minimum(boxes[current, 3], boxes[rest, 3])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        union = areas[current] + areas[rest] - inter
        # union == 0 only for degenerate zero-area boxes; treat those
        # as fully overlapping so they get suppressed rather than
        # dividing by zero.
        iou = np.where(union > 0, inter / np.maximum(union, 1e-9), 1.0)
        order = rest[iou <= iou_threshold]
    return keep


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
