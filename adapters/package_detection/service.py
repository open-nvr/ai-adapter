# Copyright (c) 2026 OpenNVR
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
PackageDetectionService — parcel detection implementation of
``AdapterService``.

Wraps a YOLOv8n fine-tuned on doorstep parcels on ``onnxruntime`` and
returns §5.1 detections with one label, ``package``. It exists because
COCO has no package class: the general detector Tier-0 runs can tell an
app that a person came and went, but not whether a parcel is on the
step. This adapter answers that one question, on demand.

Design choices worth pinning:

* **On demand, not streamed.** The consumer this was built for
  (``examples/package-delivery`` in open-nvr) counts parcels when
  Tier-0 says somebody left the doorstep and on a slow recheck cadence
  — a handful of frames a day per door, over HTTP ``/infer``. There is
  no per-frame stream path here (``supports_stream=False``); a box that
  wants parcels tracked at frame rate should run this model as an
  ``object_detection`` adapter instead, which is a different contract
  posture and a different CPU budget.

* **§5.1 output, normalized boxes.** Unlike the pose adapter, a
  parcel IS a region: the consumer asks "is the centre of this box
  inside my porch zone", which is exactly what normalized coordinates
  are for. ``frame_dimensions`` is echoed so a consumer can go back to
  pixels, and ``labels`` is echoed so a future model with more classes
  cannot silently change what index 0 means.

* **Letterbox preprocessing**, the same as ``adapters/yolo_pose/``:
  pad to square rather than stretch, because the training pipeline
  letterboxes and a stretched frame shifts every box on a 16:9 camera.

* **The weights are acquired the way the SDK prescribes** —
  ``ensure_model_file``: a file present at the weights path always
  wins and no network is touched; a missing file is fetched once from
  ``PACKAGE_DETECTION_MODEL_URL`` into the mounted weights volume; an
  empty URL means "the operator pre-populates, never download".

* **Any single- or multi-class detection export is accepted.** The
  shipped weights have one class. A model with more (an operator's own
  fine-tune with ``package`` and ``envelope``, say) works unchanged:
  the label map comes from ``PACKAGE_DETECTION_LABELS`` (comma-separated,
  index order) and any index past its end is reported as ``class_N``
  rather than dropped.
"""
from __future__ import annotations

import hashlib
import logging
import os
import platform
import time
from datetime import datetime, timezone
from typing import Any

from opennvr_adapter_sdk import AdapterService, BODY_BYTES_KEY, ServiceError
from opennvr_adapter_sdk.contract import (
    DetectionItem,
    DetectionResult,
    ErrorCategory,
    FrameDimensions,
    HardwareEvaluationResponse,
    HardwareVerdict,
    HealthStatus,
    InferResponse,
    ModelInfo,
    NormalizedBBox,
)
from opennvr_adapter_sdk.model_fetch import ensure_model_file

logger = logging.getLogger(__name__)

MODEL_FRAMEWORK: str = "onnxruntime"

#: Model identity reported on /capabilities. Swapping the weights file
#: without changing this string is exactly the drift §11.3 catches —
#: the fingerprint moves and the name doesn't.
MODEL_NAME: str = "yolov8n-package"

#: Default request body cap for /infer (§3.8). 8 MiB comfortably holds
#: a 4K JPEG; same cap as the sibling vision adapters.
MAX_IMAGE_BYTES: int = 8 * 1024 * 1024

#: Where the ONNX file lives. The Docker image sets
#: ``PACKAGE_DETECTION_WEIGHTS_DIR=/weights`` and mounts a volume
#: there; a source checkout falls back to the repo's ``model_weights/``.
DEFAULT_WEIGHTS_DIR: str = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "model_weights")
)
WEIGHTS_FILENAME: str = "yolov8n-package.onnx"

#: First-boot download source, read at construction time. Empty means
#: "never download" (the ``sovereignty=local_only`` posture); the
#: Dockerfile and the release notes give the published URL.
MODEL_URL_ENV: str = "PACKAGE_DETECTION_MODEL_URL"

#: The class-index → label map, index order, comma-separated. The
#: shipped weights have exactly one class.
LABELS_ENV: str = "PACKAGE_DETECTION_LABELS"
DEFAULT_LABELS: tuple[str, ...] = ("package",)

# ── Inference defaults ─────────────────────────────────────────────
# All overridable per call via the /infer params block.

#: Confidence floor. 0.35: a parcel counter is asked "how many", and a
#: half-confident box on a doormat becomes a phantom delivery, which
#: is worse than a missed count that the next recheck corrects. The
#: shipped weights are precise (0.97) before they are complete (0.80
#: recall), and this threshold is set on that trade.
DEFAULT_CONF: float = 0.35
#: IoU threshold for the class-agnostic NMS.
DEFAULT_IOU: float = 0.5
#: Model input side in pixels. The shipped weights were trained at 416
#: and exported with dynamic axes (so this stays a caller's dial).
DEFAULT_IMGSZ: int = 416
IMGSZ_STRIDE: int = 32
MIN_IMGSZ: int = 160
MAX_IMGSZ: int = 1280
#: Cap on detections per frame — a payload guard, not a scene
#: assumption (a mail room can genuinely hold thirty parcels).
DEFAULT_MAX_DETECTIONS: int = 50
ABSOLUTE_MAX_DETECTIONS: int = 300


def labels_from_env() -> tuple[str, ...]:
    raw = os.getenv(LABELS_ENV, "").strip()
    if not raw:
        return DEFAULT_LABELS
    labels = tuple(part.strip().lower() for part in raw.split(",") if part.strip())
    return labels or DEFAULT_LABELS


class PackageDetectionService(AdapterService):
    """Stateful façade around the package-detector ONNX session."""

    def __init__(self, weights_path: str | None = None) -> None:
        self._weights_path = weights_path or os.path.join(
            os.getenv("PACKAGE_DETECTION_WEIGHTS_DIR", DEFAULT_WEIGHTS_DIR),
            WEIGHTS_FILENAME,
        )
        self._model_url: str = os.getenv(MODEL_URL_ENV, "")
        self._labels: tuple[str, ...] = labels_from_env()
        self._session: Any | None = None
        self._input_name: str = "images"
        self._static_imgsz: int | None = None
        self._num_classes: int | None = None
        self._load_state: HealthStatus = HealthStatus.LOADING
        self._load_error: str | None = None
        self._fingerprint_cache: str | None = None
        self._gpu_in_use: bool = False

    @property
    def labels(self) -> tuple[str, ...]:
        return self._labels

    # ── AdapterService impl ────────────────────────────────────────

    def load(self) -> None:
        """Eagerly load the ONNX weights. Idempotent."""
        if self._load_state == HealthStatus.OK:
            return
        # Domain metrics. ``adapter_package_frames_total{result}`` is
        # THE diagnostic for this adapter: a door whose every frame
        # comes back ``empty`` after a camera was re-aimed, or whose
        # every frame comes back ``packages`` because a doormat pattern
        # reads as a parcel, shows up in one scrape. Label set is fixed,
        # so cardinality is bounded.
        self.metrics.register_counter(
            "adapter_package_frames_total",
            "Frames inferred, by whether any package was found.",
            label_key="result", allowed_values=["packages", "empty"])
        self.metrics.register_counter(
            "adapter_packages_total",
            "Packages returned, by class label.",
            label_key="label", allowed_values=list(self._labels))
        try:
            import onnxruntime as ort  # optional dep: uv sync --extra yolo

            ensure_model_file(
                self._weights_path,
                self._model_url,
                label=f"{MODEL_NAME} weights",
                logger=logger,
            )
            self._session = ort.InferenceSession(
                self._weights_path,
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            )
            input_meta = self._session.get_inputs()[0]
            self._input_name = input_meta.name
            self._static_imgsz = _static_input_size(getattr(input_meta, "shape", None))
            self._fingerprint_cache = self._compute_fingerprint()
            self._gpu_in_use = self._detect_gpu_in_use()
            self._warm_up()
            self._load_state = HealthStatus.OK
            self._load_error = None
            logger.info(
                "PackageDetectionService ready: weights=%s fingerprint=%s gpu=%s "
                "imgsz=%s labels=%s",
                self._weights_path, self._fingerprint_cache, self._gpu_in_use,
                self._static_imgsz or f"dynamic (default {DEFAULT_IMGSZ})",
                ",".join(self._labels),
            )
        except Exception as exc:
            self._load_state = HealthStatus.ERROR
            self._load_error = str(exc)
            logger.exception(
                "PackageDetectionService failed to load weights %s", self._weights_path
            )

    def is_ready(self) -> bool:
        return self._load_state == HealthStatus.OK

    def fingerprint(self) -> str | None:
        """Recompute live so §11.3 drift detection sees weight rotation."""
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
            modalities_out=["bbox_classes"],
            fingerprint=self.fingerprint(),
        )

    def hardware_evaluation(self) -> HardwareEvaluationResponse:
        cpu_count = os.cpu_count() or 0
        if self._load_state == HealthStatus.OK:
            if self._gpu_in_use:
                verdict = HardwareVerdict.OK
                reasoning = "GPU detected and in use; weights loaded."
            else:
                # CPU is the design target: the adapter is called a few
                # times a day per door, so even two cores serve it.
                verdict = HardwareVerdict.OK
                reasoning = (
                    f"Running on CPU with {cpu_count} cores — the design target "
                    f"for this on-demand adapter (measured 23-34 ms per frame at "
                    f"imgsz={DEFAULT_IMGSZ} on two x86 cores)."
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
                "labels": list(self._labels),
            },
        )

    def infer(self, payload: dict[str, Any]) -> InferResponse:
        """SDK /infer entry point. The image bytes live at
        ``payload[BODY_BYTES_KEY]``; the rest of the dict is request
        params (conf, iou, imgsz, max_detections)."""
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

    # ── Inference core ─────────────────────────────────────────────

    def _infer_image_bytes(self, image_bytes: bytes, params: dict[str, Any]) -> InferResponse:
        if self._load_state != HealthStatus.OK:
            raise ServiceError(
                ErrorCategory.MODEL_ERROR,
                code=("weights_missing" if self._load_state == HealthStatus.ERROR
                      else "package_detection.model_loading"),
                message=self._load_error or "Model still loading.",
                transient=(self._load_state == HealthStatus.LOADING),
                http_status=503,
                retry_after_ms=2000 if self._load_state == HealthStatus.LOADING else None,
            )
        if len(image_bytes) > MAX_IMAGE_BYTES:
            raise ServiceError(
                ErrorCategory.TRANSPORT_ERROR,
                code="malformed_input",
                message=(f"Frame exceeds {MAX_IMAGE_BYTES}-byte limit "
                         f"({len(image_bytes)} received)."),
                transient=False,
                http_status=413,
            )

        conf = _float_param(params, "conf", DEFAULT_CONF, 0.0, 1.0,
                            aliases=("confidence_threshold",))
        iou = _float_param(params, "iou", DEFAULT_IOU, 0.0, 1.0,
                           aliases=("iou_threshold", "nms_threshold"))
        imgsz = self._resolve_imgsz(params)
        max_detections = _int_param(params, "max_detections", DEFAULT_MAX_DETECTIONS,
                                    1, ABSOLUTE_MAX_DETECTIONS)

        start = time.monotonic()
        # Decode, inference AND post-processing inside one guard so every
        # exit is a typed ServiceError with a §7 envelope (the SDK route
        # only translates ServiceError; anything else is a bare 500).
        try:
            img, width, height = _decode_image(image_bytes)
            raw, scale, pad_x, pad_y = self._run_inference(img, imgsz)
            items = self._shape_detections(
                raw, scale=scale, pad_x=pad_x, pad_y=pad_y, width=width,
                height=height, conf=conf, iou=iou, max_detections=max_detections,
            )
        except DecodeError as exc:
            raise ServiceError(
                ErrorCategory.TRANSPORT_ERROR, code="malformed_input",
                message=str(exc), transient=False, http_status=400,
            ) from exc
        except ServiceError:
            raise
        except Exception as exc:
            logger.exception("package-detection inference raised unexpectedly")
            raise ServiceError(
                ErrorCategory.MODEL_ERROR, code="inference_runtime_crash",
                message="Inference failed.", transient=False, http_status=500,
            ) from exc

        inference_ms = int((time.monotonic() - start) * 1000)

        try:
            self.metrics.inc_counter(
                "adapter_package_frames_total",
                label_value="packages" if items else "empty")
            for item in items:
                if item.label in self._labels:
                    self.metrics.inc_counter("adapter_packages_total", label_value=item.label)
        except Exception:  # pragma: no cover - metrics must never break infer
            logger.debug("package-detection domain metrics recording failed", exc_info=True)

        result = DetectionResult(
            detections=items,
            frame_dimensions=FrameDimensions(w=width, h=height),
        ).model_dump(mode="json")
        # Echoed so a consumer never hard-codes what index 0 means, and
        # so a future multi-class export cannot silently change it.
        result["labels"] = list(self._labels)
        # The question this adapter is usually asked. Same number a
        # consumer would get from len(detections); spelled out because
        # the package-delivery app reads a count, not a list.
        result["count"] = len(items)
        return InferResponse(
            model_name=MODEL_NAME,
            model_version=self._adapter_model_version(),
            inference_ms=inference_ms,
            result=result,
        )

    def _resolve_imgsz(self, params: dict[str, Any]) -> int:
        """A fixed-size export dictates the size when the caller didn't
        ask; a caller who did ask for something else is told why not
        (same rule as ``adapters/yolo_pose/``)."""
        requested = _imgsz_param(params)
        if self._static_imgsz is None:
            return requested
        if "imgsz" in params and requested != self._static_imgsz:
            raise ServiceError(
                ErrorCategory.TRANSPORT_ERROR,
                code="malformed_input",
                message=(f"This model was exported with a fixed {self._static_imgsz}px "
                         f"input, so imgsz={requested} cannot be served. Use "
                         f"imgsz={self._static_imgsz}, or re-export with dynamic=True."),
                transient=False,
                http_status=400,
            )
        return self._static_imgsz

    def _run_inference(self, img: Any, imgsz: int) -> tuple[Any, float, float, float]:
        """Preprocess, run the session, return ``(N, 4 + nc)`` rows
        plus the letterbox geometry. Ultralytics detection exports emit
        ``(1, 4 + nc, anchors)``; every consumer wants ``(anchors, ·)``."""
        import numpy as np

        blob, scale, pad_x, pad_y = _letterbox_blob(img, imgsz)
        outputs = self._session.run(None, {self._input_name: blob})
        raw = np.transpose(outputs[0], (0, 2, 1)).squeeze(0)
        if raw.ndim == 1:
            raw = np.expand_dims(raw, axis=0)
        features = int(raw.shape[-1])
        # A pose export (56 features) or a segmentation export pointed
        # at this adapter would otherwise produce nonsense scores. A
        # detection export has 4 box values plus one score per class;
        # anything with fewer than 5 features cannot be one.
        if features < 5:
            raise ServiceError(
                ErrorCategory.MODEL_ERROR,
                code="package_detection.unexpected_model_output",
                message=(f"Model emitted {features} features per prediction, "
                         f"expected 4 box + N class scores. Is "
                         f"{self._weights_path} a detection export?"),
                transient=False,
                http_status=500,
            )
        self._num_classes = features - 4
        return raw, scale, pad_x, pad_y

    def _shape_detections(
        self, raw: Any, *, scale: float, pad_x: float, pad_y: float,
        width: int, height: int, conf: float, iou: float, max_detections: int,
    ) -> list[DetectionItem]:
        """Raw rows → §5.1 items: best class per row, confidence floor,
        letterbox unmapped to source pixels, class-agnostic NMS in
        source pixels, normalized to [0, 1], sorted by score, capped."""
        import numpy as np

        class_scores = raw[:, 4:]
        class_ids = np.argmax(class_scores, axis=1)
        scores = class_scores[np.arange(raw.shape[0]), class_ids].astype(float)
        keep_mask = scores >= conf
        if not bool(keep_mask.any()):
            return []
        rows = raw[keep_mask]
        scores = scores[keep_mask]
        class_ids = class_ids[keep_mask]

        cx, cy, bw, bh = rows[:, 0], rows[:, 1], rows[:, 2], rows[:, 3]
        x1 = (cx - bw / 2.0 - pad_x) / scale
        y1 = (cy - bh / 2.0 - pad_y) / scale
        x2 = (cx + bw / 2.0 - pad_x) / scale
        y2 = (cy + bh / 2.0 - pad_y) / scale
        boxes = np.stack([x1, y1, x2, y2], axis=1).astype(float)
        np.clip(boxes[:, 0::2], 0.0, float(width), out=boxes[:, 0::2])
        np.clip(boxes[:, 1::2], 0.0, float(height), out=boxes[:, 1::2])

        order = _nms(boxes, scores, iou)[:max_detections]

        items: list[DetectionItem] = []
        for index in order:
            box = boxes[index]
            bx, by = box[0] / width, box[1] / height
            bwn, bhn = (box[2] - box[0]) / width, (box[3] - box[1]) / height
            if bwn <= 0.0 or bhn <= 0.0:
                continue
            class_id = int(class_ids[index])
            label = (self._labels[class_id] if class_id < len(self._labels)
                     else f"class_{class_id}")
            items.append(DetectionItem(
                label=label,
                confidence=round(min(max(float(scores[index]), 0.0), 1.0), 4),
                bbox=NormalizedBBox(
                    x=round(min(max(bx, 0.0), 1.0), 5),
                    y=round(min(max(by, 0.0), 1.0), 5),
                    w=round(min(max(bwn, 0.0), 1.0 - min(max(bx, 0.0), 1.0)), 5),
                    h=round(min(max(bhn, 0.0), 1.0 - min(max(by, 0.0), 1.0)), 5),
                ),
                track_id=None,
                attributes={"class_id": class_id},
            ))
        return items

    # ── Helpers ────────────────────────────────────────────────────

    def _warm_up(self) -> None:
        """One throwaway inference so the first real request does not
        pay onnxruntime's arena allocation. Never fatal."""
        try:
            import numpy as np

            imgsz = self._static_imgsz or DEFAULT_IMGSZ
            blank = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
            blob, _, _, _ = _letterbox_blob(blank, imgsz)
            self._session.run(None, {self._input_name: blob})
        except Exception:  # pragma: no cover - never fatal
            logger.debug("package-detection warm-up failed", exc_info=True)

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
        try:
            return "CUDAExecutionProvider" in self._session.get_providers()
        except Exception:
            return False


# ── Request-parameter parsing ───────────────────────────────────────
# Every caller-supplied number is validated here so a string ``conf``
# comes back as a typed 400, not a ValueError the SDK turns into a 500.


def _float_param(params: dict[str, Any], name: str, default: float, low: float,
                 high: float, *, aliases: tuple[str, ...] = ()) -> float:
    raw = params.get(name)
    for alias in aliases:
        if raw is None:
            raw = params.get(alias)
    if raw is None:
        return default
    if isinstance(raw, bool):
        raw = repr(raw)
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise ServiceError(
            ErrorCategory.TRANSPORT_ERROR, code="malformed_input",
            message=f"{name} must be a number, got {raw!r}.",
            transient=False, http_status=400,
        ) from exc
    if not low <= value <= high:
        raise ServiceError(
            ErrorCategory.TRANSPORT_ERROR, code="malformed_input",
            message=f"{name} must be between {low} and {high}.",
            transient=False, http_status=400,
        )
    return value


def _int_param(params: dict[str, Any], name: str, default: int, low: int, high: int) -> int:
    raw = params.get(name)
    if raw is None:
        return default
    if isinstance(raw, bool):
        raw = repr(raw)
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise ServiceError(
            ErrorCategory.TRANSPORT_ERROR, code="malformed_input",
            message=f"{name} must be an integer, got {raw!r}.",
            transient=False, http_status=400,
        ) from exc
    if not low <= value <= high:
        raise ServiceError(
            ErrorCategory.TRANSPORT_ERROR, code="malformed_input",
            message=f"{name} must be between {low} and {high}.",
            transient=False, http_status=400,
        )
    return value


def _imgsz_param(params: dict[str, Any]) -> int:
    value = _int_param(params, "imgsz", DEFAULT_IMGSZ, MIN_IMGSZ, MAX_IMGSZ)
    if value % IMGSZ_STRIDE != 0:
        raise ServiceError(
            ErrorCategory.TRANSPORT_ERROR, code="malformed_input",
            message=(f"imgsz must be a multiple of {IMGSZ_STRIDE} "
                     f"(got {value}; try {value - value % IMGSZ_STRIDE})."),
            transient=False, http_status=400,
        )
    return value


# ── Image helpers ───────────────────────────────────────────────────


class DecodeError(Exception):
    """Raised when the request bytes cannot be decoded as an image."""


def _decode_image(image_bytes: bytes) -> tuple[Any, int, int]:
    import cv2
    import numpy as np

    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise DecodeError("Could not decode frame as JPEG/PNG.")
    height, width = img.shape[:2]
    return img, width, height


def _static_input_size(shape: Any) -> int | None:
    """The fixed square input an ONNX graph demands, or None for a
    dynamic-axes export (symbolic dims come back as strings/None)."""
    if not isinstance(shape, (list, tuple)) or len(shape) < 2:
        return None
    height, width = shape[-2], shape[-1]
    if isinstance(height, int) and isinstance(width, int) and height == width > 0:
        return int(height)
    return None


def _letterbox_blob(img: Any, imgsz: int) -> tuple[Any, float, float, float]:
    """Resize into an ``imgsz × imgsz`` letterbox (grey 114 padding —
    the preprocessing Ultralytics trains with) and return the NCHW
    float32 blob plus ``(scale, pad_x, pad_y)``. Mapping back is
    ``source = (model - pad) / scale``."""
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

    blob = canvas[:, :, ::-1].astype(np.float32) / 255.0
    blob = np.ascontiguousarray(blob.transpose(2, 0, 1)[None, ...])
    return blob, scale, float(left), float(top)


def _nms(boxes: Any, scores: Any, iou_threshold: float) -> list[int]:
    """Greedy class-agnostic NMS over xyxy boxes; kept indices in
    descending-score order. Class-agnostic on purpose: two classes
    claiming the same parcel is one parcel."""
    import numpy as np

    areas = (boxes[:, 2] - boxes[:, 0]).clip(min=0) * (boxes[:, 3] - boxes[:, 1]).clip(min=0)
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
        iou = np.where(union > 0, inter / np.maximum(union, 1e-9), 1.0)
        order = rest[iou <= iou_threshold]
    return keep
