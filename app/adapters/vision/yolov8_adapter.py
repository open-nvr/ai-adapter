# Copyright (c) 2026 OpenNVR
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
YOLOv8 adapter focused on raw model inference output.

Two pieces are shared with the contract service in ``adapters/yolov8``
and live here so both paths decode the model the same way:

* :func:`letterbox` — aspect-preserving resize + grey padding to the
  model's square input, returning the geometry needed to map boxes back.
  The old ``cv2.dnn.blobFromImage`` squash stretched a 16:9 frame 1.78x
  horizontally; the model is trained on letterboxed inputs, so recall and
  box tightness suffered on every wide camera frame.
* :func:`decode_predictions` — vectorised decode of the ``(N, 4+nc)``
  output with per-class non-maximum suppression. The Ultralytics ONNX
  export used by ``download_models.py`` emits every anchor (8400 at 640),
  and without NMS one object came back as a cluster of overlapping boxes.
"""
import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List

# cv2 and numpy are intentionally NOT imported at module level.
# They are deferred into the methods that use them so that
# PluginManager can discover this class without importing the
# full OpenCV/NumPy stack (~150 MB) for adapters that may
# never be called in a lightweight deployment.
from app.config import INPUT_SIZE, MODEL_WEIGHTS_DIR
from app.adapters.base import BaseAdapter
from app.utils.image_utils import load_image_from_uri

logger = logging.getLogger(__name__)

# Ultralytics' own defaults: keep a box at >= 0.25 confidence, and suppress
# a same-class box overlapping a stronger one at IoU > 0.45.
DEFAULT_CONFIDENCE_THRESHOLD: float = 0.25
DEFAULT_IOU_THRESHOLD: float = 0.45

# Padding colour for the letterbox borders (Ultralytics uses 114 grey).
_LETTERBOX_FILL = (114, 114, 114)


@dataclass(frozen=True)
class LetterboxGeometry:
    """How a source frame was placed inside the square model input.

    ``scale`` is source-pixels → input-pixels; ``pad_x``/``pad_y`` are the
    left/top borders in input pixels. Inverting it maps a model-space box
    back onto the source frame.
    """

    scale: float
    pad_x: float
    pad_y: float
    src_w: int
    src_h: int
    input_size: int


def letterbox(img: Any, input_size: int = INPUT_SIZE) -> tuple[Any, LetterboxGeometry]:
    """Resize ``img`` to fit ``input_size`` square without distortion and
    pad the remainder. Returns the padded BGR image and its geometry."""
    import cv2

    src_h, src_w = img.shape[:2]
    scale = min(input_size / src_w, input_size / src_h)
    new_w = max(1, int(round(src_w * scale)))
    new_h = max(1, int(round(src_h * scale)))
    resized = img if (new_w, new_h) == (src_w, src_h) else cv2.resize(
        img, (new_w, new_h), interpolation=cv2.INTER_LINEAR
    )
    pad_x = (input_size - new_w) / 2.0
    pad_y = (input_size - new_h) / 2.0
    left, top = int(round(pad_x - 0.1)), int(round(pad_y - 0.1))
    right, bottom = input_size - new_w - left, input_size - new_h - top
    padded = cv2.copyMakeBorder(
        resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=_LETTERBOX_FILL
    )
    return padded, LetterboxGeometry(
        scale=scale, pad_x=float(left), pad_y=float(top),
        src_w=src_w, src_h=src_h, input_size=input_size,
    )


def decode_predictions(
    raw: Any,
    *,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    iou_threshold: float = DEFAULT_IOU_THRESHOLD,
) -> tuple[Any, Any, Any]:
    """Decode a ``(N, 4+nc)`` YOLOv8 output into NMS'd detections.

    Returns ``(boxes_xyxy, class_ids, confidences)`` as numpy arrays in
    model-input coordinates, sorted by confidence descending. Boxes are
    ``[x1, y1, x2, y2]``. All arrays are empty when nothing survives.
    """
    import numpy as np

    empty = (
        np.zeros((0, 4), dtype=np.float32),
        np.zeros((0,), dtype=np.int64),
        np.zeros((0,), dtype=np.float32),
    )
    arr = np.asarray(raw, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] < 5:
        return empty

    class_scores = arr[:, 4:]
    class_ids = np.argmax(class_scores, axis=1)
    confidences = class_scores[np.arange(arr.shape[0]), class_ids]
    keep = confidences >= confidence_threshold
    if not np.any(keep):
        return empty
    boxes_cxcywh = arr[keep, :4]
    class_ids = class_ids[keep]
    confidences = confidences[keep]

    half_w = boxes_cxcywh[:, 2] / 2.0
    half_h = boxes_cxcywh[:, 3] / 2.0
    x1 = boxes_cxcywh[:, 0] - half_w
    y1 = boxes_cxcywh[:, 1] - half_h
    boxes_xyxy = np.stack(
        [x1, y1, boxes_cxcywh[:, 0] + half_w, boxes_cxcywh[:, 1] + half_h], axis=1
    )
    # NMSBoxes wants [x, y, w, h]; class-batched so a person standing in
    # front of a car keeps both boxes while duplicate persons collapse.
    rects = np.stack([x1, y1, boxes_cxcywh[:, 2], boxes_cxcywh[:, 3]], axis=1)
    idxs = _nms(rects, confidences, class_ids, iou_threshold)
    if len(idxs) == 0:
        return empty
    idxs = idxs[np.argsort(-confidences[idxs], kind="stable")]
    return boxes_xyxy[idxs], class_ids[idxs], confidences[idxs]


def _nms(rects: Any, confidences: Any, class_ids: Any, iou_threshold: float) -> Any:
    """Per-class NMS → indices into the inputs. Uses OpenCV's batched
    variant when the build has it (4.7+), else one NMS pass per class.

    The confidence gate has already been applied by the caller with
    ``>=``; cv2's own score threshold is a strict ``>``, so it is passed
    as 0.0 here — otherwise a box scoring exactly the threshold would
    pass the gate and then vanish inside NMS.
    """
    import cv2
    import numpy as np

    rects32 = np.ascontiguousarray(rects, dtype=np.float32)
    conf32 = np.ascontiguousarray(confidences, dtype=np.float32)
    ids32 = np.ascontiguousarray(class_ids, dtype=np.int32)
    batched = getattr(cv2.dnn, "NMSBoxesBatched", None)
    if batched is not None:
        out = batched(rects32, conf32, ids32, 0.0, iou_threshold)
        return np.asarray(out, dtype=np.int64).reshape(-1)
    kept: list[int] = []
    for cid in np.unique(ids32):
        members = np.flatnonzero(ids32 == cid)
        out = cv2.dnn.NMSBoxes(rects32[members], conf32[members], 0.0, iou_threshold)
        kept.extend(int(members[i]) for i in np.asarray(out).reshape(-1))
    return np.asarray(kept, dtype=np.int64)


def box_to_source(box_xyxy: Any, geometry: LetterboxGeometry) -> tuple[float, float, float, float]:
    """Map a model-space ``[x1, y1, x2, y2]`` box (model-input pixels,
    as the Ultralytics export emits them) onto the source frame, clipped
    to it. Returns ``(x1, y1, x2, y2)`` in source pixels.
    """
    x1, y1, x2, y2 = (float(v) for v in box_xyxy)
    scale = geometry.scale or 1.0
    x1 = (x1 - geometry.pad_x) / scale
    x2 = (x2 - geometry.pad_x) / scale
    y1 = (y1 - geometry.pad_y) / scale
    y2 = (y2 - geometry.pad_y) / scale
    x1 = min(max(x1, 0.0), float(geometry.src_w))
    x2 = min(max(x2, 0.0), float(geometry.src_w))
    y1 = min(max(y1, 0.0), float(geometry.src_h))
    y2 = min(max(y2, 0.0), float(geometry.src_h))
    return x1, y1, x2, y2


def _unit_float(params: Dict[str, Any], name: str, default: float) -> float:
    """A [0, 1] float request parameter for the legacy ``infer_local``
    path. Raises ``ValueError`` with the parameter named — the contract
    service has its own typed 400 for the same check."""
    raw = params.get(name, default)
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a number, got {raw!r}.") from exc
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be between 0.0 and 1.0.")
    return value


class YOLOv8Adapter(BaseAdapter):
    name = "yolov8_adapter"
    type = "vision"

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.session = None
        self.model = None
        self._model_path = self.config.get(
            "weights_path",
            os.path.join(MODEL_WEIGHTS_DIR, "yolov8n.onnx"),
        )
        if not os.path.isabs(self._model_path):
            self._model_path = os.path.join(MODEL_WEIGHTS_DIR, self._model_path)

    def load_model(self) -> None:
        import onnxruntime as ort  # optional dep: uv sync --extra yolo

        if not os.path.exists(self._model_path):
            raise FileNotFoundError(f"YOLOv8 model not found at {self._model_path}")

        self.session = ort.InferenceSession(
            self._model_path,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        self.model = self.session
        logger.info("YOLOv8 model loaded from %s", self._model_path)

    def _preprocess(self, img: Any) -> tuple[Any, LetterboxGeometry]:
        """Letterbox ``img`` into the model input and build the NCHW blob.
        Returns ``(blob, geometry)``; the geometry maps boxes back."""
        import cv2  # deferred: only loaded when inference actually runs

        padded, geometry = letterbox(img, INPUT_SIZE)
        blob = cv2.dnn.blobFromImage(
            padded,
            1 / 255.0,
            (INPUT_SIZE, INPUT_SIZE),
            swapRB=True,
            crop=False,
        )
        return blob, geometry

    def _run_inference(self, blob: Any) -> Any:
        import numpy as np  # deferred: only loaded when inference actually runs
        outputs = self.session.run(None, {self.session.get_inputs()[0].name: blob})
        predictions = np.transpose(outputs[0], (0, 2, 1)).squeeze()
        if predictions.ndim == 1:
            predictions = np.expand_dims(predictions, axis=0)
        return predictions

    def _convert_bbox(self, box_xyxy: Any, geometry: LetterboxGeometry) -> List[int]:
        """Model-space box → ``[left, top, width, height]`` source pixels."""
        x1, y1, x2, y2 = box_to_source(box_xyxy, geometry)
        left, top = int(x1), int(y1)
        return [left, top, max(0, int(x2) - left), max(0, int(y2) - top)]

    def infer_local(self, input_data: Any) -> Dict[str, Any]:
        if self.session is None:
            self.load_model()

        start_time = time.time()
        uri = input_data["frame"]["uri"]
        img = load_image_from_uri(uri)

        blob, geometry = self._preprocess(img)
        raw_predictions = self._run_inference(blob)

        confidence_threshold = _unit_float(
            input_data, "confidence_threshold", DEFAULT_CONFIDENCE_THRESHOLD
        )
        iou_threshold = _unit_float(input_data, "iou_threshold", DEFAULT_IOU_THRESHOLD)
        boxes, class_ids, confidences = decode_predictions(
            raw_predictions,
            confidence_threshold=confidence_threshold,
            iou_threshold=iou_threshold,
        )
        detections = [
            {
                "bbox": self._convert_bbox(box, geometry),
                "class_id": int(cid),
                "confidence": round(float(conf), 4),
            }
            for box, cid, conf in zip(boxes, class_ids, confidences)
        ]

        return {
            "task": input_data.get("task", "yolov8_raw_inference"),
            "predictions": detections,
            "raw_prediction_count": int(raw_predictions.shape[0]),
            "executed_at": int(time.time() * 1000),
            "latency_ms": int((time.time() - start_time) * 1000),
        }

    @property
    def schema(self) -> Dict[str, Any]:
        return {
            "task": "yolov8_raw_inference",
            "description": "Returns raw YOLOv8 detections (bbox, class_id, confidence).",
        }

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "type": self.type,
            "model": "yolov8n",
            "framework": "onnx",
            "tasks": ["person_detection", "person_counting"],
            "model_path": self._model_path,
            "model_loaded": self.session is not None,
        }

    def health_check(self) -> Dict[str, Any]:
        return {
            "status": "healthy",
            "type": self.type,
            "model_loaded": self.session is not None,
            "model_info": self.get_model_info(),
        }
