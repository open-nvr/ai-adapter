# Copyright (c) 2026 OpenNVR
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
YOLOv8 adapter focused on raw model inference output.
"""
import logging
import os
import time
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

    def _preprocess(self, img: Any) -> Any:
        import cv2  # deferred: only loaded when inference actually runs
        return cv2.dnn.blobFromImage(
            img,
            1 / 255.0,
            (INPUT_SIZE, INPUT_SIZE),
            swapRB=True,
            crop=False,
        )

    def _run_inference(self, blob: Any) -> Any:
        import numpy as np  # deferred: only loaded when inference actually runs
        outputs = self.session.run(None, {self.session.get_inputs()[0].name: blob})
        predictions = np.transpose(outputs[0], (0, 2, 1)).squeeze()
        if predictions.ndim == 1:
            predictions = np.expand_dims(predictions, axis=0)
        return predictions

    def _convert_bbox(
        self, cx: float, cy: float, w: float, h: float, img_width: int, img_height: int
    ) -> List[int]:
        if w < 1.0:
            left = int((cx - w / 2) * img_width)
            top = int((cy - h / 2) * img_height)
            width = int(w * img_width)
            height = int(h * img_height)
        else:
            x_scale = img_width / INPUT_SIZE
            y_scale = img_height / INPUT_SIZE
            left = int((cx - w / 2) * x_scale)
            top = int((cy - h / 2) * y_scale)
            width = int(w * x_scale)
            height = int(h * y_scale)

        return [max(0, left), max(0, top), max(0, width), max(0, height)]

    def infer_local(self, input_data: Any) -> Dict[str, Any]:
        import numpy as np  # deferred: only loaded when inference actually runs

        if self.session is None:
            self.load_model()

        start_time = time.time()
        uri = input_data["frame"]["uri"]
        img = load_image_from_uri(uri)
        img_height, img_width = img.shape[:2]

        blob = self._preprocess(img)
        raw_predictions = self._run_inference(blob)

        confidence_threshold = float(input_data.get("confidence_threshold", 0.25))
        detections = []
        for pred in raw_predictions:
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
                    "bbox": self._convert_bbox(cx, cy, w, h, img_width, img_height),
                    "class_id": class_id,
                    "confidence": round(confidence, 4),
                }
            )

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
