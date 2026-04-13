# Copyright (c) 2026 OpenNVR
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
YOLOv11 Adapter with ByteTrack for person counting using Clean Architecture.
Implements lazy loading via BaseAdapter.load_model().
"""
import logging
import os
import re
import time
import pathlib
import random
from typing import Dict, Any, List

# cv2 is intentionally NOT imported at module level — deferred into
# _count_persons() so PluginManager can discover this class without
# loading OpenCV for deployments that don't use person-counting.
from app.adapters.base import BaseAdapter
from app.utils.image_utils import load_image_from_uri
from app.config import BASE_FRAMES_DIR, MODEL_WEIGHTS_DIR

logger = logging.getLogger(__name__)

# Optimized parameters from testing (90-95% accuracy)
COUNTING_CONFIDENCE_THRESHOLD = 0.42
COUNTING_IOU_THRESHOLD = 0.35
DEFAULT_MODEL_NAME = "yolo11m.pt"

# Colorful palette for bounding boxes
COLORS = [
    (255, 0, 0),      # Blue
    (0, 255, 0),      # Green
    (0, 0, 255),      # Red
    (255, 255, 0),    # Cyan
    (255, 0, 255),    # Magenta
    (0, 255, 255),    # Yellow
    (128, 0, 255),    # Purple
    (255, 128, 0),    # Orange
    (0, 128, 255),    # Light Blue
    (255, 0, 128),    # Pink
    (128, 255, 0),    # Lime
    (0, 255, 128),    # Spring Green
]


class YOLOv11Adapter(BaseAdapter):
    """
    Adapter for YOLOv11 model with ByteTrack support for person counting.
    Model is loaded lazily on first inference call.
    
    Features:
    - ByteTrack temporal tracking for consistent person IDs across frames
    - Optimized for CPU with medium model (90-95% accuracy)
    - Colorful bounding boxes with tracking IDs
    """
    
    name = "yolov11_adapter"
    type = "vision"
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self._yolo_model = None  # Ultralytics YOLO model, loaded lazily
        
        model_name = self.config.get("weights_path", DEFAULT_MODEL_NAME)
        if os.path.isabs(model_name):
            self._model_path = model_name
        else:
            self._model_path = os.path.join(MODEL_WEIGHTS_DIR, model_name)
    
    def load_model(self):
        """
        Heavy model loading - only called on first inference.
        Loads YOLOv11 model with ByteTrack support.
        """
        from ultralytics import YOLO
        
        os.makedirs(os.path.dirname(self._model_path), exist_ok=True)
        
        if os.path.exists(self._model_path):
            self._yolo_model = YOLO(self._model_path)
            logger.info(f"YOLOv11-Medium loaded from {self._model_path}")
        else:
            logger.info(f"Downloading YOLOv11-Medium (target: {self._model_path})...")
            self._yolo_model = YOLO(DEFAULT_MODEL_NAME)
            
            if os.path.abspath(DEFAULT_MODEL_NAME) != os.path.abspath(self._model_path):
                import shutil
                if os.path.exists(DEFAULT_MODEL_NAME):
                    try:
                        shutil.move(DEFAULT_MODEL_NAME, self._model_path)
                        logger.info(f"Moved model to {self._model_path}")
                        self._yolo_model = YOLO(self._model_path)
                    except Exception as e:
                        logger.warning(f"Could not move model file: {e}")
            
            logger.info("YOLOv11-Medium ready with ByteTrack tracking")
        
        self.model = self._yolo_model  # Set model reference for BaseAdapter
    
    def infer_local(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute person counting inference.
        
        Args:
            input_data: Dict containing 'frame' with 'uri' key
            
        Returns:
            Counting result with count and detections
        """
        task = input_data.get("task", "person_counting")
        if task != "person_counting":
            raise ValueError(f"YOLOv11Adapter only supports person_counting, got: {task}")
        
        return self._count_persons(input_data)
    
    def _get_random_color(self):
        """Get a random vibrant color for bounding boxes."""
        return random.choice(COLORS)
    
    def _count_persons(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Count all persons detected in the image using ByteTrack tracking."""
        import cv2  # deferred: only loaded when inference actually runs

        start_time = time.time()
        
        uri = input_data["frame"]["uri"]
        img = load_image_from_uri(uri)
        h_img, w_img = img.shape[:2]
        
        results = self._yolo_model.track(
            img,
            conf=COUNTING_CONFIDENCE_THRESHOLD,
            iou=COUNTING_IOU_THRESHOLD,
            tracker="bytetrack.yaml",
            persist=True,
            verbose=False
        )
        
        person_count = 0
        tracked_ids = set()
        annotated_frame = img.copy()
        detections = []
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls = int(box.cls[0])
                if cls == 0:  # Person class
                    track_id = int(box.id[0]) if box.id is not None else -1
                    
                    if track_id not in tracked_ids:
                        tracked_ids.add(track_id)
                        person_count += 1
                    
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    
                    color = self._get_random_color()
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 3)
                    
                    if track_id != -1:
                        label = f"ID:{track_id} ({conf:.2f})"
                    else:
                        label = f"Person ({conf:.2f})"
                    
                    (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(annotated_frame, (x1, y1 - label_h - 10), (x1 + label_w, y1), color, -1)
                    cv2.putText(annotated_frame, label, (x1, y1 - 5),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    
                    bbox = [x1, y1, x2 - x1, y2 - y1]
                    detections.append({
                        "bbox": bbox,
                        "confidence": round(conf, 2),
                        "track_id": track_id if track_id != -1 else None
                    })
        
        count_text = f"Persons Detected: {person_count}"
        (text_w, text_h), _ = cv2.getTextSize(count_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
        cv2.rectangle(annotated_frame, (5, 5), (15 + text_w, 40 + text_h), (0, 0, 0), -1)
        cv2.rectangle(annotated_frame, (5, 5), (15 + text_w, 40 + text_h), (0, 255, 255), 2)
        cv2.putText(annotated_frame, count_text, (10, 35),
                  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
        avg_confidence = (
            sum(d["confidence"] for d in detections) / len(detections)
            if detections else 0.0
        )
        
        result = {
            "task": "person_counting",
            "count": person_count,
            "confidence": round(avg_confidence, 2),
            "detections": detections,
            "executed_at": int(time.time() * 1000),
            "latency_ms": int((time.time() - start_time) * 1000)
        }
        
        if person_count > 0:
            uri_parts = uri.replace("opennvr://frames/", "").split("/")
            raw_dir = uri_parts[0] if uri_parts else "camera_0"
            camera_dir = re.sub(r'[^a-zA-Z0-9_\-]', '_', raw_dir) or "camera_0"

            base = pathlib.Path(BASE_FRAMES_DIR).resolve()
            annotated_path = (base / camera_dir / "person_counting_tracked.jpg").resolve()
            try:
                annotated_path.relative_to(base)
            except ValueError:
                camera_dir = "camera_0"
                annotated_path = base / camera_dir / "person_counting_tracked.jpg"

            os.makedirs(annotated_path.parent, exist_ok=True)
            cv2.imwrite(str(annotated_path), annotated_frame)
            
            result["annotated_image_uri"] = f"opennvr://frames/{camera_dir}/person_counting_tracked.jpg"
            logger.info(f"[YOLO11-BYTETRACK] Count={person_count}, Avg Conf={avg_confidence:.2f}, Tracked IDs={len(tracked_ids)}")
        else:
            logger.info("[YOLO11-BYTETRACK] No persons detected")
        
        return result
    
    @property
    def schema(self) -> dict:
        """Return OpenNVR UI schema for person_counting task."""
        return {
            "task": "person_counting",
            "description": "Count all persons in the image with tracking IDs",
            "response_fields": {
                "task": {
                    "type": "string",
                    "description": "Task name identifier",
                    "example": "person_counting"
                },
                "count": {
                    "type": "integer",
                    "description": "Total number of persons detected in the frame",
                    "example": 3
                },
                "confidence": {
                    "type": "float",
                    "description": "Average confidence across all detections (0.0 to 1.0)",
                    "example": 0.82
                },
                "detections": {
                    "type": "array[object]",
                    "description": "List of detected persons with bounding boxes and tracking IDs",
                    "item_schema": {
                        "bbox": {
                            "type": "array[int]",
                            "description": "[left, top, width, height]",
                            "example": [100, 150, 200, 300]
                        },
                        "confidence": {
                            "type": "float",
                            "description": "Detection confidence (0.0 to 1.0)",
                            "example": 0.85
                        },
                        "track_id": {
                            "type": "integer",
                            "description": "Unique tracking ID for this person",
                            "example": 1
                        }
                    }
                },
                "annotated_image_uri": {
                    "type": "string",
                    "description": "URI to annotated image with all detections drawn",
                    "example": "opennvr://frames/camera_0/person_counting_tracked.jpg",
                    "optional": True
                },
                "executed_at": {
                    "type": "integer",
                    "description": "Timestamp in milliseconds when inference was executed",
                    "example": 1735546430000
                },
                "latency_ms": {
                    "type": "integer",
                    "description": "Inference latency in milliseconds",
                    "example": 180
                }
            },
            "example_response": {
                "task": "person_counting",
                "count": 3,
                "confidence": 0.82,
                "detections": [
                    {"bbox": [100, 150, 200, 300], "confidence": 0.85, "track_id": 1},
                    {"bbox": [400, 200, 180, 280], "confidence": 0.78, "track_id": 2},
                    {"bbox": [700, 100, 150, 250], "confidence": 0.92, "track_id": 3}
                ],
                "annotated_image_uri": "opennvr://frames/camera_0/person_counting_tracked.jpg",
                "executed_at": 1735546430000,
                "latency_ms": 180
            }
        }

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "type": self.type,
            "model": "yolo11m",
            "framework": "pytorch",
            "tasks": ["person_counting"],
            "tracker": "bytetrack",
            "model_path": self._model_path,
            "model_loaded": self._yolo_model is not None,
        }
