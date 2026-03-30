# Copyright (c) 2026 OpenNVR
# This file is part of OpenNVR.
# 
# OpenNVR is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# OpenNVR is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU Affero General Public License
# along with OpenNVR.  If not, see <https://www.gnu.org/licenses/>.

﻿"""
YOLOv8 model handler for person detection and counting.
"""
import onnxruntime as ort
import cv2
import numpy as np
import pathlib
import re
import time
import os
from typing import List, Dict, Any
from fastapi import HTTPException

from .base_handler import BaseModelHandler
from ..utils.image_utils import load_image_from_uri
from ..utils.visualization import draw_bounding_boxes
from ..config import CONFIDENCE_THRESHOLD, INPUT_SIZE, BASE_FRAMES_DIR


class YOLOv8Handler(BaseModelHandler):
    """
    Handler for YOLOv8 model supporting:
    - person_detection: Detect the person with highest confidence
    - person_counting: Count all persons in the image
    """
    
    def __init__(self, model_path: str):
        """
        Initialize YOLOv8 handler and load the ONNX model.
        
        Args:
            model_path: Path to yolov8n.onnx file
        """
        super().__init__(model_path)
        # Load ONNX model
        self.session = ort.InferenceSession(
            model_path, 
            providers=['CPUExecutionProvider']
        )
        print(f"âœ“ YOLOv8 model loaded from {model_path}")
    
    def get_supported_tasks(self) -> List[str]:
        """Return list of tasks supported by YOLOv8."""
        return ["person_detection"]  # Counting moved to YOLOv11

    def get_model_info(self) -> Dict[str, Any]:
        """Return YOLOv8 model metadata."""
        return {
            "model": "yolov8n",
            "framework": "onnx",
            "device": "cpu",
            "tasks": self.get_supported_tasks(),
        }

    def infer(self, task: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Route inference request to appropriate task method.
        
        Args:
            task: Task name ("person_detection" or "person_counting")
            input_data: Request data containing frame URI
            
        Returns:
            Task-specific result dictionary
        """
        if not self.validate_task(task):
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported task: {task}"
            )
        
        # Route to appropriate method
        if task == "person_detection":
            return self._detect_persons(input_data)
    
    def _preprocess(self, img: np.ndarray) -> np.ndarray:
        """
        Preprocess image for YOLOv8 inference.
        
        Args:
            img: Input image as numpy array
            
        Returns:
            Preprocessed blob ready for inference
        """
        blob = cv2.dnn.blobFromImage(
            img, 
            1/255.0, 
            (INPUT_SIZE, INPUT_SIZE), 
            swapRB=True, 
            crop=False
        )
        return blob
    
    def _run_inference(self, blob: np.ndarray) -> np.ndarray:
        """
        Run ONNX inference on preprocessed blob.
        
        Args:
            blob: Preprocessed image blob
            
        Returns:
            Model predictions
        """
        outputs = self.session.run(
            None, 
            {self.session.get_inputs()[0].name: blob}
        )
        predictions = np.transpose(outputs[0], (0, 2, 1)).squeeze()
        return predictions
    
    def _filter_person_detections(self, predictions: np.ndarray) -> np.ndarray:
        """
        Filter predictions to get only person class with adaptive confidence threshold.
        
        Uses size-based adaptive threshold:
        - Small objects (distant people): lower threshold (0.15)
        - Large objects (close people): higher threshold (0.25)
        
        Args:
            predictions: Raw model predictions
            
        Returns:
            Filtered person detections
        """
        # Adaptive threshold based on bounding box size
        # Small bbox (w*h < 0.05 of image) = distant person â†’ use lower threshold (0.15)
        # Large bbox = close person â†’ use higher threshold (0.25)
        
        filtered_rows = []
        for pred in predictions:
            cx, cy, w, h, conf = pred[:5]
            
            # Calculate bbox area (normalized, where 1.0 = full image)
            if w < 1.0:  # Normalized coordinates
                bbox_area = w * h
            else:  # Pixel coordinates (convert to normalized)
                bbox_area = (w / INPUT_SIZE) * (h / INPUT_SIZE)
            
            # Adaptive threshold based on size
            # Small objects (area < 0.05): threshold = 0.15
            # Large objects (area >= 0.05): threshold = 0.25
            threshold = 0.15 if bbox_area < 0.05 else 0.25
            
            # Keep detection if confidence > adaptive threshold
            if conf > threshold:
                filtered_rows.append(pred)
        
        return np.array(filtered_rows) if filtered_rows else np.array([])
    
    def _convert_bbox(self, cx: float, cy: float, w: float, h: float, 
                      img_width: int, img_height: int) -> List[int]:
        """
        Convert YOLOv8 bbox format to [left, top, width, height].
        
        Args:
            cx, cy: Center coordinates
            w, h: Width and height
            img_width, img_height: Original image dimensions
            
        Returns:3
            Bbox in [left, top, width, height] format
        """
        if w < 1.0:  # Normalized coordinates
            left = int((cx - w/2) * img_width)
            top = int((cy - h/2) * img_height)
            width = int(w * img_width)
            height = int(h * img_height)
        else:  # Pixel coordinates from 640x640 input
            x_scale = img_width / INPUT_SIZE
            y_scale = img_height / INPUT_SIZE
            left = int((cx - w/2) * x_scale)
            top = int((cy - h/2) * y_scale)
            width = int(w * x_scale)
            height = int(h * y_scale)
        
        return [max(0, left), max(0, top), width, height]
    
    def _apply_nms(self, detections: List[Dict], iou_threshold: float = 0.3) -> List[Dict]:
        """
        Apply Non-Maximum Suppression to remove duplicate detections.
        
        Args:
            detections: List of detection dictionaries with 'bbox' and 'confidence'
            iou_threshold: IoU threshold for NMS (default 0.45)
            
        Returns:
            Filtered list of detections after NMS
        """
        if not detections:
            return []
        
        # Convert to format for cv2.dnn.NMSBoxes
        boxes = [d["bbox"] for d in detections]
        confidences = [d["confidence"] for d in detections]
        
        # Apply OpenCV's NMS
        indices = cv2.dnn.NMSBoxes(
            boxes, 
            confidences, 
            score_threshold=CONFIDENCE_THRESHOLD,
            nms_threshold=iou_threshold
        )
        
        # Filter detections based on NMS results
        if len(indices) > 0:
            indices = indices.flatten()
            return [detections[i] for i in indices]
        return []
    
    def _detect_persons(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Detect the person with highest confidence in the image.
        
        Args:
            input_data: Request containing frame URI
            
        Returns:
            Detection result with bbox and confidence
        """
        start_time = time.time()
        
        # Load image
        uri = input_data["frame"]["uri"]
        img = load_image_from_uri(uri)
        h_img, w_img = img.shape[:2]
        
        # Preprocess and run inference
        blob = self._preprocess(img)
        predictions = self._run_inference(blob)
        
        # Filter person detections
        person_rows = self._filter_person_detections(predictions)
        
        # Default result
        result = {
            "label": "person",
            "confidence": 0.0,
            "bbox": [0, 0, 0, 0],
            "executed_at": int(time.time() * 1000),
            "latency_ms": 0
        }
        
        # Get best detection if any
        if len(person_rows) > 0:
            best_row = person_rows[np.argmax(person_rows[:, 4])]
            cx, cy, w, h = best_row[:4]
            conf = float(best_row[4])
            
            bbox = self._convert_bbox(cx, cy, w, h, w_img, h_img)
            result["bbox"] = bbox
            result["confidence"] = round(conf, 2)
            
            # Generate annotated image with bounding box
            detection = {"bbox": bbox, "confidence": round(conf, 2)}
            annotated_img = draw_bounding_boxes(
                img,
                [detection],  # Single detection
                count=1,
                show_labels=True,
                show_count=False  # Don't show count for detection
            )
            
            # Add "Person Detected" overlay instead of count
            cv2.rectangle(annotated_img, (5, 5), (280, 45), (0, 0, 0), -1)
            cv2.rectangle(annotated_img, (5, 5), (280, 45), (0, 255, 0), 2)
            cv2.putText(annotated_img, "Person Detected", (15, 32),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # Save annotated image â€” sanitize camera_dir to prevent path traversal
            uri_parts = uri.replace("opennvr://frames/", "").split("/")
            raw_dir = uri_parts[0] if uri_parts else "camera_0"
            camera_dir = re.sub(r'[^a-zA-Z0-9_\-]', '_', raw_dir) or "camera_0"

            base = pathlib.Path(BASE_FRAMES_DIR).resolve()
            annotated_path = (base / camera_dir / "person_detection_annotated.jpg").resolve()
            try:
                annotated_path.relative_to(base)
            except ValueError:
                camera_dir = "camera_0"
                annotated_path = base / camera_dir / "person_detection_annotated.jpg"

            os.makedirs(annotated_path.parent, exist_ok=True)
            cv2.imwrite(str(annotated_path), annotated_img)
            
            result["annotated_image_uri"] = f"opennvr://frames/{camera_dir}/person_detection_annotated.jpg"
            print(f"[VISUALIZATION] Saved to {annotated_path}")
        
        result["latency_ms"] = int((time.time() - start_time) * 1000)
        
        print(f"[DETECTION] Conf={result['confidence']}, BBox={result['bbox']}")
        return result
    
    # person_counting task moved to YOLOv11Handler with ByteTrack tracking
