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

"""
YOLOv11 model handler with ByteTrack for person counting.
"""
import cv2
import pathlib
import re
import time
import os
import random
from typing import List, Dict, Any
from fastapi import HTTPException
from ultralytics import YOLO

from .base_handler import BaseModelHandler
from ..utils.image_utils import load_image_from_uri
from ..config import BASE_FRAMES_DIR, MODEL_WEIGHTS_DIR


# Optimized parameters from testing (90-95% accuracy)
CONFIDENCE_THRESHOLD = 0.42
IOU_THRESHOLD = 0.35
MODEL_NAME = "yolo11m.pt"  # Medium model - optimal for CPU

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


class YOLOv11Handler(BaseModelHandler):
    """
    Handler for YOLOv11 model with ByteTrack support for person counting.
    
    Features:
    - ByteTrack temporal tracking for consistent person IDs across frames
    - Optimized for CPU with medium model (90-95% accuracy)
    - Colorful bounding boxes with tracking IDs
    - Handles occlusion and overlapping people
    """
    
    def __init__(self, model_path: str = None):
        """
        Initialize YOLOv11 handler and load the model.
        
        Args:
            model_path: Path to model (default: None, falls back to config or download)
        """
        # Use provided path or fall back to default location
        self.model_path = model_path if model_path else os.path.join(MODEL_WEIGHTS_DIR, MODEL_NAME)
        self.session = None  # Not used for ultralytics models
        
        # Ensure target directory exists
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        
        # Load YOLOv11 model
        if os.path.exists(self.model_path):
            # Load directly from our persistent path
            self.model = YOLO(self.model_path)
            print(f"âœ“ YOLOv11-Medium loaded from {self.model_path}")
        else:
            # File missing, download it
            print(f"  Downloading YOLOv11-Medium (target: {self.model_path})...")
            # Ultralytics downloads to CWD by default if given a filename
            self.model = YOLO(MODEL_NAME) 
            
            # Move to centralized location if it's not already there
            if os.path.abspath(MODEL_NAME) != os.path.abspath(self.model_path):
                import shutil
                if os.path.exists(MODEL_NAME):
                    try:
                        shutil.move(MODEL_NAME, self.model_path)
                        print(f"  âœ“ Moved to {self.model_path}")
                        # Reload from final path
                        self.model = YOLO(self.model_path)
                    except Exception as e:
                        print(f"  ! Warning: Could not move model file: {e}")
            
            print(f"âœ“ YOLOv11-Medium ready with ByteTrack tracking")
    
    def get_supported_tasks(self) -> List[str]:
        """Return list of tasks supported by YOLOv11."""
        return ["person_counting"]  # Only counting, not detection

    def get_model_info(self) -> Dict[str, Any]:
        """Return YOLOv11 model metadata."""
        return {
            "model": "yolo11m",
            "framework": "pytorch",
            "device": "cpu",
            "tasks": self.get_supported_tasks(),
        }

    def infer(self, task: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Route inference request to appropriate task method.
        
        Args:
            task: Task name ("person_counting")
            input_data: Request data containing frame URI
            
        Returns:
            Task-specific result dictionary
        """
        if not self.validate_task(task):
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported task: {task}. YOLOv11 only supports person_counting"
            )
        
        # Route to counting method
        if task == "person_counting":
            return self._count_persons(input_data)
    
    def _get_random_color(self):
        """Get a random vibrant color for bounding boxes"""
        return random.choice(COLORS)
    
    def _count_persons(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Count all persons detected in the image using ByteTrack tracking.
        
        Args:
            input_data: Request containing frame URI
            
        Returns:
            Count result with total count and all detections with tracking IDs
        """
        start_time = time.time()
        
        # Load image
        uri = input_data["frame"]["uri"]
        img = load_image_from_uri(uri)
        h_img, w_img = img.shape[:2]
        
        # Run YOLOv11 inference with ByteTrack tracking
        results = self.model.track(
            img, 
            conf=CONFIDENCE_THRESHOLD, 
            iou=IOU_THRESHOLD, 
            tracker="bytetrack.yaml",  # Use ByteTrack for improved tracking
            persist=True,  # Maintain tracker state across frames
            verbose=False
        )
        
        # Create annotated frame with colorful boxes
        person_count = 0
        tracked_ids = set()  # Track unique IDs to avoid duplicates
        annotated_frame = img.copy()
        detections = []
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls = int(box.cls[0])
                if cls == 0:  # Person class
                    # Get tracking ID if available
                    track_id = int(box.id[0]) if box.id is not None else -1
                    
                    # Only count unique tracked IDs
                    if track_id not in tracked_ids:
                        tracked_ids.add(track_id)
                        person_count += 1
                    
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    
                    # Get a colorful color for this detection
                    color = self._get_random_color()
                    
                    # Draw thick colorful rectangle
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 3)
                    
                    # Create label with tracking ID
                    if track_id != -1:
                        label = f"ID:{track_id} ({conf:.2f})"
                    else:
                        label = f"Person ({conf:.2f})"
                    
                    (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(annotated_frame, (x1, y1 - label_h - 10), (x1 + label_w, y1), color, -1)
                    
                    # Draw label text
                    cv2.putText(annotated_frame, label, (x1, y1 - 5),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    
                    # Add to detections list
                    bbox = [x1, y1, x2 - x1, y2 - y1]  # [left, top, width, height]
                    detections.append({
                        "bbox": bbox,
                        "confidence": round(conf, 2),
                        "track_id": track_id if track_id != -1 else None
                    })
        
        # Add count overlay with background
        count_text = f"Persons Detected: {person_count}"
        (text_w, text_h), _ = cv2.getTextSize(count_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
        cv2.rectangle(annotated_frame, (5, 5), (15 + text_w, 40 + text_h), (0, 0, 0), -1)
        cv2.rectangle(annotated_frame, (5, 5), (15 + text_w, 40 + text_h), (0, 255, 255), 2)
        cv2.putText(annotated_frame, count_text, (10, 35),
                  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
        # Calculate average confidence
        avg_confidence = (
            sum(d["confidence"] for d in detections) / len(detections)
            if detections else 0.0
        )
        
        # Build result
        result = {
            "task": "person_counting",
            "count": person_count,
            "confidence": round(avg_confidence, 2),
            "detections": detections,
            "executed_at": int(time.time() * 1000),
            "latency_ms": int((time.time() - start_time) * 1000)
        }
        
        # Save annotated image
        if person_count > 0:
            # Sanitize camera_dir from URI to prevent path traversal
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
            print(f"[YOLO11-BYTETRACK] Count={person_count}, Avg Conf={avg_confidence:.2f}, Tracked IDs={len(tracked_ids)}")
        else:
            print(f"[YOLO11-BYTETRACK] No persons detected")
        
        return result
