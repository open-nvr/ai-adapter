# app/adapters/vision/yolov11_adapter.py
import cv2
import time
from typing import Dict, Any

from app.adapters.base import BaseAdapter

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False
    print("⚠️ ultralytics not installed.")

import numpy as np
def load_image_from_uri(uri: str) -> np.ndarray:
    try:
        if uri.startswith("opennvr://"):
            return cv2.imread(uri.replace("opennvr://", "/tmp/opennvr/"))
    except:
        pass
    return np.zeros((640, 640, 3), dtype=np.uint8)

class YOLOv11Adapter(BaseAdapter):
    name: str = "yolov11_adapter"
    type: str = "vision"
    
    CONFIDENCE_THRESHOLD = 0.42
    IOU_THRESHOLD = 0.35

    def load_model(self):
        if not ULTRALYTICS_AVAILABLE:
            raise RuntimeError("ultralytics not installed.")
            
        self.model_path = self.config.get("weights_path", "yolo11m.pt")
        self.model = YOLO(self.model_path)
        
    def infer_local(self, input_data: Any) -> Dict[str, Any]:
        task = input_data.get("task", "person_counting")
        if task == "person_counting":
            return self._count_persons(input_data)
        else:
            raise ValueError(f"YOLOv11Adapter only supports person_counting, got: {task}")

    def _count_persons(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        start_time = time.time()
        uri = input_data.get("frame", {}).get("uri", "")
        img = load_image_from_uri(uri)
        
        results = self.model.track(
            img, 
            conf=self.CONFIDENCE_THRESHOLD, 
            iou=self.IOU_THRESHOLD, 
            tracker="bytetrack.yaml",
            persist=True,
            verbose=False
        )
        
        person_count = 0
        tracked_ids = set()
        detections = []
        
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for box in boxes:
                cls = int(box.cls[0])
                if cls == 0:  # person class
                    track_id = int(box.id[0]) if box.id is not None else -1
                    if track_id not in tracked_ids:
                        tracked_ids.add(track_id)
                        person_count += 1
                        
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    
                    detections.append({
                        "bbox": [x1, y1, x2 - x1, y2 - y1],
                        "confidence": round(conf, 2),
                        "track_id": track_id if track_id != -1 else None
                    })
                    
        avg_confidence = sum(d["confidence"] for d in detections) / len(detections) if detections else 0.0
        
        return {
            "task": "person_counting",
            "count": person_count,
            "confidence": round(avg_confidence, 2),
            "detections": detections,
            "latency_ms": int((time.time() - start_time) * 1000)
        }
