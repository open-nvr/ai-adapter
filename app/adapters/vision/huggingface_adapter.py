# app/adapters/vision/huggingface_adapter.py
import time
from typing import Dict, Any

from app.adapters.base import BaseAdapter

try:
    import torch
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("⚠️ transformers/torch not installed.")

class HuggingFaceAdapter(BaseAdapter):
    name: str = "huggingface_adapter"
    type: str = "vision"
    
    def load_model(self):
        if not TRANSFORMERS_AVAILABLE:
            raise RuntimeError("transformers and torch not installed.")
            
        self.model_id = "google/owlvit-base-patch32"
        self.device = 0 if torch.cuda.is_available() else -1
        self.detector = pipeline("zero-shot-object-detection", model=self.model_id, device=self.device)

    def infer_local(self, input_data: Any) -> Dict[str, Any]:
        task = input_data.get("task", "zero_shot_detection")
        if task == "zero_shot_detection":
            return self._detect_zero_shot(input_data)
        else:
            raise ValueError(f"HuggingFaceAdapter only supports zero_shot_detection, got: {task}")

    def _detect_zero_shot(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        start_time = time.time()
        uri = input_data.get("frame", {}).get("uri", "")
        # The labels to detect
        candidate_labels = input_data.get("labels", ["person", "car"])
        
        try:
            from PIL import Image
            image_path = uri.replace("opennvr://", "/tmp/opennvr/")
            image = Image.open(image_path).convert("RGB")
            
            predictions = self.detector(
                image,
                candidate_labels=candidate_labels,
            )
            
            detections = []
            for pred in predictions:
                if pred["score"] > 0.1: # Default threshold
                    box = pred["box"]
                    detections.append({
                        "label": pred["label"],
                        "confidence": round(pred["score"], 2),
                        "bbox": [box["xmin"], box["ymin"], box["xmax"] - box["xmin"], box["ymax"] - box["ymin"]]
                    })
                    
        except Exception as e:
            detections = []
            
        return {
            "task": "zero_shot_detection",
            "detections": detections,
            "latency_ms": int((time.time() - start_time) * 1000)
        }
