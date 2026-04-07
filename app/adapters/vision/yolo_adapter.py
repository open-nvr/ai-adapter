import logging
import time
from typing import Dict, Any
from app.adapters.base import BaseAdapter

logger = logging.getLogger(__name__)

class YOLOAdapter(BaseAdapter):
    name: str = "yolo_adapter"
    type: str = "vision"
    
    def load_model(self):
        """Simulate heavy model loading logic. This is lazy-loaded (called once)."""
        logger.info(f"[{self.name}] Connecting to CUDA device and loading weights into memory...")
        time.sleep(1) # Simulate high IO operation
        self.model = "YOLO_V8_MODEL_IN_MEMORY_CACHE"
        
    def infer_local(self, input_data: Any) -> Dict[str, Any]:
        """Execute detection. By the time this runs, self.model is guaranteed to be loaded."""
        return {
            "objects": [
                {"label": "person", "confidence": 0.95, "bbox": [12, 45, 100, 250]},
                {"label": "car", "confidence": 0.88, "bbox": [300, 200, 150, 100]}
            ]
        }
