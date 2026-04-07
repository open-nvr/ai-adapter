# app/adapters/example_adapter.py
"""
Example Adapter Template
Use this template to add new intelligence models (Vision, LLMs, Audio) to OpenNVR in under 30 minutes!

Steps to add a new model:
1. Copy this file into app/adapters/vision/ (or llm/ or audio/).
2. Rename the class, `name`, and `type`.
3. Implement `load_model()` to load your weights into memory (this guarantees Lazy Loading).
4. Implement `infer_local()` to format your predictions.
5. Register it in `app/config/config.py`.
"""
import time
import logging
from typing import Dict, Any

from app.adapters.base import BaseAdapter

logger = logging.getLogger(__name__)

class ExampleAdapter(BaseAdapter):
    # Unique identifier matching your config.py entry
    name: str = "example_adapter" 
    # Must be 'vision', 'llm', 'audio', etc.
    type: str = "vision"

    def load_model(self):
        """
        Runs ONLY ONCE when the first inference request is made.
        Load your heavy PyTorch, ONNX, or TensorFlow model here.
        """
        logger.info(f"[{self.name}] ⏳ Loading weights into VRAM/RAM...")
        
        weights_path = self.config.get("weights_path", "default_weights.pt")
        # Example PyTorch: self.model = torch.load(weights_path)
        # Example ONNX: self.session = ort.InferenceSession(weights_path)
        
        self.model = "LOADED_MOCK_MODEL"
        logger.info(f"[{self.name}] ✅ Model Ready.")

    def infer_local(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Runs on every request. Model is guaranteed to be loaded at this point.
        """
        start_time = time.time()
        
        # 1. Parse input
        task = input_data.get("task", "custom_task")
        uri = input_data.get("frame", {}).get("uri", "")
        
        if task != "custom_task":
            raise ValueError(f"{self.name} does not support task: {task}")
            
        # 2. Preprocess
        # e.g., image = load_image(uri)
        
        # 3. Predict
        # e.g., output = self.model(image)
        prediction = f"Processed {uri} successfully"
        
        # 4. Format and Return Response
        return {
            "task": task,
            "prediction": prediction,
            "confidence": 0.99,
            "latency_ms": int((time.time() - start_time) * 1000)
        }
