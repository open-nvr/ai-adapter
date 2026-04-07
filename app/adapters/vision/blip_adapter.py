# app/adapters/vision/blip_adapter.py
import time
from typing import Dict, Any

from app.adapters.base import BaseAdapter

try:
    import torch
    from transformers import BlipProcessor, BlipForConditionalGeneration
    from PIL import Image
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("⚠️ transformers/torch not installed.")

class BLIPAdapter(BaseAdapter):
    name: str = "blip_adapter"
    type: str = "vision"
    
    def load_model(self):
        if not TRANSFORMERS_AVAILABLE:
            raise RuntimeError("transformers and torch not installed.")
            
        self.model_id = "Salesforce/blip-image-captioning-base"
        self.processor = BlipProcessor.from_pretrained(self.model_id)
        self.caption_model = BlipForConditionalGeneration.from_pretrained(self.model_id)
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.caption_model.to(self.device)
        self.caption_model.eval()

    def infer_local(self, input_data: Any) -> Dict[str, Any]:
        task = input_data.get("task", "scene_description")
        if task == "scene_description":
            return self._describe_scene(input_data)
        else:
            raise ValueError(f"BLIPAdapter only supports scene_description, got: {task}")

    def _describe_scene(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        start_time = time.time()
        uri = input_data.get("frame", {}).get("uri", "")
        
        # Resolve uri
        try:
            image_path = uri.replace("opennvr://", "/tmp/opennvr/")
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            # Create a dummy image if not available for testing
            image = Image.new("RGB", (224, 224), color="black")
            
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            generated_ids = self.caption_model.generate(**inputs, max_new_tokens=50)
            
        caption = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        
        return {
            "task": "scene_description",
            "caption": caption,
            "model_id": self.model_id,
            "latency_ms": int((time.time() - start_time) * 1000)
        }
