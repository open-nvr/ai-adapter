# app/adapters/vision/yolov8_adapter.py
import cv2
import numpy as np
import time
from typing import Dict, Any

from app.adapters.base import BaseAdapter

try:
    import onnxruntime as ort
    ORT_AVAILABLE = True
except ImportError:
    ORT_AVAILABLE = False
    print("⚠️ onnxruntime not installed.")

def load_image_from_uri(uri: str) -> np.ndarray:
    try:
        if uri.startswith("opennvr://"):
            return cv2.imread(uri.replace("opennvr://", "/tmp/opennvr/"))
    except:
        pass
    return np.zeros((640, 640, 3), dtype=np.uint8)

class YOLOv8Adapter(BaseAdapter):
    name: str = "yolov8_adapter"
    type: str = "vision"
    INPUT_SIZE = 640

    def load_model(self):
        if not ORT_AVAILABLE:
            raise RuntimeError("onnxruntime not installed.")
        
        self.model_path = self.config.get("weights_path", "yolov8m.onnx")
        self.session = ort.InferenceSession(self.model_path, providers=['CPUExecutionProvider'])
        
    def infer_local(self, input_data: Any) -> Dict[str, Any]:
        task = input_data.get("task", "person_detection")
        if task == "person_detection":
            return self._detect_persons(input_data)
        else:
            raise ValueError(f"YOLOv8Adapter does not support task: {task}")

    def _convert_bbox(self, cx, cy, w, h, img_width, img_height):
        if w < 1.0:
            left, top = int((cx - w/2) * img_width), int((cy - h/2) * img_height)
            width, height = int(w * img_width), int(h * img_height)
        else:
            x_scale, y_scale = img_width / self.INPUT_SIZE, img_height / self.INPUT_SIZE
            left, top = int((cx - w/2) * x_scale), int((cy - h/2) * y_scale)
            width, height = int(w * x_scale), int(h * y_scale)
        return [max(0, left), max(0, top), width, height]

    def _detect_persons(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        start_time = time.time()
        uri = input_data.get("frame", {}).get("uri", "")
        img = load_image_from_uri(uri)
        h_img, w_img = img.shape[:2]
        
        blob = cv2.dnn.blobFromImage(img, 1/255.0, (self.INPUT_SIZE, self.INPUT_SIZE), swapRB=True, crop=False)
        outputs = self.session.run(None, {self.session.get_inputs()[0].name: blob})
        predictions = np.transpose(outputs[0], (0, 2, 1)).squeeze()
        
        filtered = []
        for pred in predictions:
            cx, cy, w, h, conf = pred[:5]
            bbox_area = w*h if w < 1.0 else (w/self.INPUT_SIZE)*(h/self.INPUT_SIZE)
            threshold = 0.15 if bbox_area < 0.05 else 0.25
            if conf > threshold:
                filtered.append(pred)
                
        result = {"label": "person", "confidence": 0.0, "bbox": [0, 0, 0, 0], "latency_ms": 0}
        
        if filtered:
            filtered = np.array(filtered)
            best = filtered[np.argmax(filtered[:, 4])]
            cx, cy, w, h, conf = best[:5]
            result["bbox"] = self._convert_bbox(cx, cy, w, h, w_img, h_img)
            result["confidence"] = round(float(conf), 2)
            
        result["latency_ms"] = int((time.time() - start_time) * 1000)
        return result
