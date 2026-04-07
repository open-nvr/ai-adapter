# app/adapters/vision/insightface_adapter.py
import os
import cv2
import pathlib
import re
import time
import numpy as np
from typing import Dict, Any, Optional

from app.adapters.base import BaseAdapter
from app.db.face_db import FaceDatabase

try:
    from insightface.app import FaceAnalysis
    INSIGHTFACE_AVAILABLE = True
except ImportError:
    INSIGHTFACE_AVAILABLE = False
    print("⚠️ InsightFace not installed.")

# Mocking config variables for demo purposes
BASE_FRAMES_DIR = "/tmp/opennvr/frames"

def load_image_from_uri(uri: str) -> np.ndarray:
    """Mock implementation to load image from URI."""
    try:
        if uri.startswith("opennvr://"):
            return cv2.imread(uri.replace("opennvr://", "/tmp/opennvr/"))
    except:
        pass
    # return a dummy image if file not found
    return np.zeros((640, 640, 3), dtype=np.uint8)

class InsightFaceAdapter(BaseAdapter):
    name: str = "insightface_adapter"
    type: str = "vision"

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self._face_db = FaceDatabase()
        
    def load_model(self):
        """Loads Buffalo-L model on first use."""
        if not INSIGHTFACE_AVAILABLE:
            raise RuntimeError("InsightFace not installed. Run: pip install insightface onnxruntime-gpu")
        
        self.model_path = self.config.get("weights_path", "/tmp/models/insightface")
        os.makedirs(self.model_path, exist_ok=True)
        
        self.model = FaceAnalysis(
            name="buffalo_l",
            root=self.model_path,
            providers=['CPUExecutionProvider']
        )
        self.model.prepare(ctx_id=0, det_size=(640, 640))
        
    def infer_local(self, input_data: Any) -> Dict[str, Any]:
        task = input_data.get("task")
        
        if task == "face_detection":
            return self._detect_faces(input_data)
        elif task == "face_embedding":
            return self._get_embedding(input_data)
        elif task == "face_recognition":
            return self._recognize_face(input_data)
        elif task == "face_verify":
            return self._verify_faces(input_data)
        elif task == "watchlist_check":
            return self._check_watchlist(input_data)
        else:
            raise ValueError(f"Unsupported task for insightface: {task}")

    def _detect_faces(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        start_time = time.time()
        uri = input_data.get("frame", {}).get("uri", "")
        img = load_image_from_uri(uri)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        faces = self.model.get(img_rgb)
        face_results = []
        for face in faces:
            face_data = {
                "bbox": [int(x) for x in face.bbox.tolist()],
                "confidence": round(float(face.det_score), 3),
            }
            if face.kps is not None:
                face_data["landmarks"] = [[int(x), int(y)] for x, y in face.kps.tolist()]
            if hasattr(face, 'age') and face.age is not None:
                face_data["age"] = int(face.age)
            if hasattr(face, 'gender') and face.gender is not None:
                face_data["gender"] = "M" if face.gender == 1 else "F"
            face_results.append(face_data)
            
        return {
            "faces": face_results,
            "face_count": len(face_results),
            "latency_ms": int((time.time() - start_time) * 1000)
        }
        
    def _get_embedding(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        start_time = time.time()
        uri = input_data.get("frame", {}).get("uri", "")
        img = load_image_from_uri(uri)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        faces = self.model.get(img_rgb)
        if len(faces) == 0:
            return {"embedding": None, "face_bbox": None, "latency_ms": int((time.time() - start_time) * 1000)}
            
        best_face = max(faces, key=lambda f: f.det_score)
        return {
            "embedding": best_face.embedding.tolist(),
            "face_bbox": [int(x) for x in best_face.bbox.tolist()],
            "latency_ms": int((time.time() - start_time) * 1000)
        }

    def _recognize_face(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        emb_result = self._get_embedding(input_data)
        if emb_result["embedding"] is None:
            return {"recognized": False, "message": "No face"}
            
        q_emb = np.array(emb_result["embedding"], dtype=np.float32)
        match = self._face_db.get_best_match(q_emb, threshold=input_data.get("threshold", 0.5))
        
        if match:
            return {"recognized": True, "person_id": match["person_id"], "similarity": match["similarity"]}
        return {"recognized": False}

    def _verify_faces(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        emb1 = self._get_embedding({"frame": input_data.get("frame1")})
        emb2 = self._get_embedding({"frame": input_data.get("frame2")})
        
        if emb1["embedding"] is None or emb2["embedding"] is None:
            return {"error": "Missing face in one of the images"}
            
        e1 = np.array(emb1["embedding"], dtype=np.float32)
        e2 = np.array(emb2["embedding"], dtype=np.float32)
        similarity = float(np.dot(e1/(np.linalg.norm(e1)+1e-8), e2/(np.linalg.norm(e2)+1e-8)))
        
        return {"is_same_person": similarity >= input_data.get("threshold", 0.5), "similarity": similarity}

    def _check_watchlist(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        emb_result = self._get_embedding(input_data)
        if emb_result["embedding"] is None:
            return {"is_on_watchlist": False}
            
        q_emb = np.array(emb_result["embedding"], dtype=np.float32)
        matches = self._face_db.search_similar(q_emb, threshold=input_data.get("threshold", 0.5), category="watchlist")
        
        return {"is_on_watchlist": len(matches) > 0, "matches": matches}
