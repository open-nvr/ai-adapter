# Copyright (c) 2026 OpenNVR
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
InsightFace Buffalo-L Adapter for face analysis tasks using Clean Architecture.
Implements lazy loading via BaseAdapter.load_model().

Supports:
- face_detection: Detect faces with bounding boxes, landmarks, age, gender
- face_embedding: Generate 512-dimensional face embeddings
- face_recognition: Identify person from registered faces (requires DB)
- face_verify: Compare two faces (1:1 verification)
- watchlist_check: Check if face matches anyone on watchlist (requires DB)
"""
import logging
import os
import re
import time
import pathlib
from typing import Dict, Any, List, Optional

# cv2, numpy, insightface, and FaceDatabase are intentionally NOT imported
# at module level. They are deferred into the methods that need them so that:
#   1. PluginManager can discover InsightFaceAdapter without loading ~300MB
#      of insightface/onnxruntime/scipy at startup.
#   2. If `insightface` is not installed (e.g. yolo-only deployment),
#      the adapter is gracefully skipped by PluginManager rather than
#      crashing the entire server.
#
# Install optional deps:  uv sync --extra face
from app.adapters.base import BaseAdapter
from app.utils.image_utils import load_image_from_uri
from app.config import BASE_FRAMES_DIR, MODEL_WEIGHTS_DIR

logger = logging.getLogger(__name__)



class InsightFaceAdapter(BaseAdapter):
    """
    Adapter for InsightFace Buffalo-L model.
    Model is loaded lazily on first inference call.
    
    Features:
    - Lazy loading: Model loads on first inference, not at startup
    - GPU acceleration: Uses CUDA if available, falls back to CPU
    - In-memory face matching: Fast cosine similarity search
    """
    
    name = "insightface_adapter"
    type = "vision"
    
    SUPPORTED_TASKS = [
        "face_detection",
        "face_embedding",
        "face_recognition",
        "face_verify",
        "watchlist_check"
    ]
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self._face_model: Optional[FaceAnalysis] = None
        self._face_db = None
        
        weights_path = self.config.get("weights_path", "insightface")
        if os.path.isabs(weights_path):
            self._model_path = weights_path
        else:
            self._model_path = os.path.join(MODEL_WEIGHTS_DIR, weights_path)
    
    def load_model(self):
        """
        Heavy model loading - only called on first inference.
        Loads InsightFace Buffalo-L model.
        """
        # Import insightface here, not at module level, so that
        # deployments without insightface installed can still run
        # other adapters (yolov8, blip, etc.) without crashing.
        try:
            from insightface.app import FaceAnalysis  # optional dep: uv sync --extra face
        except ImportError as exc:
            raise ImportError(
                "InsightFace is not installed. "
                "Install it with: uv sync --extra face"
            ) from exc

        logger.info("Loading InsightFace Buffalo-L model (first use, may take a moment)...")
        start_time = time.time()

        os.makedirs(self._model_path, exist_ok=True)

        self._face_model = FaceAnalysis(
            name="buffalo_l",
            root=self._model_path,
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        self._face_model.prepare(ctx_id=0, det_size=(640, 640))

        self.model = self._face_model  # Set model reference for BaseAdapter

        # Import FaceDatabase here so a missing db module doesn't
        # prevent the adapter from loading.
        try:
            from app.db.face_db import FaceDatabase
            self._face_db = FaceDatabase()
        except ImportError:
            self._face_db = None
            logger.warning("FaceDatabase not available — face recognition/watchlist disabled")

        load_time = time.time() - start_time
        logger.info(f"InsightFace model loaded in {load_time:.1f}s")
    
    def infer_local(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Route to appropriate face analysis task.
        
        Args:
            input_data: Dict containing 'frame' with 'uri' key
            
        Returns:
            Task-specific result dictionary
        """
        task = input_data.get("task", "face_detection")
        
        if task not in self.SUPPORTED_TASKS:
            raise ValueError(f"Unsupported task: {task}. Supported: {self.SUPPORTED_TASKS}")
        
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
    
    def _detect_faces(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Detect all faces in image with bounding boxes, landmarks, age, gender."""
        import cv2      # deferred: only loaded when inference actually runs
        import numpy as np  # deferred: only loaded when inference actually runs

        start_time = time.time()
        
        uri = input_data["frame"]["uri"]
        img = load_image_from_uri(uri)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        faces = self._face_model.get(img_rgb)
        
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
        
        annotated_img = img.copy()
        for face_data in face_results:
            x1, y1, x2, y2 = face_data["bbox"]
            conf = face_data["confidence"]
            
            cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            label = f"Face {conf:.2f}"
            if "age" in face_data:
                label += f" | {face_data['gender']}/{face_data['age']}y"
            
            cv2.putText(annotated_img, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            if "landmarks" in face_data:
                for lm in face_data["landmarks"]:
                    cv2.circle(annotated_img, tuple(lm), 2, (0, 0, 255), -1)
        
        uri_parts = uri.replace("opennvr://frames/", "").split("/")
        raw_dir = uri_parts[0] if uri_parts else "camera_0"
        camera_dir = re.sub(r'[^a-zA-Z0-9_\-]', '_', raw_dir) or "camera_0"

        base = pathlib.Path(BASE_FRAMES_DIR).resolve()
        annotated_path = (base / camera_dir / "face_detection_annotated.jpg").resolve()
        try:
            annotated_path.relative_to(base)
        except ValueError:
            camera_dir = "camera_0"
            annotated_path = base / camera_dir / "face_detection_annotated.jpg"

        os.makedirs(annotated_path.parent, exist_ok=True)
        cv2.imwrite(str(annotated_path), annotated_img)
        
        latency = int((time.time() - start_time) * 1000)
        
        result = {
            "faces": face_results,
            "face_count": len(face_results),
            "annotated_image_uri": f"opennvr://frames/{camera_dir}/face_detection_annotated.jpg",
            "executed_at": int(time.time() * 1000),
            "latency_ms": latency
        }
        
        logger.info(f"[FACE_DETECTION] Found {len(face_results)} faces in {latency}ms")
        return result
    
    def _get_embedding(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract face embedding (512-dimensional vector)."""
        import numpy as np  # deferred: only loaded when inference actually runs

        start_time = time.time()
        
        uri = input_data["frame"]["uri"]
        img = load_image_from_uri(uri)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        faces = self._face_model.get(img_rgb)
        
        if len(faces) == 0:
            return {
                "embedding": None,
                "face_bbox": None,
                "message": "No face detected",
                "latency_ms": int((time.time() - start_time) * 1000)
            }
        
        best_face = max(faces, key=lambda f: f.det_score)
        
        result = {
            "embedding": best_face.embedding.tolist(),
            "face_bbox": [int(x) for x in best_face.bbox.tolist()],
            "embedding_size": len(best_face.embedding),
            "latency_ms": int((time.time() - start_time) * 1000)
        }
        
        logger.info(f"[FACE_EMBEDDING] Extracted {result['embedding_size']}-dim embedding")
        return result
    
    def _recognize_face(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Identify person from registered faces."""
        import numpy as np  # deferred: only loaded when inference actually runs

        start_time = time.time()
        
        embedding_result = self._get_embedding(input_data)
        
        if embedding_result["embedding"] is None:
            return {
                "recognized": False,
                "message": "No face detected",
                "latency_ms": int((time.time() - start_time) * 1000)
            }
        
        if self._face_db is None:
            return {
                "recognized": False,
                "message": "Face database not available",
                "face_bbox": embedding_result["face_bbox"],
                "latency_ms": int((time.time() - start_time) * 1000)
            }
        
        query_embedding = np.array(embedding_result["embedding"], dtype=np.float32)
        
        try:
            threshold = float(input_data.get("threshold", 0.5))
        except (TypeError, ValueError):
            threshold = 0.5
        if not (0.0 < threshold <= 1.0):
            threshold = 0.5
        
        match = self._face_db.get_best_match(query_embedding, threshold=threshold)
        
        if match:
            result = {
                "recognized": True,
                "person_id": match["person_id"],
                "name": match["name"],
                "category": match["category"],
                "similarity": match["similarity"],
                "face_bbox": embedding_result["face_bbox"],
                "latency_ms": int((time.time() - start_time) * 1000)
            }
            logger.info(f"[FACE_RECOGNITION] Matched: {match['name']} ({match['similarity']:.2%})")
        else:
            result = {
                "recognized": False,
                "message": f"No match found (threshold: {threshold})",
                "face_bbox": embedding_result["face_bbox"],
                "latency_ms": int((time.time() - start_time) * 1000)
            }
            logger.info("[FACE_RECOGNITION] No match found")
        
        return result
    
    def _verify_faces(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Compare two faces (1:1 verification)."""
        import numpy as np  # deferred: only loaded when inference actually runs

        start_time = time.time()
        
        emb1_result = self._get_embedding({"frame": input_data["frame1"]})
        emb2_result = self._get_embedding({"frame": input_data["frame2"]})
        
        if emb1_result["embedding"] is None:
            return {"error": "No face detected in first image"}
        if emb2_result["embedding"] is None:
            return {"error": "No face detected in second image"}
        
        emb1 = np.array(emb1_result["embedding"], dtype=np.float32)
        emb2 = np.array(emb2_result["embedding"], dtype=np.float32)
        
        emb1_norm = emb1 / (np.linalg.norm(emb1) + 1e-8)
        emb2_norm = emb2 / (np.linalg.norm(emb2) + 1e-8)
        similarity = float(np.dot(emb1_norm, emb2_norm))
        
        try:
            threshold = float(input_data.get("threshold", 0.5))
        except (TypeError, ValueError):
            threshold = 0.5
        if not (0.0 < threshold <= 1.0):
            threshold = 0.5
        
        is_same = similarity >= threshold
        
        result = {
            "is_same_person": is_same,
            "similarity": round(similarity, 4),
            "threshold": threshold,
            "face1_bbox": emb1_result["face_bbox"],
            "face2_bbox": emb2_result["face_bbox"],
            "latency_ms": int((time.time() - start_time) * 1000)
        }
        
        logger.info(f"[FACE_VERIFY] Similarity: {similarity:.2%}, Same: {is_same}")
        return result
    
    def _check_watchlist(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Check if face matches anyone on watchlist."""
        import numpy as np  # deferred: only loaded when inference actually runs

        start_time = time.time()
        
        embedding_result = self._get_embedding(input_data)
        
        if embedding_result["embedding"] is None:
            return {
                "is_on_watchlist": False,
                "message": "No face detected",
                "latency_ms": int((time.time() - start_time) * 1000)
            }
        
        if self._face_db is None:
            return {
                "is_on_watchlist": False,
                "message": "Face database not available",
                "face_bbox": embedding_result["face_bbox"],
                "latency_ms": int((time.time() - start_time) * 1000)
            }
        
        query_embedding = np.array(embedding_result["embedding"], dtype=np.float32)
        
        try:
            threshold = float(input_data.get("threshold", 0.5))
        except (TypeError, ValueError):
            threshold = 0.5
        if not (0.0 < threshold <= 1.0):
            threshold = 0.5
        
        matches = self._face_db.search_similar(
            query_embedding,
            threshold=threshold,
            category="watchlist"
        )
        
        result = {
            "is_on_watchlist": len(matches) > 0,
            "matches": matches,
            "face_bbox": embedding_result["face_bbox"],
            "latency_ms": int((time.time() - start_time) * 1000)
        }
        
        if matches:
            logger.warning(f"[WATCHLIST] ALERT! Matched {len(matches)} watchlist entries")
        else:
            logger.info("[WATCHLIST] No watchlist matches")
        
        return result
    
    def register_face(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Register a face for future recognition."""
        if self._face_db is None:
            return {"success": False, "message": "Face database not available"}
        
        embedding_result = self._get_embedding(input_data)
        
        if embedding_result["embedding"] is None:
            return {"success": False, "message": "No face detected in image"}
        
        result = self._face_db.register_face(
            person_id=input_data["person_id"],
            name=input_data["name"],
            embedding=np.array(embedding_result["embedding"], dtype=np.float32),
            category=input_data.get("category", "unknown"),
            metadata=input_data.get("metadata", {})
        )
        
        logger.info(f"[REGISTER] Registered face: {input_data['name']} ({input_data['person_id']})")
        return result
    
    def list_faces(self, category: str = None) -> Dict[str, Any]:
        """List all registered faces, optionally filtered by category."""
        if self._face_db is None:
            return {"faces": [], "message": "Face database not available"}
        
        faces = self._face_db.list_faces(category=category)
        return {"faces": faces, "count": len(faces)}
    
    def get_face(self, person_id: str) -> Dict[str, Any]:
        """Get a specific registered face by person_id."""
        if self._face_db is None:
            return {"face": None, "message": "Face database not available"}
        
        face = self._face_db.get_face(person_id)
        if face:
            return {"face": face}
        return {"face": None, "message": f"Person {person_id} not found"}
    
    def delete_face(self, person_id: str) -> Dict[str, Any]:
        """Delete a registered face by person_id."""
        if self._face_db is None:
            return {"success": False, "message": "Face database not available"}
        
        return self._face_db.delete_face(person_id)
    
    @property
    def schema(self) -> dict:
        """Return OpenNVR UI schema for face analysis tasks."""
        return {
            "face_detection": {
                "task": "face_detection",
                "description": "Detect faces with bounding boxes, landmarks, age, and gender",
                "response_fields": {
                    "faces": {
                        "type": "array[object]",
                        "description": "List of detected faces",
                        "item_schema": {
                            "bbox": {"type": "array[int]", "description": "Bounding box [x1, y1, x2, y2]"},
                            "confidence": {"type": "float", "description": "Detection confidence (0.0 to 1.0)"},
                            "landmarks": {"type": "array[array[int]]", "description": "5-point facial landmarks [[x,y], ...]"},
                            "age": {"type": "integer", "description": "Estimated age"},
                            "gender": {"type": "string", "description": "M or F"}
                        }
                    },
                    "face_count": {"type": "integer", "description": "Number of faces detected", "example": 2},
                    "annotated_image_uri": {"type": "string", "description": "URI to annotated image", "optional": True},
                    "executed_at": {"type": "integer", "description": "Timestamp in milliseconds"},
                    "latency_ms": {"type": "integer", "description": "Inference latency in milliseconds"}
                },
                "example_response": {
                    "faces": [{"bbox": [120, 80, 220, 200], "confidence": 0.99, "landmarks": [[150, 120], [190, 120], [170, 145], [155, 170], [185, 170]], "age": 25, "gender": "M"}],
                    "face_count": 1,
                    "annotated_image_uri": "opennvr://frames/camera_0/face_detection_annotated.jpg",
                    "executed_at": 1735546430000,
                    "latency_ms": 45
                }
            },
            "face_recognition": {
                "task": "face_recognition",
                "description": "Identify person from registered faces (1:N matching)",
                "response_fields": {
                    "recognized": {"type": "boolean", "description": "Whether a match was found"},
                    "person_id": {"type": "string", "description": "Matched person ID", "example": "emp_001"},
                    "name": {"type": "string", "description": "Person name", "example": "John Doe"},
                    "similarity": {"type": "float", "description": "Match similarity score (0.0 to 1.0)"},
                    "executed_at": {"type": "integer", "description": "Timestamp in milliseconds"},
                    "latency_ms": {"type": "integer", "description": "Inference latency in milliseconds"}
                }
            },
            "face_verify": {
                "task": "face_verify",
                "description": "Compare two faces (1:1 verification)",
                "response_fields": {
                    "is_same_person": {"type": "boolean", "description": "Whether faces match"},
                    "similarity": {"type": "float", "description": "Similarity score (0.0 to 1.0)"},
                    "threshold": {"type": "float", "description": "Threshold used for matching"}
                }
            },
            "watchlist_check": {
                "task": "watchlist_check",
                "description": "Check if face matches anyone on watchlist",
                "response_fields": {
                    "is_on_watchlist": {"type": "boolean", "description": "Whether person is on watchlist"},
                    "matches": {"type": "array[object]", "description": "List of matching watchlist entries"}
                }
            }
        }

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "type": self.type,
            "model": "buffalo_l",
            "framework": "onnxruntime",
            "tasks": self.SUPPORTED_TASKS,
            "model_path": self._model_path,
            "face_db_available": self._face_db is not None,
            "model_loaded": self._face_model is not None,
        }
