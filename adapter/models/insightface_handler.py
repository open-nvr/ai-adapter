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
InsightFace Buffalo-L handler for face analysis tasks.

Supports:
- face_detection: Detect faces with bounding boxes, landmarks, age, gender
- face_embedding: Generate 512-dimensional face embeddings
- face_recognition: Identify person from registered faces (requires DB)
- face_verify: Compare two faces (1:1 verification)
- watchlist_check: Check if face matches anyone on watchlist (requires DB)
"""
import os
import cv2
import pathlib
import re
import time
import numpy as np
from typing import List, Dict, Any, Optional

from .base_handler import BaseModelHandler
from ..utils.image_utils import load_image_from_uri
from ..config import MODEL_WEIGHTS_DIR, BASE_FRAMES_DIR
from ..database.face_db import FaceDatabase

# InsightFace imports (will fail gracefully if not installed)
try:
    from insightface.app import FaceAnalysis
    INSIGHTFACE_AVAILABLE = True
except ImportError:
    INSIGHTFACE_AVAILABLE = False
    print("âš  InsightFace not installed. Run: pip install insightface onnxruntime-gpu")


class InsightFaceHandler(BaseModelHandler):
    """
    Handler for InsightFace Buffalo-L model.
    
    Features:
    - Lazy loading: Model loads on first inference, not at startup
    - GPU acceleration: Uses CUDA if available, falls back to CPU
    - In-memory face matching: Fast cosine similarity search
    """
    
    def __init__(self, model_path: str = None):
        """
        Initialize InsightFace handler.
        
        Args:
            model_path: Path to store InsightFace models (downloads automatically)
        """
        if model_path is None:
            model_path = os.path.join(MODEL_WEIGHTS_DIR, "insightface")
        
        super().__init__(model_path)
        
        # Lazy loading - model loads on first use
        self._model: Optional[FaceAnalysis] = None
        self._model_loaded = False
        
        # Face database for recognition/watchlist (in-memory for now)
        self._face_db = FaceDatabase()
        
        print(f"âœ“ InsightFace handler initialized (model will load on first use)")
    
    @property
    def model(self) -> 'FaceAnalysis':
        """Lazy load the InsightFace model on first access."""
        if not self._model_loaded:
            self._load_model()
        return self._model
    
    def _load_model(self):
        """Load the Buffalo-L model."""
        if not INSIGHTFACE_AVAILABLE:
            raise RuntimeError("InsightFace not installed. Run: pip install insightface onnxruntime-gpu")
        
        print("â³ Loading InsightFace Buffalo-L model (first use, may take a moment)...")
        start_time = time.time()
        
        # Create model directory if needed
        os.makedirs(self.model_path, exist_ok=True)
        
        # Initialize FaceAnalysis with buffalo_l model
        self._model = FaceAnalysis(
            name="buffalo_l",
            root=self.model_path,
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        
        # Prepare model (downloads if needed, ~183MB)
        self._model.prepare(ctx_id=0, det_size=(640, 640))
        
        self._model_loaded = True
        load_time = time.time() - start_time
        print(f"âœ“ InsightFace model loaded in {load_time:.1f}s")
    
    def get_supported_tasks(self) -> List[str]:
        """Return list of supported face analysis tasks."""
        return [
            "face_detection",
            "face_embedding",
            "face_recognition",
            "face_verify",
            "watchlist_check"
        ]

    def get_model_info(self) -> Dict[str, Any]:
        """Return InsightFace model metadata."""
        return {
            "model": "buffalo_l",
            "framework": "onnx",
            "device": "cuda" if self._model_loaded and hasattr(self, '_model') else "cpu",
            "tasks": self.get_supported_tasks(),
        }

    def infer(self, task: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Route to appropriate task handler.
        
        Args:
            task: Task name
            input_data: Request data with frame URI
            
        Returns:
            Task-specific result dictionary
        """
        if not self.validate_task(task):
            raise ValueError(f"Unsupported task: {task}")
        
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
        """
        Detect all faces in image with bounding boxes, landmarks, age, gender.
        
        Input: {"frame": {"uri": "opennvr://frames/camera_0/latest.jpg"}}
        
        Returns:
            {
                "faces": [
                    {
                        "bbox": [x1, y1, x2, y2],
                        "confidence": 0.99,
                        "landmarks": [[x,y], ...],  # 5-point landmarks
                        "age": 25,
                        "gender": "M"
                    }
                ],
                "face_count": 1,
                "latency_ms": 45
            }
        """
        start_time = time.time()
        
        # Load image from URI
        uri = input_data["frame"]["uri"]
        img = load_image_from_uri(uri)
        
        # Convert BGR to RGB (InsightFace expects RGB)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Run face detection
        faces = self.model.get(img_rgb)
        
        # Format results
        face_results = []
        for face in faces:
            face_data = {
                "bbox": [int(x) for x in face.bbox.tolist()],
                "confidence": round(float(face.det_score), 3),
            }
            
            # Add landmarks if available (5-point)
            if face.kps is not None:
                face_data["landmarks"] = [[int(x), int(y)] for x, y in face.kps.tolist()]
            
            # Add age/gender if available
            if hasattr(face, 'age') and face.age is not None:
                face_data["age"] = int(face.age)
            if hasattr(face, 'gender') and face.gender is not None:
                face_data["gender"] = "M" if face.gender == 1 else "F"
            
            face_results.append(face_data)
        
        # Draw bounding boxes on image
        annotated_img = img.copy()
        for face_data in face_results:
            x1, y1, x2, y2 = face_data["bbox"]
            conf = face_data["confidence"]
            
            # Draw box
            cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label
            label = f"Face {conf:.2f}"
            if "age" in face_data:
                label += f" | {face_data['gender']}/{face_data['age']}y"
            
            cv2.putText(annotated_img, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Draw landmarks
            if "landmarks" in face_data:
                for lm in face_data["landmarks"]:
                    cv2.circle(annotated_img, tuple(lm), 2, (0, 0, 255), -1)
        
        # Save annotated image â€” sanitize camera_dir to prevent path traversal
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
            "annotated_image_uri": f"opennvr://frames/{camera_dir}/face_detection_annotated.jpg",            "executed_at": int(time.time() * 1000),
            "latency_ms": latency
        }
        
        print(f"[FACE_DETECTION] Found {len(face_results)} faces in {latency}ms")
        return result
    
    def _get_embedding(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract face embedding (512-dimensional vector).
        
        Uses the largest/most confident face if multiple detected.
        
        Returns:
            {
                "embedding": [0.123, -0.456, ...],  # 512 floats
                "face_bbox": [x1, y1, x2, y2],
                "latency_ms": 50
            }
        """
        start_time = time.time()
        
        uri = input_data["frame"]["uri"]
        img = load_image_from_uri(uri)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        faces = self.model.get(img_rgb)
        
        if len(faces) == 0:
            return {
                "embedding": None,
                "face_bbox": None,
                "message": "No face detected",
                "latency_ms": int((time.time() - start_time) * 1000)
            }
        
        # Use face with highest confidence
        best_face = max(faces, key=lambda f: f.det_score)
        
        result = {
            "embedding": best_face.embedding.tolist(),
            "face_bbox": [int(x) for x in best_face.bbox.tolist()],
            "embedding_size": len(best_face.embedding),
            "latency_ms": int((time.time() - start_time) * 1000)
        }
        
        print(f"[FACE_EMBEDDING] Extracted {result['embedding_size']}-dim embedding")
        return result
    
    def _recognize_face(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Identify person from registered faces.
        
        Compares detected face against all faces in database.
        
        Returns:
            {
                "recognized": true/false,
                "person_id": "emp_001",
                "name": "John Doe",
                "similarity": 0.89,
                "face_bbox": [x1, y1, x2, y2]
            }
        """
        start_time = time.time()
        
        # First get embedding
        embedding_result = self._get_embedding(input_data)
        
        if embedding_result["embedding"] is None:
            return {
                "recognized": False,
                "message": "No face detected",
                "latency_ms": int((time.time() - start_time) * 1000)
            }
        
        query_embedding = np.array(embedding_result["embedding"], dtype=np.float32)

        # Search in database
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
            print(f"[FACE_RECOGNITION] Matched: {match['name']} ({match['similarity']:.2%})")
        else:
            result = {
                "recognized": False,
                "message": f"No match found (threshold: {threshold})",
                "face_bbox": embedding_result["face_bbox"],
                "latency_ms": int((time.time() - start_time) * 1000)
            }
            print(f"[FACE_RECOGNITION] No match found")
        
        return result
    
    def _verify_faces(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compare two faces (1:1 verification).
        
        Input: {
            "frame1": {"uri": "opennvr://frames/camera_0/face1.jpg"},
            "frame2": {"uri": "opennvr://frames/camera_0/face2.jpg"}
        }
        
        Returns:
            {
                "is_same_person": true/false,
                "similarity": 0.92,
                "threshold": 0.5
            }
        """
        start_time = time.time()
        
        # Get embeddings for both images
        emb1_result = self._get_embedding({"frame": input_data["frame1"]})
        emb2_result = self._get_embedding({"frame": input_data["frame2"]})
        
        if emb1_result["embedding"] is None:
            return {"error": "No face detected in first image"}
        if emb2_result["embedding"] is None:
            return {"error": "No face detected in second image"}
        
        # Calculate cosine similarity
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
        
        print(f"[FACE_VERIFY] Similarity: {similarity:.2%}, Same: {is_same}")
        return result
    
    def _check_watchlist(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check if face matches anyone on watchlist.
        
        Same as recognition but filters by category="watchlist".
        
        Returns:
            {
                "is_on_watchlist": true/false,
                "matches": [...]
            }
        """
        start_time = time.time()
        
        # Get embedding
        embedding_result = self._get_embedding(input_data)
        
        if embedding_result["embedding"] is None:
            return {
                "is_on_watchlist": False,
                "message": "No face detected",
                "latency_ms": int((time.time() - start_time) * 1000)
            }
        
        query_embedding = np.array(embedding_result["embedding"], dtype=np.float32)
        
        # Search only in watchlist category
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
            print(f"[WATCHLIST] âš  ALERT! Matched {len(matches)} watchlist entries")
        else:
            print(f"[WATCHLIST] No watchlist matches")
        
        return result
    
    # =========================================================================
    # FACE REGISTRATION (for recognition/watchlist to work)
    # =========================================================================
    
    def register_face(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Register a face for future recognition.
        
        Input: {
            "frame": {"uri": "opennvr://frames/camera_0/person.jpg"},
            "person_id": "emp_001",
            "name": "John Doe",
            "category": "employee"  # or "watchlist", "vip", etc.
        }
        """
        # Get embedding
        embedding_result = self._get_embedding(input_data)
        
        if embedding_result["embedding"] is None:
            return {"success": False, "message": "No face detected in image"}
        
        # Register in database
        result = self._face_db.register_face(
            person_id=input_data["person_id"],
            name=input_data["name"],
            embedding=np.array(embedding_result["embedding"], dtype=np.float32),
            category=input_data.get("category", "unknown"),
            metadata=input_data.get("metadata", {})
        )
        
        print(f"[REGISTER] Registered face: {input_data['name']} ({input_data['person_id']})")
        return result
    
    def list_registered_faces(self, category: str = None) -> Dict[str, Any]:
        """List all registered faces."""
        faces = self._face_db.list_faces(category=category)
        return {
            "faces": faces,
            "total_count": len(faces)
        }
    
    def delete_registered_face(self, person_id: str) -> Dict[str, Any]:
        """Delete a registered face."""
        return self._face_db.delete_face(person_id)
    
    def get_response_schema(self, task: str) -> Dict[str, Any]:
        """Return the expected response schema for each task."""
        schemas = {
            "face_detection": {
                "description": "Detect faces with bounding boxes, landmarks, age, gender",
                "response": {
                    "faces": [{"bbox": "[x1,y1,x2,y2]", "confidence": "float", "age": "int", "gender": "M/F"}],
                    "face_count": "int",
                    "annotated_image_uri": "string",
                    "latency_ms": "int"
                }
            },
            "face_embedding": {
                "description": "Extract 512-dimensional face embedding vector",
                "response": {
                    "embedding": "[512 floats]",
                    "face_bbox": "[x1,y1,x2,y2]",
                    "latency_ms": "int"
                }
            },
            "face_recognition": {
                "description": "Identify person from registered faces",
                "response": {
                    "recognized": "bool",
                    "person_id": "string",
                    "name": "string",
                    "similarity": "float 0-1",
                    "latency_ms": "int"
                }
            },
            "face_verify": {
                "description": "Compare two faces (1:1 verification)",
                "request": {
                    "frame1": {"uri": "opennvr://..."},
                    "frame2": {"uri": "opennvr://..."},
                    "threshold": "float (optional, default 0.5)"
                },
                "response": {
                    "is_same_person": "bool",
                    "similarity": "float 0-1",
                    "latency_ms": "int"
                }
            },
            "watchlist_check": {
                "description": "Check if face matches anyone on watchlist",
                "response": {
                    "is_on_watchlist": "bool",
                    "matches": "[{person_id, name, similarity}]",
                    "latency_ms": "int"
                }
            }
        }
        return schemas.get(task, {"error": f"Unknown task: {task}"})
