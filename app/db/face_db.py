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
SQLite/MySQL database for storing face embeddings and metadata.

STATUS: COMMENTED OUT FOR NOW - Will enable when MySQL is configured.

This module provides persistent storage for registered faces used in:
- face_recognition: Identify who a person is
- watchlist_check: Check if person is on a watchlist
"""

# =============================================================================
# DATABASE CODE - COMMENTED OUT FOR FUTURE USE
# =============================================================================
# 
# Uncomment and configure MySQL credentials when ready to use database features.
# 
# import mysql.connector
# import os
# import json
# import numpy as np
# from typing import List, Dict, Any, Optional
# from datetime import datetime
# 
# 
# class FaceDatabase:
#     """
#     MySQL-based face embedding database with in-memory caching.
#     
#     Stores face embeddings (512-dim vectors) along with:
#     - person_id: Unique identifier (e.g., "emp_001", "visitor_123")
#     - name: Display name
#     - category: Classification (employee, visitor, watchlist, vip, etc.)
#     - metadata: Additional JSON data
#     """
#     
#     def __init__(self, host="localhost", user="aiuser", password=None, database="ai_adapter"):
#         """
#         Initialize MySQL connection.
#         Credentials must be supplied via environment variables — do NOT hardcode them.
#         Use: user=os.environ["FACE_DB_USER"], password=os.environ["FACE_DB_PASSWORD"]
#         """
#         self.conn = mysql.connector.connect(
#             host=host,
#             user=user,
#             password=password,
#             database=database
#         )
#         self._init_database()
#         self._cache = {}
#         self._load_cache()
#     
#     def _init_database(self):
#         """Create tables if they don't exist."""
#         cursor = self.conn.cursor()
#         cursor.execute("""
#             CREATE TABLE IF NOT EXISTS faces (
#                 person_id VARCHAR(100) PRIMARY KEY,
#                 name VARCHAR(255) NOT NULL,
#                 category VARCHAR(50) DEFAULT 'unknown',
#                 embedding BLOB NOT NULL,
#                 metadata JSON,
#                 created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
#                 updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
#             )
#         """)
#         self.conn.commit()
#     
#     def register_face(self, person_id, name, embedding, category="unknown", metadata=None):
#         """Register a new face."""
#         pass  # TODO: Implement
#     
#     def search_similar(self, query_embedding, threshold=0.5, top_k=5):
#         """Find similar faces using cosine similarity."""
#         pass  # TODO: Implement
#     
#     def get_best_match(self, query_embedding, threshold=0.5):
#         """Get the single best matching face."""
#         pass  # TODO: Implement


# =============================================================================
# TEMPORARY IN-MEMORY STORAGE FOR TESTING
# =============================================================================

import numpy as np
from typing import Dict, List, Any, Optional


class FaceDatabase:
    """
    Temporary in-memory face storage for testing.
    
    Will be replaced with MySQL implementation later.
    """
    
    def __init__(self):
        """Initialize empty in-memory storage."""
        self._faces: Dict[str, Dict[str, Any]] = {}
        print("✓ Face database initialized (in-memory mode for testing)")
    
    def register_face(
        self,
        person_id: str,
        name: str,
        embedding: np.ndarray,
        category: str = "unknown",
        metadata: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Register a face (stored in memory only)."""
        self._faces[person_id] = {
            "person_id": person_id,
            "name": name,
            "category": category,
            "embedding": embedding.astype(np.float32),
            "metadata": metadata or {}
        }
        return {
            "success": True,
            "person_id": person_id,
            "name": name,
            "message": "Face registered (in-memory, will be lost on restart)"
        }
    
    def get_face(self, person_id: str) -> Optional[Dict[str, Any]]:
        """Get face by ID."""
        if person_id in self._faces:
            face = self._faces[person_id].copy()
            face.pop("embedding", None)
            return face
        return None
    
    def list_faces(self, category: str = None) -> List[Dict[str, Any]]:
        """List all faces."""
        faces = []
        for data in self._faces.values():
            if category and data["category"] != category:
                continue
            faces.append({
                "person_id": data["person_id"],
                "name": data["name"],
                "category": data["category"]
            })
        return faces
    
    def delete_face(self, person_id: str) -> Dict[str, Any]:
        """Delete a face."""
        if person_id not in self._faces:
            return {"success": False, "message": f"Person {person_id} not found"}
        del self._faces[person_id]
        return {"success": True, "message": f"Face {person_id} deleted"}
    
    def search_similar(
        self,
        query_embedding: np.ndarray,
        threshold: float = 0.5,
        top_k: int = 5,
        category: str = None
    ) -> List[Dict[str, Any]]:
        """Find similar faces using cosine similarity."""
        if len(self._faces) == 0:
            return []
        
        query = query_embedding.astype(np.float32)
        query_norm = query / (np.linalg.norm(query) + 1e-8)
        
        matches = []
        for data in self._faces.values():
            if category and data["category"] != category:
                continue
            
            db_embedding = data["embedding"]
            db_norm = db_embedding / (np.linalg.norm(db_embedding) + 1e-8)
            similarity = float(np.dot(query_norm, db_norm))
            
            if similarity >= threshold:
                matches.append({
                    "person_id": data["person_id"],
                    "name": data["name"],
                    "category": data["category"],
                    "similarity": round(similarity, 4)
                })
        
        matches.sort(key=lambda x: x["similarity"], reverse=True)
        return matches[:top_k]
    
    def get_best_match(
        self,
        query_embedding: np.ndarray,
        threshold: float = 0.5
    ) -> Optional[Dict[str, Any]]:
        """Get the single best matching face."""
        matches = self.search_similar(query_embedding, threshold, top_k=1)
        return matches[0] if matches else None
    
    @property
    def count(self) -> int:
        """Get total number of registered faces."""
        return len(self._faces)
