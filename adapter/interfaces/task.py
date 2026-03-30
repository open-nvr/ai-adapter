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
BaseTask â€” The Developer Contract for AI Adapter Task Plugins.

Every task plugin MUST inherit from this class and implement the required methods.
This is the ONLY file a contributor needs to read to understand how to add
a new AI capability to OpenNVR.

HOW TO CREATE A NEW TASK PLUGIN:
    1. Create a folder:  adapter/tasks/your_task_name/
    2. Create task.py inside it with a class called Task that extends BaseTask
    3. Implement: name, description, run(), schema(), get_model_info()
    4. Drop the folder in â€” the system auto-discovers it at startup

EXAMPLE:

    from adapter.interfaces import BaseTask

    class Task(BaseTask):

        name = "fire_detection"
        description = "Detect fire or flames in camera frame"

        def setup(self):
            # Load your model here (called once at startup)
            from ultralytics import YOLO
            self.model = YOLO("fire_model.pt")

        def run(self, image, params: dict) -> dict:
            results = self.model.predict(image)
            return {
                "label": "fire",
                "confidence": float(results[0].boxes.conf[0]),
                "bbox": results[0].boxes.xywh[0].tolist(),
            }

        def schema(self) -> dict:
            return {
                "task": "fire_detection",
                "description": self.description,
                "response_fields": {
                    "label": {"type": "string"},
                    "confidence": {"type": "float"},
                    "bbox": {"type": "array[int]"},
                },
            }

        def get_model_info(self) -> dict:
            return {
                "model": "fire_yolov8",
                "framework": "pytorch",
                "device": "cpu",
                "tasks": ["fire_detection"],
            }
"""

from abc import ABC, abstractmethod
from typing import Dict, Any
import numpy as np


class BaseTask(ABC):
    """
    Abstract base class that ALL task plugins must implement.

    Class Attributes (REQUIRED):
        name (str):        Unique task identifier, e.g. "person_detection"
        description (str): Human-readable description of what this task does

    Methods to Implement:
        run(image, params)   â€” Execute inference and return results
        schema()             â€” Return the response schema for this task
        get_model_info()     â€” Return model metadata for monitoring

    Optional Override:
        setup()              â€” Called once at startup to load models / weights
        cleanup()            â€” Called at shutdown to release resources
    """

    # --- Class attributes (override in subclass) --------------------------

    name: str = ""
    description: str = ""

    # --- Lifecycle ---------------------------------------------------------

    def __init__(self):
        """Initialize the task. Calls setup() for model loading."""
        if not self.name:
            raise ValueError(
                f"{self.__class__.__name__} must define a 'name' attribute"
            )
        self.setup()

    def setup(self):
        """
        Called once at startup. Override this to load models / weights.

        This keeps __init__ clean and gives a single place for heavy
        initialization like loading ONNX sessions or PyTorch models.
        """
        pass  # Default: nothing to load

    def cleanup(self):
        """
        Called at shutdown. Override to release GPU memory or temp files.
        """
        pass  # Default: nothing to clean

    # --- Abstract methods (MUST implement) ---------------------------------

    @abstractmethod
    def run(self, image: np.ndarray, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute inference on a single image.

        This is the MAIN method. Receive an image, run your model, return
        results in the format described by schema().

        Args:
            image:  Input image as a numpy array (BGR, HxWxC).
                    Already decoded â€” no need to read from disk.
            params: Optional parameters from the request, e.g.
                    {"confidence_threshold": 0.5, "camera_id": "cam_1"}

        Returns:
            Dictionary matching the structure defined in schema().
            Must include at minimum the fields declared there.

        Example:
            return {
                "label": "person",
                "confidence": 0.92,
                "bbox": [100, 150, 200, 300],
            }
        """
        pass

    def schema(self) -> Dict[str, Any]:
        """
        Return the response schema for this task.
        Reads from schema.json in the task's directory.
        """
        import os
        import json
        import inspect

        module_path = inspect.getfile(self.__class__)
        task_dir = os.path.dirname(os.path.abspath(module_path))
        schema_path = os.path.join(task_dir, "schema.json")
        
        try:
            with open(schema_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            return {
                "task": self.name,
                "description": self.description,
                "error": f"Failed to load schema.json: {e}",
                "response_fields": {}
            }

    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """
        Return metadata about the underlying model.

        Used by monitoring dashboards, health checks, and the GET /tasks
        endpoint so operators can see what's running.

        Returns:
            Dictionary with at least these keys:
            {
                "model":     str   â€“ model name, e.g. "yolov8n"
                "framework": str   â€“ "onnx", "pytorch", "tensorflow", etc.
                "device":    str   â€“ "cpu" or "cuda:0"
                "tasks":     list  â€“ task names this model serves
            }

        Example:
            return {
                "model": "yolov8n",
                "framework": "onnx",
                "device": "cpu",
                "tasks": ["person_detection"],
            }
        """
        pass

    # --- Helpers (available to all task plugins) ----------------------------

    def validate_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and apply defaults to incoming params.

        Override this if your task accepts specific parameters.
        Default implementation just returns params as-is.

        Args:
            params: Raw params dict from the request

        Returns:
            Validated / defaulted params dict
        """
        return params or {}

    def __repr__(self) -> str:
        return f"<Task: {self.name} ({self.__class__.__name__})>"
