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
Base model handler interface that all model handlers must implement.

This file defines an Abstract Base Class (ABC) that serves as a CONTRACT
for all model handlers in the system. Any new model handler (YOLOv8, YOLO11, etc.)
MUST inherit from this class and implement the required methods.

WHY THIS EXISTS:
- Ensures all handlers have the same interface (methods)
- Makes it easy to add new models without breaking existing code
- Provides type safety and prevents forgotten method implementations
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any
import os


class BaseModelHandler(ABC):
    """
    Abstract base class for all model handlers.
    
    This is a TEMPLATE that defines what ALL handlers must have:
    1. get_supported_tasks() - Returns list of tasks the model can do
    2. infer() - Runs the model inference for a given task
    
    Any class that inherits from this MUST implement these methods,
    otherwise Python will refuse to create an instance of it.
    
    USAGE:
        class MyModelHandler(BaseModelHandler):
            def get_supported_tasks(self):
                return ["task1", "task2"]
            
            def infer(self, task, input_data):
                # implementation
                return result
    """
    
    def __init__(self, model_path: str):
        """
        Initialize the model handler.
        
        This is called when creating a new handler instance.
        Child classes should call super().__init__(model_path) and then
        load their specific model.
        
        Args:
            model_path: Absolute path to the model file (e.g., .onnx, .pt)
        """
        self.model_path = model_path  # Store model file path
        self.session = None  # Placeholder for model session (ONNX, TensorFlow, etc.)
    
    @abstractmethod
    def get_supported_tasks(self) -> List[str]:
        """
        Return list of task names supported by this model.
        
        This method MUST be implemented by child classes.
        The @abstractmethod decorator enforces this requirement.
        
        Returns:
            List of task name strings that this handler can execute
            Example: ["person_detection", "person_counting"]
        
        IMPORTANT: The task names returned here will be used as keys
        in the model_registry to route requests to this handler.
        """
        pass  # Child class MUST replace this
    
    @abstractmethod
    def infer(self, task: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run inference for the specified task.
        
        This is the MAIN method that executes the model.
        Child classes must implement this to:
        1. Validate input
        2. Preprocess image
        3. Run model inference
        4. Postprocess results
        5. Return formatted output
        
        Args:
            task: Task name (must be in supported_tasks list)
            input_data: Dictionary containing request data, typically:
                {
                    "frame": {"uri": "opennvr://frames/camera_0/latest.jpg"}
                }
            
        Returns:
            Result dictionary with inference results. Format depends on task,
            but typically includes:
            {
                "task": str,
                "confidence": float,
                "bbox": [x, y, w, h] or similar,
                "executed_at": timestamp,
                "latency_ms": int
            }
        """
        pass  # Child class MUST replace this
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Return metadata about this model handler.

        Used by monitoring dashboards, health checks, and the GET /tasks
        endpoint so operators can see what's running.

        Override this in child classes to provide accurate info.
        Default implementation returns basic info from instance attributes.

        Returns:
            Dictionary with model metadata:
            {
                "model":     str  - model name / identifier
                "framework": str  - "onnx", "pytorch", "tensorflow", etc.
                "device":    str  - "cpu" or "cuda:0"
                "tasks":     list - task names this handler serves
            }
        """
        return {
            "model": os.path.basename(self.model_path) if self.model_path else "unknown",
            "framework": "unknown",
            "device": "cpu",
            "tasks": self.get_supported_tasks(),
        }

    def validate_task(self, task: str) -> bool:
        """
        Check if a task is supported by this handler.
        
        This is a HELPER method (not abstract, so optional to override).
        It checks if the given task name is in the list of supported tasks.
        
        Args:
            task: Task name to validate (e.g., "person_detection")
            
        Returns:
            True if this handler supports the task, False otherwise
        
        USAGE:
            if handler.validate_task("person_counting"):
                result = handler.infer("person_counting", input_data)
        """
        return task in self.get_supported_tasks()

