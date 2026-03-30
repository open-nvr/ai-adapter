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
FastAPI Application - Main Entry Point for AI Adapter API

This file defines the REST API server that:
1. Auto-discovers and loads task plugins at startup
2. Registers tasks in a registry (Dictionary mapping)
3. Provides endpoints for health checks, capabilities, and inference
4. Routes incoming requests to the appropriate task plugin based on task name

KEY CONCEPTS:
- task_registry: Dictionary that maps task names â†’ BaseTask instances (plugin system)
- model_registry: Backward-compatible dict mapping task names â†’ handler instances
- Startup event: Loads task plugins when server starts (not on every request)
- Routing: Uses task name from request to find the right handler

ENDPOINTS:
- GET  /health        - Check if server is running
- GET  /capabilities  - List all available tasks
- GET  /tasks         - List tasks with metadata, schemas, and model info
- POST /infer         - Run model inference for a specific task
"""


from fastapi import FastAPI, HTTPException
from typing import Dict
import logging
import re
import time
import os

# Import model handlers (still needed for face mgmt endpoints & cloud inference)
from .models import InsightFaceHandler, HuggingFaceHandler
from .config import MODEL_CONFIGS, ENABLED_TASKS
from .response_schemas import get_schema_for_task, get_all_schemas

# Dynamic task plugin loader
from .loader import load_tasks
from .interfaces import BaseTask

# Global reference to InsightFace handler (needed for face management endpoints)
insightface_handler: InsightFaceHandler = None

# Global reference to HuggingFace handler (lazy-loaded for cloud inference)
huggingface_handler: HuggingFaceHandler = None


# Create FastAPI application instance
# This is the main app object that handles all HTTP requests
app = FastAPI()

logger = logging.getLogger(__name__)

# TASK REGISTRY - Maps task names to BaseTask plugin instances (new plugin system)
task_registry: Dict[str, BaseTask] = {}

# MODEL REGISTRY - Maps task names to handler instances (backward-compatible)
# Populated from task_registry so existing /infer endpoint keeps working
model_registry: Dict[str, any] = {}


@app.on_event("startup")
async def startup_event():
    """
    Server startup handler - Runs ONCE when server starts.
    
    This function:
    1. Auto-discovers task plugins from adapter/tasks/
    2. Instantiates each plugin (loading models via setup())
    3. Populates task_registry and model_registry
    
    WHY PLUGIN-BASED?
    - Zero configuration: drop a folder in adapter/tasks/ â†’ restart â†’ it works
    - Each plugin is self-contained with its own schema and model info
    - Errors in one plugin don't crash the server
    
    FLOW:
    1. load_tasks() scans adapter/tasks/ for subdirectories
    2. Each task.py with a Task(BaseTask) class is instantiated
    3. task_registry[task.name] = task_instance
    4. model_registry[task.name] = task_instance._handler (backward compat)
    5. Server is ready to accept requests
    """
    print("\nðŸš€ Initializing AI Adapter (plugin system)...")
    
    # --- Phase 1: Auto-discover and load task plugins ---
    global insightface_handler
    
    loaded_tasks = load_tasks()
    
    for name, task_instance in loaded_tasks.items():
        task_registry[name] = task_instance
        
        # Populate model_registry for backward compatibility with /infer
        # Task plugins expose their handler via _handler attribute
        if hasattr(task_instance, '_handler'):
            model_registry[name] = task_instance._handler
            
            # Capture insightface_handler for face management endpoints
            if isinstance(task_instance._handler, InsightFaceHandler):
                insightface_handler = task_instance._handler
        else:
            # Task plugin handles inference directly (no separate handler)
            model_registry[name] = task_instance
    
    print(f"\nâœ… Adapter ready with {len(task_registry)} tasks (plugin system)\n")


@app.get("/health")
def health():
    """
    Health check endpoint.
    
    Simple endpoint to verify the server is running and responding.
    Used by monitoring systems, load balancers, etc.
    
    Returns:
        {"status": "ok"} if server is alive
    
    USAGE:
        curl http://localhost:9100/health
    """
    return {"status": "ok"}


@app.get("/capabilities")
def capabilities():
    """
    Return list of supported tasks.
    
    Dynamically generates list based on:
    - Which handlers are loaded
    - Which tasks are enabled in config
    
    This is determined at runtime from model_registry.keys(),
    so it always reflects the current state.
    
    Returns:
        {"tasks": ["person_detection", "person_counting", ...]}
    
    USAGE:
        curl http://localhost:9100/capabilities
    """
    return {"tasks": list(model_registry.keys())}


@app.get("/tasks")
def list_tasks():
    """
    Return detailed info about all loaded task plugins.

    Unlike /capabilities (which returns just task names), this endpoint
    returns full metadata: description, response schema, and model info.

    Useful for:
    - Monitoring dashboards to see what models/devices are running
    - Camera UI to auto-discover available AI capabilities
    - API consumers to understand response formats before calling /infer

    Returns:
        {
            "tasks": {
                "person_detection": {
                    "name": "person_detection",
                    "description": "Detect the person with highest confidence ...",
                    "schema": { ... },
                    "model_info": {"model": "yolov8n", "framework": "onnx", ...}
                },
                ...
            },
            "count": 5
        }

    USAGE:
        curl http://localhost:9100/tasks
    """
    tasks_info = {}
    for name, task_instance in task_registry.items():
        tasks_info[name] = {
            "name": task_instance.name,
            "description": task_instance.description,
            "schema": task_instance.schema(),
            "model_info": task_instance.get_model_info(),
        }
    return {"tasks": tasks_info, "count": len(tasks_info)}


@app.get("/schema")
def get_schema(task: str = None):
    """
    Get response schema for a specific task or all tasks.
    
    This endpoint returns the structure/format of the response
    that will be returned by each model/task when you call /infer.
    
    Useful for:
    - Understanding what fields each task returns
    - API documentation and integration
    - Validating response structure
    
    Args:
        task: Optional task name. If provided, returns schema for that task only.
              If omitted, returns schemas for all available tasks.
    
    Returns:
        Schema dictionary describing response structure with:
        - Field names and types
        - Field descriptions
        - Example values
        - Example complete response
    
    USAGE:
        # Get schema for specific task
        curl http://localhost:9100/schema?task=person_detection
        
        # Get schemas for all tasks
        curl http://localhost:9100/schema
    
    Example Response (for person_detection):
    {
        "task": "person_detection",
        "description": "Detects the person with highest confidence in the image",
        "response_fields": {
            "label": {
                "type": "string",
                "description": "Object class label",
                "example": "person"
            },
            "confidence": {
                "type": "float",
                "description": "Confidence score (0.0 to 1.0)",
                "example": 0.85
            },
            ...
        },
        "example_response": {
            "label": "person",
            "confidence": 0.85,
            "bbox": [100, 150, 200, 300],
            ...
        }
    }
    """
    if task:
        # Return schema for specific task
        if task not in model_registry:
            available_tasks = list(model_registry.keys())
            raise HTTPException(
                status_code=400,
                detail=f"Unknown task: '{task}'. Available tasks: {available_tasks}"
            )
        
        # Try task plugin first, fall back to legacy response_schemas
        if task in task_registry:
            return task_registry[task].schema()
        
        try:
            return get_schema_for_task(task)
        except KeyError:
            raise HTTPException(
                status_code=500,
                detail=f"Schema not defined for task: '{task}'"
            )
    else:
        # Return schemas for all tasks (merge plugin + legacy)
        schemas = {}
        # Plugin schemas take priority
        for name, task_instance in task_registry.items():
            schemas[name] = task_instance.schema()
        # Legacy schemas as fallback for tasks not yet migrated
        for name, schema in get_all_schemas().items():
            if name not in schemas:
                schemas[name] = schema
        return {"schemas": schemas}



@app.post("/infer")
def infer(req: dict):
    """
    Main inference endpoint - Routes requests to appropriate handler.
    
    This is the MAIN API endpoint for running model inference.
    It works as a ROUTER that:
    1. Extracts task name from request
    2. Looks up handler in registry
    3. Calls handler.infer() with the task and input
    4. Returns result
    
    REQUEST FORMAT:
    {
        "task": "person_detection" or "person_counting",
        "input": {
            "frame": {
                "uri": "opennvr://frames/camera_0/latest.jpg"
            }
        }
    }
    
    RESPONSE FORMAT (depends on task):
    {
        "task": "person_counting",
        "count": 2,
        "confidence": 0.75,
        "detections": [...],
        "annotated_image_uri": "opennvr://frames/camera_0/annotated.jpg",
        "executed_at": 1234567890,
        "latency_ms": 150
    }
    
    FLOW:
    Client â†’ POST /infer â†’ Extract task name â†’ model_registry[task] â†’ handler.infer() â†’ Result
    
    Args:
        req: Request dictionary containing task and input fields
        
    Returns:
        Result dictionary from handler.infer()
        
    Raises:
        HTTPException 400: Invalid request (missing fields, unsupported task)
        HTTPException 500: Inference execution error
    """
    # Step 1: Validate request has 'task' field
    task = req.get("task")
    if not task:
        raise HTTPException(status_code=400, detail="Missing 'task' field")
    
    # Check if this is a cloud inference request (has input_data instead of input)
    is_cloud_request = "input_data" in req
    
    if is_cloud_request:
        # Cloud inference path (Hugging Face)
        global huggingface_handler
        if huggingface_handler is None:
            huggingface_handler = HuggingFaceHandler()
        
        # Validate task is supported by HF handler
        if not huggingface_handler.validate_task(task):
            available_tasks = huggingface_handler.get_supported_tasks()
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported cloud task: '{task}'. Available tasks: {available_tasks}"
            )
        
        # Run cloud inference
        try:
            result = huggingface_handler.infer(task, req["input_data"])
            return result
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception:
            logger.exception("Cloud inference error for task '%s'", task)
            raise HTTPException(status_code=500, detail="Cloud inference failed")
    
    # Step 2: Check if task is supported and enabled (local models)
    # This checks if task exists as a key in model_registry
    if task not in model_registry:
        available_tasks = list(model_registry.keys())
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported or disabled task: '{task}'. Available tasks: {available_tasks}"
        )
    
    # Step 3: Validate request has frame input
    if "input" not in req:
        raise HTTPException(status_code=400, detail="Missing 'input' field")

    has_frame = "frame" in req["input"]
    has_verify_frames = "frame1" in req["input"] and "frame2" in req["input"]
    if not has_frame and not has_verify_frames:
        raise HTTPException(
            status_code=400,
            detail="Missing 'input.frame' field (or 'input.frame1'/'input.frame2' for face_verify)"
        )
    
    # Step 4: Get handler from registry
    handler = model_registry[task]
    
    # Step 5: Run inference using the handler
    try:
        result = handler.infer(task, req["input"])
        return result
    except HTTPException:
        raise
    except Exception:
        logger.exception("Inference error for task '%s'", task)
        raise HTTPException(status_code=500, detail="Inference failed")


# =============================================================================
# FACE MANAGEMENT ENDPOINTS
# =============================================================================

@app.post("/faces/register")
def register_face(req: dict):
    """
    Register a new face for recognition/watchlist.
    
    Request:
    {
        "frame": {"uri": "opennvr://frames/camera_0/person.jpg"},
        "person_id": "emp_001",
        "name": "John Doe",
        "category": "employee"  # employee, visitor, watchlist, vip
    }
    
    Returns:
        {"success": true, "person_id": "emp_001", "message": "..."}
    """
    if insightface_handler is None:
        raise HTTPException(status_code=503, detail="InsightFace not initialized")
    
    required = ["frame", "person_id", "name"]
    for field in required:
        if field not in req:
            raise HTTPException(status_code=400, detail=f"Missing '{field}' field")

    person_id = req["person_id"]
    if not re.match(r'^[a-zA-Z0-9_\-]{1,64}$', str(person_id)):
        raise HTTPException(
            status_code=400,
            detail="Invalid person_id: use only letters, digits, hyphens, underscores (max 64 chars)"
        )

    name = str(req.get("name", "")).strip()[:255]
    if not name:
        raise HTTPException(status_code=400, detail="name must not be empty")
    req = dict(req)
    req["name"] = name

    return insightface_handler.register_face(req)


@app.get("/faces/list")
def list_faces(category: str = None):
    """
    List all registered faces.
    
    Query params:
        category: Optional filter (employee, watchlist, etc.)
    
    Returns:
        {"faces": [{person_id, name, category}, ...], "total_count": N}
    """
    if insightface_handler is None:
        raise HTTPException(status_code=503, detail="InsightFace not initialized")
    
    return insightface_handler.list_registered_faces(category=category)


@app.get("/faces/{person_id}")
def get_face(person_id: str):
    """
    Get details for a specific registered face.
    
    Returns:
        {"person_id": "...", "name": "...", "category": "...", "metadata": {}}
    """
    if insightface_handler is None:
        raise HTTPException(status_code=503, detail="InsightFace not initialized")
    
    face = insightface_handler._face_db.get_face(person_id)
    if face is None:
        raise HTTPException(status_code=404, detail=f"Person {person_id} not found")
    
    return face


@app.delete("/faces/{person_id}")
def delete_face(person_id: str):
    """
    Delete a registered face.
    
    Returns:
        {"success": true, "message": "Face deleted"}
    """
    if insightface_handler is None:
        raise HTTPException(status_code=503, detail="InsightFace not initialized")
    
    result = insightface_handler.delete_registered_face(person_id)
    if not result.get("success"):
        raise HTTPException(status_code=404, detail=result.get("message"))
    
    return result
