import asyncio
import logging
from fastapi import APIRouter, HTTPException, Query, Depends
from typing import Dict, Any, List, Optional
from app.api.auth import get_api_key

logger = logging.getLogger("OpenNVR_Adapter")

public_router = APIRouter()
router = APIRouter(dependencies=[Depends(get_api_key)])

# We will inject the global router instance here upon app startup
_global_router = None
_global_engine = None

def set_global_router(model_router, pipeline_engine):
    global _global_router, _global_engine
    _global_router = model_router
    _global_engine = pipeline_engine

try:
    from app.utils.hardware import get_system_specs
except ImportError:
    get_system_specs = lambda: {"error": "hardware utility missing"}

@public_router.get("/health")
def health_check():
    """System health check and adapter status."""
    if not _global_router:
        return {"status": "starting"}
    
    adapter_health = {}
    for name, adapter in _global_router.adapters.items():
        health_fn = getattr(adapter, "health_check", None)
        if callable(health_fn):
            adapter_health[name] = health_fn()
        else:
            adapter_health[name] = {
                "status": "healthy",
                "model_loaded": getattr(adapter, "model", None) is not None,
            }
        
    return {
        "status": "healthy",
        "active_adapters": len(_global_router.adapters),
        "system_hardware": get_system_specs(),
        "adapter_details": adapter_health
    }

@public_router.get("/capabilities")
def capabilities():
    """List of all valid tasks this NVR node can process based on router mapping."""
    if not _global_router:
        return {"tasks": []}
    
    # Return all tasks defined in config.routing
    return {"tasks": list(_global_router.config_module.CONFIG.get("routing", {}).keys())}

@router.get("/tasks")
def list_tasks():
    """
    Returns a list of all available tasks with their metadata.
    Similar to legacy list_tasks() - includes task name, adapter, and description.
    """
    if not _global_router:
        return {"tasks": []}
    
    routing = _global_router.config_module.CONFIG.get("routing", {})
    tasks_list = []
    
    for task_name, adapter_name in routing.items():
        adapter = _global_router.adapters.get(adapter_name)
        task_info = {
            "task": task_name,
            "adapter": adapter_name,
            "enabled": _global_router.config_module.is_task_enabled(task_name),
            "model_loaded": adapter.model is not None if adapter else False
        }
        
        # Try to get description from adapter's schema
        if adapter:
            schema = adapter.schema
            if isinstance(schema, dict):
                if task_name in schema:
                    task_info["description"] = schema[task_name].get("description", "")
                elif "description" in schema:
                    task_info["description"] = schema["description"]
        
        tasks_list.append(task_info)
    
    return {"tasks": tasks_list, "count": len(tasks_list)}

@router.get("/schema")
async def get_schema(task: Optional[str] = Query(None, description="Task name to get schema for")):
    """
    Returns the schema for a specific task or all schemas if no task specified.
    The schema describes the expected input/output format for OpenNVR UI.
    """
    if not _global_router:
        return {"schema": {}}

    if task:
        # Get schema for a specific task
        adapter_name = _global_router.config_module.get_adapter_for_task(task)
        if not adapter_name:
            raise HTTPException(status_code=404, detail=f"No routing rule found for task '{task}'")

        try:
            adapter = await _global_router.get_or_create_adapter(adapter_name)
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))

        schema = adapter.schema
        # If schema is nested by task name, return the specific task schema
        if isinstance(schema, dict) and task in schema:
            return {"task": task, "adapter": adapter_name, "schema": schema[task]}
        return {"task": task, "adapter": adapter_name, "schema": schema}

    # Return all schemas
    all_schemas = {}
    routing = _global_router.config_module.CONFIG.get("routing", {})

    for task_name, adapter_name in routing.items():
        try:
            adapter = await _global_router.get_or_create_adapter(adapter_name)
            schema = adapter.schema
            if isinstance(schema, dict) and task_name in schema:
                all_schemas[task_name] = schema[task_name]
            else:
                all_schemas[task_name] = schema
        except Exception:
            continue

    return {"schemas": all_schemas}

@router.post("/infer")
async def infer(req: dict):
    """Run a single AI task."""
    task = req.get("task")
    data = req.get("input")
    if not task or data is None:
        raise HTTPException(status_code=400, detail="Missing 'task' or 'input'")
        
    try:
        result = await _global_router.route_task(task, data)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Inference failed for task %r", task)
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

@router.get("/adapters")
def list_adapters():
    """List all available loaded adapters."""
    if not _global_router:
        return {"adapters": []}
    return {"adapters": _global_router.get_available_adapters()}

@router.post("/pipeline/run")
async def run_pipeline(req: dict):
    """Run a sequenced AI pipeline."""
    steps = req.get("steps")
    data = req.get("data")
    if not isinstance(steps, list) or len(steps) == 0:
        raise HTTPException(status_code=400, detail="Missing or invalid 'steps' array")
        
    try:
        return await _global_engine.run_pipeline(steps, data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# FACE MANAGEMENT API ENDPOINTS
# =============================================================================

async def _get_insightface_adapter():
    """Helper to get InsightFace adapter with proper error handling."""
    if not _global_router:
        raise HTTPException(status_code=503, detail="Router not initialized")

    try:
        adapter = await _global_router.get_or_create_adapter("insightface_adapter")
        await asyncio.to_thread(_global_router.ensure_adapter_loaded, adapter)
        return adapter
    except ValueError as e:
        raise HTTPException(status_code=503, detail=str(e))

@router.post("/faces/register")
async def register_face(req: dict):
    """
    Register a face for future recognition.
    
    Request body:
    {
        "person_id": "emp_001",
        "name": "John Doe",
        "category": "employee",  // optional: "employee", "visitor", "watchlist", "vip"
        "frame": {"uri": "opennvr://frames/camera_0/face.jpg"},
        "metadata": {}  // optional additional data
    }
    """
    adapter = await _get_insightface_adapter()

    required_fields = ["person_id", "name", "frame"]
    for field in required_fields:
        if field not in req:
            raise HTTPException(status_code=400, detail=f"Missing required field: {field}")

    try:
        result = await asyncio.to_thread(adapter.register_face, req)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Registration error: {str(e)}")

@router.get("/faces/list")
async def list_faces(category: Optional[str] = Query(None, description="Filter by category (employee, visitor, watchlist, etc.)")):
    """
    List all registered faces, optionally filtered by category.
    """
    adapter = await _get_insightface_adapter()

    try:
        result = await asyncio.to_thread(adapter.list_faces, category=category)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing faces: {str(e)}")

@router.get("/faces/{person_id}")
async def get_face(person_id: str):
    """
    Get details of a specific registered face.
    """
    adapter = await _get_insightface_adapter()

    try:
        result = await asyncio.to_thread(adapter.get_face, person_id)
        if result.get("face") is None and "not found" in result.get("message", "").lower():
            raise HTTPException(status_code=404, detail=f"Person {person_id} not found")
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting face: {str(e)}")

@router.delete("/faces/{person_id}")
async def delete_face(person_id: str):
    """
    Delete a registered face by person_id.
    """
    adapter = await _get_insightface_adapter()

    try:
        result = await asyncio.to_thread(adapter.delete_face, person_id)
        if not result.get("success"):
            raise HTTPException(status_code=404, detail=result.get("message", "Delete failed"))
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting face: {str(e)}")
