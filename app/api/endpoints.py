from fastapi import APIRouter, HTTPException
from typing import Dict, Any, List

router = APIRouter()

# We will inject the global router instance here upon app startup
_global_router = None
_global_engine = None

def set_global_router(model_router, pipeline_engine):
    global _global_router, _global_engine
    _global_router = model_router
    _global_engine = pipeline_engine

@router.post("/infer")
async def infer(req: dict):
    """Run a single AI task."""
    task = req.get("task")
    data = req.get("data")
    if not task or data is None:
        raise HTTPException(status_code=400, detail="Missing 'task' or 'data'")
        
    try:
        result = _global_router.route_task(task, data)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
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
        return _global_engine.run_pipeline(steps, data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
