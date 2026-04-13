import pytest
import asyncio
from app.pipelines.engine import PipelineEngine

class DummyRouter:
    async def route_task(self, task_name: str, input_data: dict):
        if task_name == "detect":
            return {"detections": [{"tag": "person"}], "confidence": 0.9}
        if task_name == "count":
            return {"person_count": 1}
        raise ValueError(f"Unknown task {task_name}")

@pytest.mark.asyncio
async def test_pipeline_engine_success():
    engine = PipelineEngine(DummyRouter())
    
    steps = ["detect", "count"]
    input_data = {"frame": {"uri": "mock://frame/1"}}
    
    result = await engine.run_pipeline(steps, input_data)
    
    assert result["status"] == "success"
    assert "detect" in result["results"]
    assert "count" in result["results"]
    assert result["results"]["count"]["person_count"] == 1

@pytest.mark.asyncio
async def test_pipeline_engine_failure():
    engine = PipelineEngine(DummyRouter())
    
    steps = ["detect", "unknown"]
    input_data = {"frame": {"uri": "mock://frame/1"}}
    
    result = await engine.run_pipeline(steps, input_data)
    
    assert result["status"] == "error"
    assert result["failed_at"] == "unknown"
    assert result["error"] == "Unknown task unknown"
    # Should still contain partial results for earlier steps
    assert "detect" in result["partial_results"]
