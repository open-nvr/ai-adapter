import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.api.endpoints import set_global_router

class FakeConfig:
    CONFIG = {"routing": {"person_counting": "yolov8"}}

class FakeRouter:
    def __init__(self):
        self.adapters = {"yolov8": {}}
        self.config_module = FakeConfig()

    async def route_task(self, task_name: str, input_data: dict):
        return {"status": "success", "task": task_name, "mocked": True}

    def get_available_adapters(self) -> list:
        return [{"id": "yolov8", "loaded": True}]

    def get_available_tasks(self) -> list:
        return [{"id": "person_counting", "adapter": "yolov8"}]

class FakeEngine:
    async def run_pipeline(self, pipeline: list, input_data: dict):
        return {"status": "success", "steps": len(pipeline)}

from unittest.mock import patch

async def dummy_startup():
    set_global_router(FakeRouter(), FakeEngine())

@pytest.fixture
def client():
    with patch("app.main.startup_event", dummy_startup):
        with TestClient(app) as c:
            set_global_router(FakeRouter(), FakeEngine())
            yield c

def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert "active_adapters" in response.json()

def test_get_capabilities(client):
    response = client.get("/capabilities")
    assert response.status_code == 200
    assert "tasks" in response.json()

def test_post_infer(client):
    payload = {
        "task": "person_counting",
        "input": {"frame": {"uri": "opennvr://frame/1"}}  # endpoint reads req.get("input"), not "data"
    }
    response = client.post("/infer", json=payload)
    print("response infer:", response.text)
    assert response.status_code == 200
    assert response.json()["task"] == "person_counting"
    assert response.json()["mocked"] is True


def test_post_infer_missing_task(client):
    payload = {
        "data": {"frame": {"uri": "opennvr://frame/1"}}
    }
    response = client.post("/infer", json=payload)
    assert response.status_code == 400

def test_post_pipeline(client):
    payload = {
        "steps": ["detect", "recognize"],
        "data": {"frame": {"uri": "opennvr://frame/1"}}
    }
    response = client.post("/pipeline/run", json=payload)
    print("response pipeline:", response.text)
    assert response.status_code == 200
    assert response.json()["status"] == "success"
