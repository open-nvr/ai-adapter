import json, pytest
from _yolo_pose_service_fixtures import *  # noqa
from fastapi.testclient import TestClient

def test_unexpected_exception_in_shaping(yolo_pose_app, square_jpeg, monkeypatch):
    import adapters.yolo_pose.service as svc
    client, service = yolo_pose_app
    monkeypatch.setattr(type(service), "_shape_persons",
                        lambda self, *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    r = client.post("/infer", files={"image": ("f.jpg", square_jpeg, "image/jpeg")})
    print("HTTP status:", r.status_code, r.text[:300])

def test_unexpected_exception_in_stream(yolo_pose_app, square_jpeg, monkeypatch):
    client, service = yolo_pose_app
    monkeypatch.setattr(type(service), "_shape_persons",
                        lambda self, *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    with client.websocket_connect("/infer/stream") as ws:
        ws.send_text(json.dumps({"type":"handshake","client_id":"c","protocol_version":"1"}))
        print("ack:", ws.receive_text()[:120])
        ws.send_text(json.dumps({"type":"frame","seq":1,"ts_ms":1}))
        ws.send_bytes(square_jpeg)
        try:
            print("result:", ws.receive_text()[:200])
        except Exception as e:
            print("STREAM BLEW UP:", type(e).__name__, str(e)[:200])
