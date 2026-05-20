# YOLOv8 Object-Detection Adapter (Contract v1)

Reference implementation of the [AI Adapter Contract v1](../../../open-nvr/docs/AI_ADAPTER_CONTRACT.md) wrapping the YOLOv8 ONNX object detector. This is the **second adapter migrated** (A2.2 milestone) and the **first one to exercise the §6 WebSocket streaming protocol**.

## What it does

POST a JPEG/PNG frame (or open a WebSocket stream), get COCO-80 object detections back. Returns the §5.1 `DetectionResult` shape: normalized `[0, 1]` bounding boxes, COCO labels (`person`, `car`, `dog`, …), confidence scores, and optional frame dimensions.

## Endpoints

| Endpoint | Status | Notes |
|---|---|---|
| `GET /health` | required | auth-exempt |
| `GET /capabilities` | required | sha256 fingerprint, `tasks_advertised=["object_detection"]`, `gpu=true`, `fair_queuing=per_camera` |
| `GET /hardware/evaluation` | required | reports GPU detection + onnxruntime providers |
| `GET /metrics` | required | Prometheus exposition incl. `adapter_stream_connections_active` |
| `POST /infer` | required | multipart (`frame` file) or JSON (`frame_b64`) |
| `POST /infer/stream` (WS) | required | full §6 protocol — handshake → frame_meta + bytes → result loop |

**Shared-memory fast path** (§6.2) is deferred to a follow-up. Adapter advertises `supports_shared_memory: false`; clients that offer `frame_transport: "shared_memory"` see the ack downgrade to `"websocket"`.

## Run locally

```bash
# Install deps
uv sync --extra yolo

# Download yolov8n weights (one-time, ~6 MB)
mkdir -p model_weights
cd model_weights
curl -LO https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.onnx
cd ..

# Start the service
OPENNVR_ADAPTER_TOKEN=dev-token \
  uvicorn adapters.yolov8.main:app --host 0.0.0.0 --port 9002
```

## Run with Docker

```bash
docker build -f adapters/yolov8/Dockerfile -t opennvr/yolov8-adapter:1.0.0 .

# CPU
docker run --rm -p 9002:9002 \
  -e OPENNVR_ADAPTER_TOKEN=$(openssl rand -hex 16) \
  -v $(pwd)/model_weights:/weights:ro \
  opennvr/yolov8-adapter:1.0.0

# GPU passthrough (requires NVIDIA Container Toolkit on the host)
docker run --rm --gpus all -p 9002:9002 \
  -e OPENNVR_ADAPTER_TOKEN=$(openssl rand -hex 16) \
  -v $(pwd)/model_weights:/weights:ro \
  opennvr/yolov8-adapter:1.0.0
```

## Verify conformance

```bash
python -m conformance http://localhost:9002 --token $OPENNVR_ADAPTER_TOKEN
```

This runs the full conformance suite **including** the §6 WebSocket roundtrip (handshake → real JPEG frame → result message → close). Green = KAI-C will accept it.

## Try it

### HTTP — multipart (§3.5 canonical)

```bash
curl -X POST http://localhost:9002/infer \
  -H "Authorization: Bearer $OPENNVR_ADAPTER_TOKEN" \
  -H "X-Correlation-Id: $(uuidgen)" \
  -F "frame=@/path/to/image.jpg" \
  -F 'params={"confidence_threshold": 0.4, "classes": ["person", "car"]};type=application/json'
```

### HTTP — JSON (base64 fallback)

```bash
B64=$(base64 -i /path/to/image.jpg)
curl -X POST http://localhost:9002/infer \
  -H "Authorization: Bearer $OPENNVR_ADAPTER_TOKEN" \
  -H "Content-Type: application/json" \
  -d "{\"frame_b64\": \"$B64\", \"confidence_threshold\": 0.4}"
```

### WebSocket (Python client)

```python
import asyncio, json, websockets

async def main():
    headers = {"Authorization": "Bearer dev-token"}
    async with websockets.connect("ws://localhost:9002/infer/stream", extra_headers=headers) as ws:
        # Handshake
        await ws.send(json.dumps({
            "type": "handshake", "client_id": "demo", "camera_id": "cam-1",
            "frame_transport": "websocket",
        }))
        ack = json.loads(await ws.recv())
        print("session", ack["session_id"])

        # Send a frame
        with open("/path/to/image.jpg", "rb") as f:
            frame = f.read()
        await ws.send(json.dumps({"type": "frame", "seq": 1, "ts_ms": 0, "content_type": "image/jpeg"}))
        await ws.send(frame)
        result = json.loads(await ws.recv())
        print("detections:", result["result"]["detections"])

        await ws.send(json.dumps({"type": "close", "reason": "done"}))

asyncio.run(main())
```

## Layout

```
adapters/yolov8/
├── main.py          FastAPI app — HTTP routes + WS streaming loop
├── service.py       YoloV8Service — wraps legacy YOLOv8Adapter, normalizes bboxes
├── auth.py          AuthAndCorrelationMiddleware + websocket_auth_failure
├── metrics.py       Prometheus exposition with stream-connection gauge
├── coco_classes.py  COCO-80 label table (class_id → "person"/"car"/etc.)
├── Dockerfile       Self-contained image; CPU + GPU compatible
└── README.md        you are here
```

## Why a new service vs. extending the legacy monolith

Same reason as Piper (A2.1): "one adapter wraps one model" per §1 of the contract. Each adapter is its own container with its own declared permissions. The legacy `app/main.py` keeps running until every adapter has its own contract-compliant service.

## Tests

```bash
pytest tests/test_yolov8_service.py tests/test_conformance_against_yolov8.py
```

34 tests covering every endpoint, full WS protocol (handshake / frame roundtrip / pause-resume / stats / close codes), auth (HTTP + WS), correlation_id, and the conformance kit pointed back at this service.
