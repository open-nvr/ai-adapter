# API Reference

The AI Adapter REST API runs on port **9100** and communicates via JSON.

---

## Authentication

Authentication is opt-in, controlled by environment variables:

| Variable | Default | Description |
|---|---|---|
| `REQUIRE_AUTH` | `false` | Set to `true` to require API key on all requests |
| `API_KEY` | *(none)* | The secret key. Required if `REQUIRE_AUTH=true` |

When auth is enabled, pass the key in the `X-API-Key` header:
```bash
curl -H "X-API-Key: your-secret-key" http://localhost:9100/health
```

If `REQUIRE_AUTH=true` but `API_KEY` is not set, the server refuses to start.

---

## System Endpoints

### `GET /health`

Server health, loaded adapters, and system hardware info.

```bash
curl http://localhost:9100/health
```

**Response:**
```json
{
  "status": "healthy",
  "active_adapters": 3,
  "system_hardware": {
    "cpu": "Intel i7-12700",
    "ram_gb": 32,
    "gpu": "NVIDIA RTX 3060"
  },
  "adapter_details": {
    "yolov8_adapter": {
      "status": "healthy",
      "model_loaded": true,
      "model_info": {"model": "yolov8n", "framework": "onnx"}
    }
  }
}
```

### `GET /capabilities`

List of all task names registered in the config routing.

```bash
curl http://localhost:9100/capabilities
```

**Response:**
```json
{
  "tasks": [
    "person_detection",
    "person_counting",
    "face_detection",
    "face_recognition",
    "scene_description"
  ]
}
```

### `GET /tasks`

Detailed metadata for all tasks including adapter mapping and descriptions.

```bash
curl http://localhost:9100/tasks
```

**Response:**
```json
{
  "tasks": [
    {
      "task": "person_detection",
      "adapter": "yolov8_adapter",
      "enabled": true,
      "model_loaded": true,
      "description": "Detects the person with highest confidence"
    },
    {
      "task": "face_detection",
      "adapter": "insightface_adapter",
      "enabled": true,
      "model_loaded": false,
      "description": "Detects all faces with landmarks"
    }
  ],
  "count": 2
}
```

### `GET /schema`

Response schema for a specific task or all tasks. Used by the OpenNVR dashboard to understand response formats.

```bash
# All schemas
curl http://localhost:9100/schema

# Specific task
curl "http://localhost:9100/schema?task=person_detection"
```

**Response (single task):**
```json
{
  "task": "person_detection",
  "adapter": "yolov8_adapter",
  "schema": {
    "description": "Returns raw YOLOv8 detections",
    "task": "yolov8_raw_inference"
  }
}
```

### `GET /adapters`

List of currently instantiated (loaded) adapters.

```bash
curl http://localhost:9100/adapters
```

**Response:**
```json
{
  "adapters": ["yolov8_adapter", "insightface_adapter"]
}
```

---

## Inference Endpoints

### `POST /infer`

The primary inference endpoint. Pass a task name and input data. The ModelRouter handles adapter lookup, lazy loading, and task pipeline execution.

**Request format:**
```json
{
  "task": "<task_name>",
  "input": {
    "frame": {"uri": "opennvr://frames/<camera_id>/latest.jpg"}
  }
}
```

#### Person Detection

```bash
curl -X POST http://localhost:9100/infer \
  -H "Content-Type: application/json" \
  -d '{
    "task": "person_detection",
    "input": {
      "frame": {"uri": "opennvr://frames/camera_0/latest.jpg"}
    }
  }'
```

**Response:**
```json
{
  "label": "person",
  "confidence": 0.92,
  "bbox": [120, 80, 200, 350],
  "annotated_image_uri": "opennvr://frames/camera_0/person_detection_annotated.jpg",
  "executed_at": 1735546430000,
  "latency_ms": 145
}
```

#### Person Counting

```bash
curl -X POST http://localhost:9100/infer \
  -H "Content-Type: application/json" \
  -d '{
    "task": "person_counting",
    "input": {
      "frame": {"uri": "opennvr://frames/camera_0/latest.jpg"}
    }
  }'
```

**Response:**
```json
{
  "task": "person_counting",
  "count": 3,
  "detections": [
    {"bbox": {"left": 100, "top": 150, "width": 200, "height": 300}, "confidence": 0.85, "class_id": 0, "track_id": 1},
    {"bbox": {"left": 400, "top": 200, "width": 180, "height": 280}, "confidence": 0.78, "class_id": 0, "track_id": 2},
    {"bbox": {"left": 700, "top": 100, "width": 150, "height": 250}, "confidence": 0.92, "class_id": 0, "track_id": 3}
  ],
  "raw_prediction_count": 8400,
  "executed_at": 1735546430000,
  "latency_ms": 180
}
```

#### Face Detection

```bash
curl -X POST http://localhost:9100/infer \
  -H "Content-Type: application/json" \
  -d '{
    "task": "face_detection",
    "input": {
      "frame": {"uri": "opennvr://frames/camera_0/latest.jpg"}
    }
  }'
```

**Response:**
```json
{
  "faces": [
    {
      "bbox": [220, 80, 340, 220],
      "confidence": 0.98,
      "landmarks": [[250, 130], [310, 130], [280, 160], [260, 190], [300, 190]],
      "age": 32,
      "gender": "M"
    }
  ],
  "face_count": 1,
  "executed_at": 1735546430000,
  "latency_ms": 210
}
```

#### Face Recognition

```bash
curl -X POST http://localhost:9100/infer \
  -H "Content-Type: application/json" \
  -d '{
    "task": "face_recognition",
    "input": {
      "frame": {"uri": "opennvr://frames/camera_0/latest.jpg"}
    }
  }'
```

**Response (match found):**
```json
{
  "recognized": true,
  "person_id": "emp_001",
  "name": "Jane Doe",
  "category": "employee",
  "similarity": 0.87,
  "face_bbox": [220, 80, 340, 220],
  "latency_ms": 350
}
```

**Response (no match):**
```json
{
  "recognized": false,
  "message": "No matching face found in database",
  "latency_ms": 280
}
```

#### Scene Description

```bash
curl -X POST http://localhost:9100/infer \
  -H "Content-Type: application/json" \
  -d '{
    "task": "scene_description",
    "input": {
      "frame": {"uri": "opennvr://frames/camera_0/latest.jpg"}
    }
  }'
```

**Response:**
```json
{
  "task": "scene_description",
  "caption": "a person sitting at a desk with a computer monitor",
  "model_id": "Salesforce/blip-image-captioning-base",
  "executed_at": 1735546430000,
  "latency_ms": 2100
}
```

---

### `POST /pipeline/run`

Execute a multi-step AI pipeline. The output of step N becomes the input to step N+1.

**Request:**
```bash
curl -X POST http://localhost:9100/pipeline/run \
  -H "Content-Type: application/json" \
  -d '{
    "steps": ["face_detection", "face_recognition"],
    "data": {
      "frame": {"uri": "opennvr://frames/camera_0/latest.jpg"}
    }
  }'
```

**Response (success):**
```json
{
  "status": "success",
  "results": {
    "face_detection": {
      "faces": [{"bbox": [220, 80, 340, 220], "confidence": 0.98}],
      "face_count": 1,
      "latency_ms": 210
    },
    "face_recognition": {
      "recognized": true,
      "person_id": "emp_001",
      "name": "Jane Doe",
      "latency_ms": 350
    }
  }
}
```

**Response (partial failure):**
```json
{
  "status": "error",
  "failed_at": "face_recognition",
  "error": "InsightFace model not available",
  "partial_results": {
    "face_detection": {"faces": [...], "face_count": 1}
  }
}
```

---

## Face Registry Endpoints

Manage the in-memory face database for recognition and watchlist features.

### `POST /faces/register`

```bash
curl -X POST http://localhost:9100/faces/register \
  -H "Content-Type: application/json" \
  -d '{
    "frame": {"uri": "opennvr://frames/camera_0/person.jpg"},
    "person_id": "emp_001",
    "name": "Jane Doe",
    "category": "employee"
  }'
```

**Response:**
```json
{
  "success": true,
  "person_id": "emp_001",
  "message": "Face registered successfully"
}
```

### `GET /faces/list`

```bash
# All faces
curl http://localhost:9100/faces/list

# Filter by category
curl "http://localhost:9100/faces/list?category=watchlist"
```

**Response:**
```json
{
  "faces": [
    {"person_id": "emp_001", "name": "Jane Doe", "category": "employee"},
    {"person_id": "watch_01", "name": "Unknown", "category": "watchlist"}
  ],
  "total_count": 2
}
```

### `GET /faces/{person_id}`

```bash
curl http://localhost:9100/faces/emp_001
```

### `DELETE /faces/{person_id}`

```bash
curl -X DELETE http://localhost:9100/faces/emp_001
```

**Response:**
```json
{
  "success": true,
  "message": "Face deleted"
}
```

---

## Error Responses

All errors follow a consistent format:

```json
{
  "detail": "Human-readable error message"
}
```

| Status Code | Meaning | Example |
|---|---|---|
| `400` | Invalid request | `Missing 'task' or 'input'` |
| `401` | Auth failure | `Invalid or missing API Key` |
| `404` | Not found | `No routing rule found for task 'unknown'` |
| `500` | Server error | `Inference error: CUDA out of memory` |
| `503` | Service unavailable | `Router not initialized` / `Adapter disabled` |

---

## For Contributors

When you add a new adapter/task, please update this document with your task's expected response format under the appropriate section above. This helps frontend developers and API consumers integrate with your new capability.

See [PLUGIN_DEVELOPMENT.md](PLUGIN_DEVELOPMENT.md) for the full tutorial on adding new AI capabilities.
