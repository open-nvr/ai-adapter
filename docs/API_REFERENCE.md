# 📖 API Reference: AI Adapters

Welcome to the **AIAdapters Inference API** documentation. This RESTful API allows the core Open-NVR server (via KAI-C) or any independent third-party client to orchestrate AI tasks effortlessly. 

The server runs on port **9100** by default and communicates exclusively via JSON.

---

## ⚡ Base URL & Authentication

- **Base URL:** `http://localhost:9100` (or your configured Docker hostname)
- **Content-Type:** `application/json`
- **Authentication:** Currently restricted to local network execution. To expose this API externally, we highly recommend utilizing Open-NVR's zero-trust proxy or placing it behind a reverse proxy (Nginx/Traefik) with API Keys.

---

## 🔍 System Discovery Endpoints

### `GET /health`
Liveness check to ensure the inference engine is running and ready to accept frames.

**Response `200 OK`**:
```json
{
  "status": "ok",
  "uptime_seconds": 3600
}
```

### `GET /capabilities`
Returns a lightweight array of all dynamically loaded AI task plugins. Use this to quickly verify if your custom plugin was loaded successfully.

**Response `200 OK`**:
```json
{
  "tasks": [
    "person_detection",
    "person_counting",
    "scene_description",
    "face_detection"
  ]
}
```

### `GET /tasks`
Provides deep metadata regarding installed tasks, including the underlying model framework (e.g., ONNX, PyTorch) and hardware execution targets (CPU vs GPU).

**Response `200 OK`**:
```json
{
  "tasks": {
    "person_detection": {
      "name": "person_detection",
      "description": "Detect the person with highest confidence in the image",
      "model_info": { "model": "yolov8n", "framework": "onnx", "device": "cpu" }
    }
  },
  "count": 1
}
```

### `GET /schema`
Outputs the exact JSON Schema required by KAI-C or your client to process the response. Add `?task=<task_name>` to filter.

---

## 🧠 Core Inference Engine

### `POST /infer`
The primary routing endpoint. Pass a defined `task` alongside an `opennvr://` isolated URI. The engine will automatically route the frame to the correct local Model Handler or Cloud Proxy.

#### Example 1: Edge Inference (Local Vision Model)
Run a lightweight task (like person counting) directly on edge hardware.

**Request**:
```bash
curl -X POST http://localhost:9100/infer \
  -H "Content-Type: application/json" \
  -d '{
    "task": "person_counting",
    "input": {
      "frame": { "uri": "opennvr://frames/camera_0/latest.jpg" }
    }
  }'
```

**Response `200 OK`**:
```json
{
  "task": "person_counting",
  "count": 3,
  "detections": [
    { "bbox": [100, 150, 200, 300], "confidence": 0.85, "track_id": 1 }
  ],
  "annotated_image_uri": "opennvr://frames/camera_0/person_counting_tracked.jpg",
  "latency_ms": 180
}
```

#### Example 2: Cloud Inference (Hugging Face Integration)
Route heavy lifting (like Vision-Language Models) securely to the cloud using the `input_data` schema.

**Request**:
```bash
curl -X POST http://localhost:9100/infer \
  -H "Content-Type: application/json" \
  -d '{
    "task": "image-classification",
    "input_data": {
      "model_name": "Salesforce/blip-image-captioning-base",
      "inputs": { "image": "opennvr://frames/camera_2/latest.jpg" },
      "api_token": "hf_your_secure_token"
    }
  }'
```

**Response `200 OK`**:
```json
{
  "task": "image-classification",
  "model_name": "Salesforce/blip-image-captioning-base",
  "result": [
    { "label": "a person walking through a server room", "score": 0.92 }
  ],
  "latency_ms": 1200
}
```

---

## 👤 Face Registry Management

The `face_db` module allows temporary in-memory registration of faces against which the `face_recognition` task can compare live streams.

### `POST /faces/register`
Register a face encoding to a specific ID and watchlist category.

**Request**:
```json
{
  "frame": { "uri": "opennvr://frames/camera_0/person.jpg" },
  "person_id": "emp_001",
  "name": "Jane Doe",
  "category": "vip"
}
```

### `GET /faces/list`
List all registered identies. Optionally append `?category=watchlist` to filter results.

### `DELETE /faces/{person_id}`
Remove a registered identity instantly from the scoring database.

---

## ⚠️ Error Handling

The API uses standardized HTTP status codes to indicate the success or failure of an API request.

| Status Code | Description | Example Payload |
|-------------|-------------|-----------------|
| `400 Bad Request` | Malformed parameters | `{ "detail": "Missing 'task' field" }` |
| `404 Not Found` | Unknown task requested | `{ "detail": "Unsupported task: 'unknown'" }` |
| `500 Server Error` | Model crash or OOM | `{ "detail": "CUDA out of memory" }` |

---

## 🛠️ For Open-Source Contributors: How to Update This Document

Welcome to the community! 🎉 We are incredibly excited that you are building a new AI Adapter for Open-NVR.

Since the AI inference engine auto-discovers your plugins natively, you don't need to write any new API routing code in Python. However, the UI developers need to know what shape your data comes back in! 

**Before submitting your Pull Request, please add a quick example of your new AI task's JSON response to the `POST /infer` section above.**

### Example: Adding a "License Plate Reader"
If you built a new adapter task called `license_plate_reader`, just scroll up to the `POST /infer` section and paste your expected JSON output like this:

```json
**Response (license_plate_reader):**
{
  "task": "license_plate_reader",
  "plate_number": "KA-01-AB-1234",
  "confidence": 0.98,
  "state": "Karnataka",
  "latency_ms": 210
}
```

That's it! By adding this single block, you ensure that anyone building dashboards for Open-NVR knows exactly how to parse the `plate_number` from your awesome new model! 🚀

> **A Quick Rule:** The AIAdapters engine relies on the universal `POST /infer` endpoint for modularity. **Please do not hardcode custom endpoints** (like `/infer/my_custom_model`) directly into `main.py`. Your model will magically work through the main router automatically!

