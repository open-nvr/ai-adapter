# OpenNVR AI Adapter — Modular Inference Engine

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.127.0-009688.svg?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-FFD21E.svg)](https://huggingface.co/models)

A plug-and-play AI inference server for the **OpenNVR** ecosystem. Drop in any AI model — ONNX, PyTorch, or HuggingFace cloud — and it becomes available as a REST API, automatically discovered and lazily loaded.

> **v2.0 — Anti-Bloat Architecture:** Dependencies are now split into optional groups. A minimal deployment installs ~9 packages and only grows when you explicitly ask for more adapters.

---

## Key Features

- **Auto-Discovery** — Place an adapter file in `app/adapters/`, restart. The system finds it automatically. No imports, no registration code.
- **Lazy Model Loading** — Models load into memory only when their first request arrives. Idle models use zero memory.
- **Deferred Library Imports** — Heavy ML libraries (`cv2`, `numpy`, `onnxruntime`, `insightface`, `ultralytics`, `transformers`) are imported inside `load_model()`, not at module level. Discovery never touches them.
- **Graceful Adapter Skipping** — If an adapter's optional dependencies aren't installed, `PluginManager` skips it cleanly with a helpful install hint. Other adapters continue to load normally.
- **Optional Dependency Groups** — Install only what you need: `uv sync --extra yolo` gives you person detection without pulling in torch/transformers/insightface.
- **Adapter + Task Separation** — Adapters handle raw model inference. Tasks apply business logic (filtering, validation) on top. Swap models by changing config.
- **Pydantic-Validated Responses** — Response schemas catch invalid data at the boundary (confidence out of range, missing fields, count mismatches).
- **Async Pipeline Engine** — Chain multiple tasks sequentially via `POST /pipeline/run`. Output of step N feeds into step N+1.
- **API Key Authentication** — Opt-in `X-API-Key` header auth via environment variables.
- **Multi-stage Docker** — Minimal runtime image with no build tools. Built-in `HEALTHCHECK`.

---

## Quick Start

```bash
# 1. Create virtual environment
uv venv

# 2. Activate
source .venv/bin/activate       # Linux/Mac
# .venv\Scripts\activate        # Windows

# 3. Install core + chosen adapters (see Installation Profiles below)
uv sync --extra all             # everything — recommended for first-time setup
# OR use a lean profile, e.g. detection + faces only (~750 MB):
# uv sync --extra yolo --extra face

# 4. Download model weights
uv run python download_models.py

# 5. Start server
uv run uvicorn app.main:app --reload --port 9100
```

The server auto-discovers all adapters and tasks at startup:
```
INFO - Server ready. Discovered tasks=5 adapters=5
```

---

## Project Structure

```
ai-adapter/
├── app/                              # Application source
│   ├── main.py                       # FastAPI entry point + startup
│   │
│   ├── adapters/                     # MODEL WRAPPERS (auto-discovered)
│   │   ├── base.py                   # BaseAdapter — lazy loading contract
│   │   ├── vision/                   # Vision model adapters
│   │   │   ├── yolov8_adapter.py     #   YOLOv8 ONNX (person detection)
│   │   │   ├── yolov11_adapter.py    #   YOLOv11 PyTorch (person counting)
│   │   │   ├── insightface_adapter.py#   InsightFace (face detection/recognition)
│   │   │   ├── blip_adapter.py       #   BLIP (scene description)
│   │   │   └── huggingface_adapter.py#   HuggingFace cloud proxy
│   │   └── llm/                      # LLM adapters
│   │       ├── blip_adapter.py       #   BLIP captioning
│   │       └── huggingface_adapter.py#   HuggingFace inference API
│   │
│   ├── pipelines/                    # TASK BUSINESS LOGIC (auto-discovered)
│   │   ├── engine.py                 # PipelineEngine — chains tasks
│   │   ├── person_detection/task.py  #   Picks best person from YOLO output
│   │   ├── person_counting/task.py   #   Counts persons, validates consistency
│   │   ├── face_detection/task.py    #   Parses face detections
│   │   ├── face_recognition/task.py  #   Matches against face database
│   │   └── scene_description/task.py #   Wraps BLIP captions
│   │
│   ├── api/                          # REST API
│   │   ├── endpoints.py              # All route definitions
│   │   └── auth.py                   # API key middleware
│   │
│   ├── config/config.py              # Routing, adapter settings, constants
│   ├── router/model_router.py        # Task→Adapter routing + lazy instantiation
│   ├── interfaces/                   # Abstract contracts (BaseAdapter, BaseTask)
│   ├── schemas/responses.py          # Pydantic response models
│   ├── db/face_db.py                 # In-memory face database
│   └── utils/                        # Image loading, plugin discovery
│
├── docs/                             # Documentation
│   ├── ARCHITECTURE.md               # System design deep-dive
│   ├── PLUGIN_DEVELOPMENT.md         # How to add custom adapters/tasks
│   ├── API_REFERENCE.md              # REST API documentation
│   ├── MODELS.md                     # Supported models reference
│   └── RUNNER_GUIDE.md               # OpenNVR runner integration
│
├── model_weights/                    # Downloaded .onnx/.pt model files
├── tests/                            # pytest test suite
├── Dockerfile                        # Multi-stage Docker build
├── pyproject.toml                    # Dependencies (uv/pip)
└── start.py                          # Programmatic uvicorn launcher
```

---

## Installation Profiles

The project uses **optional dependency groups** to keep deployments lean. You only install the libraries that your adapters actually need.

| Profile | Command | What you get | Approx size |
|---|---|---|---|
| **Core only** | `uv sync` | FastAPI, uvicorn, pydantic, numpy, opencv | ~50 MB |
| **YOLOv8 detection** | `uv sync --extra yolo` | + onnxruntime | ~250 MB |
| **YOLOv11 counting** | `uv sync --extra yolo11 --extra cpu` | + ultralytics + torch (CPU) | ~2.5 GB |
| **Face recognition** | `uv sync --extra face` | + insightface + onnxruntime + scipy | ~500 MB |
| **Scene captioning** | `uv sync --extra blip --extra cpu` | + transformers + torch (CPU) | ~3 GB |
| **HuggingFace cloud** | `uv sync --extra huggingface` | + huggingface_hub | ~60 MB |
| **Full (all adapters)** | `uv sync --extra all --extra cpu` | everything | ~4 GB |
| **Full GPU** | `uv sync --extra all --extra gpu` | everything + CUDA torch | ~6 GB |

> **Tip:** Most NVR deployments only need `--extra yolo --extra face` (~750 MB) for person detection + face recognition. The full 4 GB install is only needed if you want scene captioning or cloud inference too.

---

## How It Works

### The Two Plugin Types

```
┌─────────────────────────────────┐     ┌──────────────────────────────────┐
│         ADAPTER                  │     │           TASK                    │
│  Wraps a model, runs inference   │     │  Business logic on top of adapter │
│                                  │     │                                   │
│  Lives in: app/adapters/         │     │  Lives in: app/pipelines/         │
│  Extends: BaseAdapter            │     │  Extends: BaseTask                │
│  Must implement:                 │     │  Must implement:                  │
│    load_model()                  │     │    process(image, adapter)        │
│    infer_local(input_data)       │     │  Returns: Pydantic model          │
│  Returns: raw dict               │     │                                   │
└─────────────────────────────────┘     └──────────────────────────────────┘
```

### Request Flow

```
POST /infer {"task": "person_detection", "input": {"frame": {"uri": "..."}}}
    │
    ▼
ModelRouter.route_task("person_detection")
    │
    ├── Config lookup → adapter = "yolov8_adapter"
    ├── Lazy-create YOLOv8Adapter (first time only)
    ├── Lazy-load ONNX model (first time only)
    │
    ├── Task "person_detection" exists in TASK_REGISTRY?
    │   YES → PersonDetectionTask.process(data, adapter)
    │          └── Calls adapter.predict() → gets raw detections
    │          └── Picks highest confidence person
    │          └── Returns PersonDetectionResponse (Pydantic-validated)
    │
    └── Response: {"label": "person", "confidence": 0.92, "bbox": [...], ...}
```

---

## Contributing: Add Your Own AI Model

Adding a new AI capability takes **3 files** and **zero changes to existing code**:

### 1. Create an Adapter

```python
# app/adapters/vision/fire_adapter.py
from app.adapters.base import BaseAdapter

class FireAdapter(BaseAdapter):
    name = "fire_adapter"
    type = "vision"

    def __init__(self, config=None):
        self.config = config or {}
        self.model = None

    def load_model(self):
        import onnxruntime as ort
        self.model = ort.InferenceSession("model_weights/fire.onnx")

    def infer_local(self, input_data):
        # Load image, run model, return raw results
        return {"label": "fire", "confidence": 0.95, "bbox": [100, 80, 200, 150]}
```

### 2. Register Routing

```python
# app/config/config.py
TASK_ADAPTER_MAP = {
    # ... existing ...
    "fire_detection": "fire_adapter",
}
CONFIG["adapters"]["fire_adapter"] = {"enabled": True, "weights_path": "fire.onnx"}
```

### 3. (Optional) Add a Task Pipeline for Validated Output

```python
# app/pipelines/fire_detection/task.py
from app.interfaces.task import BaseTask

class FireDetectionTask(BaseTask):
    name = "fire_detection"

    def process(self, image, adapter):
        raw = adapter.predict(image)
        # Apply business logic, return Pydantic model
        return FireDetectionResponse(**raw)
```

**That's it.** No imports in `main.py`. No registration calls. The `PluginManager` auto-discovers your classes at startup.

> **Important:** Always declare your adapter's ML libraries as a new `[project.optional-dependencies]` group in `pyproject.toml` — never add them to `[project.dependencies]`. Import them inside `load_model()`, not at the module top level. This keeps the project lean for everyone. See [Dependency Hygiene](docs/PLUGIN_DEVELOPMENT.md#dependency-hygiene--keep-the-project-lean) for full rules.

→ Full tutorial with working code: **[docs/PLUGIN_DEVELOPMENT.md](docs/PLUGIN_DEVELOPMENT.md)**
→ Architecture deep-dive: **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)**
→ API reference: **[docs/API_REFERENCE.md](docs/API_REFERENCE.md)**

---

## API Overview

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Server health + loaded adapter status + hardware info |
| `/capabilities` | GET | List all available task names |
| `/tasks` | GET | Task metadata with adapter info and descriptions |
| `/schema` | GET | Response schema for one or all tasks |
| `/infer` | POST | Run a single task inference |
| `/pipeline/run` | POST | Run a chain of tasks sequentially |
| `/adapters` | GET | List currently loaded adapters |
| `/faces/register` | POST | Register a face for recognition |
| `/faces/list` | GET | List registered faces |
| `/faces/{person_id}` | GET/DELETE | Get or delete a registered face |

---

## Docker

```bash
# CPU build (default)
docker build -t ai-adapter .

# GPU build
docker build --build-arg USE_GPU=true -t ai-adapter:gpu .

# Run
docker run -p 9100:9100 -v ./model_weights:/app/model_weights ai-adapter
```

Built-in health check:
```dockerfile
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:9100/health || exit 1
```

---

## License

Licensed under the **GNU Affero General Public License v3.0 (AGPL v3)**.
All forks and derivative works must share source code under the same terms.
See the `LICENSE` file for details.

> For commercial licensing or enterprise support: **[contact@cryptovoip.in](mailto:contact@cryptovoip.in)**
