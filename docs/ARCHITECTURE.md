# Architecture Overview

This document explains how the AI Adapter server is designed, how data flows through the system, how the plugin discovery system works, and — crucially — **how the architecture prevents bloat as the community adds more adapters**.

Read this before the [Plugin Development Guide](PLUGIN_DEVELOPMENT.md).

---

## System Design: Clean Architecture with Lazy Loading + Anti-Bloat

The AI Adapter uses a layered architecture where each layer has a single responsibility:

```
┌────────────────────────────────────────────────────────────────────────┐
│                        AI Adapter Server                               │
│                                                                        │
│  ┌──────────────┐   ┌──────────────┐   ┌───────────────────────────┐  │
│  │   API Layer  │──▶│    Router    │──▶│     Adapter Layer          │  │
│  │  endpoints.py│   │ ModelRouter  │   │                            │  │
│  │              │   │              │   │  YOLOv8Adapter ──▶ ONNX   │  │
│  │  POST /infer │   │   Lazy       │   │  InsightFace   ──▶ ArcFace│  │
│  │  GET /health │   │   Loading    │   │  BLIPAdapter   ──▶ HF     │  │
│  │  GET /tasks  │   │   + Caching  │   │  YourAdapter   ──▶ ???    │  │
│  └──────────────┘   └──────────────┘   └───────────────────────────┘  │
│          │                 │                        ▲                   │
│          │          ┌──────┴──────┐                 │                   │
│          │          │  Pipeline   │     ┌───────────┴────────┐         │
│          │          │   Layer     │     │  Plugin Discovery   │         │
│          │          │             │     │  (PluginManager)    │         │
│          │          │ Task pipes  │     │  Scans app/adapters │         │
│          │          │ transform   │     │  Scans app/pipelines│         │
│          │          │ raw output  │     └────────────────────┘         │
│          │          └─────────────┘                                     │
│  ┌───────┴─────────────────────────────────────────────────────────┐   │
│  │  Config Layer — TASK_ADAPTER_MAP, ENABLED_TASKS, CONFIG         │   │
│  └─────────────────────────────────────────────────────────────────┘   │
└────────────────────────────────────────────────────────────────────────┘
```

---

## Anti-Bloat Architecture

This is the most important section for contributors. The four mechanisms that prevent the project from bloating as the community adds adapters:

### Mechanism 1: Optional Dependency Groups

Before this refactor, all ML libraries were in `[project.dependencies]` — every deployment installed everything:

```
❌ BEFORE — every deployment installs ~3 GB regardless of which adapters are used
[project.dependencies]
insightface, ultralytics, transformers, onnxruntime, scipy, torch, torchvision...
```

Now each adapter's libraries are in their own optional group:

```toml
✅ AFTER — install only what your deployment actually needs
[project.optional-dependencies]
yolo     = ["onnxruntime==1.23.2", "onnx", "onnxslim"]          # ~250 MB
yolo11   = ["ultralytics==8.3.240", "matplotlib", ...]           # +torch = ~2.5 GB
face     = ["insightface>=0.7.3", "scipy==1.16.3", ...]          # ~500 MB
blip     = ["transformers==5.0.0"]                               # +torch = ~3 GB
huggingface = ["huggingface_hub==1.3.4"]                         # ~60 MB
```

A person-detection-only deployment now installs `uv sync --extra yolo` — **250 MB instead of 3 GB**.

### Mechanism 2: Deferred Library Imports

Before this refactor, `cv2`, `numpy`, `insightface`, and other heavy libraries were imported at the **module top level** of each adapter. This meant `PluginManager` pulled them all into memory the moment it scanned the files during discovery — even for adapters that are never called.

```python
❌ BEFORE — cv2 loaded for ALL deployments at startup (even ones that don't do vision)
import cv2        # top-level in yolov8_adapter.py
import numpy as np

❌ BEFORE — insightface loaded at startup even if face tasks are never used
try:
    from insightface.app import FaceAnalysis  # top-level in insightface_adapter.py
    INSIGHTFACE_AVAILABLE = True
except ImportError:
    INSIGHTFACE_AVAILABLE = False
```

Now all heavy imports are deferred into the methods that actually need them:

```python
✅ AFTER — cv2 only loaded when inference actually runs (not at discovery time)
def _preprocess(self, img):
    import cv2  # deferred: only loaded when inference actually runs
    return cv2.dnn.blobFromImage(...)

✅ AFTER — insightface only loaded on first real inference request
def load_model(self):
    from insightface.app import FaceAnalysis  # optional dep: uv sync --extra face
    self._face_model = FaceAnalysis(...)
```

**Result:** `PluginManager` can now scan and register every adapter class without loading a single ML library.

### Mechanism 3: Graceful Import-Failure Skipping

Before this refactor, if an adapter's dependency wasn't installed, the `ImportError` was logged as a WARNING and the entire server could behave unpredictably.

```
❌ BEFORE — noisy warning, unclear message
WARNING - Skipping module 'app.adapters.vision.insightface_adapter': No module named 'insightface'
```

Now `PluginManager` distinguishes two cases:

```
✅ AFTER — ImportError (expected, missing optional dep) → INFO + install hint
INFO - Skipping 'app.adapters.vision.insightface_adapter': optional dependency
       'insightface' not installed. Install with: uv sync --extra face

✅ AFTER — Other errors (real bugs) → WARNING
WARNING - Skipping module 'app.adapters.vision.broken_adapter' due to
          unexpected error: SyntaxError at line 42
```

And at end of startup, a clean summary:
```
INFO - Discovered 2 adapters: [yolov8_adapter, huggingface_adapter]
INFO - 3 adapter module(s) skipped (optional dep not installed).
       Run 'uv sync --extra all' to enable everything.
```

### Mechanism 4: Config-Aware Model Downloads

Before this refactor, `download_models.py` downloaded everything in `MODEL_URLS` regardless of what was enabled:

```python
❌ BEFORE — downloads ALL models always
MODEL_URLS = {
    "yolov8n.onnx": "...",
    # add more → always downloaded
}
```

Now it reads `CONFIG["adapters"]` and only fetches weights for enabled adapters:

```bash
✅ AFTER — only downloads what's enabled in config.py
uv run python download_models.py           # config-aware (skips disabled adapters)
uv run python download_models.py --all    # override: download everything (for CI/Docker)
```

---

## The Five Layers

### Layer 1: API (`app/api/`)

| File | Purpose |
|---|---|
| `endpoints.py` | FastAPI route definitions (`/infer`, `/health`, `/tasks`, `/pipeline/run`, `/faces/*`) |
| `auth.py` | API key authentication via `X-API-Key` header (opt-in via `REQUIRE_AUTH` env var) |

**Rules:**
- No business logic here — only HTTP request/response handling
- All inference work is delegated to the `ModelRouter` and `PipelineEngine`
- Endpoints are async; blocking model work is offloaded via `asyncio.to_thread()`

### Layer 2: Router (`app/router/`)

| File | Purpose |
|---|---|
| `model_router.py` | Maps task names → adapters, lazy-creates instances, orchestrates inference |

**ModelRouter is the brain of the system.** When a request arrives for task `X`:

1. Looks up `config.TASK_ADAPTER_MAP["X"]` → adapter name
2. Lazy-creates the adapter instance (or uses cached one)
3. Ensures the model is loaded (`ensure_model_loaded()`)
4. Checks if a Task pipeline exists in `TASK_REGISTRY["X"]`
   - **Yes** → calls `task.process(data, adapter)` — task drives the inference
   - **No** → calls `adapter.predict(data)` directly — raw output returned

### Layer 3: Adapters (`app/adapters/`)

| File | Purpose |
|---|---|
| `base.py` | `BaseAdapter` abstract class — defines lazy loading, `infer()` → `infer_local()` flow |
| `vision/*.py` | Concrete vision adapters (YOLOv8, InsightFace, BLIP, HuggingFace, etc.) |
| `llm/*.py` | LLM/text adapters |

**Rules:**
- One class per model/framework
- Models load in `load_model()`, NOT in `__init__()`
- Heavy ML libraries (`cv2`, `numpy`, `torch`, `onnxruntime`, etc.) imported **inside** `load_model()` or the inference methods — **never at module top level**
- Adapters know nothing about task names or business logic

### Layer 4: Pipelines / Tasks (`app/pipelines/`)

| File | Purpose |
|---|---|
| `engine.py` | `PipelineEngine` — chains multiple tasks sequentially (`POST /pipeline/run`) |
| `<task_name>/task.py` | Task-specific business logic classes |

**Rules:**
- Tasks receive an adapter instance via dependency injection
- Tasks call `adapter.predict()` to get raw data, then apply domain logic
- Tasks return Pydantic models for validated, typed responses
- Tasks are optional — adapters work fine without them

### Layer 5: Schemas (`app/schemas/`)

| File | Purpose |
|---|---|
| `responses.py` | Pydantic `BaseModel` classes for every response type |

These models provide:
- Runtime validation (e.g., `confidence` must be 0.0–1.0, `bbox` must have 4 elements)
- Cross-field validation (e.g., `PersonCountResponse` checks `count == len(detections)`)
- Auto-serialization with null stripping
- Type documentation for API consumers

---

## Boot Sequence

```python
# app/main.py startup_event()

1. PluginManager.discover_plugins()
   │
   ├── Scans app/pipelines/**/*.py
   │   For each .py file:
   │   ├── Try importlib.import_module(module_name)
   │   │   ├── ImportError → skip gracefully, log install hint  (missing optional dep)
   │   │   ├── Other error → skip with WARNING                  (real bug)
   │   │   └── Success → inspect classes, register BaseTask subclasses
   │   └── Result: TASK_REGISTRY = {"person_detection": PersonDetectionTask, ...}
   │
   ├── Scans app/adapters/**/*.py (same process)
   │   └── Result: ADAPTER_REGISTRY = {"yolov8_adapter": YOLOv8Adapter, ...}
   │
   └── Logs summary:
       "Discovered 2 adapters: [yolov8_adapter, huggingface_adapter]"
       "3 adapter module(s) skipped (optional dep not installed)."

   NOTE: Only CLASS REFERENCES stored. No instances. No models loaded.
         No ML libraries imported. Zero memory consumed by models.

2. ModelRouter(config_module)
   └── Stores reference to config for routing lookups
   └── Creates empty caches: self.adapters = {}, self.tasks = {}

3. PipelineEngine(router)
   └── Stores reference to ModelRouter for chaining tasks

4. Server opens port 9100 ✓
```

---

## Request Flow: Single Task Inference

```
Client                 API                  Router                Adapter              Task
  │                     │                     │                     │                    │
  │  POST /infer        │                     │                     │                    │
  │  {task, input}      │                     │                     │                    │
  │────────────────────▶│                     │                     │                    │
  │                     │  route_task()       │                     │                    │
  │                     │────────────────────▶│                     │                    │
  │                     │                     │  config lookup      │                    │
  │                     │                     │  task→adapter name  │                    │
  │                     │                     │                     │                    │
  │                     │                     │  get_or_create      │                    │
  │                     │                     │  adapter ──────────▶│ __init__(config)   │
  │                     │                     │  (lazy create)      │                    │
  │                     │                     │                     │                    │
  │                     │                     │  ensure_model       │                    │
  │                     │                     │  loaded ───────────▶│ load_model()       │
  │                     │                     │                     │ (first time only)  │
  │                     │                     │                     │ ← imports cv2,     │
  │                     │                     │                     │   numpy, ort HERE  │
  │                     │                     │                     │                    │
  │                     │                     │  task.process       │                    │
  │                     │                     │  (if task exists)───┼────────────────────▶
  │                     │                     │                     │◀─ predict(data) ───│
  │                     │                     │◀─ PydanticModel ────┼────────────────────│
  │◀────────────────────│◀────────────────────│                     │                    │
  │   JSON response     │                     │                     │                    │
```

---

## Request Flow: Multi-Step Pipeline

```
POST /pipeline/run {"steps": ["face_detection", "face_recognition"], "data": {...}}
    │
    ▼
PipelineEngine.run_pipeline(["face_detection", "face_recognition"], data)
    │
    ├── Step 1: router.route_task("face_detection", data)
    │   └── Returns: {faces: [...], face_count: 3, ...}
    │
    ├── Step 2: router.route_task("face_recognition", step_1_result)
    │   └── Input IS the output of step 1
    │   └── Returns: {recognized: true, person_id: "emp_001", ...}
    │
    └── Final: {"status": "success", "results": {"face_detection": {...}, "face_recognition": {...}}}
```

---

## File-by-File Reference

### Core Application

| File | Purpose |
|---|---|
| `app/main.py` | FastAPI app creation, startup hook, plugin discovery |
| `app/config/config.py` | All configuration: task→adapter routing, adapter settings, constants |
| `app/router/model_router.py` | Maps tasks to adapters, lazy instantiation, inference orchestration |
| `app/pipelines/engine.py` | Sequential multi-task pipeline executor |
| `app/utils/loader.py` | `PluginManager` — auto-discovers adapter/task plugins, graceful skip on missing deps |

### Interfaces (Contracts)

| File | Purpose |
|---|---|
| `app/interfaces/adapter.py` | `BaseAdapter` ABC: `load_model()`, `predict()` |
| `app/interfaces/task.py` | `BaseTask` ABC: `process(image, adapter)` |

### Adapters (Model Wrappers)

| File | Purpose | Optional Extra |
|---|---|---|
| `app/adapters/base.py` | Concrete `BaseAdapter` with lazy loading, health checks | core |
| `app/adapters/vision/yolov8_adapter.py` | YOLOv8 ONNX inference | `--extra yolo` |
| `app/adapters/vision/yolov11_adapter.py` | YOLOv11 PyTorch with ByteTrack | `--extra yolo11 --extra cpu` |
| `app/adapters/vision/insightface_adapter.py` | InsightFace face detection/recognition | `--extra face` |
| `app/adapters/llm/blip_adapter.py` | BLIP image captioning | `--extra blip --extra cpu` |
| `app/adapters/llm/huggingface_adapter.py` | HuggingFace cloud inference proxy | `--extra huggingface` |

### Task Pipelines (Business Logic)

| File | Purpose |
|---|---|
| `app/pipelines/person_detection/task.py` | Picks highest-confidence person from YOLOv8 output |
| `app/pipelines/person_counting/task.py` | Counts all person detections, validates count consistency |
| `app/pipelines/face_detection/task.py` | Parses raw face detections into validated response |
| `app/pipelines/face_recognition/task.py` | Matches detected face against registered face database |
| `app/pipelines/scene_description/task.py` | Wraps BLIP caption output |

### Supporting Files

| File | Purpose |
|---|---|
| `app/api/endpoints.py` | All FastAPI endpoint definitions |
| `app/api/auth.py` | API key authentication middleware |
| `app/db/face_db.py` | In-memory face embedding database |
| `app/utils/image_utils.py` | Secure image loading from `opennvr://` URIs |
| `download_models.py` | Downloads model weight files for enabled adapters only |
| `start.py` | Programmatic uvicorn launcher |

---

## Key Design Decisions

### Why Lazy Loading?
A production NVR may have 10+ adapters configured but only use 2-3 regularly. Eager loading all models at startup would consume gigabytes of memory for unused models. Lazy loading means models only occupy memory when actively needed.

### Why Deferred Library Imports?
`PluginManager` needs to *import* adapter modules to discover their classes. If heavy libraries (`cv2`, `insightface`, `torch`) are at module top level, they are loaded during discovery — even for adapters that are disabled or never called. Deferring imports into `load_model()` keeps discovery zero-cost.

### Why Optional Dependency Groups?
A deployment running only YOLOv8 person detection needs ~250 MB. Without optional groups, it would have to install the full 3-4 GB stack (torch, transformers, insightface) even though none of those are used. Optional groups let each deployment install exactly what it needs.

### Why Separate Adapters and Tasks?
- **Adapters** are reusable. One `YOLOv8Adapter` serves `person_detection`, `person_counting`, and potentially `vehicle_detection`.
- **Tasks** are domain-specific. `PersonDetectionTask` applies the business rule "return only the highest-confidence person". This rule is independent of which model ran the inference.
- You can swap the underlying model (e.g., YOLOv8 → YOLOv11) by changing config, without touching any task code.

### Why Pydantic Response Models?
Raw dicts are error-prone. Pydantic catches these at response time, preventing corrupt data from reaching the NVR dashboard.

### Why Async Endpoints?
Model inference is CPU-bound and can take 50-500ms. Sync endpoints would block the FastAPI event loop, starving health checks and other requests. `asyncio.to_thread()` offloads blocking inference to a thread pool while keeping the event loop responsive.

---

Ready to build your own adapter? Head to the [Plugin Development Guide](PLUGIN_DEVELOPMENT.md).
