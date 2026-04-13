# Plugin Development Guide — Adding Custom AI Capabilities

This guide shows you how to add new AI capabilities to OpenNVR. There are **two types** of plugins you can create:

| Plugin Type | What It Does | Where It Lives | Auto-Discovered From |
|---|---|---|---|
| **Adapter** | Wraps a model (ONNX, PyTorch, API). Handles raw inference. | `app/adapters/vision/` or `app/adapters/llm/` | `app.adapters` |
| **Task** | Business logic layer. Takes adapter output → validates → shapes the final response. | `app/pipelines/<task_name>/task.py` | `app.pipelines` |

> **Key Concept:** Adapters do the heavy lifting (loading models, running inference). Tasks apply domain logic on top (e.g., "find the single best person detection from raw YOLO output"). You can have an adapter without a task (raw output goes straight to the client), or a task that transforms the adapter's raw output into a validated Pydantic response.

---

## Dependency Hygiene — Keep the Project Lean

This project uses **optional dependency groups** so that a deployment only installs what it actually needs. If you add a new adapter, follow these rules:

### Rule 1: Never add a heavy library to `[project.dependencies]`

`[project.dependencies]` in `pyproject.toml` is the **core** install — always small. Heavy ML libraries (`torch`, `onnxruntime`, `transformers`, etc.) belong in optional groups only.

```toml
# ❌ WRONG — forces every deployment to install PyTorch even if unused
[project.dependencies]
"torch==2.9.1"

# ✅ CORRECT — only installed when user asks for it
[project.optional-dependencies]
my_adapter = ["torch==2.9.1"]
```

### Rule 2: Add your adapter's deps as a new optional group

```toml
# pyproject.toml — add under [project.optional-dependencies]
my_adapter = [
    "my-ml-library>=1.0.0",
]

# Also add it to the "all" group so `uv sync --extra all` still works:
all = [
    "opennvr-ai-adapter[yolo,yolo11,face,blip,huggingface,my_adapter]",
]
```

### Rule 3: Import heavy libraries inside `load_model()`, not at the top

```python
# ❌ WRONG — loads my_ml_library for EVERY deployment at startup
import my_ml_library

class MyAdapter(BaseAdapter):
    def load_model(self):
        self.model = my_ml_library.load("weights.bin")

# ✅ CORRECT — loads my_ml_library only when first request arrives
class MyAdapter(BaseAdapter):
    def load_model(self):
        import my_ml_library  # optional dep: uv sync --extra my_adapter
        self.model = my_ml_library.load("weights.bin")
```

### Result: PluginManager handles missing deps gracefully

If a user hasn't installed your extra, PluginManager will:
- Catch the `ImportError` when it tries to import your adapter module
- Log a helpful message: `Skipping 'app.adapters.vision.my_adapter': optional dependency 'my_ml_library' not installed. Install with: uv sync --extra my_adapter`
- Continue — other adapters load normally
- Your adapter simply won't appear in `/capabilities`

No crash. No ugly stack trace. Just a clean skip.

---

## How Auto-Discovery Works

You **never** need to manually import or register your plugin. Here's exactly how the system finds it:

### Step-by-Step Discovery Flow

```
Server starts
    │
    ▼
app/main.py → startup_event()
    │
    ▼
PluginManager.discover_plugins()
    │
    ├── Scan "app.adapters" package (recursive)
    │   └── For every .py file (except those starting with _):
    │       └── Find all classes where:
    │           ✓ Defined in this module (not imported from elsewhere)
    │           ✓ Subclass of BaseAdapter
    │           ✓ Not abstract
    │       └── Register: ADAPTER_REGISTRY[class.name] = class
    │
    └── Scan "app.pipelines" package (recursive)
        └── Same rules, but looks for subclasses of BaseTask
        └── Register: TASK_REGISTRY[class.name] = class
    │
    ▼
ModelRouter + PipelineEngine created
    │
    ▼
Server ready — no models loaded yet (lazy loading!)
```

### The Actual Code (from `app/utils/loader.py`)

```python
class PluginManager:
    TASK_REGISTRY: dict[str, type[BaseTask]] = {}
    ADAPTER_REGISTRY: dict[str, type[BaseAdapter]] = {}

    @classmethod
    def discover_plugins(cls, force_reload: bool = False) -> None:
        cls._scan_package("app.pipelines", BaseTask, cls.TASK_REGISTRY)
        cls._scan_package("app.adapters", BaseAdapter, cls.ADAPTER_REGISTRY)

    @classmethod
    def _register_module_classes(cls, module_name, base_class, registry):
        module = importlib.import_module(module_name)
        for _, discovered_class in inspect.getmembers(module, inspect.isclass):
            # Only register classes DEFINED in this module (not imported)
            if discovered_class.__module__ != module.__name__:
                continue
            # Must be a subclass of the base (but not the base itself)
            if discovered_class is base_class or not issubclass(discovered_class, base_class):
                continue
            # Skip abstract classes
            if inspect.isabstract(discovered_class):
                continue

            plugin_name = cls._resolve_plugin_name(discovered_class)
            registry[plugin_name] = discovered_class  # Stores the CLASS, not an instance
```

**What this means for you:**

1. Your class just needs to exist in a `.py` file under `app/adapters/` or `app/pipelines/`
2. It must extend `BaseAdapter` or `BaseTask`
3. It must NOT be abstract
4. It must define `name` (a class attribute used as the registry key)
5. That's it. No imports in `main.py`, no registration calls, nothing.

### How the `name` Attribute Becomes the Registry Key

```python
@staticmethod
def _resolve_plugin_name(discovered_class: type) -> str:
    name = getattr(discovered_class, "name", None)
    if isinstance(name, str) and name.strip():
        return name           # Uses your class.name attribute
    return discovered_class.__name__  # Fallback to class name
```

So if your adapter has `name = "fire_adapter"`, it will be registered as `ADAPTER_REGISTRY["fire_adapter"]`.

---

## Tutorial 1: Adding a New Adapter (Adapter-Only, No Custom Task)

This is the simplest approach. Your adapter returns a dict, and it goes directly to the client. Good for generic models where you don't need special post-processing.

### Step 1: Create the Adapter File

```
app/adapters/vision/fire_adapter.py
```

### Step 2: Write the Adapter

```python
# app/adapters/vision/fire_adapter.py

import logging
import os
import time
from typing import Any, Dict

from app.adapters.base import BaseAdapter
from app.config import MODEL_WEIGHTS_DIR
from app.utils.image_utils import load_image_from_uri

logger = logging.getLogger(__name__)


class FireAdapter(BaseAdapter):
    """
    Fire/flame detection adapter using a custom YOLO model.

    HOW AUTO-DISCOVERY FINDS THIS:
    - This file is under app/adapters/ (scanned package)
    - FireAdapter extends BaseAdapter (matches base class check)
    - FireAdapter is NOT abstract (passes abstract check)
    - name = "fire_adapter" → ADAPTER_REGISTRY["fire_adapter"] = FireAdapter
    """

    # These two class attributes are REQUIRED by BaseAdapter.__init__()
    # If either is empty, it raises ValueError at instantiation time
    name = "fire_adapter"
    type = "vision"

    def __init__(self, config: Dict[str, Any] = None):
        """
        Called LAZILY — only when the first request for this adapter arrives.
        Do NOT load models here. Just store configuration.
        """
        self.config = config or {}
        self.model = None  # Must be None initially for lazy loading to work
        self._model_path = self.config.get(
            "weights_path",
            os.path.join(MODEL_WEIGHTS_DIR, "fire_yolov8.onnx"),
        )
        if not os.path.isabs(self._model_path):
            self._model_path = os.path.join(MODEL_WEIGHTS_DIR, self._model_path)

    def load_model(self) -> None:
        """
        Called ONCE, on the first inference request (lazy loading).
        This is where you load heavy model weights into memory.

        WHY NOT IN __init__?
        Because __init__ runs when ModelRouter first creates this adapter.
        If we loaded the model there, ALL adapters would load ALL models at
        startup — eating memory for tasks that may never be called.
        """
        import onnxruntime as ort

        if not os.path.exists(self._model_path):
            raise FileNotFoundError(f"Fire model not found at {self._model_path}")

        self.session = ort.InferenceSession(
            self._model_path,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        self.model = self.session  # Set self.model so ensure_model_loaded() knows it's loaded
        logger.info("Fire detection model loaded from %s", self._model_path)

    def infer_local(self, input_data: Any) -> Dict[str, Any]:
        """
        Called on EVERY inference request after the model is loaded.

        Args:
            input_data: Dict from the API request, always has at minimum:
                {
                    "task": "fire_detection",
                    "frame": {"uri": "opennvr://frames/camera_0/latest.jpg"}
                }

        Returns:
            Dict with your detection results. This goes directly to the client
            if no Task pipeline exists for this task name.
        """
        start_time = time.time()

        # Load the image from OpenNVR's internal frame store
        uri = input_data["frame"]["uri"]
        img = load_image_from_uri(uri)

        # Run your model inference here
        # ... (your model-specific preprocessing + inference + postprocessing) ...

        # Example result
        fire_detected = True
        confidence = 0.95
        bbox = [120, 80, 200, 150]

        return {
            "task": "fire_detection",
            "label": "fire" if fire_detected else "no_fire",
            "confidence": round(confidence, 4),
            "bbox": bbox,
            "executed_at": int(time.time() * 1000),
            "latency_ms": int((time.time() - start_time) * 1000),
        }

    @property
    def schema(self) -> Dict[str, Any]:
        """
        Describes what this adapter returns. Used by GET /schema and GET /tasks.
        The OpenNVR dashboard reads this to know how to display your results.
        """
        return {
            "fire_detection": {
                "description": "Detects active fire/flames in camera frames",
                "response_fields": {
                    "label": {"type": "string", "description": "'fire' or 'no_fire'"},
                    "confidence": {"type": "float", "description": "0.0 to 1.0"},
                    "bbox": {"type": "array[int]", "description": "[left, top, width, height]"},
                },
            }
        }

    def get_model_info(self) -> Dict[str, Any]:
        """Optional: provides metadata for /health endpoint."""
        return {
            "name": self.name,
            "type": self.type,
            "model": "fire_yolov8",
            "framework": "onnx",
            "tasks": ["fire_detection"],
            "model_loaded": self.model is not None,
        }
```

### Step 3: Register the Routing in Config

Open `app/config/config.py` and add two things:

```python
# 1. Add to TASK_ADAPTER_MAP — maps task name → adapter name
TASK_ADAPTER_MAP = {
    # ... existing entries ...
    "fire_detection": "fire_adapter",      # ← ADD THIS
}

# 2. Add to CONFIG["adapters"] — adapter-specific settings
CONFIG = {
    "adapters": {
        # ... existing entries ...
        "fire_adapter": {                   # ← ADD THIS
            "enabled": True,
            "weights_path": "fire_yolov8.onnx"
        },
    },
    # ...
}
```

### Step 4: Restart and Verify

```bash
uv run uvicorn app.main:app --reload --port 9100
```

Check the startup logs — you should see:
```
Server ready. Discovered tasks=5 adapters=6
```

Verify your adapter is live:
```bash
# Check if your task appears in capabilities
curl http://localhost:9100/capabilities

# Test inference
curl -X POST http://localhost:9100/infer \
  -H "Content-Type: application/json" \
  -d '{
    "task": "fire_detection",
    "input": {
      "frame": {"uri": "opennvr://frames/camera_0/latest.jpg"}
    }
  }'
```

### What Happens at Runtime (Full Trace)

```
Client: POST /infer {"task": "fire_detection", "input": {"frame": {"uri": "..."}}}
    │
    ▼
endpoints.py: infer() extracts task="fire_detection", data={...}
    │
    ▼
ModelRouter.route_task("fire_detection", data)
    │
    ├── config.get_adapter_for_task("fire_detection") → "fire_adapter"
    │
    ├── get_or_create_adapter("fire_adapter")
    │   ├── First time? → PluginManager.ADAPTER_REGISTRY["fire_adapter"] → FireAdapter class
    │   ├── FireAdapter(config={"enabled": True, "weights_path": "..."}) → instance created
    │   └── Instance cached in self.adapters["fire_adapter"]
    │
    ├── ensure_adapter_loaded(adapter)
    │   ├── adapter.model is None? → adapter.load_model() → ONNX session created
    │   └── adapter.model is not None? → skip (already loaded)
    │
    ├── _get_or_create_task("fire_detection")
    │   └── PluginManager.TASK_REGISTRY has no "fire_detection" → returns None
    │       (because we didn't create a Task pipeline for this)
    │
    └── No task found → _predict_with_adapter(adapter, ..., data)
        └── adapter.predict(data) → adapter.infer(data) → adapter.infer_local(data)
            └── Returns {"task": "fire_detection", "label": "fire", ...}
                │
                ▼
            JSON response sent to client
```

---

## Tutorial 2: Adding an Adapter + Task Pipeline (Full Example)

When you need to transform adapter output — e.g., pick the highest-confidence detection, validate the response shape, or return a Pydantic model — you create a **Task**.

For this example, let's build a **License Plate Reader** with validated Pydantic output.

### Step 1: Create the Pydantic Response Schema

```python
# app/schemas/responses.py (add to the existing file)

class LicensePlateResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    task: Literal["license_plate_detection"] = "license_plate_detection"
    plate_number: str = Field(min_length=1, max_length=20)
    confidence: float = Field(ge=0.0, le=1.0)
    bbox: list[int]
    region: str | None = None
    executed_at: int = Field(ge=0)
    latency_ms: int = Field(ge=0)

    @field_validator("bbox")
    @classmethod
    def validate_bbox(cls, value: list[int]) -> list[int]:
        if len(value) != 4:
            raise ValueError("bbox must contain [left, top, width, height]")
        if any(v < 0 for v in value):
            raise ValueError("bbox values must be >= 0")
        return value
```

### Step 2: Create the Adapter (Raw Model Wrapper)

```python
# app/adapters/vision/plate_adapter.py

import logging
import os
import time
from typing import Any, Dict

from app.adapters.base import BaseAdapter
from app.config import MODEL_WEIGHTS_DIR
from app.utils.image_utils import load_image_from_uri

logger = logging.getLogger(__name__)


class PlateAdapter(BaseAdapter):
    """Raw ALPR model adapter. Returns unprocessed detections."""

    name = "plate_adapter"
    type = "vision"

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.model = None
        self._model_path = self.config.get(
            "weights_path",
            os.path.join(MODEL_WEIGHTS_DIR, "plate_detector.onnx"),
        )
        if not os.path.isabs(self._model_path):
            self._model_path = os.path.join(MODEL_WEIGHTS_DIR, self._model_path)

    def load_model(self) -> None:
        import onnxruntime as ort
        self.session = ort.InferenceSession(self._model_path)
        self.model = self.session
        logger.info("Plate detection model loaded from %s", self._model_path)

    def infer_local(self, input_data: Any) -> Dict[str, Any]:
        start_time = time.time()
        uri = input_data["frame"]["uri"]
        img = load_image_from_uri(uri)

        # ... run your plate detection model ...

        return {
            "task": input_data.get("task", "plate_raw"),
            "plates": [
                {
                    "text": "KA-01-AB-1234",
                    "confidence": 0.97,
                    "bbox": [100, 200, 180, 60],
                    "region": "Karnataka",
                },
                {
                    "text": "MH-02-CD-5678",
                    "confidence": 0.82,
                    "bbox": [400, 210, 170, 55],
                    "region": "Maharashtra",
                },
            ],
            "executed_at": int(time.time() * 1000),
            "latency_ms": int((time.time() - start_time) * 1000),
        }
```

### Step 3: Create the Task Pipeline

The task takes the adapter's raw `plates` list and applies business logic (pick the highest confidence plate, validate with Pydantic).

```
app/pipelines/license_plate_detection/
├── __init__.py
└── task.py
```

```python
# app/pipelines/license_plate_detection/__init__.py
from .task import LicensePlateTask
```

```python
# app/pipelines/license_plate_detection/task.py

import time
from typing import Any

from pydantic import BaseModel

from app.interfaces.adapter import BaseAdapter
from app.interfaces.task import BaseTask
from app.schemas.responses import LicensePlateResponse


class LicensePlateTask(BaseTask):
    """
    Business logic for license plate detection.
    
    Takes raw plate detections from the adapter and returns
    the highest-confidence plate as a validated Pydantic model.

    HOW AUTO-DISCOVERY FINDS THIS:
    - This file is under app/pipelines/ (scanned package)
    - LicensePlateTask extends BaseTask (matches base class check)
    - It is NOT abstract (concrete class)
    - name = "license_plate_detection" → TASK_REGISTRY["license_plate_detection"]
    """

    name = "license_plate_detection"

    def process(self, image: Any, adapter: BaseAdapter) -> LicensePlateResponse:
        """
        Called by ModelRouter when a request for "license_plate_detection" arrives
        AND a task with this name exists in the TASK_REGISTRY.

        Args:
            image: The input data dict from the API request
            adapter: The adapter instance (already loaded with model)

        Returns:
            LicensePlateResponse — Pydantic validates this at construction time.
            If validation fails (e.g., confidence > 1.0), the server returns 500.
        """
        # Step 1: Prepare payload for the adapter
        payload = image
        if isinstance(image, dict):
            payload = dict(image)
            payload.setdefault("task", "license_plate_detection")

        # Step 2: Get raw results from the adapter
        raw_result = adapter.predict(payload)

        # Step 3: Normalize to dict (adapter might return Pydantic model or dict)
        if isinstance(raw_result, BaseModel):
            raw_payload = raw_result.model_dump()
        elif isinstance(raw_result, dict):
            raw_payload = raw_result
        else:
            raise TypeError(f"Expected dict/BaseModel, got {type(raw_result).__name__}")

        # Step 4: Apply business logic — find highest confidence plate
        plates = raw_payload.get("plates", [])
        best_plate = None
        best_confidence = -1.0

        for plate in plates:
            conf = float(plate.get("confidence", 0))
            if conf > best_confidence:
                best_confidence = conf
                best_plate = plate

        executed_at = int(raw_payload.get("executed_at", int(time.time() * 1000)))
        latency_ms = int(raw_payload.get("latency_ms", 0))

        if best_plate is None:
            return LicensePlateResponse(
                plate_number="NONE",
                confidence=0.0,
                bbox=[0, 0, 0, 0],
                executed_at=executed_at,
                latency_ms=latency_ms,
            )

        # Step 5: Return validated Pydantic model
        # If any field is invalid, Pydantic raises ValidationError → 500
        return LicensePlateResponse(
            plate_number=best_plate["text"],
            confidence=round(best_confidence, 2),
            bbox=best_plate["bbox"],
            region=best_plate.get("region"),
            executed_at=executed_at,
            latency_ms=latency_ms,
        )
```

### Step 4: Register the Routing in Config

```python
# app/config/config.py

TASK_ADAPTER_MAP = {
    # ... existing entries ...
    "license_plate_detection": "plate_adapter",
}

CONFIG = {
    "adapters": {
        # ... existing entries ...
        "plate_adapter": {
            "enabled": True,
            "weights_path": "plate_detector.onnx"
        },
    },
    # ...
}
```

### Step 5: Restart and Test

```bash
curl -X POST http://localhost:9100/infer \
  -H "Content-Type: application/json" \
  -d '{
    "task": "license_plate_detection",
    "input": {
      "frame": {"uri": "opennvr://frames/camera_0/latest.jpg"}
    }
  }'
```

Expected response (Pydantic-validated, nulls stripped):
```json
{
  "task": "license_plate_detection",
  "plate_number": "KA-01-AB-1234",
  "confidence": 0.97,
  "bbox": [100, 200, 180, 60],
  "region": "Karnataka",
  "executed_at": 1735546430000,
  "latency_ms": 150
}
```

### What Happens at Runtime (With a Task)

```
Client POST /infer {"task": "license_plate_detection", ...}
    │
    ▼
ModelRouter.route_task("license_plate_detection", data)
    │
    ├── config → adapter_name = "plate_adapter"
    ├── get_or_create_adapter("plate_adapter") → PlateAdapter instance
    ├── ensure_adapter_loaded → load_model() if first time
    │
    ├── _get_or_create_task("license_plate_detection")
    │   └── TASK_REGISTRY["license_plate_detection"] → LicensePlateTask class
    │   └── LicensePlateTask() → instance created & cached
    │
    └── task IS found → task.process(data, adapter)
        │
        ├── adapter.predict(data) → raw plates list from PlateAdapter
        ├── Business logic: find best plate
        └── LicensePlateResponse(...) → Pydantic validates → JSON to client
```

**Key difference from Tutorial 1:** When a Task exists in `TASK_REGISTRY` with the same name as the requested task, `ModelRouter.route_task()` calls `task.process(data, adapter)` instead of `adapter.predict(data)` directly. The task gets the adapter injected so it can call `adapter.predict()` itself and then transform the result.

---

## The Complete Class Hierarchy

```
app/interfaces/adapter.py:BaseAdapter (ABC)        ← Interface: load_model(), predict()
    │
    └── app/adapters/base.py:BaseAdapter (ABC)      ← Implementation: lazy loading, health check
            │
            ├── app/adapters/vision/yolov8_adapter.py:YOLOv8Adapter
            ├── app/adapters/vision/insightface_adapter.py:InsightFaceAdapter
            ├── app/adapters/vision/fire_adapter.py:FireAdapter         ← YOUR ADAPTER
            └── app/adapters/llm/blip_adapter.py:BLIPAdapter

app/interfaces/task.py:BaseTask (ABC)               ← Interface: process(image, adapter)
    │
    ├── app/pipelines/person_detection/task.py:PersonDetectionTask
    ├── app/pipelines/face_detection/task.py:FaceDetectionTask
    └── app/pipelines/license_plate_detection/task.py:LicensePlateTask  ← YOUR TASK
```

---

## Reference: BaseAdapter Contract

Every adapter inherits from `app/adapters/base.py:BaseAdapter` and gets these methods:

| Attribute / Method | Required? | Description |
|---|---|---|
| `name: str` | **Yes** | Unique adapter identifier. Used as config key and registry key. |
| `type: str` | **Yes** | `"vision"` or `"llm"`. Used for hardware validation. |
| `__init__(self, config)` | **Yes** | Store config, set `self.model = None`. Do NOT load model here. |
| `load_model(self)` | **Yes (abstract)** | Load model weights into `self.model`. Called lazily. |
| `infer_local(self, input_data)` | **Yes (abstract)** | Run the actual inference. Return dict or Pydantic model. |
| `predict(self, input_data)` | Inherited | Entry point called by ModelRouter. Calls `ensure_model_loaded()` → `infer_local()`. |
| `infer(self, input_data)` | Inherited | Same as `predict()`. Both exist for compatibility. |
| `ensure_model_loaded(self)` | Inherited | Calls `load_model()` if `self.model is None`. |
| `schema` (property) | Optional | Returns dict describing response format for `/schema` endpoint. |
| `get_model_info(self)` | Optional | Returns metadata dict for `/health` endpoint. |
| `health_check(self)` | Optional | Returns health status dict. |

## Reference: BaseTask Contract

Every task inherits from `app/interfaces/task.py:BaseTask`:

| Attribute / Method | Required? | Description |
|---|---|---|
| `name: str` | **Yes** | Must match the task name in `TASK_ADAPTER_MAP` keys. |
| `process(self, image, adapter)` | **Yes (abstract)** | Receives input data + adapter. Call `adapter.predict()`, apply logic, return response. |

---

## Checklist Before Submitting a Pull Request

- [ ] **Adapter file** exists in `app/adapters/vision/` or `app/adapters/llm/`
- [ ] Adapter has `name` and `type` class attributes set
- [ ] Adapter implements `load_model()` and `infer_local()`
- [ ] Model is NOT loaded in `__init__()` (lazy loading pattern)
- [ ] **Config updated**: task added to `TASK_ADAPTER_MAP` and adapter added to `CONFIG["adapters"]`
- [ ] *(Optional)* Task pipeline created in `app/pipelines/<task_name>/task.py`
- [ ] *(Optional)* Pydantic response model added to `app/schemas/responses.py`
- [ ] *(Optional)* Model download URL added to `download_models.py`
- [ ] Tested with `curl POST /infer` locally
- [ ] Updated `docs/API_REFERENCE.md` with your response JSON example

---

## FAQ

**Q: Do I need to import my adapter anywhere?**
No. `PluginManager` auto-discovers all concrete subclasses of `BaseAdapter` in `app/adapters/`. Just create the file.

**Q: Do I need to touch `main.py`?**
No. The discovery and routing happens automatically.

**Q: Can one adapter serve multiple tasks?**
Yes. Add multiple entries in `TASK_ADAPTER_MAP` pointing to the same adapter name:
```python
TASK_ADAPTER_MAP = {
    "fire_detection": "fire_adapter",
    "smoke_detection": "fire_adapter",  # same adapter, different task name
}
```
The adapter receives `input_data["task"]` so it can branch logic internally.

**Q: What if I don't create a Task pipeline?**
The adapter's `predict()` output goes directly to the client as JSON. This is fine for simple use cases.

**Q: What if I only want a Task pipeline (using an existing adapter)?**
Just create the task in `app/pipelines/<name>/task.py` and add the routing in config pointing to the existing adapter. The task's `process()` method will receive that adapter instance.

**Q: How do I disable my adapter without deleting code?**
Set `"enabled": False` in `CONFIG["adapters"]`:
```python
"fire_adapter": {"enabled": False, "weights_path": "fire.onnx"}
```

**Q: Why isn't my adapter being discovered?**
Check these common issues:
1. File name starts with `_` (e.g., `_my_adapter.py`) — these are skipped
2. Class is abstract (has unimplemented abstract methods)
3. Class isn't a subclass of `BaseAdapter`
4. Class was imported from another module (discovery only registers classes defined in the current module)
5. Import error in your file — check server logs for `Skipping module ... during plugin discovery`
