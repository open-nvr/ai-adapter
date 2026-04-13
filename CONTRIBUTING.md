# Contributing to OpenNVR AI Adapter

Thank you for your interest in contributing! This guide gets you from zero to a working AI adapter in minutes, and explains **how to do it without bloating the project for everyone else**.

---

## Quick Overview

The AI Adapter uses a **plugin-based architecture** with two types of plugins:

1. **Adapters** — Wrap AI models (ONNX, PyTorch, HuggingFace). Handle raw inference. Live in `app/adapters/`.
2. **Tasks** — Apply business logic on adapter output (filtering, validation). Live in `app/pipelines/`. Optional.

Both are **auto-discovered** at startup. You never need to edit `main.py` or manually register anything.

---

## Development Setup

```bash
# Clone and enter the repo
git clone <repo-url>
cd ai-adapter

# Create venv
uv venv
source .venv/bin/activate   # or .venv\Scripts\activate on Windows

# Install everything + dev tools
uv sync --extra all --extra dev

# Download model weights
uv run python download_models.py

# Run the server
uv run uvicorn app.main:app --reload --port 9100

# Run tests
uv run pytest tests/ -v
```

---

## The Anti-Bloat Rules — Read This First

The project uses **optional dependency groups** so a deployment only installs what it needs. Before adding any code, understand these three rules:

### Rule 1: Your adapter's libraries MUST go in a new optional group

Never add heavy ML libraries to `[project.dependencies]`. That's shared core — always small.

```toml
# pyproject.toml

# ❌ WRONG — forces ~500 MB on every deployment
[project.dependencies]
"my-ml-library>=1.0.0"

# ✅ CORRECT — only installed when user requests it
[project.optional-dependencies]
my_adapter = ["my-ml-library>=1.0.0"]

# Also add to "all" for backward compat:
all = ["opennvr-ai-adapter[yolo,yolo11,face,blip,huggingface,my_adapter]"]
```

### Rule 2: Import your heavy libraries inside `load_model()`, not at the top of the file

`PluginManager` imports every adapter file during discovery. If your library is at the top level, it loads for EVERY deployment even if your adapter is never used.

```python
# ❌ WRONG — my_lib loads for all deployments at startup
import my_lib

class MyAdapter(BaseAdapter):
    def load_model(self):
        self.model = my_lib.load("weights.bin")

# ✅ CORRECT — my_lib loads only when first inference request arrives
class MyAdapter(BaseAdapter):
    def load_model(self):
        import my_lib  # optional dep: uv sync --extra my_adapter
        self.model = my_lib.load("weights.bin")
```

### Rule 3: What happens if users don't install your extra

If someone runs `uv sync --extra yolo` (without your extra), `PluginManager` will:
- Catch the `ImportError` when importing your file
- Log: `Skipping 'app.adapters.vision.my_adapter': optional dependency 'my_lib' not installed. Install with: uv sync --extra my_adapter`
- Continue — all other adapters load fine
- Your adapter simply won't appear in `/capabilities`

**No crash. No stack trace. Just a clean, helpful skip.**

---

## Adding a New AI Capability

### Option A: Adapter Only (Simplest)

Raw model output goes directly to the client.

**Step 1 — Create `app/adapters/vision/my_adapter.py`:**

```python
from app.adapters.base import BaseAdapter

class MyAdapter(BaseAdapter):
    name = "my_adapter"       # Must be unique. Used in config routing.
    type = "vision"           # "vision" or "llm"

    def __init__(self, config=None):
        self.config = config or {}
        self.model = None     # Must start as None for lazy loading

    def load_model(self):
        # ← Import heavy libraries HERE, not at the top of the file
        import my_ml_library  # optional dep: uv sync --extra my_adapter
        self.model = my_ml_library.load(self.config.get("weights_path"))

    def infer_local(self, input_data):
        from app.utils.image_utils import load_image_from_uri
        img = load_image_from_uri(input_data["frame"]["uri"])
        result = self.model.predict(img)
        return {"label": "...", "confidence": 0.95, "bbox": [0, 0, 100, 100]}
```

**Step 2 — Add your optional dep group to `pyproject.toml`:**

```toml
[project.optional-dependencies]
my_adapter = ["my-ml-library>=1.0.0"]
all = ["opennvr-ai-adapter[yolo,yolo11,face,blip,huggingface,my_adapter]"]
```

**Step 3 — Register routing in `app/config/config.py`:**

```python
TASK_ADAPTER_MAP["my_task"] = "my_adapter"
CONFIG["adapters"]["my_adapter"] = {"enabled": True, "weights_path": "my_model.bin"}
```

**Step 4 — Add download entry to `download_models.py` (if local weights needed):**

```python
MODEL_REGISTRY["my_adapter"] = [
    {
        "filename": "my_model.bin",
        "url": "https://github.com/me/my-models/releases/download/v1/my_model.bin",
        "size_hint": "~50 MB",
    }
]
```

**Step 5 — Restart. Done.**

---

### Option B: Adapter + Task Pipeline (Validated Output)

Adds a task that transforms raw adapter output into a Pydantic-validated response.

**Step 1** — Create the adapter *(same as Option A)*

**Step 2 — Create Pydantic response model in `app/schemas/responses.py`:**

```python
class MyResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    label: str
    confidence: float = Field(ge=0.0, le=1.0)
    bbox: list[int]
    executed_at: int = Field(ge=0)
    latency_ms: int = Field(ge=0)
```

**Step 3 — Create `app/pipelines/my_task/task.py`:**

```python
from app.interfaces.task import BaseTask
from app.schemas.responses import MyResponse

class MyTask(BaseTask):
    name = "my_task"          # Must match the key in TASK_ADAPTER_MAP

    def process(self, image, adapter):
        raw = adapter.predict(image)
        return MyResponse(
            label=raw["label"],
            confidence=raw["confidence"],
            bbox=raw["bbox"],
            executed_at=raw.get("executed_at", 0),
            latency_ms=raw.get("latency_ms", 0),
        )
```

**Step 4** — Follow Steps 2-5 from Option A.

→ Full tutorial with detailed code: **[docs/PLUGIN_DEVELOPMENT.md](docs/PLUGIN_DEVELOPMENT.md)**

---

## How Auto-Discovery Works

At startup, `PluginManager.discover_plugins()`:

1. Walks every `.py` file (non-private) in `app/adapters/` and `app/pipelines/`
2. For each file, attempts `importlib.import_module(module_name)`
   - `ImportError` → logs a helpful skip message (missing optional dep — expected)
   - Other error → WARNING (real bug in your code)
   - Success → inspects classes, registers any `BaseAdapter` / `BaseTask` subclasses
3. Registers each class by its `name` attribute (or class name as fallback)
4. Logs a summary: `Discovered 3 adapters: [insightface_adapter, yolov8_adapter, yolov11_adapter]`

No imports in `main.py`. No registration calls. Just create the file and restart.

---

## Installation Profiles

| Profile | Command | Approx Size |
|---|---|---|
| Core only | `uv sync` | ~50 MB |
| YOLOv8 detection | `uv sync --extra yolo` | ~250 MB |
| YOLOv11 counting | `uv sync --extra yolo11 --extra cpu` | ~2.5 GB |
| Face recognition | `uv sync --extra face` | ~500 MB |
| Detection + recognition | `uv sync --extra yolo --extra face` | ~750 MB |
| Scene captioning | `uv sync --extra blip --extra cpu` | ~3 GB |
| HuggingFace cloud | `uv sync --extra huggingface` | ~60 MB |
| Everything | `uv sync --extra all --extra cpu` | ~4 GB |
| Dev (with tests) | `uv sync --extra all --extra cpu --extra dev` | ~4 GB |

---

## Testing

```bash
# Run all tests
uv run pytest tests/ -v

# Run with coverage
uv run pytest tests/ --cov=app --cov-report=term-missing

# Test a specific file
uv run pytest tests/test_endpoints.py -v
```

When adding a new adapter/task, please add tests in `tests/`.

---

## Checklist Before PR

- [ ] Adapter has `name` and `type` class attributes
- [ ] **Model loads in `load_model()`, not `__init__()`**
- [ ] **Heavy libraries imported inside `load_model()` / methods, NOT at module top level**
- [ ] **Optional dep group added to `pyproject.toml` + added to `all` group**
- [ ] Config updated: `TASK_ADAPTER_MAP` + `CONFIG["adapters"]`
- [ ] If local weights needed: `MODEL_REGISTRY` entry added in `download_models.py`
- [ ] Tested with `curl POST /infer` locally
- [ ] Added response JSON example to `docs/API_REFERENCE.md`
- [ ] *(If applicable)* Pydantic response model added to `app/schemas/responses.py`
- [ ] All tests pass: `uv run pytest tests/ -v`

---

## License

By contributing, you agree that your contributions will be licensed under the AGPL v3.0 license.
