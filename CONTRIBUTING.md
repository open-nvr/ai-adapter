# Contributing to OpenNVR AI Adapter

Thanks for being here. This repo is two things in one: the `opennvr-adapter-sdk` PyPI package (the boilerplate-free way to ship a contract-compliant adapter) and a reference monolith server that bundles several adapters in one process. There's a contribution path for each, and the recommended path depends on what you're trying to do.

## Which path are you on?

If you're shipping a new adapter, use **Path A — the SDK plus a standalone container.** Your adapter becomes its own service with its own `pyproject.toml`, declared permissions, and lifecycle. Nothing in this repo needs to change, and the adapter ships under whatever licence you choose. The seven shipped adapters in `adapters/<name>/` are the reference implementations — find the one with the closest body shape and copy the structure. Bug fixes to those services also live on this path.

If you're extending the bundled reference monolith — `app/adapters/<name>/` — use **Path B**, the `BaseAdapter` plugin pattern. This path stays supported for contributors keeping the monolith useful, but new adapters should target Path A: a standalone container is contract-compliant out of the box, ships independently, and survives any future change to the monolith's internals.

If you're improving the SDK itself, work in [`opennvr_adapter_sdk/`](opennvr_adapter_sdk/) and read its [README](opennvr_adapter_sdk/README.md) for the surface and conventions.

## Development setup

```bash
git clone https://github.com/open-nvr/ai-adapter.git
cd ai-adapter

uv venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
uv sync --extra all --extra cpu --extra dev

# Download model weights for the adapters you'll touch
uv run python download_models.py

# Run the legacy monolith server (still the simplest way to dogfood everything)
uv run uvicorn app.main:app --reload --port 9100

# Run tests
uv run pytest tests/ -v
```

Per-adapter services live in `adapters/<name>/` and are their own
deployable units. Each one has its own `pyproject.toml`, `service.py`,
`main.py`, and tests. They're built and published as separate GHCR
images by `.github/workflows/publish-images.yml`.

## Anti-bloat rules — read this first

These apply to **both paths**. They exist because the project becomes
unusable for everyone if every adapter drags in PyTorch, transformers,
and onnxruntime by default.

### Rule 1: Heavy ML libraries go in optional dependency groups

Never add an ML library to `[project.dependencies]`. That section is
shared core — keep it small.

```toml
# pyproject.toml

# WRONG — forces ~500 MB on every deployment
[project.dependencies]
"my-ml-library>=1.0.0"

# RIGHT — only installed when the user opts in
[project.optional-dependencies]
my_adapter = ["my-ml-library>=1.0.0"]

# Also add to the "all" group so `uv sync --extra all` still works
all = ["opennvr-ai-adapter[yolo,yolo11,face,blip,huggingface,my_adapter]"]
```

For Path A (standalone container), the equivalent is your per-adapter
`pyproject.toml` — keep it tight, list only what the adapter actually
needs. The Dockerfile is built from this so every megabyte counts.

### Rule 2: Import heavy libraries inside `load()` / `load_model()`, not at the top

In Path A, `AdapterService.load()` is the lifecycle hook the SDK
guarantees is called once before the service goes ready. In Path B,
`BaseAdapter.load_model()` is the equivalent. Either way, defer heavy
imports until you actually need them.

```python
# WRONG — torch loads even if this adapter is never used
import torch
from transformers import AutoModel

class MyService(AdapterService):
    def load(self):
        self._model = AutoModel.from_pretrained("...")

# RIGHT — torch loads only when the adapter is actually instantiated
class MyService(AdapterService):
    def load(self):
        import torch
        from transformers import AutoModel
        self._model = AutoModel.from_pretrained("...")
```

### Rule 3: What happens when a dep isn't installed

Path A (standalone): your container's `pip install` lists everything;
the adapter either runs or fails to start. No surprises.

Path B (monolith): `PluginManager` catches the `ImportError` at
discovery time and logs `Skipping 'app.adapters.vision.my_adapter':
optional dependency 'my_lib' not installed. Install with: uv sync
--extra my_adapter`. All other adapters load fine; yours just doesn't
appear in `/capabilities`. No crash, no stack trace.

## Path A — SDK + standalone container

The end state is a Docker image that runs a single `AdapterService`
behind the FastAPI app `AdapterApp` builds for you. The SDK README
walks the minimum viable example in detail; the short version:

```python
# my_adapter/main.py
from datetime import datetime, timezone
from opennvr_adapter_sdk import (
    AdapterApp, AdapterService, BodyShape, BODY_BYTES_KEY,
    HardwareEvaluationResponse, HardwareVerdict,
    InferResponse, ModelInfo,
)

class MyService(AdapterService):
    def load(self):
        import onnxruntime as ort
        self._model = ort.InferenceSession("/weights/my-model.onnx")

    def is_ready(self) -> bool:
        return self._model is not None

    def fingerprint(self) -> str | None:
        return "sha256:..."     # sha256 of the weights file

    def model_info(self) -> ModelInfo:
        return ModelInfo(
            name="my-model", version="1.0.0",
            framework="onnx", fingerprint=self.fingerprint(),
        )

    def hardware_evaluation(self) -> HardwareEvaluationResponse:
        return HardwareEvaluationResponse(
            verdict=HardwareVerdict.OK, reasoning="ok",
            checked_at=datetime.now(timezone.utc), details={},
        )

    def infer(self, payload) -> InferResponse:
        frame_bytes = payload[BODY_BYTES_KEY]
        # ... run your model ...
        return InferResponse(
            model_name="my-model", model_version="1.0.0",
            inference_ms=42,
            result={"detections": [{"label": "fire", "confidence": 0.95}]},
        )

app = AdapterApp(
    service=MyService(),
    name="my-adapter", version="1.0.0", vendor="me", license="MIT",
    tasks_advertised=["fire_detection"],
    body_shape=BodyShape.IMAGE,
).fastapi_app
```

```bash
OPENNVR_ADAPTER_TOKEN=dev-token \
  uvicorn my_adapter.main:app --host 0.0.0.0 --port 9001
```

Then verify your adapter is contract-compliant:

```bash
python -m conformance http://localhost:9001 --token dev-token
```

The conformance suite checks every required endpoint, the `/health`
state machine, the failure envelope shape, and the multipart/JSON body
parser. If your service passes conformance, KAI-C will register it.

**Reference implementations** to copy-from in this repo:

- `adapters/yolov8/` — `BodyShape.IMAGE`, WebSocket streaming
- `adapters/whisper/` — `BodyShape.AUDIO`, audio decode pipeline
- `adapters/piper/` — `BodyShape.TEXT`, custom `/voices` route
- `adapters/insightface/` — `BodyShape.IMAGE`, face DB enrollment endpoints
- `adapters/blip/` — `BodyShape.IMAGE`, transformer model
- `adapters/fast_plate_ocr/` — `BodyShape.IMAGE`, OCR over a pre-cropped plate
- `adapters/bytetrack/` — `BodyShape.TEXT`, stateful post-processor over an upstream detector

Full SDK reference: [`opennvr_adapter_sdk/README.md`](opennvr_adapter_sdk/README.md).

## Path B — Add to the reference monolith

The monolith uses a plugin pattern that predates the contract. It's
still supported because it's simpler for contributors who just want to
extend the bundled server without standing up a separate container.

### Option B1 — Adapter only (raw output)

The simplest shape: your adapter's raw output goes straight to the
client. No Pydantic validation.

```python
# app/adapters/vision/my_adapter.py
from app.adapters.base import BaseAdapter

class MyAdapter(BaseAdapter):
    name = "my_adapter"     # Unique; used in TASK_ADAPTER_MAP routing
    type = "vision"         # "vision" or "llm"

    def __init__(self, config=None):
        self.config = config or {}
        self.model = None   # Must start as None for lazy loading

    def load_model(self):
        # Import heavy libraries HERE, not at the top of the file.
        import my_ml_library
        self.model = my_ml_library.load(self.config.get("weights_path"))

    def infer_local(self, input_data):
        from app.utils.image_utils import load_image_from_uri
        img = load_image_from_uri(input_data["frame"]["uri"])
        return self.model.predict(img)
```

```toml
# pyproject.toml — add the optional dep group
[project.optional-dependencies]
my_adapter = ["my-ml-library>=1.0.0"]
all = ["opennvr-ai-adapter[yolo,yolo11,face,blip,huggingface,my_adapter]"]
```

```python
# app/config/config.py — register routing
TASK_ADAPTER_MAP["my_task"] = "my_adapter"
CONFIG["adapters"]["my_adapter"] = {"enabled": True, "weights_path": "my_model.bin"}
```

```python
# download_models.py — only if your model needs a local weights file
MODEL_REGISTRY["my_adapter"] = [
    {
        "filename": "my_model.bin",
        "url": "https://github.com/me/my-models/releases/download/v1/my_model.bin",
        "size_hint": "~50 MB",
    }
]
```

Restart the server. `PluginManager` auto-discovers your class. Done.

### Option B2 — Adapter + Pydantic-validated task

Same as B1 plus a task pipeline that validates the adapter output
against a Pydantic schema before returning it.

```python
# app/schemas/responses.py
class MyResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    label: str
    confidence: float = Field(ge=0.0, le=1.0)
    bbox: list[int]
```

```python
# app/pipelines/my_task/task.py
from app.interfaces.task import BaseTask
from app.schemas.responses import MyResponse

class MyTask(BaseTask):
    name = "my_task"        # Must match the key in TASK_ADAPTER_MAP

    def process(self, image, adapter):
        raw = adapter.predict(image)
        return MyResponse(**raw)
```

Full Path B tutorial with all the details:
[`docs/PLUGIN_DEVELOPMENT.md`](docs/PLUGIN_DEVELOPMENT.md).

## How auto-discovery works (Path B only)

At startup `PluginManager.discover_plugins()`:

1. Walks every non-private `.py` file under `app/adapters/` and `app/pipelines/`.
2. Attempts `importlib.import_module(...)` on each.
   - `ImportError` → logs a helpful skip message (missing optional dep — expected).
   - Other error → WARNING (real bug in your code).
   - Success → registers any `BaseAdapter` / `BaseTask` subclasses by their `name`.
3. Logs a summary: `Discovered 3 adapters: [insightface_adapter, yolov8_adapter, yolov11_adapter]`.

No imports in `main.py`. No registration calls. Just create the file
and restart. (Path A doesn't need this — your service is the only
class in the container.)

## Installation profiles (monolith)

Per-profile dependencies and sizes are documented in the main [README](README.md#install-profiles). The relevant addition for contributors is the `--extra dev` group, which adds the test suite and tooling (around 4 GB total with `--extra all --extra cpu --extra dev`).

## Testing

```bash
# Monolith suite
uv run pytest tests/ -v

# Per-adapter suites
uv run pytest adapters/yolov8/tests/ -v
uv run pytest adapters/whisper/tests/ -v
# ...etc

# With coverage
uv run pytest tests/ --cov=app --cov-report=term-missing

# Conformance check against a running adapter
python -m conformance http://localhost:9001 --token dev-token
```

When adding a new adapter, please add tests in either `tests/` (Path B)
or `adapters/<name>/tests/` (Path A).

## Pre-PR checklist

### For both paths

- [ ] Heavy ML libraries imported inside `load()` / `load_model()`, not module top.
- [ ] Optional dep group declared (or per-adapter `pyproject.toml` updated).
- [ ] Tests added for the new behaviour.
- [ ] Tested with `curl POST /infer` locally.
- [ ] All tests pass: `uv run pytest -v`.

### Path A additions

- [ ] Adapter passes the conformance suite (`python -m conformance ...`).
- [ ] Per-adapter `pyproject.toml` lists only what the adapter actually needs.
- [ ] Per-adapter `Dockerfile` builds and produces a runnable image.

### Path B additions

- [ ] `BaseAdapter` subclass has `name` and `type` class attributes.
- [ ] Model loads in `load_model()`, not `__init__()`.
- [ ] Config updated: `TASK_ADAPTER_MAP` + `CONFIG["adapters"]`.
- [ ] If local weights needed: `MODEL_REGISTRY` entry added in `download_models.py`.
- [ ] If returning structured output: Pydantic response model added to `app/schemas/responses.py`.
- [ ] Response JSON example added to `docs/API_REFERENCE.md`.

## Commit messages

[Conventional Commits](https://www.conventionalcommits.org/). Common
scopes: `sdk`, `adapters/<name>`, `app`, `conformance`, `docs`.

```text
feat(adapters/yolov8): support WebSocket streaming protocol
fix(sdk): correct BodyShape.AUDIO multipart field name
docs(contributing): clarify Path A vs Path B
```

## Code of conduct

We want this to stay a project people enjoy contributing to. The expectation is welcoming language and respectful disagreement; harassment, doxxing, and inflammatory off-topic posting are not part of that. Report violations to **contact@cryptovoip.in** — reports are confidential.

## License

By contributing, you agree your contributions are licensed under:

- **AGPL v3** for code in `app/`, `adapters/<name>/`, and the rest of
  the reference server.
- **Apache-2.0** for code in `opennvr_adapter_sdk/` — the SDK is
  intentionally permissive so adapter authors can publish under any
  compatible license, including proprietary.
