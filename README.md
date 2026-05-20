# OpenNVR AI Adapter — Modular Inference Engine

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.127.0-009688.svg?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-FFD21E.svg)](https://huggingface.co/models)

A plug-and-play AI inference server for the **[OpenNVR](https://github.com/open-nvr/open-nvr)** ecosystem. Drop in any AI model — ONNX, PyTorch, or HuggingFace cloud — and it becomes available as a REST API, automatically discovered and lazily loaded.

> **v2.0 — Anti-Bloat Architecture:** Dependencies are now split into optional groups. A minimal deployment installs ~9 packages and only grows when you explicitly ask for more adapters.

---

## Why use ai-adapter (vs loading your model directly)?

The honest answer first: **for a single-camera hobby project, you don't need it.** `uv add ultralytics && python -c "from ultralytics import YOLO; YOLO('yolov8n.onnx')(frame)"` is shorter, faster, and totally fine.

ai-adapter exists because a **security product** has cross-cutting concerns that none of the underlying ML libraries (`ultralytics`, `faster-whisper`, `piper-tts`) provide. The same trade-off as `docker run` vs Kubernetes: the abstractions earn their keep at the scale and stakes where direct calls become liabilities.

### What this layer adds that loading YOLO directly does NOT

| # | Concern | What you get | Direct YOLO | ai-adapter |
|---|---------|--------------|-------------|------------|
| 1 | **Audit + correlation_id** | Every inference traceable from alert → middleware (KAI-C) → adapter line | not built in | one `X-Correlation-Id` per call, joined across the chain |
| 2 | **Fingerprint drift detection** | sha256 of weights polled every 60s — spots tampering / accidental rotation | impossible without re-rolling | §11.3 of the contract; surfaced as `adapter.fingerprint_mismatch` audit events |
| 3 | **Sovereignty (local_only)** | KAI-C refuses to register adapters that declare `network_egress` under `AI_SOVEREIGNTY=local_only` | self-enforce in your own code | enforced at the middleware boundary; verifiable from the audit log |
| 4 | **Operator permissions (§8)** | GPU / filesystem / network egress declared in `/capabilities`, surfaced for approval | implicit, ungated | explicit, gated, drift-detected |
| 5 | **Process isolation** | Model crash doesn't take down alerts pipeline | one Python process, one OOM | each adapter is its own container with its own resource limits |
| 6 | **Hot-swap any model** | Operator changes `kaic_adapter_name`; no monitoring-app rewrite | rewrite the script | YOLOv8 → YOLOv11 → cloud, one config line |
| 7 | **Multi-tenant fair queuing** | N monitoring apps share one adapter; one chatty camera can't starve the rest | each app owns its model | `scheduling.fair_queuing="per_camera"` in `/capabilities` (§9) |
| 8 | **Typed §7 error envelope** | Same wire shape across HTTP + WebSocket; clients write one parser | invent your own | `FailureEnvelope` with `category`, `code`, `transient`, `retry_after_ms` |
| 9 | **Prometheus metrics** | `adapter_infer_total{outcome=...}`, latency histograms, in-flight gauges — no glue | wire it yourself | built into the SDK; same labels across every adapter |

### How the [intrusion-detection example](https://github.com/open-nvr/open-nvr/tree/main/examples/intrusion-detection) uses it

```
┌─────────────────────┐  every poll_interval_seconds (or WS @ frame rate)
│  Camera (RTSP/HTTP) │ ─────────────────────────────┐
└─────────────────────┘                              │
                                                     ▼
                                       ┌──────────────────────────┐
                                       │  intrusion-detection     │  watch_labels, zones,
                                       │  monitoring app          │  restricted_hours
                                       └──────────┬───────────────┘
                                                  │ frame bytes + correlation_id
                                                  ▼
                                       ┌──────────────────────────┐
                                       │  KAI-C                   │  registry · sovereignty
                                       │  POST /api/v1/infer/...  │  · audit · authz · WS
                                       └──────────┬───────────────┘  proxy (A2.4b)
                                                  │
                                                  ▼
                                       ┌──────────────────────────┐
                                       │  YOLOv8 adapter (SDK)    │  §5.1 DetectionResult
                                       │  /infer · /infer/stream  │  (normalized bboxes)
                                       └──────────────────────────┘
```

When an operator investigates *"why did this alert fire at 22:14?"*, they can join:

```
alert correlation_id  →  KAI-C inference event log  →  adapter audit line
       a4f1b...                same a4f1b...               same a4f1b...
                                                       + model fingerprint sha256:...
                                                       + latency 38ms
                                                       + outcome=ok
```

If the weights file was tampered with, the fingerprint at the alert time differs from the registered fingerprint — visible as an `adapter.fingerprint_mismatch` drift event. If you'd just `from ultralytics import YOLO` in your script, you couldn't prove anything about which weights actually produced the detection.

### So when do you actually need it?

| If you... | Skip ai-adapter, just `uv add ultralytics` | Use ai-adapter |
|-----------|--------------------------------------------|----------------|
| One camera in your garage | ✓ | overkill |
| Prototype a detection idea | ✓ | overkill |
| Ship a security product to operators | dangerous | ✓ |
| Need to prove "AI didn't lie" in an incident | impossible | the audit chain is the proof |
| Swap YOLOv8 → YOLOv11 without rewriting alerts | painful | one config line |
| Run on `local_only` (no cloud calls ever) | self-enforce | KAI-C refuses non-compliant adapters |
| Run multiple monitoring apps on shared GPU | each owns its model | one adapter, fair-queued |

The wire contract is spec'd in [`open-nvr/docs/AI_ADAPTER_CONTRACT.md`](https://github.com/open-nvr/open-nvr/blob/main/docs/AI_ADAPTER_CONTRACT.md); the boilerplate-free SDK lives in [`opennvr_adapter_sdk/`](./opennvr_adapter_sdk/).

---

## Why build adapters here?

- **Real users, real cameras.** OpenNVR is a self-hosted NVR — every adapter you ship runs against live RTSP/ONVIF streams on real hardware, not a toy benchmark.
- **Three files, zero plumbing.** Auto-discovery, lazy loading, config routing, Pydantic validation, Docker packaging, and REST surface are all handled. You focus on the model.
- **Ship without forking.** Add your adapter as its own file in `app/adapters/<vision|llm>/`. No edits to `main.py`, no central registry, no breaking other adapters.
- **Lean by design.** Your adapter declares its own `[project.optional-dependencies]` group. Users who don't need your model don't pay for it at install time.
- **AGPLv3-licensed.** Good-faith contributions stay open. Commercial licensing is available if you need to ship adapters under closed terms — see [License](#license) below.

### Adapters we'd love to see

Contributions in any of these areas are on the roadmap and explicitly welcome:

| Category | Ideas |
|---|---|
| **Safety / security** | Weapons detection, fire/smoke detection, fall detection, PPE compliance (hard hat / vest / mask) |
| **Access & identity** | License-plate recognition (ANPR), uniform / ID-badge detection, gait recognition |
| **Analytics** | Crowd density estimation, queue length, dwell-time heatmaps, vehicle classification |
| **Audio** | Glass-break detection, gunshot detection, aggression detection, diarisation (Whisper STT ships as `whisper_adapter`; Piper TTS ships as `piper_adapter`) |
| **Conversational agents** | Function-calling LLM adapters, RAG-over-events, voicemail bots, on-call escalation flows (Ollama LLM ships as `ollama_adapter`; real-time streaming is the next frontier) |
| **Animals & wildlife** | Pet / livestock detection, wildlife classification, bird-species ID |
| **Edge optimisation** | TensorRT / OpenVINO / CoreML variants of existing adapters for Jetson, Intel NUC, Apple Silicon |

Have another idea? Open a [discussion](https://github.com/open-nvr/ai-adapter/discussions) before you start coding — we'll help scope it.

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
| **Whisper STT** | `uv sync --extra stt` | + faster-whisper (CTranslate2, CPU-ok) | ~300 MB |
| **Piper TTS** | `uv sync --extra tts` | + piper-tts (ONNX voices, CPU-ok) | ~100 MB + 25 MB/voice |
| **Ollama LLM** | (no extras needed — uses core `httpx`) | requires an Ollama daemon running locally | ~0 MB (model lives in Ollama) |
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

## Community & Support

- 💬 **Discussions:** [github.com/open-nvr/ai-adapter/discussions](https://github.com/open-nvr/ai-adapter/discussions) — questions, adapter proposals, RFCs.
- 🐛 **Issues:** [github.com/open-nvr/ai-adapter/issues](https://github.com/open-nvr/ai-adapter/issues) — bug reports, feature requests. Look for the `good first issue` label to pick up something bite-sized.
- 📘 **Parent project:** [github.com/open-nvr/open-nvr](https://github.com/open-nvr/open-nvr) — the NVR that consumes these adapters.

---

## License

Licensed under the **GNU Affero General Public License v3.0 (AGPL v3)**.

**What this means for adapter authors:**
- ✅ You can publish your adapter under AGPLv3 (or any AGPLv3-compatible license).
- ✅ Your model *weights* are not AGPL-bound — only the adapter source is. You may ship weights under any license the model permits (including proprietary).
- ⚠️ If you integrate a **GPL-incompatible library** (e.g. a commercial SDK that forbids copyleft linking), you cannot distribute that adapter in-tree. Keep it as a third-party repo and we'll link to it.
- ⚠️ Running OpenNVR + your adapter as a **network service** triggers AGPL source-disclosure: any modified source must be made available to users of the service.

See the `LICENSE` file for the full terms.

> For commercial licensing (closed-source adapters, proprietary redistribution, or enterprise support): **[contact@cryptovoip.in](mailto:contact@cryptovoip.in)**
