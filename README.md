<div align="center">

<img src=".github/opennvr-logo.svg" alt="OpenNVR" width="260" />

# AI Adapter

### Any model. Any framework. One governed contract.

The pluggable inference layer for [OpenNVR™](https://github.com/open-nvr/open-nvr). Drop any model behind an HTTP or WebSocket endpoint and it becomes a first-class, *governed* capability — with end-to-end audit, fingerprint drift detection, and sovereignty enforcement that the underlying model never had to know about. Governance lives in the contract itself, not bolted on after: an adapter must declare what it touches (GPU, filesystem, network egress) before it can run, and an operator's sovereignty policy can refuse it at registration. That is what lets you run third-party — even proprietary or classified — AI on regulated, air-gapped sites and still answer "which model, which weights, on which frame, under whose authority?"

[![CI](https://github.com/open-nvr/ai-adapter/actions/workflows/ci.yml/badge.svg)](https://github.com/open-nvr/ai-adapter/actions/workflows/ci.yml)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](LICENSE)
[![SDK on PyPI](https://img.shields.io/badge/PyPI-opennvr--adapter--sdk-3775A9?logo=pypi&logoColor=white)](https://pypi.org/project/opennvr-adapter-sdk/)
[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)](https://www.python.org)

[Write an adapter](#write-your-own-adapter) · [Install profiles](#install-profiles) · [What ships](#what-ships) · [Contract spec](https://github.com/open-nvr/open-nvr/blob/main/docs/AI_ADAPTER_CONTRACT.md)

</div>

---

## What this is

OpenNVR is the offline-first NVR underneath; this repo is the layer that lets *any* AI model become one of its detectors. The wire spec — the [AI Adapter Contract v1](https://github.com/open-nvr/open-nvr/blob/main/docs/AI_ADAPTER_CONTRACT.md) — describes six HTTP endpoints (plus one WebSocket) every adapter implements. An adapter is whatever software implements those endpoints. The SDK published here makes that around thirty lines of Python.

What you get back, for free, is the part that's actually hard: an `X-Correlation-Id` threading every inference from the alert that fired to the model that ran, a sha256 fingerprint of the model weights polled every 60 seconds for drift, sovereignty enforcement (`local_only` mode refuses to register adapters that declare network egress), per-camera fair queuing, typed error envelopes, and Prometheus metrics with the same label shape across every adapter. The middleware does that work once so you don't ship it in each model wrapper.

For a one-camera homelab project, `pip install ultralytics` and a YOLO loop is shorter. The adapter layer earns its keep where the stakes go up — multiple cameras, multiple models, audit obligations, compliance evidence, or a threat model where "the vendor's cloud is part of the attack surface."

## Quickstart — ship an adapter in thirty seconds

```bash
pip install opennvr-adapter-sdk
```

```python
from opennvr_adapter_sdk import (
    AdapterApp, AdapterService, BodyShape, BODY_BYTES_KEY,
    HardwareEvaluationResponse, HardwareVerdict,
    InferResponse, ModelInfo,
)

class MyDetector(AdapterService):
    def load(self):                                  # eagerly load your weights
        ...

    def is_ready(self) -> bool:
        return True

    def fingerprint(self) -> str | None:             # sha256 of the weights on disk
        return "sha256:..."

    def model_info(self) -> ModelInfo:
        return ModelInfo(name="my-model", version="1.0.0",
                         framework="onnx", fingerprint=self.fingerprint())

    def hardware_evaluation(self) -> HardwareEvaluationResponse:
        return HardwareEvaluationResponse(verdict=HardwareVerdict.OK, details="")

    def infer(self, payload) -> InferResponse:
        frame_bytes = payload[BODY_BYTES_KEY]
        # ... run your model ...
        return InferResponse(result={"detections": [
            {"label": "person", "confidence": 0.93, "bbox": [10, 20, 100, 200]},
        ]})

app = AdapterApp(
    service=MyDetector(),
    name="my-detector", version="1.0.0", vendor="me", license="MIT",
    tasks_advertised=["object_detection"],
    body_shape=BodyShape.IMAGE,
).fastapi_app
```

`uvicorn my_module:app --port 9100`, register the URL with KAI-C's `/api/v1/adapters/register`, and your adapter is online — hot-swappable, audit-chained, fingerprint-tracked. The SDK is Apache-2.0, so your adapter ships under whatever licence you choose — including proprietary or classified for the organisations where that matters.

## What ships

The repo is two things in one. The [`opennvr_adapter_sdk/`](opennvr_adapter_sdk/) directory is the small Apache-2.0 SDK published to PyPI — three classes (`AdapterService`, `AdapterApp`, `ServiceError`), a handful of result models, zero ML dependencies. Adapter authors install it from PyPI and never need to clone this repo. The [`adapters/`](adapters/) directory is the eight reference adapters that ship as standalone Docker images on GHCR (`ghcr.io/open-nvr/*-adapter`) — each is a working example of an SDK-based adapter, and Tier 0 of OpenNVR pulls them by tag.

```
ai-adapter/
├── opennvr_adapter_sdk/      The SDK (Apache-2.0) — what PyPI publishes
├── adapters/                 Contract v1 per-adapter services (one image each)
│   ├── yolov8/               Object detection — ONNX runtime, CPU + GPU
│   ├── piper/                Text-to-speech — inline-audio response option
│   ├── whisper/              Speech-to-text — faster-whisper, CPU + GPU
│   ├── fast_plate_ocr/       License-plate recognition
│   ├── insightface/          Face detection + recognition with REST face DB
│   ├── blip/                 Scene captioning — used by the camera-agent
│   ├── vlm/                  Open-vocabulary detection — OWL-ViT v2, detects free-text queries ("red truck")
│   └── bytetrack/            Multi-object tracking — stateful post-processor over an upstream detector
├── templates/adapter-template/   Scaffold a new adapter in one command
├── conformance/              Wire-contract conformance test suite
├── app/                      Legacy monolithic reference server
└── docs/                     Architecture, plugin dev, API reference
```

Each of the eight shipped adapters lives in its own directory with its own `pyproject.toml`, Dockerfile, README, and tests. Replicate the shape, swap the model, and you have a new adapter.

## Write your own adapter

There are two paths and the recommended one is the SDK. The standalone-container path means your adapter is its own service with its own `pyproject.toml`, declared permissions, and lifecycle. Nothing in this repo needs to change. The shipped adapters in [`adapters/`](adapters/) are the reference implementations — pick the one with the closest body shape and copy the structure. `adapters/yolov8/` is the canonical `BodyShape.IMAGE` example with WebSocket streaming; `adapters/whisper/` covers `BodyShape.AUDIO` with multipart decode; `adapters/piper/` covers `BodyShape.TEXT` with a custom route and inline audio response; `adapters/bytetrack/` is the stateful post-processor reference, sitting over an upstream detector's output. The [SDK README](opennvr_adapter_sdk/README.md) is the authoring walkthrough; the [template scaffold](templates/adapter-template/) generates a working adapter directory from one command.

The legacy `app/` monolith ships HuggingFace and the original Ollama wrapper as in-process plugins via a `BaseAdapter` pattern that predates the contract. It stays supported for contributors extending that bundled server, but new adapters should target the SDK — a standalone container is contract-compliant out of the box, ships independently, and exposes only the wire-contract surface, which is what KAI-C actually depends on. Several of the shipped reference adapters in [`adapters/`](adapters/) currently delegate to legacy classes from `app/` as their backing implementation; a fully fresh adapter wouldn't carry that coupling, and the wire surface is the same either way. The full monolith authoring walkthrough is in [`docs/PLUGIN_DEVELOPMENT.md`](docs/PLUGIN_DEVELOPMENT.md).

## Declaring permissions

Your adapter's `/capabilities` card carries a `permissions` block — the host scopes your service needs (GPU device, network egress hosts, host filesystem paths, shared memory, host metadata). These are **authorization requests, not hardware requirements**: `gpu=True` means "may I be handed the GPU device", and any declared scope puts your adapter into a fail-closed `pending` state on registration — visible and health-polled, but refused on every inference path until an operator grants each scope from the OpenNVR UI. Whether the hardware is actually present is `/hardware/evaluation`'s job, not a permission. The full model — grantable keys, the approval lifecycle, the 60-second drift diff — is [§8 of the contract](https://github.com/open-nvr/open-nvr/blob/main/docs/AI_ADAPTER_CONTRACT.md#8-permission-declaration--sandbox-enforcement); this is the short version for authors.

You declare permissions in the `AdapterApp` constructor. The shipped YOLOv8 adapter ([`adapters/yolov8/main.py`](adapters/yolov8/main.py)) is the reference:

```python
from opennvr_adapter_sdk import AdapterApp, Permissions

_adapter_app = AdapterApp(
    ...,
    permissions=Permissions(
        # §8 — declare build-accurately: gpu=True (an operator-approval
        # gate at KAI-C registration) only when this build can actually
        # use CUDA. The stock CPU image therefore declares gpu=False.
        gpu=_cuda_provider_available(),
        network_egress=[],
        # No host_filesystem entry: the weights come from a
        # container-owned named volume, not a host bind-mount.
        host_filesystem=[],
        shared_memory_paths=[],
        host_metadata=False,
    ),
)
```

Four rules keep the operator's approval dialog honest and your adapter friction-free:

1. **Declare build-accurately.** Describe what *this image* touches at runtime, not what the model family could use. If you ship separate CPU and GPU builds, the CPU build must declare `gpu=False`; only the GPU build declares `gpu=True` — YOLOv8 does this by checking the installed onnxruntime build for `CUDAExecutionProvider` at startup. Weights baked into the image at build time or supplied via a container-owned volume are **not** `host_filesystem` — only host bind-mounts are. Compare the shipped references: BLIP bakes its weights and YOLOv8 reads them from a named volume, so neither declares a path; declare `host_filesystem` only when your compose wiring genuinely bind-mounts a host directory into the container.
2. **Declare minimally.** Every scope is one operator decision at install time and one permanent line of audit surface. If a build change (bake the weights, cache at build time) removes a scope, remove it.
3. **The empty set is the zero-friction default.** An adapter that declares no permissions auto-approves and serves the moment it registers — no dialog, no waiting. That's the target state for most local adapters.
4. **Never declare egress you don't strictly need.** Under `local_only` sovereignty — the default posture for the deployments OpenNVR targets — any declared `network_egress` entry means your adapter is refused at registration outright. Cloud-proxy adapters should declare every host they call, explicitly, and expect to run only under `federated` / `cloud_allowed`.

One more reason to get the declaration right the first time: KAI-C re-polls `/capabilities` every 60 seconds and treats a *newly appearing* permission as a tamper signal — the adapter flips back to `pending` and stops serving until an operator re-approves. Shipping an update that quietly widens your scope will take your users' deployments offline until they consent.

## Run the reference server

If you want to run the bundled monolith end-to-end against a local OpenNVR:

```bash
git clone https://github.com/open-nvr/ai-adapter.git
cd ai-adapter
uv venv && source .venv/bin/activate          # Windows: .venv\Scripts\activate
uv sync --extra all --extra cpu               # full install, CPU torch
uv run python download_models.py              # fetch model weights
uv run uvicorn app.main:app --reload --port 9100
```

On boot you should see `Server ready. Discovered tasks=N adapters=M`. The server auto-discovers everything under `app/adapters/` and `app/pipelines/` — add a file, restart, you're live. For the standalone per-adapter containers each subdirectory under `adapters/` builds via its own Dockerfile (`docker build -f adapters/yolov8/Dockerfile .`).

## Install profiles

Dependencies are split into optional groups so a minimal deployment installs around nine packages and only grows when you ask for more. The core install (`uv sync`) is FastAPI plus uvicorn plus pydantic plus numpy plus opencv — about 50 MB. From there each capability adds its own extra: `--extra yolo` adds ONNX runtime (~250 MB) for YOLOv8; `--extra yolo11 --extra cpu` adds ultralytics plus CPU torch (~2.5 GB) for YOLOv11; `--extra face` adds InsightFace plus scipy (~500 MB); `--extra lpr` adds fast-plate-ocr's ONNX runtime (~30 MB) for license-plate recognition; `--extra blip --extra cpu` adds transformers plus torch (~3 GB); `--extra huggingface` adds the HuggingFace Hub client (~60 MB); `--extra stt` adds faster-whisper (~300 MB); `--extra tts` adds piper-tts (~100 MB plus ~25 MB per voice). `--extra all --extra cpu` pulls everything with CPU torch (~4 GB); `--extra all --extra gpu` swaps in CUDA torch (~6 GB). Ollama is reached over HTTP via the core `httpx` dependency — install Ollama itself separately.

Most NVR deployments only need `--extra yolo --extra face` (around 750 MB) for person detection plus face recognition.

## API surface

The reference server exposes the following routes; full request and response shapes live in [`docs/API_REFERENCE.md`](docs/API_REFERENCE.md) and the wire spec is the [AI Adapter Contract](https://github.com/open-nvr/open-nvr/blob/main/docs/AI_ADAPTER_CONTRACT.md).

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Server health, loaded adapter status, hardware info |
| `/capabilities` | GET | Wire-contract capability advertisement |
| `/tasks` | GET | Available tasks with adapter info |
| `/schema` | GET | Response schema for one or all tasks |
| `/infer` | POST | Single-task inference |
| `/pipeline/run` | POST | Chain of tasks run sequentially |
| `/adapters` | GET | List currently loaded adapters |
| `/faces/register` | POST | Register a face for recognition |
| `/faces/{person_id}` | GET / DELETE | Get or delete a registered face |

WebSocket streaming (`/infer/stream`) is part of the contract and supported by the SDK's `AdapterApp` for every SDK-based adapter; the bundled reference server currently exposes only the HTTP surface above.

## Adapters we'd like next

Contributions in any of these areas are explicitly welcome and tracked in the [discussions](https://github.com/open-nvr/ai-adapter/discussions). Safety and security: weapon detection, fire and smoke detection, fall detection (pose-based), PPE compliance (hard hat, vest, harness). Access and identity: license-plate recognition variants for non-Latin scripts, uniform and ID-badge detection, gait recognition. Analytics: crowd density, queue length, dwell-time heatmaps, vehicle classification. Audio: glass-break and gunshot detection, aggression detection, speaker diarisation. Conversational agents: function-calling LLM adapters, RAG-over-events, on-call escalation flows. Animals and wildlife: pet and livestock detection, wildlife and bird-species ID. Edge optimisation: TensorRT, OpenVINO, and CoreML variants for Jetson, NUC, and Apple Silicon. Have a different idea? Open a [discussion](https://github.com/open-nvr/ai-adapter/discussions) before you start coding and we'll help scope it.

## Docker

```bash
docker build -t ai-adapter .                                  # CPU build
docker build --build-arg USE_GPU=true -t ai-adapter:gpu .     # CUDA build
docker run -p 9100:9100 -v ./model_weights:/app/model_weights ai-adapter
```

Built-in healthcheck pings `/health` every 30 seconds.

## Community

Adapter proposals and design discussions go in [Discussions](https://github.com/open-nvr/ai-adapter/discussions); bug reports and feature requests in [Issues](https://github.com/open-nvr/ai-adapter/issues) — look for `good first issue` if you're new. The parent project is [open-nvr/open-nvr](https://github.com/open-nvr/open-nvr) — the NVR that consumes these adapters.

## License

The reference server in this repo is **AGPLv3**. The SDK at [`opennvr_adapter_sdk/`](opennvr_adapter_sdk/) is **Apache-2.0** so adapter authors can publish under any compatible licence — including proprietary or classified. Adapter model weights you ship are not AGPL-bound; they remain under whatever licence the model itself permits. Running OpenNVR with your adapter as a network service triggers AGPL source-disclosure for the server-side modifications you've made. Full terms in [`LICENSE`](LICENSE).

"OpenNVR" and the OpenNVR logo are trademarks of the project; an adapter may describe itself as "compatible with OpenNVR" but should not use the name as its own. See the [trademark policy](https://github.com/open-nvr/open-nvr/blob/main/TRADEMARK.md).

For commercial licensing — closed-source adapters, proprietary redistribution, enterprise support — write to **[contact@cryptovoip.in](mailto:contact@cryptovoip.in)**.

---

<div align="center">

**[Install the SDK](#quickstart--ship-an-adapter-in-thirty-seconds) · [Write an adapter](#write-your-own-adapter) · [Read the contract](https://github.com/open-nvr/open-nvr/blob/main/docs/AI_ADAPTER_CONTRACT.md) · [Browse OpenNVR](https://github.com/open-nvr/open-nvr)**

</div>
