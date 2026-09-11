# opennvr-adapter-sdk

Publish a model on [OpenNVR](https://opennvr.org). An adapter wraps a
model and answers the [AI Adapter Contract v1](https://github.com/open-nvr/open-nvr/blob/main/docs/AI_ADAPTER_CONTRACT.md),
so any deployment can route work to it and any app can ask for its task
by name.

**Apache-2.0** — your adapter is yours, under any licence you choose.

## Install

```bash
uv add opennvr-adapter-sdk
# with the uvicorn server bundled too:
uv add 'opennvr-adapter-sdk[serve]'
```

`pip install opennvr-adapter-sdk` works the same. We recommend
[uv](https://docs.astral.sh/uv/) because adapter projects grow heavy ML
dependencies fast.

## A whole adapter

```python
from opennvr_adapter_sdk import Adapter

adapter = Adapter(
    "fall-detection",
    version="1.0.0",
    vendor="ACME",
    license="Apache-2.0",
    tasks=["object_detection"],
    framework="onnxruntime",
    weights="models/fall.onnx",
)

@adapter.load()
def load():
    import onnxruntime as ort
    return ort.InferenceSession(adapter.weights)

@adapter.on_image()
def detect(call):
    boxes = call.model.run(None, {"images": preprocess(call.image)})
    return [call.detection("fallen", score, x, y, w, h)
            for score, (x, y, w, h) in boxes]

app = adapter.app          # uvicorn my_model:app
```

That is a complete, conformant adapter. The SDK derives what the
contract needs and you would otherwise hand-write: the **fingerprint**
from the weights file (and never null — KAI-C silently skips drift
detection on a null one), **health** from the loader, the **hardware
verdict**, the **modalities** and **body shape** from which handler you
registered, and the **error taxonomy** — a `ValueError` becomes a 400 so
KAI-C does not retry a bad frame, anything else a 500, and `Overloaded`
a 503 with a retry hint.

## The tools

```bash
opennvr-adapter new my-model     # a runnable adapter project + tests
cd my-model
opennvr-adapter dev              # drive it in-process — no Docker, no stack
opennvr-adapter validate .       # the full conformance run
opennvr-adapter spec             # its OpenAPI 3.1 document
opennvr-adapter conform URL      # check a running adapter
```

`validate` is the one that matters: it runs the same checks KAI-C will,
in-process, so a green run means a deployment will accept your adapter.

## What your adapter publishes about itself

| Endpoint | |
|---|---|
| `GET /health` | Liveness and model-load state. |
| `GET /capabilities` | Identity, model info, fingerprint, tasks, permissions. |
| `GET /hardware/evaluation` | Whether this host can run the model well. |
| `GET /metrics` | Prometheus exposition. |
| `POST /infer` | One inference. |
| `GET /openapi.json` | **OpenAPI 3.1**, every response typed — Swagger UI at `/docs`. |
| `GET /asyncapi.json` | **AsyncAPI 3.0** — the `/infer/stream` protocol. |

Both specs are generated from the contract types the adapter actually
returns, so they cannot drift from it.

## …or the classes underneath

`Adapter` compiles to `AdapterService` + `AdapterApp`. Use them directly
when a model outgrows the decorators — the process, the endpoints, the
metrics and the specs are identical.

## The minimum viable adapter

```python
# my_adapter/main.py
from datetime import datetime, timezone

from opennvr_adapter_sdk import (
    AdapterApp, AdapterService, BodyShape, ErrorCategory,
    HardwareEvaluationResponse, HardwareVerdict, InferResponse,
    ModelInfo, ServiceError,
)

class MyService(AdapterService):
    def __init__(self):
        self._ready = False

    def load(self):
        # Heavy lifting goes here.
        self._ready = True

    def is_ready(self): return self._ready

    def fingerprint(self):
        return "sha256:..."

    def model_info(self):
        return ModelInfo(
            name="my-model", version="1.0",
            framework="numpy", modalities_in=["text"],
            modalities_out=["text"], fingerprint=self.fingerprint(),
        )

    def hardware_evaluation(self):
        return HardwareEvaluationResponse(
            verdict=HardwareVerdict.OK, reasoning="ready",
            checked_at=datetime.now(timezone.utc), details={},
        )

    def infer(self, payload):
        if "text" not in payload:
            raise ServiceError(
                ErrorCategory.TRANSPORT_ERROR, code="malformed_input",
                message="'text' required", transient=False, http_status=400,
            )
        return InferResponse(
            model_name="my-model", model_version="1.0",
            inference_ms=1, result={"echoed": payload["text"]},
        )

app = AdapterApp(
    service=MyService(),
    name="my-adapter", version="1.0.0",
    vendor="me", license="MIT",
    tasks_advertised=["echo"],
    body_shape=BodyShape.TEXT,
).fastapi_app
```

Run it:

```bash
OPENNVR_ADAPTER_TOKEN=dev-token \
  uvicorn my_adapter.main:app --host 0.0.0.0 --port 9001
```

Verify conformance:

```bash
python -m conformance http://localhost:9001 --token dev-token
```

That's a complete contract-compliant adapter. The SDK handles `/health`, `/capabilities`, `/hardware/evaluation`, `/metrics`, `/infer`, `/infer/stream` (HTTP 501 refusal), auth, correlation_id, multipart + JSON body parsing, Prometheus metrics, lifespan startup. You write only the model wrapper.

## Body shapes

| Shape | Wire | Use for |
|---|---|---|
| `BodyShape.TEXT` | JSON + multipart (text-only fields) | TTS, LLM chat, any text-only adapter |
| `BodyShape.IMAGE` | multipart `frame` file + JSON `frame_b64` | Vision detection, classification, OCR |
| `BodyShape.AUDIO` | multipart `audio` file + JSON `audio_b64` | ASR, audio classification, TTS post-process |
| `BodyShape.GENERIC` | multipart `data` file + JSON `data_b64` | Anything else with binary input |

For non-TEXT shapes, the SDK puts the binary content at `payload[BODY_BYTES_KEY]` (bytes) and merges any `params` JSON into the dict. `BODY_BYTES_KEY` is re-exported from the SDK root — import it rather than hard-coding the literal so future renames don't silently break adapters. Caller-supplied params that shadow this key are rejected with `malformed_input` so collisions surface at the wire, not as silently-overwritten values.

## Streaming adapters

Add `supports_stream=True` to `AdapterApp(...)` and override `AdapterService.handle_stream(websocket)`:

```python
class MyDetector(AdapterService):
    async def handle_stream(self, websocket):
        await websocket.accept()
        # ... §6 protocol ...

app = AdapterApp(
    service=MyDetector(),
    ...
    supports_stream=True,
    stream_max_concurrent=16,
    stream_supports_shared_memory=False,
).fastapi_app
```

The SDK handles auth on the WebSocket upgrade (§6.5 close code 4001 on auth failure) and delegates to your handler. The §6 protocol itself — handshake → frame_meta + binary → result_message — is the adapter's responsibility (YOLOv8 has the reference implementation under `adapters/yolov8/`).

## Constructor reference

```python
AdapterApp(
    # Service — exactly one required:
    service: AdapterService | None,           # eager construction
    service_factory: Callable[[], AdapterService] | None,  # lazy (lifespan startup)

    # Adapter identity (required):
    name: str, version: str,
    vendor: str, license: str,
    tasks_advertised: Sequence[str],

    # Body shape + size cap:
    body_shape: BodyShape = BodyShape.TEXT,
    max_body_bytes: int = 32 * 1024 * 1024,

    # Capabilities metadata (optional, with defaults):
    permissions: Permissions = Permissions(),
    scheduling: Scheduling = Scheduling(),  # default max_inflight=1
    cost: Cost = Cost(),
    model_card_url: str | None = None,
    supported_contract_versions: Sequence[str] = ("1",),

    # Streaming (default off):
    supports_stream: bool = False,
    stream_max_concurrent: int = 0,
    stream_supports_shared_memory: bool = False,

    # Tuning:
    latency_buckets_seconds: tuple[float, ...] = (...),  # Prometheus buckets
    cors_origins: Sequence[str] = ("*",),
)
```

## Real-world examples

`opennvr-adapter-sdk` is the production runtime for the eight adapters shipped in this repo: `adapters/yolov8/` for object detection (`BodyShape.IMAGE` with WebSocket streaming); `adapters/piper/` for text-to-speech (`BodyShape.TEXT` with a custom `/voices` route and inline-audio response); `adapters/whisper/` for speech-to-text (`BodyShape.AUDIO`, multipart decode); `adapters/fast_plate_ocr/` for license-plate text recognition on a pre-cropped plate image (`BodyShape.IMAGE`, designed to chain downstream of YOLOv8); `adapters/insightface/` for face detection plus recognition with a REST face DB; `adapters/blip/` for scene captioning, used by the OpenNVR camera-agent; `adapters/vlm/` for open-vocabulary detection (OWL-ViT v2, detects free-text queries like "red truck"); and `adapters/bytetrack/` for stateful multi-object tracking as a post-processor over an upstream detector's results. Their `main.py` files are non-trivial reference implementations worth reading before authoring your own.

## Versioning

SDK ships with the same major version as the contract. SDK v1.x targets contract v1; a future contract v2 ships SDK v2.x. `AdapterApp.supported_contract_versions` defaults to `["1"]` — bump when you implement multi-version support.

## Why this isn't a "framework"

`AdapterService` is an ABC, not a metaclass. `AdapterApp` is a builder, not a base class. The SDK lives between you and FastAPI — your service code never imports FastAPI directly, but you can still drop down to `app.add_route(...)` for adapter-specific endpoints (see Piper's `/voices` route).

The contract is the source of truth. The SDK is a convenience layer on top of it. If the SDK gets in your way, write the service hand-rolled the way the early reference adapters did before the SDK was extracted — the contract is implementable without it, just more boilerplate.
