# opennvr-adapter-sdk

The boilerplate-free way to write an [AI Adapter Contract v1](../../open-nvr/docs/AI_ADAPTER_CONTRACT.md) service.

A minimal adapter is now **~30 lines of FastAPI**, per §3.7 of the contract design. The SDK provides:

- `AdapterService` — the ABC every adapter implements (4 abstract methods)
- `AdapterApp` — wraps the service in a FastAPI app with the six mandatory endpoints, auth, correlation_id, Prometheus metrics, multipart + JSON body parsing
- `ServiceError` — typed §7 failure envelope
- Re-exports of every contract Pydantic type, so adapter authors get one import line

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

`opennvr-adapter-sdk` is currently the production runtime for:

- `adapters/piper/` — TTS adapter (`BodyShape.TEXT`, no streaming)
- (A2.3b) `adapters/yolov8/` — object detection (`BodyShape.IMAGE`, streaming via WS)
- (A2.3c) `adapters/whisper/` — ASR (`BodyShape.AUDIO`, no streaming)

Read those `main.py` files for non-trivial reference implementations.

## Versioning

SDK ships with the same major version as the contract. SDK v1.x targets contract v1; a future contract v2 ships SDK v2.x. `AdapterApp.supported_contract_versions` defaults to `["1"]` — bump when you implement multi-version support.

## Why this isn't a "framework"

`AdapterService` is an ABC, not a metaclass. `AdapterApp` is a builder, not a base class. The SDK lives between you and FastAPI — your service code never imports FastAPI directly, but you can still drop down to `app.add_route(...)` for adapter-specific endpoints (see Piper's `/voices` route).

The contract is the source of truth. The SDK is a convenience layer on top of it. If the SDK gets in your way, write the service hand-rolled the way A2.1 / A2.2 / A2.3-prep did initially — the contract is implementable without it, just more boilerplate.
