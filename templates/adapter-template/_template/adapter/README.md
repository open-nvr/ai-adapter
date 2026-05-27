# __SLUG__ Adapter (Contract v1)

<!-- TODO: replace this opening paragraph with what your adapter actually
     does. Lead with the user value, not the technical detail. Look at
     adapters/yolov8/README.md or adapters/bytetrack/README.md for the
     shape every adapter README follows. -->

Reference implementation of the [AI Adapter Contract v1](../../../open-nvr/docs/AI_ADAPTER_CONTRACT.md)
wrapping <!-- TODO: model name + provenance -->. <!-- TODO: one sentence
about what kind of input it takes and what output it returns. -->

## What it does

<!-- TODO: explain the inference call. A `curl` example with realistic
     input + output is the single most useful piece of this README. -->

```bash
curl -sS http://127.0.0.1:__PORT__/infer \
  -H "Authorization: Bearer $OPENNVR_ADAPTER_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{...}'
```

## Endpoints

| Endpoint | Status | Notes |
|---|---|---|
| `GET /health` | required | auth-exempt |
| `GET /capabilities` | required | <!-- TODO: list your tasks_advertised + key permissions --> |
| `GET /hardware/evaluation` | required | reports load state + adapter-specific diagnostics |
| `GET /metrics` | required | Prometheus exposition incl. `adapter_infer_seconds` |
| `POST /infer` | required | <!-- TODO: multipart? JSON? both? --> |
| `POST /infer/stream` (WS) | <!-- TODO: required / not supported --> | <!-- TODO: protocol notes --> |

## Operational notes

<!-- TODO: bullet points covering anything an operator needs to know:
- model size / download path
- CPU vs GPU posture
- cold-start latency
- input size limits
- per-camera state / TTL / memory bound
- sovereignty implications (network egress for weight download?)
-->

## Run locally

```bash
# Install deps (from the ai-adapter repo root)
uv venv && source .venv/bin/activate
uv sync --extra dev

# TODO: add any extra pip install lines your adapter needs

# Start the service
OPENNVR_ADAPTER_TOKEN=dev-token \
  uv run uvicorn adapters.__DIR_NAME__.main:app --host 0.0.0.0 --port __PORT__
```

## Run with Docker

```bash
# Build (from ai-adapter repo root)
docker build -f adapters/__DIR_NAME__/Dockerfile -t opennvr/__IMAGE_NAME__:1.0.0 .

# Run
docker run --rm -p __PORT__:__PORT__ \
  -e OPENNVR_ADAPTER_TOKEN=dev-token \
  opennvr/__IMAGE_NAME__:1.0.0
```

Pre-built images are published to `ghcr.io/open-nvr/__IMAGE_NAME__`
on every tagged release — see
[`.github/workflows/publish-images.yml`](../../.github/workflows/publish-images.yml).
<!-- TODO: append your adapter to the workflow's matrix once you're
     ready to publish. -->

## Contract conformance

```bash
python -m conformance http://localhost:__PORT__ --token $OPENNVR_ADAPTER_TOKEN
```

The conformance suite exercises every required endpoint, the `/health`
state machine, the §7 failure envelope, and the body parser for your
declared `BodyShape`. A green result means the adapter will register
cleanly with KAI-C.

## Tests

```bash
pytest tests/test___DIR_NAME___service.py
```

<!-- TODO: list what the test suite covers — load lifecycle, malformed-
     input rejection, per-camera state if applicable, fingerprint
     stability across loads, etc. -->

## Why this model

<!-- TODO: justify the design choice. Why this model over alternatives?
     What trade-offs do operators inherit? What ships better than the
     thing they'd write in a weekend? This section is where the project
     learns from your adapter even after you've moved on. -->
