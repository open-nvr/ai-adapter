# BLIP Scene-Caption Adapter (Contract v1)

Reference implementation of the [AI Adapter Contract v1](https://github.com/open-nvr/open-nvr/blob/main/docs/AI_ADAPTER_CONTRACT.md)
wrapping Salesforce's BLIP image-captioning model. Drops a JPEG / PNG in,
gets a one-sentence natural-language description of the scene back. Useful
for the camera-agent example's "describe what you see" tool, and for any
downstream consumer that wants a semantic summary alongside the structured
detection output from YOLOv8 or InsightFace.

## What it does

POST a frame, get a caption back. Returns the §5.x `SceneCaptionResult`
shape with a free-form `caption` string. Caption length and decoding
parameters are tuned by the model card; this adapter exposes the
unmodified BLIP `image-captioning-base` defaults.

```bash
curl -sS http://127.0.0.1:9006/infer \
  -H "Authorization: Bearer $OPENNVR_ADAPTER_TOKEN" \
  -F frame=@porch.jpg
# {"model_name":"blip-scene-caption","inference_ms":421,
#  "result":{"caption":"a brown package sitting on the front steps of a house"}}
```

## Endpoints

| Endpoint | Status | Notes |
|---|---|---|
| `GET /health` | required | auth-exempt |
| `GET /capabilities` | required | sha256 fingerprint, `tasks_advertised=["scene_caption"]`, `gpu=false`, `fair_queuing=per_camera` |
| `GET /hardware/evaluation` | required | reports GPU detection (CUDA auto-selected when available) |
| `GET /metrics` | required | Prometheus exposition incl. `adapter_infer_seconds` |
| `POST /infer` | required | multipart (`frame` file) or JSON (`frame_b64`) |
| `POST /infer/stream` (WS) | not supported | returns 501 — captioning is single-shot, not streaming |

## Operational notes

- **First-run weights download.** The HuggingFace `transformers` library
  fetches `Salesforce/blip-image-captioning-base` (~990 MB) on first
  `load()`. Subsequent restarts read from the local HuggingFace cache.
  Operators with strict sovereignty either pre-bake the model into the
  image or run a private HuggingFace mirror.
- **CPU is the default; GPU is opt-in.** The service uses
  `torch.cuda.is_available()` at load time — pass NVIDIA devices via the
  standard Docker `--gpus all` flag and torch picks them up. There are no
  adapter-specific device knobs.
- **Inference latency.** ~400 ms / image on a modern CPU,
  ~50 ms / image on an entry-level NVIDIA GPU. The `fair_queuing=per_camera`
  capability lets KAI-C round-robin frames across cameras under load.
- **Image size cap.** 8 MiB per request, matching the YOLOv8 adapter.
  Larger frames get a `malformed_input` rejection before reaching the
  model — keeps memory predictable under churn.

## Run locally

```bash
# Install deps (from the ai-adapter repo root)
uv venv && source .venv/bin/activate
uv sync --extra blip --extra cpu      # ~3 GB; pulls torch + transformers

# Start the service
OPENNVR_ADAPTER_TOKEN=dev-token \
  uv run uvicorn adapters.blip.main:app --host 0.0.0.0 --port 9006
```

The first request triggers the model load — expect a ~30-second pause
on the first `/infer` while weights stream from HuggingFace into the
local cache. `/health` returns `loading` during that window.

## Run with Docker

```bash
# Build (from ai-adapter repo root)
docker build -f adapters/blip/Dockerfile -t opennvr/blip-adapter:1.0.0 .

# Run
docker run --rm -p 9006:9006 \
  -e OPENNVR_ADAPTER_TOKEN=dev-token \
  -v $(pwd)/model_weights/blip:/root/.cache/huggingface:rw \
  opennvr/blip-adapter:1.0.0
```

Pre-built images are published to `ghcr.io/open-nvr/blip-adapter` on
every tagged release — see
[`.github/workflows/publish-images.yml`](../../.github/workflows/publish-images.yml).

## Contract conformance

```bash
python -m conformance http://localhost:9006 --token $OPENNVR_ADAPTER_TOKEN
```

The conformance suite exercises every required endpoint, the `/health`
state machine across load → ok / load → error, the §7 failure envelope,
and the multipart + JSON body parsers. A green result means the adapter
will register cleanly with KAI-C.

## Tests

```bash
pytest tests/test_blip_service.py
```

The test suite covers caption shaping, oversize-frame rejection, the
loading-state behaviour, and the `__getattr__` shim that keeps the
`main._service` test fixture working.

## Why a hosted captioning model

For the camera-agent example, BLIP fills a different role than YOLOv8 or
InsightFace: structured detection answers "what objects?" and
"whose face?", BLIP answers "what's happening?" in natural language.
That semantic summary is what makes a voice LLM grounded — without it
the model would have to infer scene context from coordinate lists.

The choice of BLIP over a larger VLM (LLaVA, Qwen2-VL, Florence-2) is
deliberate: BLIP runs CPU-friendly at sub-second latencies, which keeps
the camera-agent loop interactive on homelab hardware. The contract
doesn't pin this choice — operators wanting richer captions can swap in
a per-adapter Dockerfile wrapping any captioning model that takes
`(image) → text`.
