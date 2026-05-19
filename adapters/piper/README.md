# Piper TTS Adapter (Contract v1)

Reference implementation of the [AI Adapter Contract v1](../../../open-nvr/docs/AI_ADAPTER_CONTRACT.md) wrapping Piper neural TTS voices.

This is the **first adapter ported to the contract** (A2.1 milestone). It serves as the canonical example for community contributors: ~450 lines of code, six mandatory contract endpoints, bearer-token auth, correlation-id wiring, Prometheus metrics, and a passing conformance run.

## What it does

POST text in JSON, get a WAV file URI back. The underlying [Piper](https://github.com/rhasspy/piper) library produces phone-quality neural TTS on CPU at faster-than-realtime speed (~25 MB models).

## Endpoints

| Endpoint | Status | Notes |
|---|---|---|
| `GET /health` | required | auth-exempt; `loading` → `ok` → `error` |
| `GET /capabilities` | required | includes sha256 model fingerprint |
| `GET /hardware/evaluation` | required | CPU verdict |
| `GET /metrics` | required | Prometheus exposition |
| `POST /infer` | required | JSON: `{"text": "...", "voice": "...", "length_scale": 1.0}` |
| `POST /infer/stream` | refused | HTTP 501 — Piper is one-shot, not real-time |
| `GET /voices` | extra | lists installed `.onnx` voices |

## Run locally

```bash
# Install deps
uv sync --extra tts

# Download a voice (one-time)
mkdir -p model_weights/piper
cd model_weights/piper
curl -LO https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/libritts/high/en_US-libritts-high.onnx
curl -LO https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/libritts/high/en_US-libritts-high.onnx.json
cd ../..

# Start the service
OPENNVR_ADAPTER_TOKEN=dev-token \
  uvicorn adapters.piper.main:app --host 0.0.0.0 --port 9001
```

## Run with Docker

```bash
docker build -f adapters/piper/Dockerfile -t opennvr/piper-adapter:1.0.0 .

docker run --rm -p 9001:9001 \
  -e OPENNVR_ADAPTER_TOKEN=$(openssl rand -hex 16) \
  -v $(pwd)/model_weights/piper:/voices:ro \
  -v $(pwd)/audio:/audio \
  opennvr/piper-adapter:1.0.0
```

## Verify conformance

```bash
python -m conformance http://localhost:9001 --token $OPENNVR_ADAPTER_TOKEN
```

A green run means KAI-C will accept this adapter.

## Try it

Per §3.5 of the contract, adapters MUST accept `multipart/form-data`
and MAY also accept `application/json`. Piper accepts both:

```bash
# Multipart (§3.5 canonical) — params field carries the JSON payload
curl -X POST http://localhost:9001/infer \
  -H "Authorization: Bearer $OPENNVR_ADAPTER_TOKEN" \
  -H "X-Correlation-Id: $(uuidgen)" \
  -F 'params={"text": "Hello from the OpenNVR Piper adapter."};type=application/json'

# JSON (convenience — equivalent payload, no multipart framing)
curl -X POST http://localhost:9001/infer \
  -H "Authorization: Bearer $OPENNVR_ADAPTER_TOKEN" \
  -H "Content-Type: application/json" \
  -H "X-Correlation-Id: $(uuidgen)" \
  -d '{"text": "Hello from the OpenNVR Piper adapter."}'
```

Response:

```json
{
  "status": "ok",
  "model_name": "en_US-libritts-high",
  "model_version": "piper-tts/en_US-libritts-high",
  "inference_ms": 412,
  "result": {
    "audio_uri": "opennvr://audio/tts/c4f3a1...wav",
    "duration_seconds": 1.85,
    "sample_rate": 22050,
    "voice": "en_US-libritts-high",
    "text_length": 38
  }
}
```

## Layout

```
adapters/piper/
├── main.py          FastAPI app + routes
├── service.py       PiperService — contract semantics around legacy PiperAdapter
├── auth.py          AuthAndCorrelationMiddleware — §3.8 wire spec
├── metrics.py       Prometheus exposition
├── Dockerfile       Self-contained image; versions track pyproject.toml
└── README.md        you are here
```

## Why a new service instead of refactoring the legacy monolith

The existing `app/main.py` bundles all 8 adapters into one FastAPI service. Per the contract, "one adapter wraps one model" — so each adapter gets its own container with its own declared permissions. The legacy monolith stays running and untouched until every adapter has a contract-compliant service (A2.2 through A2.4); then it retires. See §13 of the design doc for the migration roadmap.

## Tests

```bash
pytest tests/test_piper_service.py tests/test_conformance_against_piper.py
```

28 tests covering every endpoint, the auth grace window, correlation-id echo, and the full conformance run.
