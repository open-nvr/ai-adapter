# Whisper ASR Adapter (Contract v1)

Reference implementation of the [AI Adapter Contract v1](../../../open-nvr/docs/AI_ADAPTER_CONTRACT.md) wrapping faster-whisper / CTranslate2. **Third adapter migrated** (A2.3-prep, after A2.1 Piper and A2.2 YOLOv8) — validates the contract on a third modality (audio) and grounds the SDK design that A2.3 will extract.

## What it does

POST an audio clip (WAV/MP3/FLAC/M4A/OGG — anything ffmpeg decodes), get a transcript back. Returns the §5.3 ASR convention:

```json
{
  "transcript": "the room is clear",
  "language": "en",
  "segments": [
    {"start_ms": 0, "end_ms": 1800, "text": "the room is clear"}
  ],
  "language_confidence": 0.97,
  "duration_seconds": 1.85,
  "translated_to_english": false,
  "model": "whisper-base"
}
```

Two tasks:
- **`audio_transcription`** — preserves the source language (auto-detected or set via `language` param)
- **`audio_translation`** — transcribes and translates to English

## Endpoints

| Endpoint | Status | Notes |
|---|---|---|
| `GET /health` | required | auth-exempt; `loading` → `ok` → `error` |
| `GET /capabilities` | required | sha256 fingerprint computed live; `tasks_advertised=["audio_transcription","audio_translation"]`; `permissions.gpu=true` |
| `GET /hardware/evaluation` | required | reports device (cuda/cpu) + compute_type |
| `GET /metrics` | required | Prometheus exposition; latency buckets tuned for ASR (10ms - 60s) |
| `POST /infer` | required | multipart (`audio` file) or JSON (`audio_b64`) |
| `POST /infer/stream` | refused | HTTP 501 — streaming ASR (overlap windows, partial-result emission, VAD gating) lands in A2.3b |

## Operational notes

### Pre-caching models under `sovereignty=local_only` (contract honesty)

The adapter declares `permissions.network_egress=[]` in `/capabilities` — i.e., it claims to make no outbound network calls. That's true once a Whisper model is cached on disk. But `faster-whisper` will fetch weights from HuggingFace on first load if `{download_root}/models--Systran--faster-whisper-{model_size}/` doesn't exist.

**Under KAI-C `AI_SOVEREIGNTY=local_only`**: the adapter's first-run download would violate the no-egress declaration. KAI-C only checks the *declared* permissions at registration; it doesn't intercept actual network calls. So:

- **Operators MUST pre-cache** model weights before deploying under `local_only`:
  ```bash
  # On a machine with internet, run once:
  python -c "from faster_whisper import WhisperModel; WhisperModel('base', download_root='./model_weights/whisper')"
  # Then ship the resulting model_weights/whisper/ to the air-gapped host.
  ```
- Under `federated` or `cloud_allowed`, the first-run download is acceptable but the declaration should be updated to `permissions.network_egress=["huggingface.co"]` for honesty. A follow-up commit will surface this as a config flag.

### CUDA vs CPU

- `tiny` and `base` are usable on CPU (~2x realtime, ~5x realtime respectively)
- `small`, `medium` need GPU for realistic latency
- `large-v3` is GPU-only in practice

The adapter declares `permissions.gpu=true` so KAI-C asks operator approval at registration — the operator can deny GPU and the adapter falls back to CPU automatically.

## Streaming deferred

Streaming ASR isn't just "send chunks." It needs:
- Overlap-window decoding so word boundaries don't get cut mid-token
- Partial-result emission with revision support (early words may flip after context arrives)
- VAD-gated emission so silence doesn't produce blank result messages
- Backpressure when the decoder falls behind the audio stream

That's its own design milestone. v1 advertises `endpoints.infer_stream.supported = false`; KAI-C never tries to upgrade.

## Run locally

```bash
# Install deps
uv sync --extra stt

# faster-whisper auto-downloads models on first load. Or pre-cache:
mkdir -p model_weights/whisper

# Start the service
OPENNVR_ADAPTER_TOKEN=dev-token \
WHISPER_MODEL_SIZE=base \
  uvicorn adapters.whisper.main:app --host 0.0.0.0 --port 9003
```

## Run with Docker

```bash
docker build -f adapters/whisper/Dockerfile -t opennvr/whisper-adapter:1.0.0 .

# CPU (base ~2x realtime; small ~5x; large ~30x slower than CUDA)
docker run --rm -p 9003:9003 \
  -e OPENNVR_ADAPTER_TOKEN=$(openssl rand -hex 16) \
  -e WHISPER_MODEL_SIZE=base \
  -v $(pwd)/model_weights/whisper:/weights:ro \
  opennvr/whisper-adapter:1.0.0

# GPU passthrough
docker run --rm --gpus all -p 9003:9003 \
  -e OPENNVR_ADAPTER_TOKEN=$(openssl rand -hex 16) \
  -e WHISPER_MODEL_SIZE=large-v3 \
  -v $(pwd)/model_weights/whisper:/weights:ro \
  opennvr/whisper-adapter:1.0.0
```

## Verify conformance

```bash
python -m conformance http://localhost:9003 --token $OPENNVR_ADAPTER_TOKEN
```

The conformance kit automatically detects `modalities_in: ["audio"]` in /capabilities and drives `/infer` with multipart-audio (added in A2.3-prep alongside the image-modality multipart from A2.2).

## Try it

```bash
# Multipart (§3.5 canonical)
curl -X POST http://localhost:9003/infer \
  -H "Authorization: Bearer $OPENNVR_ADAPTER_TOKEN" \
  -H "X-Correlation-Id: $(uuidgen)" \
  -F "audio=@/path/to/clip.wav" \
  -F 'params={"task": "audio_transcription", "language": "en", "beam_size": 5};type=application/json'

# JSON (base64 fallback)
B64=$(base64 -i /path/to/clip.wav)
curl -X POST http://localhost:9003/infer \
  -H "Authorization: Bearer $OPENNVR_ADAPTER_TOKEN" \
  -H "Content-Type: application/json" \
  -d "{\"audio_b64\": \"$B64\", \"task\": \"audio_transcription\"}"
```

## Layout

```
adapters/whisper/
├── main.py          FastAPI app — six contract endpoints
├── service.py       WhisperService — wraps legacy WhisperAdapter, §5.3 translation
├── auth.py          Bearer-token + correlation_id middleware
├── metrics.py       Prometheus exposition (ASR-tuned histogram buckets)
├── Dockerfile       Self-contained image; CPU + GPU compatible
└── README.md        you are here
```

## Tests

```bash
pytest tests/test_whisper_service.py tests/test_conformance_against_whisper.py
```

27 tests: 22 service tests (HTTP /infer multipart + JSON, §5.3 shape, translation forces English, params validation, auth, correlation_id) + 5 conformance tests (full kit run including the new audio-multipart path).

## What's next

This is the **third grounded data point** for A2.3 SDK extraction. With Piper + YOLOv8 + Whisper migrated, the duplicated boilerplate (auth.py, metrics.py, service.py skeleton) is concrete enough to extract into `opennvr-adapter-sdk` without speculative API design.
