# ByteTrack Multi-Object Tracking Adapter (Contract v1)

Reference implementation of the [AI Adapter Contract v1](../../../open-nvr/docs/AI_ADAPTER_CONTRACT.md)
wrapping [supervision](https://github.com/roboflow/supervision)'s
ByteTrack tracker. Takes a frame's detections in, returns the same
detections with persistent `track_id` fields populated. Stateful per
camera so tracks don't bleed across cameras in a multi-camera install.

This is the canonical *post-processing* adapter — it doesn't run a
model itself, it composes with an upstream detector by chaining
through KAI-C. Pair it with YOLOv8, YOLOv11, fast-plate-ocr, or any
detection-shaped adapter; the input contract is the same.

## What it does

POST a list of detections, get the same list back with `track_id`
assigned per detection.

```bash
curl -sS http://127.0.0.1:9007/infer \
  -H "Authorization: Bearer $OPENNVR_ADAPTER_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "camera_id": "front-door",
    "detections": [
      {"label": "person", "confidence": 0.92,
       "bbox": {"x": 0.10, "y": 0.20, "w": 0.30, "h": 0.50}},
      {"label": "person", "confidence": 0.88,
       "bbox": {"x": 0.55, "y": 0.20, "w": 0.20, "h": 0.40}}
    ],
    "frame_dimensions": {"w": 1920, "h": 1080}
  }'

# {"model_name":"bytetrack","inference_ms":3,
#  "result":{"detections":[
#     {"label":"person","confidence":0.92,"bbox":{...},"track_id":1},
#     {"label":"person","confidence":0.88,"bbox":{...},"track_id":2}],
#   "frame_dimensions":{"w":1920,"h":1080}}}
```

On the next frame for the same `camera_id`, ByteTrack reuses the
existing track IDs and assigns new IDs only when a detection can't be
matched to any active track. Detections too low-confidence to activate
a track come back with `track_id: null` — that's how ByteTrack signals
"I see this box but it's not confidently part of an existing track."

## Endpoints

| Endpoint | Status | Notes |
|---|---|---|
| `GET /health` | required | auth-exempt |
| `GET /capabilities` | required | `tasks_advertised=["multi_object_tracking"]`, `gpu=false`, `fair_queuing=per_camera` |
| `GET /hardware/evaluation` | required | reports active-tracker count |
| `GET /metrics` | required | Prometheus exposition incl. `adapter_infer_seconds` |
| `POST /infer` | required | JSON only (`BodyShape.TEXT`) |
| `POST /infer/stream` (WS) | not supported | returns 501 — tracking is per-frame request/response |

## Input shape

```jsonc
{
  "camera_id": "front-door",        // REQUIRED. Non-empty string.
                                    // ByteTrack state is keyed by this.
  "detections": [                   // REQUIRED. May be [] (still ticks tracker).
    {
      "label": "person",            // Used for class consistency in tracks.
      "confidence": 0.92,
      "bbox": {                     // Normalized [0,1] coordinates (contract §5.1).
        "x": 0.10,
        "y": 0.20,
        "w": 0.30,
        "h": 0.50
      }
    }
  ],
  "frame_dimensions": {             // Optional. Defaults to (1, 1) — fine when all
    "w": 1920,                      // bboxes are already normalized.
    "h": 1080
  },
  "tracker_config": {               // Optional per-call tuning. Defaults below.
    "track_activation_threshold": 0.25,
    "lost_track_buffer": 30,
    "minimum_matching_threshold": 0.8,
    "frame_rate": 30
  }
}
```

If a call changes `tracker_config` for an existing camera, the
tracker is rebuilt fresh — track IDs reset. The honest alternative
(silently apply the new config to the existing tracker) would
desync ByteTrack's internal Kalman state.

## Operational notes

- **CPU-only.** ByteTrack runs on numpy; there's no GPU acceleration
  in supervision's implementation. A single tracker update on
  1000-detection frames is ~3 ms — well under the per-frame budget
  of any realistic inference loop.
- **Per-camera state with TTL eviction.** Each `camera_id` gets its
  own ByteTrack instance. Cameras with no inference call for
  `BYTETRACK_IDLE_TTL_SECONDS` (default 300s) get evicted to keep
  memory bounded. Operators with many transient camera IDs (e.g.
  rotating mobile cameras) can shorten the TTL via the env var.
- **No model weights.** supervision's ByteTrack is pure-algorithmic
  (Kalman filter + Hungarian assignment + IoU matching). The
  fingerprint surfaced via `/capabilities` is `supervision:<version>`
  so KAI-C's drift detection still fires if the supervision pin
  changes between deployments.
- **Track ID stability.** ByteTrack IDs are monotonically increasing
  integers per-camera. They reset on tracker eviction (idle TTL) and
  on adapter restart. Downstream consumers that need globally-stable
  IDs across restarts should layer their own UUID assignment on top.

## Run locally

```bash
# Install deps (from the ai-adapter repo root)
uv venv && source .venv/bin/activate
uv sync --extra dev

pip install "supervision>=0.21,<1.0" numpy

# Start the service
OPENNVR_ADAPTER_TOKEN=dev-token \
  uv run uvicorn adapters.bytetrack.main:app --host 0.0.0.0 --port 9007
```

## Run with Docker

```bash
# Build (from ai-adapter repo root)
docker build -f adapters/bytetrack/Dockerfile -t opennvr/bytetrack-adapter:1.0.0 .

# Run
docker run --rm -p 9007:9007 \
  -e OPENNVR_ADAPTER_TOKEN=dev-token \
  opennvr/bytetrack-adapter:1.0.0
```

Pre-built images are published to `ghcr.io/open-nvr/bytetrack-adapter`
on every tagged release — see
[`.github/workflows/publish-images.yml`](../../.github/workflows/publish-images.yml).

## Contract conformance

```bash
python -m conformance http://localhost:9007 --token $OPENNVR_ADAPTER_TOKEN
```

The conformance suite exercises every required endpoint, the `/health`
state machine, the §7 failure envelope, and the JSON body parser.
WebSocket streaming returns 501 as advertised — `supports_stream=False`
in the capability advertisement.

## Tests

```bash
pytest tests/test_bytetrack_service.py
```

The test suite covers the parser (malformed input rejection), the
per-camera state isolation, tracker rebuild on config change, idle-TTL
eviction, and that the input/output order of detections is preserved.

## Why ByteTrack

The OpenNVR contract assumes detection is per-frame — every `/infer`
call is independent. That's fine for "is there a person in this
frame?" but loses information for "this is the same person I saw five
seconds ago." Tracking restores per-object continuity, which unlocks:

- Dwell-time analytics (loitering-detection example would use this).
- Per-track state machines (package-delivery: arrive → linger →
  disappear).
- Track-stable alert deduplication (don't fire "person detected" 60
  times for the same person walking past).
- Re-identification across temporary occlusion (someone walks behind
  a car for 2 seconds and comes out the other side — same track ID).

ByteTrack was chosen over DeepSORT / BoT-SORT / OC-SORT for v0.2 for
three reasons: it's pure-CPU (no extra model weights), supervision
ships a vetted implementation (small attack surface), and the BYTE
matching algorithm handles low-confidence detections gracefully — a
real-world need for cameras that produce noisy boxes on edge lighting.
