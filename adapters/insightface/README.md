# InsightFace Recognition Adapter (Contract v1)

SDK-based reference implementation of the
[AI Adapter Contract v1](../../../open-nvr/docs/AI_ADAPTER_CONTRACT.md)
wrapping [InsightFace](https://github.com/deepinsight/insightface) for face
detection, recognition, and embedding extraction. Built for the OpenNVR
Smart Doorbell example.

## What it does

POST a JPEG / PNG image with a `task` parameter, get a face result back.
Three tasks supported:

| `task` | Result |
|---|---|
| `face_detection` (default) | List of detected faces — bbox, confidence, optional landmarks / age / gender |
| `face_recognition` | Detect the highest-confidence face and look it up in the on-disk face DB. Returns either a match (person_id / name / category / similarity) or `recognized: false` |
| `face_embedding` | Return the raw 512-d normalised embedding for the highest-confidence face. Use this if you want to do your own matching outside the adapter |

```bash
# Detection — who's at the door?
curl -X POST http://localhost:9005/infer \
  -H "Authorization: Bearer $OPENNVR_ADAPTER_TOKEN" \
  -F "frame=@doorbell-snap.jpg" \
  -F 'params={"task":"face_detection"}'

# Recognition — is it someone we know?
curl -X POST http://localhost:9005/infer \
  -H "Authorization: Bearer $OPENNVR_ADAPTER_TOKEN" \
  -F "frame=@doorbell-snap.jpg" \
  -F 'params={"task":"face_recognition","threshold":0.5}'
```

```json
{
  "result": {
    "task": "face_recognition",
    "recognized": true,
    "person_id": "alice",
    "name": "Alice Smith",
    "category": "family",
    "similarity": 0.9234,
    "face_bbox": [120, 80, 240, 240],
    "threshold": 0.5
  }
}
```

## Face DB CRUD

Beyond the contract-mandated endpoints, the adapter exposes four CRUD routes
for the local face DB. The Smart Doorbell example uses these to enroll
family members.

| Endpoint | Body | Description |
|---|---|---|
| `POST /faces/register` | multipart: `frame` (file), `person_id`, `name`, `category`, `metadata` (JSON string) | Add or update a registered face. Idempotent on `person_id` — re-registering overwrites the embedding (useful after a haircut, new glasses) |
| `GET /faces?category=family` | — | List registered faces; optional category filter |
| `GET /faces/{person_id}` | — | Get one face's metadata |
| `DELETE /faces/{person_id}` | — | Remove a registered face |

All four are protected by the same `Authorization: Bearer` token as the
contract endpoints.

Registration only stores the **embedding vector** (512 floats), category,
display name, and operator-provided metadata. The raw face image is *not*
persisted by the adapter — homelab privacy posture by default.

```bash
# Enroll Alice
curl -X POST http://localhost:9005/faces/register \
  -H "Authorization: Bearer $OPENNVR_ADAPTER_TOKEN" \
  -F "frame=@alice.jpg" \
  -F "person_id=alice" \
  -F "name=Alice Smith" \
  -F "category=family"

# List family members
curl http://localhost:9005/faces?category=family \
  -H "Authorization: Bearer $OPENNVR_ADAPTER_TOKEN"
```

## Contract conformance

| Endpoint | Status | Notes |
|---|---|---|
| `GET /health` | required | reports loading vs ok |
| `GET /capabilities` | required | sha256 fingerprint of the recognition ONNX file; `tasks_advertised=["face_detection","face_recognition","face_embedding"]`; `permissions.gpu=false` |
| `GET /hardware/evaluation` | required | reports CPU verdict + registered-face count |
| `GET /metrics` | required | Prometheus exposition with per-adapter latency buckets |
| `POST /infer` | required | multipart (`frame` file) or JSON (`frame_b64`) |
| `POST /infer/stream` | refused | HTTP 501 — face recognition is event-driven (one call per detected face from upstream), not frame-rate |

## Operational notes

### Persistence

The face DB lives at `OPENNVR_INSIGHTFACE_FACE_DB` (default
`model_weights/insightface_faces.json`). Survives container restarts when
that path is mounted from a host volume. Set the env var to empty (`""`)
for in-memory only — useful for ephemeral test deployments.

### Model swap

InsightFace ships multiple model packs — `buffalo_l` (default, most
accurate), `buffalo_s` (smaller / faster), `antelopev2` (legacy). Swap via
the `OPENNVR_INSIGHTFACE_MODEL` env var:

```bash
OPENNVR_INSIGHTFACE_MODEL=buffalo_s \
  uvicorn adapters.insightface.main:app --port 9005
```

The `/capabilities` fingerprint changes after the swap — KAI-C surfaces
that as an `adapter.fingerprint_mismatch` audit event, which is the
intended behaviour.

### Sovereignty

`permissions.network_egress=[]` — InsightFace runs entirely on-device after
the one-time model-pack download on first `prepare()`. Pre-warm the cache
in your build pipeline if you're running in air-gapped mode:

```bash
docker run --rm -v $(pwd)/insightface-cache:/root/.insightface \
  opennvr/insightface-adapter:local \
  python -c "from insightface.app import FaceAnalysis; FaceAnalysis(name='buffalo_l').prepare(ctx_id=-1)"
```

Then mount the cache at runtime.

## Try it

### Local

```bash
# from the ai-adapter repo root
uv sync --extra face --extra dev
OPENNVR_ADAPTER_TOKEN=secret \
  uv run uvicorn adapters.insightface.main:app --host 0.0.0.0 --port 9005
```

### Docker

```bash
docker build -f adapters/insightface/Dockerfile -t opennvr/insightface-adapter:local .
docker run --rm -p 9005:9005 \
  -e OPENNVR_ADAPTER_TOKEN=$(openssl rand -hex 16) \
  -v $(pwd)/face-db:/data \
  opennvr/insightface-adapter:local
```

### Conformance check

```bash
uv run python -m conformance http://localhost:9005 --token $OPENNVR_ADAPTER_TOKEN
```

## Tests

```bash
uv run pytest tests/test_insightface_service.py tests/test_conformance_against_insightface.py -v
```

All tests stub the `insightface` package so no model weights are downloaded
in CI. 31 tests covering load lifecycle, the three task paths, error
envelopes, threshold validation, and conformance.

## Layout

```
adapters/insightface/
├── main.py         SDK glue + /faces/* CRUD routes
├── service.py      InsightFaceService — AdapterService implementation
├── face_db.py      In-memory + JSON-persistence face DB
├── Dockerfile      CPU-only multi-stage build
├── README.md       you are here
└── __init__.py
```

## License

This adapter wrapper is **AGPL-3.0** (same as the parent ai-adapter repo).
InsightFace itself is **MIT-licensed**; model packs (`buffalo_l`,
`buffalo_s`, `antelopev2`) ship under InsightFace's own
[non-commercial research license](https://github.com/deepinsight/insightface/tree/master/model_zoo) —
if you're deploying for revenue-generating use, check the model-pack
licence first.
