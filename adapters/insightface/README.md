# InsightFace Recognition Adapter (Contract v1)

SDK-based reference implementation of the
[AI Adapter Contract v1](https://github.com/open-nvr/open-nvr/blob/main/docs/AI_ADAPTER_CONTRACT.md)
wrapping [InsightFace](https://github.com/deepinsight/insightface) for face
detection, recognition, and embedding extraction. Built for the OpenNVR
Smart Doorbell example.

## What it does

POST a JPEG / PNG image with a `task` parameter, get a face result back.
Four tasks supported:

| `task` | Result |
|---|---|
| `face_detection` (default) | List of detected faces — bbox, confidence, optional landmarks / age / gender |
| `face_recognition` | Detect the highest-confidence face and look it up in the on-disk face DB. Returns either a match (person_id / name / category / similarity) or `recognized: false` |
| `face_embedding` | Return the raw 512-d normalised embedding for the highest-confidence face. Use this if you want to do your own matching outside the adapter |
| `face_identify` | Look up **every** face in the frame (a classroom, a shop floor). One entry per face: bbox, confidence, recognized, person_id / name / category, similarity, second_similarity |

### `face_identify`

Params (all optional): `threshold` (as for `face_recognition`), `min_face_px`
(drop faces whose shorter bbox side is smaller; default 0), `max_faces` (keep
the most confident N, default 100, at most 500), `category` (match only people
in that category), `match` (default true; `false` runs only the detector and
looks nobody up, for a page checking webcam frames while somebody enrols).
Each face also carries `landmarks`: the detector's five points (eyes, nose,
mouth corners), which say how the head is turned.

```json
{
  "result": {
    "task": "face_identify",
    "faces": [
      {"bbox": [812, 300, 870, 372], "confidence": 0.91, "recognized": true,
       "person_id": "7f0c…", "name": "Asha Rao", "category": "attendance",
       "similarity": 0.63, "second_similarity": 0.21},
      {"bbox": [140, 410, 182, 462], "confidence": 0.74, "recognized": false,
       "person_id": null, "name": null, "category": null,
       "similarity": 0.38, "second_similarity": 0.30}
    ],
    "face_count": 2, "recognized_count": 1, "detected_count": 3,
    "dropped": {"too_small": 1, "over_limit": 0},
    "frame": {"w": 1920, "h": 1080},
    "threshold": 0.5, "registered_faces": 140
  }
}
```

Boxes are pixels in the posted frame; `frame` lets the caller normalise them.
On a miss, `similarity` is the nearest person's score but the adapter does not
say who: a near miss is not an identification. `second_similarity` is the
runner-up person's score, so a caller can demand a margin over the next-best
person.

The whole DB is matched in one matrix product per frame, so the per-face cost is
the ArcFace embedding, not the lookup.

What a crowded frame costs on CPU (1080p, 54 faces, det size 640, measured
2026-11-04 on an 8-thread laptop CPU of 2013 vintage, in this image):

| Pack | Detector | Recogniser, per face | Whole frame |
|---|---|---|---|
| `buffalo_l` (default) | ~300 ms | ~170 ms | ~10 s |
| `buffalo_s` | ~35 ms | ~17 ms | ~1 s |

Matching those 54 faces against 2,000 people × 10 samples took 26 ms. The cost
grows with the number of faces, not the size of the DB. Plan
the call rate from the faces in the room: a camera on 50 faces every 150 s is
~7 % of one `buffalo_l` instance on that CPU. `OPENNVR_INSIGHTFACE_MODEL=buffalo_s`
is ten times cheaper but its detector and recogniser are both much weaker
(InsightFace's model zoo ranks it far below `buffalo_l`); prefer more instances
or a longer interval over the smaller pack.

### Detector size

`OPENNVR_INSIGHTFACE_DET_SIZE` (default `640`, a multiple of 32 in 160–2048)
sets the detector's input size. At 640 a 1080p frame is scaled down about 3×,
so small faces at the back of a room are missed; `1280` finds them at roughly
4× the detection cost. A bad value fails the load, with the reason on `/health`.

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
    "confidence": 0.91,
    "threshold": 0.5
  }
}
```

A miss keeps its evidence: `similarity` is the nearest enrolled face
(below the threshold, and deliberately unnamed), `confidence` is the
detector's score for the face, and `face_bbox` its size. A caller that
sees `similarity: 0.45` against a threshold of 0.5 on a 30-pixel face
knows to wait for a better frame rather than announce a stranger.

```json
{
  "result": {
    "task": "face_recognition",
    "recognized": false,
    "face_bbox": [410, 92, 442, 131],
    "confidence": 0.63,
    "similarity": 0.45,
    "message": "no match above threshold 0.5",
    "threshold": 0.5,
    "registered_faces": 3
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
