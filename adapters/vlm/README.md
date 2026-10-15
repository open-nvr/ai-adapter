# vlm — open-vocabulary detection adapter

**Detect whatever you can describe.** Where YOLOv8 detects a fixed set of
COCO classes, this adapter detects free-text queries — `"red truck"`,
`"person wearing a backpack"`, `"forklift"`, `"license plate"` — and
returns a box + score for each phrase it finds. Backed by OWL-ViT v2
(`google/owlv2-base-patch16-ensemble`).

Single task: **`open_vocab_detection`**.

## Why it exists

It gives OpenNVR precise **attribute** detection — colour, clothing,
object types outside COCO — that a fixed-class detector plus a scene
captioner can only approximate. Two concrete uses:

- The [`footage-search`](https://github.com/open-nvr/open-nvr/tree/main/examples/footage-search)
  example can index against this adapter for sharper "red truck" /
  "yellow jacket" matching instead of relying on caption text.
- An application can detect site-specific objects ("pallet", "hard hat")
  with no model training — just change the query string.

## Call it

```bash
# multipart: image + comma-separated queries
curl -H "Authorization: Bearer $OPENNVR_ADAPTER_TOKEN" \
  -F frame=@street.jpg \
  -F 'queries=red truck, person on a bicycle' \
  -F 'threshold=0.15' \
  http://localhost:9012/infer
```

Response (`result`):

```json
{
  "task": "open_vocab_detection",
  "queries": ["red truck", "person on a bicycle"],
  "threshold": 0.15,
  "detections": [
    {"label": "red truck", "confidence": 0.41,
     "bbox": {"x": 0.32, "y": 0.40, "w": 0.18, "h": 0.22}}
  ]
}
```

Bboxes are normalized `{x, y, w, h}` (top-left origin, [0, 1]) — the same
§5.1 shape YOLOv8 emits, so zones, counting, line-crossing, and
footage-search consume them with no special-casing.

## Run it

```bash
OPENNVR_ADAPTER_TOKEN=secret \
  python -m uvicorn adapters.vlm.main:app --host 0.0.0.0 --port 9012

# conformance
python -m conformance http://localhost:9012 --token $OPENNVR_ADAPTER_TOKEN

# docker — the published image, weights baked in, no volume, no network
docker run --rm -p 9012:9012 -e OPENNVR_ADAPTER_TOKEN=secret \
  ghcr.io/open-nvr/open-vocab-adapter:latest

# another OWL-ViT variant is a rebuild (the runtime is pinned offline)
docker build -f adapters/vlm/Dockerfile \
  --build-arg VLM_MODEL_ID=google/owlv2-large-patch14-ensemble \
  -t open-vocab-adapter:large .
```

## Configure

- **`VLM_MODEL_ID`** (build arg) — the model baked into the image:
  `google/owlvit-base-patch32` is smaller and faster, an `owlv2-large`
  variant is more accurate. `OPENNVR_VLM_MODEL` defaults to it; the
  runtime is offline, so pointing that at a model that is not baked
  fails to load, by design.
- **`queries`** (per request) — up to 32 text phrases to detect.
- **`threshold`** (per request) — score floor, default 0.10. Open-vocab
  models are noisier than fixed-class ones; tune per scene.

## Sovereignty note

The weights are baked into the image at build time and the runtime is
pinned offline (`HF_HUB_OFFLINE=1`), so the adapter declares
`network_egress: []` and never dials anything — the same posture as
`blip` and `clip` (issue #79). It boots on a sealed box with no volume
to pre-fill. The cost is image size (~600 MB of weights); a bigger
image, bought for a container that needs no network at all.

## Performance

OWL-ViT is heavier than YOLO. On CPU expect ~3–8 s/image (more queries =
slower); a GPU is strongly recommended for anything near video rate. It's
best used **on demand** (footage-search indexing of keyframes, an agent
tool, spot checks) rather than on every frame of every camera. For
high-rate detection, keep YOLOv8 on the hot path and reach for this when a
query needs attributes YOLO can't express.

## Tests

```bash
PYTHONPATH=. python -m pytest adapters/vlm/tests/ -q
```

The unit tests stub the model and cover query/threshold validation, the
comma-separated-queries multipart path, the normalized-bbox conversion,
and the error envelopes — no transformers/torch download required.
