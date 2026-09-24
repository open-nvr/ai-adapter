# CLIP embedding adapter

**Task:** `embed` · **Port:** 9011 · **Model:** OpenCLIP `ViT-B-32` (LAION `laion2b_s34b_b79k`) · **Dimensions:** 512 · **Weights licence:** MIT

Turns a frame into a vector — and a **query in words** into a vector in
the same space.

That second half is the whole point. OpenNVR's search fuses a keyword
ranking with a similarity ranking, and a similarity ranking only helps
if the operator's sentence and the camera's frame land somewhere
comparable. It is what lets a search for *"truck"* find the visit a
captioner described as *"a lorry at the loading bay"* — a match no
keyword query can make.

## What it returns

```jsonc
{
  "result": {
    "embedding": [0.0231, -0.0114, ...],   // 512 floats, unit length
    "dim": 512,
    "model": "open_clip/ViT-B-32/laion2b_s34b_b79k",
    "modality": "image",                   // or "text"
    "normalized": true
  }
}
```

Vectors come back **unit length**, always, so cosine similarity is a
plain dot product. `dim` and `model` travel with every vector because
comparing vectors from two different models produces a confident
ordering of noise, and those two fields are the only way a store can
refuse to.

## Calling it

Two shapes, both JSON:

```bash
# a frame
curl -s localhost:9011/infer -H "Authorization: Bearer $TOKEN" \
  -H 'Content-Type: application/json' \
  -d "{\"task\":\"embed\",\"frame_b64\":\"$(base64 -w0 frame.jpg)\"}"

# a query
curl -s localhost:9011/infer -H "Authorization: Bearer $TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{"task":"embed","text":"a red truck at the loading bay"}'
```

Exactly one of `text` or `frame_b64`. Sending both is refused rather
than guessed at — this returns one vector and cannot say which modality
it came from if handed two, and silently preferring one would turn a
caller's bug into a bad ranking discovered weeks later.

> **A multipart file upload is not the image path here.** This adapter
> is `BodyShape.TEXT` so that a request carrying *no image at all* is
> accepted, and the SDK's TEXT parser collects only string form fields —
> a file part never reaches the service. Frames travel as `frame_b64`.
> The 400 you get from a multipart upload says so.

### Why `BodyShape.TEXT` and not `IMAGE`

Every other vision adapter in this repo is `BodyShape.IMAGE`, and this
one looks like it should be. It cannot be: `IMAGE` makes the binary
**mandatory**, and the SDK rejects a body without `frame_b64` before
`infer()` is ever called. That is correct for a detector, where a
request with no frame is meaningless. Here a request with no frame is
the *query* half of the feature.

The cost is that the SDK's `max_body_bytes` guard does not cover a
base64 image on this path, so the service enforces its own limit —
checked on the encoded length, before decoding, so an oversized body is
refused without being built in memory first.

## Running it

```bash
docker run --rm -p 9011:9011 \
  -e OPENNVR_ADAPTER_TOKEN=$TOKEN \
  opennvr/clip-adapter:1.0.0
```

No volume, nothing to pre-populate. **The weights are baked into the
image**, the same call `adapters/blip` makes (issue #79), and the
runtime is pinned offline (`HF_HUB_OFFLINE=1`, `CLIP_OFFLINE=1`) so a
load can never make the default metadata check to huggingface.co.

That costs ~600 MB of image and buys a container that declares **no
network egress at all** — which is the posture an air-gapped site
actually checks, and the one that would otherwise force an operator to
relax the sovereignty gate on the box whose whole reason for existing
is not having to.

**Air-gapped:** load the image and you are done.

| Variable | Default | Notes |
|---|---|---|
| `CLIP_MODEL` | `ViT-B-32` | Any OpenCLIP architecture. |
| `CLIP_PRETRAINED` | `laion2b_s34b_b79k` | **Check the licence** before changing — several popular checkpoints are non-commercial. |
| `CLIP_CACHE_DIR` | `/models/clip` | Where the baked weights live. Populated = no egress declared. |
| `CLIP_THREADS` | `2` | Torch intra-op threads. Left to torch it takes every core it sees, and on an NVR it would be competing with the detector the box exists for. |
| `CLIP_OFFLINE` | `1` in the image | Asserts the adapter will never dial out, which is stronger than a cache that happens to be full. |

The **dimension is read from the model**, not assumed — point
`CLIP_MODEL` at `ViT-L-14` and you get 768-wide vectors, reported
honestly. A store holding 512-wide vectors should refuse them rather
than compare them, which is what `dim` is for.

## Why this model

`ViT-B-32` is the small end of the CLIP family, on purpose. OpenNVR's
floor is a mini-PC, and this runs there — ~88M parameters in the image
tower, tens of milliseconds per frame on a couple of x86 cores. A
larger tower ranks slightly better and moves the floor, which is the
wrong trade for this project.

The weights licence was checked rather than assumed. Both the LAION
`laion2b_s34b_b79k` and OpenAI `openai` checkpoints for this
architecture are MIT; several otherwise-attractive embedding models
ship under CC-BY-NC, which would quietly poison OpenNVR's commercial
licensing.

## Hardware

CPU is the design target and a GPU is an accelerator, not a
requirement. The container ships the CPU torch wheel because the CUDA
wheel is several gigabytes for hardware most installs do not have; the
code detects the device at load either way, and declares `gpu=true`
only when torch actually sees one — so the default image registers
without a GPU-grant prompt.

Cost per visit is real but bounded: one embedding per finished visit,
gated by the per-camera `embed` skill assignment in OpenNVR, plus one
per search that uses words.

## Tests

```bash
# contract behaviour, stubbed model — fast, no download
pytest tests/test_clip_service.py tests/test_conformance_against_clip.py

# the real weights: does CLIP actually do what the feature needs?
CLIP_TEST_WEIGHTS=1 CLIP_CACHE_DIR=/path/to/cache pytest tests/test_clip_real_model.py
```

The split matters. The stubbed tests pin the *contract* —
normalisation, the dimension travelling with the vector, the refusals —
and would pass with the model wired backwards. `test_clip_real_model.py`
is the one that asserts words and pictures share a meaningful space,
with a margin wide enough to rank with, and that *"a lorry"* beats
*"a bicycle"* on a picture of a truck. If that fails, semantic search
ranks by noise and every other test still passes.
