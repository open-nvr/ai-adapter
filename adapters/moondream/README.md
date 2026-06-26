# Moondream2 adapter — visual question answering + captioning

A small **vision-language model** (Moondream2, ~1.6–1.9 B, **Apache-2.0**) that
answers questions *about* a frame, not just captions it. This is what makes the
camera-agent good at **conversations about video**:

- *"What is the person wearing?"* → "a blue shirt and jeans"
- *"What is he doing?"* → "working at a laptop"
- *"Is the gate open?"* → "yes, the gate is open"

BLIP can only caption ("a man at a desk"); Moondream answers the actual question.

## Tasks

| Task | When | Result key |
|------|------|------------|
| `visual_qa` | a `question` / `prompt` is in the payload | `answer` |
| `scene_caption` | no question (or `task: scene_caption`) | `caption` |

It returns whichever key applies; the camera-agent's `describe_camera` reads
`answer` then falls back to `caption`, so this is a **drop-in replacement for the
BLIP `caption_adapter`** — register it under the same name and VQA "just works".

## Run

```bash
# Standalone
OPENNVR_ADAPTER_TOKEN=secret \
  python -m uvicorn adapters.moondream.main:app --host 0.0.0.0 --port 9008

# Docker (weights baked in, offline/local_only-safe)
docker build -f adapters/moondream/Dockerfile -t opennvr/moondream-adapter:1.0.0 .
docker run --rm -p 9008:9008 -e OPENNVR_ADAPTER_TOKEN=secret opennvr/moondream-adapter:1.0.0
```

Register with KAI-C as the caption adapter:
```
POST /api/v1/adapters/register  {"name": "moondream", "url": "http://moondream-adapter:9008"}
```
and set `caption_adapter: moondream` in the camera-agent config.

## Runtime
Uses the official **`moondream` package with a quantized int8 model file**, not
transformers — it's purpose-built for fast VQA on CPU/edge, so the image is small
(no torch) and there's no remote-code / CPU-encoding fragility. Default model is
**0.5B int8** (~593 MiB, fastest on limited hardware); use **2B int8** (~1.7 GiB)
for more capable answers via `--build-arg MOONDREAM_MODEL_FILE=...`.

## Notes
- The quantized model file is baked into the image (offline) so it runs under
  `AI_SOVEREIGNTY=local_only` with no runtime egress.
- For an offline / `local_only` image, supply `--build-arg
  MOONDREAM_MODEL_URL=<url to the .mf.gz>` so the model is baked in (get the URL
  from https://moondream.ai/p/models or the HuggingFace mirror). Without it the
  image still builds (e.g. CI smoke) but has no model and reports unhealthy at
  runtime until one is provided.
- Pillow is pinned `<11` because the moondream package requires it.
- **Not yet build-verified** — confirm the `moondream` package version and the
  model-file URL with one real `docker build` before publishing.
