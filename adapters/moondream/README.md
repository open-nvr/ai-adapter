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
Uses the **`moondream==0.0.6`** package — the **onnxruntime** build with a
quantized int8 `.mf.gz` model file and **no torch**. (Pin matters: 0.1.x is
cloud-only and rejects a local model; 0.2.x pulls torch/CUDA and won't build on
CPU. 0.0.6 is the local-onnx version that works on CPU.) API: `encode_image()`
then `caption()` / `query()`. Default model is **0.5B int8** (~593 MiB, fastest
on limited hardware); use **2B int8** (~1.7 GiB) for more capable answers.

## Providing the model (three ways, most-sovereign first)
The **CI-published image is code-only** (CI doesn't pass a model URL). Supply the
int8 `.mf.gz` (from https://moondream.ai/p/models or the HF mirror) by any of:

1. **Bake at build** (offline / `local_only`, best): `--build-arg
   MOONDREAM_MODEL_URL=<url>` → fully self-contained image.
2. **Mount it** into the model volume at `OPENNVR_MOONDREAM_MODEL_PATH`
   (default `/models/moondream-0_5b-int8.mf.gz`) — offline, works with the
   code-only published image.
3. **Runtime download** — set env `OPENNVR_MOONDREAM_MODEL_URL`; the adapter
   downloads it **once** into the model path on first start (cached in the
   volume). Makes the published image **pull-and-run** with no rebuild — but it's
   a one-time network fetch, so use 1 or 2 for a strict `local_only` posture.
   (Inference itself still has no egress.)

## Notes
- Pillow is pinned `<11` (moondream requires it); runtime is `moondream==0.0.6`
  (onnxruntime, no torch).
- Confirm the model-file URL with one real run before relying on it.
