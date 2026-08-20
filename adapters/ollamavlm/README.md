# ollamavlm — Ollama-proxy VLM adapter

**Visual question answering + captioning at host-GPU speed.** Same two
tasks and result keys as the moondream adapter (`visual_qa` → `answer`,
`scene_caption` → `caption`), but inference runs on whatever **Ollama
endpoint** you point it at — by default the machine hosting Docker
(`http://host.docker.internal:11434`), where Ollama uses the real GPU
(Metal on Apple Silicon). On macOS/Windows, where containers cannot
touch the GPU, this is the difference between ~3–5 s and sub-second VQA.

The image is ~80 MB: no weights, no torch, no onnx. Models are managed by
Ollama itself — `ollama pull moondream` (or `llava`, `qwen2.5vl`, any
multimodal model) — and the adapter auto-pulls a missing model on first
boot via Ollama's API (`OPENNVR_OLLAMA_VLM_AUTOPULL=false` to disable).

## Configuration

| env | default | meaning |
|---|---|---|
| `OPENNVR_OLLAMA_VLM_URL` | `http://host.docker.internal:11434` | Ollama endpoint |
| `OPENNVR_OLLAMA_VLM_MODEL` | `moondream` | multimodal model to use |
| `OPENNVR_OLLAMA_VLM_AUTOPULL` | `true` | pull a missing model at startup |
| `OPENNVR_OLLAMA_VLM_TIMEOUT_S` | `120` | per-request timeout |

## Behavior worth knowing

- **Lazy-ready:** `/health` is OK once configuration parses. An
  unreachable endpoint or un-pulled model is a *transient* per-infer
  error (503 + retry hint), because the endpoint is an independent
  process that may start after the adapter or restart under it — the
  adapter rides it out. Live endpoint state is reported by
  `/capabilities` → hardware evaluation.
- **Declared egress:** every inference goes to the configured endpoint
  and nowhere else; frames never leave that hop. With the default URL
  the endpoint IS the local machine.
- On Linux Docker Engine add `--add-host host.docker.internal:host-gateway`
  (Docker Desktop provides it automatically).

## Use as the camera-agent captioner

In open-nvr: `CAPTION_ADAPTER=ollamavlm` (see the camera-agent compose;
pairs naturally with `OLLAMA_EXTERNAL_URL` so LLM and VLM share one
host-side Ollama).

## Run standalone

```bash
docker run --rm -p 9018:9018 \
  -e OPENNVR_ADAPTER_TOKEN=secret \
  --add-host host.docker.internal:host-gateway \
  ghcr.io/open-nvr/ollamavlm-adapter:latest
# NOTE: per contract §3.5, non-file fields travel in the JSON 'params'
# form field — a bare -F question=... is ignored by the SDK parser.
curl -H "Authorization: Bearer secret" -F frame=@street.jpg \
  -F 'params={"question":"what is the person wearing?"}' \
  http://localhost:9018/infer
```
