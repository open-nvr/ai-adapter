# llamacpp-llm adapter

Qwen2.5-3B-Instruct (GGUF Q4_K_M) served by a persistent **llama.cpp**
`llama-server` child, wrapped in the OpenNVR Adapter Contract. **Torch-free,
CPU-first, GPU-optional.** The lightweight, governed replacement for the raw
`ollama` container.

Why: same local llama.cpp engine Ollama uses, but registered/health-polled/
audited by KAI-C, and — unlike the reference `llm` (SmolLM2) adapter — it
**carries tool-calls**, which the camera-agent's tool loop needs.

## `/infer` (BodyShape.TEXT)

Request:
```json
{ "messages": [{"role":"system","content":"…"},{"role":"user","content":"list cameras"}],
  "tools": [ /* OpenAI function schemas, optional */ ],
  "max_tokens": 256, "temperature": 0.3 }
```
Response `result`:
```json
{ "text": "…", "tool_calls": [ /* when the model calls a tool */ ],
  "finish_reason": "stop|tool_calls", "usage": {…} }
```

`tool_calls` follow the OpenAI shape (`{id, type:"function", function:{name, arguments}}`).
Streaming (`/infer/stream`) returns 501 in v1.

## Config (env)

| var | default | meaning |
|---|---|---|
| `OPENNVR_LLM_MODEL_PATH` | `/app/models/Qwen2.5-3B-Instruct-Q4_K_M.gguf` | GGUF path (mount it) |
| `LLAMACPP_SERVER_BIN` | `llama-server` | native binary |
| `LLAMACPP_CTX_SIZE` | `4096` | context window |
| `LLAMACPP_THREADS` | CPU count | `-t` |
| `LLAMACPP_GPU_LAYERS` | `0` | `-ngl`; set `999` on GPU hosts |
| `OPENNVR_ADAPTER_TOKEN` | `""` | bearer token (KAI-C sets to `INTERNAL_API_KEY`) |

## Run + conformance
```bash
OPENNVR_LLM_MODEL_PATH=/models/qwen.gguf LLAMACPP_SERVER_BIN=/bin/llama-server \
  python -m uvicorn adapters.llamacpp.main:app --host 0.0.0.0 --port 9014
python -m conformance http://localhost:9014 --token $OPENNVR_ADAPTER_TOKEN
```
Sovereignty: `network_egress=[]`, GGUF mounted (never fetched) → passes `local_only`.
