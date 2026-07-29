# smolvlm-vlm adapter

SmolVLM2-2.2B-Instruct (GGUF + mmproj) served by a persistent multimodal
**llama.cpp** `llama-server`, wrapped in the OpenNVR Adapter Contract.
**Torch-free, CPU-first.** On-demand VQA/captioning of a single camera frame.

## `/infer` (BodyShape.IMAGE)
- Input: image at multipart `frame` or JSON `frame_b64` (JPEG/PNG) + optional
  `{question | prompt}`. The adapter resizes to a ≤768px long edge (Pillow) and
  re-encodes JPEG in-memory before inference.
- Output `result`: `{text, caption}`.

## Config
| var | default |
|---|---|
| `OPENNVR_VLM_MODEL_PATH` | `/app/models/SmolVLM2-2.2B-Instruct-Q4_K_M.gguf` |
| `OPENNVR_VLM_MMPROJ_PATH` | `/app/models/mmproj-SmolVLM2-2.2B-Instruct-f16.gguf` |
| `LLAMACPP_SERVER_BIN` | `llama-server` |
| `SMOLVLM_GPU_LAYERS` | `0` (set `999` on GPU hosts) |

`network_egress=[]`, weights mounted (never fetched) → passes `local_only`.
CPU vision inference is ~seconds/frame — call it only when a question is visual.

```bash
python -m conformance http://localhost:9016 --token $OPENNVR_ADAPTER_TOKEN
```
