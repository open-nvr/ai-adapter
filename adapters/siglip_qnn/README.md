# siglip-qnn — SigLIP2 embeddings on the Qualcomm Hexagon NPU

The `embed` task of the AI Adapter Contract — the same request and answer as the
CPU `clip` adapter — computed on the HTP of a Qualcomm Dragonwing / Snapdragon
board. Image and text land in one 768-dimensional space, so core's frame
embeddings, its caption embeddings and search's query vector all compare.

| | |
|---|---|
| Model | `google/siglip2-base-patch16-224` (Apache-2.0), as exported by Qualcomm AI Hub (`siglip2`, release v0.66.0, ONNX w8a16, opset 21) |
| Towers | `image_encoder.onnx` (uint16 image [1,3,224,224] → uint16 [1,768]), `text_encoder.onnx` (int32 ids [1,64] → uint16 [1,768]) |
| Tokenizer | the model's Gemma tokenizer (256k pieces), lower-cased, EOS last, padded to 64 |
| Qualcomm's numbers, IQ-8275 | image 5.1 ms, text 2.1 ms an inference on the NPU |
| Port | 9011 (the clip adapter's, so `clip-register` needs no change) |

## How it runs

The graphs are quantized ONNX, not precompiled contexts: ONNX Runtime's QNN
execution provider compiles each for the HTP at the first session build and the
SDK caches the EP context under `ADAPTER_ACCEL_CACHE_DIR`
(`/app/model_weights/siglip2-qnn/.opennvr-accel` in the image). Mount that path
to keep the compiled graphs across container recreations; without it every
recreate compiles again (minutes for the 1.1 GB text tower).

`SIGLIP_QNN_TEXT_PROVIDERS=CPUExecutionProvider` keeps the text tower on the CPU
— it carries the whole 256k-piece vocabulary and is the slow one to compile —
with the image tower on the NPU. The vectors are the same space either way; the
answer's `device` says which computed it.

## Request / answer

```json
{"task": "embed", "text": "a red truck at the loading bay"}
{"task": "embed", "frame_b64": "<jpeg>", "camera_id": "cam3"}
```

```json
{"status": "ok", "model_name": "siglip-qnn", "result": {"embedding": [768 floats],
 "dim": 768, "model": "siglip2/siglip2-base-patch16-224/w8a16-qnn", "modality": "image",
 "normalized": true, "device": "QNNExecutionProvider:htp", "encode_ms": 5.3}}
```

Exactly one of `text` or an image per request; both or neither is `malformed_input`.

## Environment

| Variable | Default | Meaning |
|---|---|---|
| `SIGLIP_QNN_ASSETS_DIR` | `/app/model_weights/siglip2-qnn` | the export's files + `tokenizer.json` |
| `ADAPTER_ONNX_PROVIDERS` | `QNNExecutionProvider:htp` | the image tower's provider |
| `SIGLIP_QNN_TEXT_PROVIDERS` | same as above | the text tower's provider; `CPUExecutionProvider` to keep it off the HTP |
| `ADAPTER_ACCEL_CACHE_DIR` | `…/.opennvr-accel` | where the compiled graphs are cached |

Tests (no Hexagon needed): `pytest tests/test_siglip_qnn_service.py`.
