# text-embed — sentence embeddings for search's caption arm

The `text_embed` task of the AI Adapter Contract: a sentence in, a unit vector
out, trained for text against text. OpenNVR's search uses it to compare a
query in words with the captions its visits carry; the `embed` adapters (clip,
siglip-qnn) keep the frame arm, where a frame and a query share one space.

Why a second adapter: an image-text model's text tower is a poor judge of text
against text. On a Qualcomm board SigLIP2's text-to-text cosines for relevant
and irrelevant caption/query pairs overlap (0.76–0.87 against 0.59–0.75,
measured 2026-11-10), so no floor separates a hit from noise. A sentence
encoder does: for the same kind of pair bge-small scores the matching caption
0.76–0.82 and the others at most 0.63 (Mac, the probe in the PR).

| | |
|---|---|
| Model | `BAAI/bge-small-en-v1.5` (MIT), 33 M parameters, the repo's own `onnx/model.onnx` (133 MB), [CLS] pooling, L2-normalised, 384 dims |
| Runtime | ONNX Runtime on the CPU, `TEXT_EMBED_THREADS` (2) intra-op threads; a caption is a few milliseconds a core |
| Port | 9013 |
| Images | `ghcr.io/open-nvr/text-embed-adapter`, linux/amd64 + linux/arm64, weights baked in, no egress |

## Request / answer

```json
{"task": "text_embed", "text": "person in a blue shirt", "kind": "query"}
{"task": "text_embed", "texts": ["a red truck parked near a loading dock", "..."], "kind": "document"}
```

```json
{"status": "ok", "model_name": "text-embed", "result": {"embedding": [384 floats],
 "dim": 384, "model": "BAAI/bge-small-en-v1.5/cls-onnx", "modality": "text", "kind": "query",
 "normalized": true, "device": "CPUExecutionProvider", "encode_ms": 3.1}}
```

`text` answers one `embedding`; `texts` (up to 64) answers an `embeddings` list in
the same order. `kind` is `query` (a search) or `document` (a caption): a model
whose card asks for a query instruction gets it prepended for queries only
(`query_prefix` in `model.json`; empty for bge v1.5, whose card says the
instruction is optional for short queries). Empty strings, both `text` and
`texts`, more than 64 texts or a text over 4096 characters are `malformed_input`.

## Another model

`TEXT_EMBED_MODEL_DIR` can point at any BERT-style sentence encoder exported to
ONNX with `input_ids` / `attention_mask` (/ `token_type_ids`) in and
`last_hidden_state` out — `model.onnx` + `tokenizer.json` + a `model.json`:

```json
{"model_id": "sentence-transformers/all-MiniLM-L6-v2", "pooling": "mean", "max_tokens": 128, "query_prefix": ""}
```

`pooling` is `cls` (BGE, GTE) or `mean` (MiniLM, E5). The answer's `model` names
the model and the pooling, so a store never compares vectors from two.

## Environment

| Variable | Default | Meaning |
|---|---|---|
| `TEXT_EMBED_MODEL_DIR` | `/app/model_weights/text-embed` | `model.onnx`, `tokenizer.json`, `model.json` |
| `ADAPTER_ONNX_PROVIDERS` | `CPUExecutionProvider` | ONNX Runtime providers, first available wins |
| `TEXT_EMBED_THREADS` | `2` | intra-op threads |

Tests (no model needed): `pytest tests/test_text_embed_service.py tests/test_conformance_against_text_embed.py`.
