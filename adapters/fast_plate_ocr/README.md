# fast-plate-ocr License-Plate Adapter (Contract v1)

Reference implementation of the
[AI Adapter Contract v1](../../../open-nvr/docs/AI_ADAPTER_CONTRACT.md)
wrapping [fast-plate-ocr](https://github.com/ankandrew/fast-plate-ocr) — a
small, plate-specific OCR model trained for license-plate recognition.

This is the **first adapter explicitly designed to be chained downstream of
another adapter.** The canonical pipeline is:

```
camera frame → YOLOv8 (vehicle + plate ROI detection)
                    → crop plate region
                          → fast-plate-ocr (this adapter)
                                  → plate text + per-character confidence
```

The orchestrating logic — vehicle detection, plate-region cropping, and the
alert payload — lives in the
[`license-plate-recognition` example app](https://github.com/open-nvr/open-nvr/tree/main/examples/license-plate-recognition);
this adapter is intentionally single-purpose so it can be reused by any
upstream that produces a plate crop.

## What it does

POST a cropped plate image, get the recognized text back:

```bash
curl -X POST http://localhost:9004/infer \
  -H "Authorization: Bearer $OPENNVR_ADAPTER_TOKEN" \
  -F "frame=@plate-crop.jpg"
```

```json
{
  "result": {
    "plate_text": "ABC1234",
    "confidence": 0.9342,
    "characters": [
      {"char": "A", "confidence": 0.9342},
      {"char": "B", "confidence": 0.9342},
      ...
    ],
    "accepted": true,
    "min_confidence_applied": 0.30,
    "model_id": "cct-xs-v1-global-model",
    "inference_ms": 18
  },
  "model": {
    "name": "cct-xs-v1-global-model",
    "version": "fast-plate-ocr/cct-xs-v1-global-model",
    "framework": "onnx",
    "fingerprint": "sha256:..."
  }
}
```

## Why a plate-specific OCR engine

Generic OCR (Tesseract, PaddleOCR) is trained on document-style text on
clean backgrounds. License plates carry a mix of dark-on-light and
light-on-dark, region-specific fonts, reflective surfaces, oblique angles,
and high motion blur. A plate-specific model trained on plate fonts is
dramatically more accurate at small image sizes and runs in ~20 ms on CPU.

`fast-plate-ocr` is Apache-2.0, runs on `onnxruntime` (no PyTorch, no
Paddle), and the whole install is ~30 MB on top of the SDK.

## Contract conformance

| Endpoint | Status | Notes |
|---|---|---|
| `GET /health` | required | reports loading vs ok |
| `GET /capabilities` | required | sha256 fingerprint of the ONNX weights, computed live; `tasks_advertised=["license_plate_recognition"]`; `permissions.gpu=false` |
| `GET /hardware/evaluation` | required | reports CPU verdict + model_id |
| `GET /metrics` | required | Prometheus exposition with per-adapter latency buckets |
| `POST /infer` | required | multipart (`frame` file) or JSON (`frame_b64`) |
| `POST /infer/stream` | refused | HTTP 501 — LPR is event-driven, not frame-rate. The orchestrator calls per vehicle detected upstream, not per frame |

## Operational notes

### Per-call confidence override

The adapter applies a default minimum-confidence floor of **0.30**. Below
that the response still carries the best candidate but `accepted=false`,
so the orchestrator can decide whether to drop the alert or surface it
as a low-confidence read.

Per-call override:

```bash
curl -X POST http://localhost:9004/infer \
  -H "Authorization: Bearer $OPENNVR_ADAPTER_TOKEN" \
  -F "frame=@plate-crop.jpg" \
  -F 'params={"min_confidence": 0.55}'
```

### Model swapping (no code change)

`fast-plate-ocr` ships multiple weight bundles for different regions
(EU, US, LATAM, global). The default (`cct-xs-v1-global-model`) is the
most general. Swap via the `OPENNVR_LPR_MODEL` env var:

```bash
OPENNVR_LPR_MODEL=cct-s-v1-global-model uvicorn adapters.fast_plate_ocr.main:app
```

The `/capabilities` fingerprint will change after the swap — KAI-C surfaces
that as an `adapter.fingerprint_mismatch` audit event, which is exactly the
intended behaviour (the operator approved a specific model; a swap requires
re-approval).

### Sovereignty (`local_only` deployments)

The adapter declares `permissions.network_egress=[]` — no inference-time
network calls. **One-time exception:** `fast-plate-ocr` downloads its ONNX
weights from a public mirror on first `load()`. That download is a
build-time concern, not an inference-time concern. For air-gapped
deployments, pre-warm the cache in your build pipeline:

```bash
docker run --rm -v $(pwd)/fp-cache:/root/.cache \
  opennvr/fast-plate-ocr-adapter:1.0.0 \
  python -c "from fast_plate_ocr import LicensePlateRecognizer; LicensePlateRecognizer('cct-xs-v1-global-model')"
```

Then mount the cache at runtime — the recognizer finds the weights locally
and never reaches out.

## Try it

### Local

```bash
# from the ai-adapter repo root
uv sync --extra lpr --extra dev
OPENNVR_ADAPTER_TOKEN=secret \
  uv run uvicorn adapters.fast_plate_ocr.main:app --host 0.0.0.0 --port 9004
```

### Docker

```bash
docker build -f adapters/fast_plate_ocr/Dockerfile -t opennvr/fast-plate-ocr-adapter:local .
docker run --rm -p 9004:9004 \
  -e OPENNVR_ADAPTER_TOKEN=$(openssl rand -hex 16) \
  opennvr/fast-plate-ocr-adapter:local
```

### Conformance check

```bash
uv run python -m conformance http://localhost:9004 --token $OPENNVR_ADAPTER_TOKEN
```

## Tests

```bash
uv run pytest tests/test_fast_plate_ocr_service.py tests/test_conformance_against_fast_plate_ocr.py -v
```

All tests stub the `fast_plate_ocr.LicensePlateRecognizer` so no model
weights are downloaded and tests run on every machine including CI.

## Layout

```
adapters/fast_plate_ocr/
├── main.py          SDK glue + AdapterApp construction
├── service.py       FastPlateOcrService: AdapterService implementation
├── Dockerfile       multi-stage CPU-only build
├── README.md        you are here
└── __init__.py
```

## License

This adapter wrapper is **AGPL-3.0** (same as the parent ai-adapter repo).
The wrapped `fast-plate-ocr` library is **Apache-2.0**; model weights are
released under fast-plate-ocr's own terms — see the
[upstream repository](https://github.com/ankandrew/fast-plate-ocr) for the
exact license on each weight bundle.
