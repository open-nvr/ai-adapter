# Whisper-QNN ASR Adapter (Contract v1)

Whisper-Small (w8a16) on the **Qualcomm Hexagon NPU**, through ONNX Runtime's
QNN execution provider, serving the §5.3 ASR convention the `whisper` and
`parakeet` adapters serve. Built for the camera agent on the Ventuno Q
(QCS8275 / Dragonwing IQ-8275), where speech-to-text on the CPU cost
0.7–2.3 s an utterance and the CPU the detector loop needed.

## What it does

POST an audio clip (16 kHz PCM WAV straight to the model; anything else
ffmpeg converts), get a transcript back:

```json
{
  "transcript": "is anyone at the gate",
  "language": "en",
  "segments": [{"start_ms": 0, "end_ms": 2140, "text": "is anyone at the gate", "no_speech_prob": 0.0021}],
  "no_speech_prob": 0.0021,
  "duration_seconds": 2.14,
  "model": "whisper_small_quantized-w8a16",
  "decode_steps": 11,
  "encoder_ms": 61.2,
  "decoder_ms": 88.5
}
```

One task, `audio_transcription`. `audio_translation` is refused with a 400
that names the whisper adapter. The decode is greedy with the English prefix
forced (`<|startoftranscript|><|en|><|transcribe|><|notimestamps|>`; another
`language` param picks that language's token when the vocabulary has it),
so the model never drifts into another language on noise — what the
free-running CPU model did. `no_speech_prob` is the softmax mass on the
no-speech token at the first decoder step: the agent's gate for clips that
are not speech.

## The model

Qualcomm AI Hub's `whisper_small_quantized` (openai/whisper-small, w8a16),
the **precompiled QNN ONNX** release: two EPContext wrappers (`encoder.onnx`,
`decoder.onnx`) around QNN context binaries (`encoder.bin` 113 MB,
`decoder.bin` 225 MB) plus `metadata.json` with every tensor's fixed-point
scale and zero point. Context binaries are compiled for ONE Hexagon
generation; the default image is for `qcs8275` (HTP v75, QAIRT 2.50; the
asset's own metadata names the Arduino VENTUNO Q as its reference device).
`--build-arg WHISPER_QNN_CHIPSET=...` rebuilds for another chip in the
release. The tokenizer and the mel filter bank come from the upstream
`openai/whisper-small` repository. Everything is baked into the image; the
container needs no network and no volume.

How it is driven (`engine.py`): the encoder takes the quantized 80 × 3000
log-mel features of a 30 s window and returns the 12 layers' cross-attention
caches; the decoder takes one token, its position, a 200-slot attention mask
and 12 self-attention caches of 199 slots, and returns the next token's
logits with the caches shifted by one. The mask opens from the end (slot
`199 − n` at step `n`); the empty cache is the fixed-point zero (the zero
point, 128), not 0. The numpy log-mel (`features.py`) was checked against
HuggingFace's `WhisperFeatureExtractor`: max abs difference 1e-6.

## Endpoints

| Endpoint | Status | Notes |
|---|---|---|
| `GET /health` | required | auth-exempt; `loading` → `ok`, or `error` with the reason (no Hexagon, assets missing) |
| `GET /capabilities` | required | `tasks_advertised=["audio_transcription"]`; `permissions.gpu=false` (the NPU is a device node, not a GPU grant) |
| `GET /hardware/evaluation` | required | `ok` on the Hexagon with chip and tool versions; `blocked` with the reason anywhere else |
| `GET /metrics` | required | Prometheus exposition |
| `POST /infer` | required | multipart (`audio` file) or JSON (`audio_b64`); params `task`, `language` |
| `POST /infer/stream` | refused | HTTP 501 |

## Where it runs — and where it refuses

Only on a Qualcomm host with the HTP lent to the container
(`--device /dev/fastrpc-cdsp`, as `docker-compose.accel-qualcomm.yml` does)
and the `qualcomm` image flavour (`onnxruntime` + the `onnxruntime-qnn`
plugin). Anywhere else the adapter loads nothing, `/health` says `error`
and `/hardware/evaluation` says `blocked` with why: a context binary that
cannot find its device would otherwise be "loaded" and run nothing, and a
CPU fallback would be a lie about the latency this adapter exists for. The
CPU alternatives are the `whisper` (faster-whisper) and `parakeet` adapters;
OpenNVR's `STT_ADAPTER` chooses.

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `WHISPER_QNN_ASSETS_DIR` | `/app/model_weights/whisper-qnn` | the asset directory (baked) |
| `ADAPTER_ONNX_PROVIDERS` | `QNNExecutionProvider:htp` | provider; CPU is refused |
| `ADAPTER_ONNX_PROVIDER_OPTIONS` | — | extra QNN options, `k=v,k=v` |
| `WHISPER_QNN_MAX_TOKENS` | `160` | generation cap per 30 s window |
| `OPENNVR_ADAPTER_TOKEN` | — | bearer token for `/infer` |

## Tests

`tests/test_whisper_qnn_service.py` (the front end, the decode loop against
the export's interface with scripted fake sessions, the contract surface) and
`tests/test_conformance_against_whisper_qnn.py`. The real binaries are
exercised only on the hardware.
