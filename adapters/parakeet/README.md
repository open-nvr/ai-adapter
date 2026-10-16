# Parakeet ASR adapter (Contract v1)

Speech to text with NVIDIA **Parakeet TDT 0.6B v3**, served through
[onnx-asr](https://github.com/istupakov/onnx-asr) on ONNX Runtime, int8,
CPU first. Same contract surface and §5.3 `AsrResult` as the
[whisper adapter](../whisper/README.md), so the camera agent's
`whisper_url` can point at either.

## Why it exists

Measured on the agent's own phrasings (29 utterances, four voices; the
numbers are in open-nvr's `examples/camera-agent/MODELS_AND_LATENCY.md`):

| | WER | Per utterance, 4 CPU cores |
|---|---|---|
| **Parakeet TDT 0.6B v3** (int8) | **1.4 %** | **0.31 s** |
| faster-whisper `base.en` (int8, the whisper adapter's default) | 3.3 % | 1.42 s |
| faster-whisper `small.en` | 3.7 % | 2.67 s |

Half the errors at a fifth of the time. A second of STT per turn was the
single largest cost in the voice loop's latency budget on CPU.

## What it does and does not do

- Task: **`audio_transcription`** only. Parakeet does not translate;
  `audio_translation` is refused with a 400 that names the whisper
  adapter.
- Languages: v3 detects 25 European languages itself (no Hindi — the
  whisper adapter covers that). A `language` param is echoed in the
  result, not enforced.
- Input: a PCM WAV at any common rate goes to the model as samples
  (it resamples; the agent sends 16 kHz mono); anything else — a
  browser's webm, an mp3 — is converted by ffmpeg in the image.
- Segments: from the model's token timestamps, split at sentence
  punctuation and at pauses over 0.8 s. No `no_speech_prob` (a
  transducer has no such number); the agent only logs it.
- Weights: ~670 MB, **baked into the image** (`PARAKEET_WEIGHTS_DIR`),
  runtime offline, `network_egress: []`. Another onnx-asr model is a
  rebuild: `--build-arg PARAKEET_MODEL=nemo-parakeet-tdt-0.6b-v2`
  (English-only).
- Licence of the weights: CC-BY-4.0 (NVIDIA); the adapter is AGPL-3.0.

## Call it

```bash
# what the agent sends: JSON with base64 audio
curl -H "Authorization: Bearer $OPENNVR_ADAPTER_TOKEN" -H "Content-Type: application/json" \
  -d "{\"task\":\"audio_transcription\",\"language\":\"en\",\"audio_b64\":\"$(base64 -w0 clip.wav)\"}" \
  http://localhost:9019/infer

# or multipart
curl -H "Authorization: Bearer $OPENNVR_ADAPTER_TOKEN" -F audio=@clip.wav http://localhost:9019/infer
```

Response (`result`):

```json
{"transcript": "Is anyone at the front door right now?", "language": "en",
 "segments": [{"start_ms": 80, "end_ms": 2310, "text": "Is anyone at the front door right now?"}],
 "duration_seconds": 2.4, "model": "nemo-parakeet-tdt-0.6b-v3"}
```

## Run it

```bash
# docker — the published image, weights baked in
docker run --rm -p 9019:9019 -e OPENNVR_ADAPTER_TOKEN=secret ghcr.io/open-nvr/parakeet-adapter:latest

# source checkout (downloads the weights from HuggingFace on first load)
uv pip install "onnx-asr[cpu,hub]"
OPENNVR_ADAPTER_TOKEN=secret python -m uvicorn adapters.parakeet.main:app --host 0.0.0.0 --port 9019

# conformance
python -m conformance http://localhost:9019 --token $OPENNVR_ADAPTER_TOKEN
```

In the OpenNVR stack: `STT_ADAPTER=parakeet` in `.env` (the default from
the next platform release) puts this image in the whisper adapter's slot;
`STT_ADAPTER=whisper` puts whisper back, for translation or Hindi.
