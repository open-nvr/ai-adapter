# whispercpp-stt adapter

ggml Whisper served by the native **whisper.cpp** engine (in-process via
`pywhispercpp`), wrapped in the OpenNVR Adapter Contract. **Torch-free,
CTranslate2-free.**

## `/infer` (BodyShape.AUDIO)
- Input: audio at multipart `audio` or JSON `audio_b64` — **16 kHz mono 16-bit WAV**
  (the adapter down-mixes + resamples if needed) + optional `{language, task}`.
- Output `result` (§5.3 AsrResult): `{transcript, language, segments:[{start_ms,end_ms,text}], duration_seconds}`.

## Config
| var | default |
|---|---|
| `OPENNVR_STT_MODEL_PATH` | `/app/models/ggml-base.en.bin` |
| `WHISPERCPP_THREADS` | CPU count |
| `WHISPERCPP_LANGUAGE` | `en` |

Mount the ggml model (`ggml-base.en.bin` ~150 MB, or `ggml-tiny.en.bin` ~75 MB).
`network_egress=[]` → passes `local_only`.

```bash
OPENNVR_STT_MODEL_PATH=/models/ggml-base.en.bin \
  python -m uvicorn adapters.whispercpp.main:app --host 0.0.0.0 --port 9013
python -m conformance http://localhost:9013 --token $OPENNVR_ADAPTER_TOKEN
```
