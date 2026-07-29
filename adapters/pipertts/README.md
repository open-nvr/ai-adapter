# pipertts-tts adapter

Piper TTS (onnxruntime) in-process, wrapped in the OpenNVR Adapter Contract.
**Torch-free.**

## `/infer` (BodyShape.TEXT)
- Input: `{text, length_scale?, noise_scale?, inline: true}`.
- Output `result`: `{audio_b64 (WAV), audio_format:"wav", sample_rate, duration_seconds, voice}`.

## Config
| var | default |
|---|---|
| `OPENNVR_TTS_VOICE_PATH` | `/app/models/en_US-amy-medium.onnx` |
| `PIPER_LENGTH_SCALE` | `1.0` (>1 slower) |

Mount both `<voice>.onnx` and `<voice>.onnx.json`. `network_egress=[]` → `local_only`.

```bash
OPENNVR_TTS_VOICE_PATH=/models/en_US-amy-medium.onnx \
  python -m uvicorn adapters.pipertts.main:app --host 0.0.0.0 --port 9012
python -m conformance http://localhost:9012 --token $OPENNVR_ADAPTER_TOKEN
```
