"""
Parakeet ASR adapter — speech to text on the AI Adapter Contract v1.

NVIDIA Parakeet TDT 0.6B v3 served through ``onnx-asr`` (ONNX Runtime,
int8, CPU first). Same contract surface and §5.3 result shape as the
whisper adapter, so the camera agent's ``whisper_url`` can point at either.
Measured on the agent's own phrasings (MODELS_AND_LATENCY.md in open-nvr):
WER 1.4 % against faster-whisper base.en's 3.3 %, at 0.31 s against
1.42 s an utterance on four CPU cores.

Run with:
    python -m uvicorn adapters.parakeet.main:app --host 0.0.0.0 --port 9019
"""
