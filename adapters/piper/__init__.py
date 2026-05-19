"""
Piper TTS adapter — A2.1 reference implementation of the AI Adapter Contract v1.

This package is the contract-compliant HTTP service that wraps Piper's
ONNX TTS voices. The inference logic lives in
``app.adapters.audio.piper_adapter.PiperAdapter`` (legacy class, untouched);
this layer adds the six mandatory endpoints, bearer-token auth, the
correlation_id wire spec, and Prometheus metrics.

Run with:
    python -m uvicorn adapters.piper.main:app --host 0.0.0.0 --port 9001
"""
