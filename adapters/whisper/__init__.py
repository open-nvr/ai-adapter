"""
Whisper ASR adapter — A2.3-prep reference implementation of the AI
Adapter Contract v1.

Third adapter migrated to the contract (after A2.1 Piper and A2.2
YOLOv8). Validates the contract on a third modality (audio) and
exercises the §5.3 ASR output convention. Same structural template as
``adapters/piper/`` and ``adapters/yolov8/`` — auth, metrics, service,
main — so future adapter migrations have an obvious pattern to copy.

Streaming ASR (partial-result emission over WebSocket with overlap
windows and VAD gating) is intentionally out of scope for v1; the
adapter advertises ``endpoints.infer_stream.supported = false`` and
refuses upgrades with HTTP 501. The §6 streaming protocol is exercised
by YOLOv8; ASR streaming is its own design problem and lands in a
follow-up.

Run with:
    python -m uvicorn adapters.whisper.main:app --host 0.0.0.0 --port 9003
"""
