# Copyright (c) 2026 OpenNVR
# SPDX-License-Identifier: Apache-2.0

"""
Piper TTS adapter — piper-tts (onnxruntime) in-process, contract-compliant.
Mount the voice at OPENNVR_TTS_VOICE_PATH (<name>.onnx + <name>.onnx.json).

Run:
    OPENNVR_TTS_VOICE_PATH=/models/en_US-amy-medium.onnx \\
    python -m uvicorn adapters.pipertts.main:app --host 0.0.0.0 --port 9012
Conformance:
    python -m conformance http://localhost:9012 --token $OPENNVR_ADAPTER_TOKEN
"""
from __future__ import annotations

from adapters.pipertts.service import PiperTtsService
from opennvr_adapter_sdk import (
    AdapterApp,
    BodyShape,
    Cost,
    FairQueuing,
    Permissions,
    Scheduling,
)

_adapter_app = AdapterApp(
    service_factory=PiperTtsService,
    name="pipertts-tts",
    version="1.0.0",
    vendor="camera-agent-lite",
    license="Apache-2.0",
    model_card_url="https://github.com/OHF-Voice/piper1-gpl",
    tasks_advertised=["speech_synthesis"],
    body_shape=BodyShape.TEXT,
    permissions=Permissions(
        gpu=False,
        network_egress=[],
        # The voice (.onnx + .json) downloads from HuggingFace ONCE on first
        # boot into the weights volume (whisper-adapter precedent).
        # Pre-populate /app/models (or set OPENNVR_TTS_VOICE_URL="") for
        # offline installs.
        host_filesystem=["/app/models"],
        shared_memory_paths=[],
        host_metadata=False,
    ),
    scheduling=Scheduling(
        max_inflight=4,       # piper is fast + re-entrant enough for a few concurrent
        preferred_batch_size=1,
        fair_queuing=FairQueuing.NONE,
    ),
    cost=Cost(currency="USD"),
    supports_stream=False,
)

app = _adapter_app.fastapi_app


def __getattr__(name: str):
    if name == "_service":
        return _adapter_app.service
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
