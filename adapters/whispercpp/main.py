# Copyright (c) 2026 OpenNVR
# SPDX-License-Identifier: Apache-2.0

"""
whisper.cpp STT adapter — ggml Whisper via pywhispercpp, contract-compliant.

Torch-free, CTranslate2-free — the native whisper.cpp/ggml engine in-process.
Mount the ggml model at OPENNVR_STT_MODEL_PATH.

Run:
    OPENNVR_STT_MODEL_PATH=/models/ggml-base.en.bin \\
    python -m uvicorn adapters.whispercpp.main:app --host 0.0.0.0 --port 9013
Conformance:
    python -m conformance http://localhost:9013 --token $OPENNVR_ADAPTER_TOKEN
"""
from __future__ import annotations

from adapters.whispercpp.service import MAX_AUDIO_BYTES, WhisperCppService
from opennvr_adapter_sdk import (
    AdapterApp,
    BodyShape,
    Cost,
    FairQueuing,
    Permissions,
    Scheduling,
)

_adapter_app = AdapterApp(
    service_factory=WhisperCppService,
    name="whispercpp-stt",
    version="1.0.0",
    vendor="camera-agent-lite",
    license="Apache-2.0",
    model_card_url="https://github.com/ggml-org/whisper.cpp",
    tasks_advertised=["audio_transcription"],
    body_shape=BodyShape.AUDIO,
    max_body_bytes=MAX_AUDIO_BYTES,
    permissions=Permissions(
        gpu=False,
        network_egress=[],
        # The ggml model downloads from HuggingFace ONCE on first boot into
        # the weights volume (whisper-adapter precedent). Pre-populate
        # /app/models (or set OPENNVR_STT_MODEL_URL="") for offline installs.
        host_filesystem=["/app/models"],
        shared_memory_paths=[],
        host_metadata=False,
    ),
    scheduling=Scheduling(
        max_inflight=1,       # whisper.cpp context is not concurrent-safe
        preferred_batch_size=1,
        fair_queuing=FairQueuing.PER_CAMERA,
    ),
    cost=Cost(currency="USD"),
    supports_stream=False,
)

app = _adapter_app.fastapi_app


def __getattr__(name: str):
    if name == "_service":
        return _adapter_app.service
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
