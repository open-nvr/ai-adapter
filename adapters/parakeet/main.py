"""
Parakeet ASR adapter — contract-compliant FastAPI service.

Run locally:
    OPENNVR_ADAPTER_TOKEN=secret \\
    python -m uvicorn adapters.parakeet.main:app --host 0.0.0.0 --port 9019

Conformance check:
    python -m conformance http://localhost:9019 --token $OPENNVR_ADAPTER_TOKEN
"""
from __future__ import annotations

from adapters.parakeet.service import MAX_AUDIO_BYTES, ParakeetService
from opennvr_adapter_sdk import (
    AdapterApp,
    BodyShape,
    Cost,
    FairQueuing,
    Permissions,
    Scheduling,
)

_adapter_app = AdapterApp(
    service_factory=ParakeetService,
    name="parakeet-asr",
    version="1.0.0",
    vendor="open-nvr",
    license="AGPL-3.0",
    model_card_url="https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3",
    # audio_transcription only: Parakeet does not translate. The agent's
    # transcribe() sends exactly this task.
    tasks_advertised=["audio_transcription"],
    body_shape=BodyShape.AUDIO,
    max_body_bytes=MAX_AUDIO_BYTES,
    permissions=Permissions(
        # CPU is the point: ~0.3 s an utterance on four cores. ONNX
        # Runtime uses CUDA when the image ships it; the stock image is
        # CPU-only and declares no GPU scope.
        gpu=False,
        # The weights are baked into the image (the Dockerfile fills
        # PARAKEET_WEIGHTS_DIR at build), so nothing is downloaded.
        network_egress=[],
        host_filesystem=[],
        shared_memory_paths=[],
        host_metadata=False,
    ),
    scheduling=Scheduling(
        max_inflight=1,
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
