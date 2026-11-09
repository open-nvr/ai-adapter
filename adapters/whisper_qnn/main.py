"""
Whisper-Small on the Qualcomm Hexagon — contract-compliant FastAPI service.

Run locally (on a Qualcomm host with the qualcomm image flavour):
    OPENNVR_ADAPTER_TOKEN=secret \\
    python -m uvicorn adapters.whisper_qnn.main:app --host 0.0.0.0 --port 9020

Conformance check:
    python -m conformance http://localhost:9020 --token $OPENNVR_ADAPTER_TOKEN
"""
from __future__ import annotations

from adapters.whisper_qnn.service import MAX_AUDIO_BYTES, WhisperQnnService
from opennvr_adapter_sdk import (
    AdapterApp,
    BodyShape,
    Cost,
    FairQueuing,
    Permissions,
    Scheduling,
)

_adapter_app = AdapterApp(
    service_factory=WhisperQnnService,
    name="whisper-qnn-asr",
    version="1.0.0",
    vendor="open-nvr",
    license="AGPL-3.0",
    model_card_url="https://huggingface.co/qualcomm/Whisper-Small-Quantized",
    # audio_transcription only: the camera agent's transcribe() sends
    # exactly this task; translation is the whisper adapter's job.
    tasks_advertised=["audio_transcription"],
    body_shape=BodyShape.AUDIO,
    max_body_bytes=MAX_AUDIO_BYTES,
    permissions=Permissions(
        # The accelerator is the Hexagon NPU, reached through the
        # /dev/fastrpc-cdsp node the compose overlay lends the container
        # — not a GPU in the contract's sense, so no GPU grant is asked.
        gpu=False,
        # The context binaries, tokenizer and mel filters are baked into
        # the image at build; nothing is downloaded at run time.
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
