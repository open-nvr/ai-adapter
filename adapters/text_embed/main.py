"""
Sentence embeddings for search's caption arm — contract-compliant FastAPI service.

Run locally:
    OPENNVR_ADAPTER_TOKEN=secret TEXT_EMBED_MODEL_DIR=/path/with/model.onnx \\
    python -m uvicorn adapters.text_embed.main:app --host 0.0.0.0 --port 9013

Conformance check:
    python -m conformance http://localhost:9013 --token $OPENNVR_ADAPTER_TOKEN
"""
from __future__ import annotations

from adapters.text_embed.service import MAX_TEXT_CHARS, TextEmbedService
from opennvr_adapter_sdk import (
    AdapterApp,
    BodyShape,
    Cost,
    FairQueuing,
    Permissions,
    Scheduling,
)

_adapter_app = AdapterApp(
    service_factory=TextEmbedService,
    name="text-embed",
    version="1.0.0",
    vendor="open-nvr",
    license="AGPL-3.0",
    model_card_url="https://huggingface.co/BAAI/bge-small-en-v1.5",
    # Not `embed`: that task is one image+text space for the FRAME arm of
    # search. This is text against text — the caption arm — and core
    # picks it by this name, falling back to the embed adapter's text
    # tower where no text_embed adapter is registered.
    tasks_advertised=["text_embed"],
    body_shape=BodyShape.TEXT,
    max_body_bytes=MAX_TEXT_CHARS * 80,
    permissions=Permissions(
        gpu=False,
        network_egress=[],
        host_filesystem=[],
        shared_memory_paths=[],
        host_metadata=False,
    ),
    scheduling=Scheduling(
        max_inflight=2,
        preferred_batch_size=16,
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
