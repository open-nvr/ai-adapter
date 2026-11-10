"""
SigLIP2 embeddings on the Qualcomm Hexagon — contract-compliant FastAPI service.

Run locally (on a Qualcomm host with the qualcomm image flavour):
    OPENNVR_ADAPTER_TOKEN=secret \\
    python -m uvicorn adapters.siglip_qnn.main:app --host 0.0.0.0 --port 9011

Conformance check:
    python -m conformance http://localhost:9011 --token $OPENNVR_ADAPTER_TOKEN
"""
from __future__ import annotations

from adapters.siglip_qnn.service import MAX_IMAGE_BYTES, SiglipQnnService
from opennvr_adapter_sdk import (
    AdapterApp,
    BodyShape,
    Cost,
    FairQueuing,
    Permissions,
    Scheduling,
)

_adapter_app = AdapterApp(
    service_factory=SiglipQnnService,
    # The same name the compose register job gives the CPU clip adapter:
    # core asks KAI-C for an adapter advertising `embed` and takes the one
    # registered; which image answers is the deployment's choice.
    name="siglip-qnn",
    version="1.0.0",
    vendor="open-nvr",
    license="AGPL-3.0",
    model_card_url="https://huggingface.co/qualcomm/SigLIP2",
    tasks_advertised=["embed"],
    body_shape=BodyShape.TEXT,
    max_body_bytes=MAX_IMAGE_BYTES,
    permissions=Permissions(
        # The accelerator is the Hexagon NPU through /dev/fastrpc-cdsp,
        # not a GPU in the contract's sense. The weights and tokenizer
        # are baked into the image: no egress, no host paths.
        gpu=False,
        network_egress=[],
        host_filesystem=[],
        shared_memory_paths=[],
        host_metadata=False,
    ),
    scheduling=Scheduling(
        max_inflight=1,
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
