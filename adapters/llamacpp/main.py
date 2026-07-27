# Copyright (c) 2026 OpenNVR
# SPDX-License-Identifier: Apache-2.0

"""
llama.cpp LLM adapter — Qwen2.5-3B-Instruct (GGUF Q4_K_M), contract-compliant.

The lightweight, governed replacement for the raw ``ollama`` container: the
same local llama.cpp engine, wrapped in the adapter contract (KAI-C registers +
health-polls + audits it) and carrying tool-calls the tool loop needs. Mount
the GGUF at OPENNVR_LLM_MODEL_PATH; bundle the ``llama-server`` binary in the
image. Torch-free, CPU-first, GPU-optional via LLAMACPP_GPU_LAYERS.

Run locally:
    OPENNVR_LLM_MODEL_PATH=/path/Qwen2.5-3B-Instruct-Q4_K_M.gguf \\
    LLAMACPP_SERVER_BIN=/path/llama-server \\
    python -m uvicorn adapters.llamacpp.main:app --host 0.0.0.0 --port 9014

Conformance:
    python -m conformance http://localhost:9014 --token $OPENNVR_ADAPTER_TOKEN
"""
from __future__ import annotations

from adapters.llamacpp.service import LlamaCppService
from opennvr_adapter_sdk import (
    AdapterApp,
    BodyShape,
    Cost,
    FairQueuing,
    Permissions,
    Scheduling,
)

_adapter_app = AdapterApp(
    service_factory=LlamaCppService,
    name="llamacpp-llm",
    version="1.0.0",
    vendor="camera-agent-lite",
    license="Apache-2.0",
    model_card_url="https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF",
    tasks_advertised=["text_generation"],
    body_shape=BodyShape.TEXT,
    permissions=Permissions(
        gpu=False,             # CPU-first; set LLAMACPP_GPU_LAYERS>0 on GPU hosts
        network_egress=[],
        # The GGUF downloads from HuggingFace ONCE on first boot into the
        # weights volume (same first-boot pattern as the whisper adapter).
        # Under sovereignty=local_only operators pre-populate /app/models
        # (or set OPENNVR_LLM_MODEL_URL="") and no egress ever happens.
        host_filesystem=["/app/models"],
        shared_memory_paths=[],
        host_metadata=False,
    ),
    scheduling=Scheduling(
        max_inflight=1,        # single llama-server slot
        preferred_batch_size=1,
        fair_queuing=FairQueuing.NONE,
    ),
    cost=Cost(currency="USD"),
    supports_stream=False,     # v1: one-shot /infer (short answers); WS streaming later
)

app = _adapter_app.fastapi_app


def __getattr__(name: str):
    if name == "_service":
        return _adapter_app.service
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
