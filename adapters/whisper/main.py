# Copyright (c) 2026 OpenNVR
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Whisper ASR adapter — contract-compliant FastAPI service.

Migrated to ``opennvr-adapter-sdk`` in A2.3c. This file is now the
minimum viable adapter §3.7 promised: ~50 lines of FastAPI app
construction. Whisper-specific logic lives in
``adapters/whisper/service.py``; the SDK provides everything else
(auth, metrics, correlation_id, all six contract endpoints, body
parsing including AUDIO multipart + JSON-base64, error envelope
translation, lifespan, HTTP 501 on /infer/stream).

Streaming ASR (overlap-window decoding + partial-result emission +
VAD gating) is its own design problem and lands in a follow-up; the
SDK's default ``supports_stream=False`` already refuses the upgrade
with the canonical 501 envelope.

Run locally:
    OPENNVR_ADAPTER_TOKEN=secret \\
    python -m uvicorn adapters.whisper.main:app --host 0.0.0.0 --port 9003

Conformance check:
    python -m conformance http://localhost:9003 --token $OPENNVR_ADAPTER_TOKEN
"""
from __future__ import annotations

import os

from adapters.whisper.service import MAX_AUDIO_BYTES, WhisperService
from app.config import MODEL_WEIGHTS_DIR
from opennvr_adapter_sdk import (
    AdapterApp,
    BodyShape,
    Cost,
    FairQueuing,
    Permissions,
    Scheduling,
)

_adapter_app = AdapterApp(
    # ``service_factory`` (lazy build at lifespan startup) instead of
    # eager ``service=`` so test fixtures that monkey-patch
    # WhisperService.__init__ between module load and TestClient
    # __enter__ take effect.
    service_factory=WhisperService,
    name="whisper-asr",
    version="1.0.0",
    vendor="open-nvr",
    license="AGPL-3.0",
    model_card_url="https://github.com/SYSTRAN/faster-whisper",
    tasks_advertised=["audio_transcription", "audio_translation"],
    body_shape=BodyShape.AUDIO,
    max_body_bytes=MAX_AUDIO_BYTES,
    permissions=Permissions(
        # GPU optional — faster-whisper uses CUDA when available;
        # declared True to capture the §8 operator-approval gate
        # since most production deployments do use CUDA.
        gpu=True,
        network_egress=[],
        # faster-whisper downloads weights from HuggingFace on first
        # load. Under sovereignty=local_only operators pre-populate
        # the weights dir; KAI-C refuses if network_egress isn't
        # empty. Advertise the Whisper-specific subdir (matches the
        # WhisperService default ``download_root``) so KAI-C policy
        # comparison stays narrow — don't widen to MODEL_WEIGHTS_DIR
        # like YOLOv8 does (YOLOv8's weights live AT the root).
        host_filesystem=[os.path.join(MODEL_WEIGHTS_DIR, "whisper")],
        shared_memory_paths=[],
        host_metadata=False,
    ),
    scheduling=Scheduling(
        # Whisper sessions are NOT thread-safe under concurrent
        # transcribe() calls — same singleton caveat as YOLOv8.
        # Honest value is 1.
        max_inflight=1,
        preferred_batch_size=1,
        fair_queuing=FairQueuing.PER_CAMERA,
    ),
    cost=Cost(currency="USD"),
)

app = _adapter_app.fastapi_app


# PEP 562 module-level __getattr__ — exposes ``_service`` as a synthetic
# attribute so tests (and other introspection code) that reach into
# ``main._service`` keep working post-SDK-refactor. Reads the live
# service from the lazily-built AdapterApp.
def __getattr__(name: str):
    if name == "_service":
        return _adapter_app.service
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
