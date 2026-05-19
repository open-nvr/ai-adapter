# Copyright (c) 2026 OpenNVR
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Piper TTS adapter — contract-compliant service.

Migrated to ``opennvr-adapter-sdk`` in A2.3a. This file is now the
minimum viable adapter §3.7 promised: ~30 lines of FastAPI app
construction. The Piper-specific logic lives in
``adapters/piper/service.py``; the SDK provides everything else
(auth, metrics, correlation_id, all six contract endpoints, body
parsing, error envelope translation, lifespan).

Run locally:
    OPENNVR_ADAPTER_TOKEN=secret \\
    python -m uvicorn adapters.piper.main:app --host 0.0.0.0 --port 9001
"""
from __future__ import annotations

from fastapi.responses import JSONResponse

from adapters.piper.service import PiperService
from opennvr_adapter_sdk import (
    AdapterApp,
    BodyShape,
    Cost,
    Permissions,
    Scheduling,
)

_adapter_app = AdapterApp(
    # ``service_factory`` (lazy build at lifespan startup) instead of
    # eager ``service=`` so test fixtures that monkey-patch
    # PiperService.__init__ between module load and TestClient.__enter__
    # take effect. Production deployments don't care which path is used.
    service_factory=PiperService,
    name="piper-tts",
    version="1.0.0",
    vendor="open-nvr",
    license="AGPL-3.0",
    model_card_url="https://github.com/rhasspy/piper",
    tasks_advertised=["speech_synthesis"],
    body_shape=BodyShape.TEXT,
    permissions=Permissions(gpu=False),
    scheduling=Scheduling(max_inflight=4, preferred_batch_size=1),
    cost=Cost(currency="USD"),
)

app = _adapter_app.fastapi_app


# Piper-specific extra endpoint. Mounted directly on the SDK-built
# FastAPI app rather than declared as an ExtraEndpoint in
# capabilities — operators discover /voices via the README, not via
# the contract. (Future SDK feature: pass an ``extra_routes=`` list to
# AdapterApp so they're declared in capabilities.endpoints.extra too.)
@app.get("/voices")
def list_voices():
    return JSONResponse(content=_adapter_app.service.list_voices())


# PEP 562 module-level __getattr__ — exposes ``_service`` as a synthetic
# attribute so tests (and other introspection code) that reach into
# ``main._service`` keep working post-SDK-refactor. Reads the live
# service from the lazily-built AdapterApp.
def __getattr__(name: str):
    if name == "_service":
        return _adapter_app.service
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
