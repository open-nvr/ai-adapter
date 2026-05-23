# Copyright (c) 2026 OpenNVR
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
InsightFace face-recognition adapter — contract-compliant FastAPI service.

The face logic lives in ``adapters/insightface/service.py``; the SDK
provides everything else — auth, metrics, correlation_id, the six
mandatory contract endpoints, body parsing, error envelope
translation, lifespan.

This adapter ALSO exposes a few face-DB CRUD routes that aren't in
the contract: register, list, get, delete. They're auth-protected
by the SDK's ``AuthAndCorrelationMiddleware`` exactly like the
contract endpoints — the middleware gates every path except the
documented open ones. The legacy adapter exposed these under
``/faces/*`` and the Smart Doorbell example expects the same URLs,
so we preserve them.

Run locally:
    OPENNVR_ADAPTER_TOKEN=secret \\
    python -m uvicorn adapters.insightface.main:app --host 0.0.0.0 --port 9005

Conformance check:
    python -m conformance http://localhost:9005 --token $OPENNVR_ADAPTER_TOKEN
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any

from fastapi import File, Form, HTTPException, UploadFile

from adapters.insightface.face_db import FaceDB
from adapters.insightface.service import (
    MAX_IMAGE_BYTES,
    InsightFaceService,
)
from opennvr_adapter_sdk import (
    AdapterApp,
    BodyShape,
    Cost,
    FairQueuing,
    Permissions,
    Scheduling,
    ServiceError,
)

logger = logging.getLogger(__name__)

# Where the face DB persists between restarts. Operators override via
# ``OPENNVR_INSIGHTFACE_FACE_DB`` (use ``""`` for in-memory only).
_FACE_DB_PATH_DEFAULT = os.path.join("model_weights", "insightface_faces.json")
_FACE_DB_PATH = os.getenv("OPENNVR_INSIGHTFACE_FACE_DB", _FACE_DB_PATH_DEFAULT)
# Resolve to absolute so KAI-C's operator-approval UI surfaces a path
# operators can interpret without knowing the container CWD.
_FACE_DB_PATH_ABS = os.path.abspath(_FACE_DB_PATH) if _FACE_DB_PATH else ""
_SHARED_FACE_DB = FaceDB(storage_path=_FACE_DB_PATH_ABS or None)


def _build_service() -> InsightFaceService:
    """Factory used by AdapterApp's lazy lifespan. Re-binds the shared
    FaceDB to the service so the inference path and the CRUD routes
    see the same registered faces."""
    return InsightFaceService(face_db=_SHARED_FACE_DB)


_adapter_app = AdapterApp(
    service_factory=_build_service,
    name="insightface-recognition",
    version="1.0.0",
    vendor="open-nvr",
    license="AGPL-3.0",
    model_card_url="https://github.com/deepinsight/insightface",
    tasks_advertised=[
        "face_detection",
        "face_recognition",
        "face_embedding",
    ],
    body_shape=BodyShape.IMAGE,
    max_body_bytes=MAX_IMAGE_BYTES,
    permissions=Permissions(
        # CPU works; GPU users override via the OPENNVR_INSIGHTFACE_CTX
        # env var. We declare gpu=False here so KAI-C registers us
        # without a GPU-grant prompt; operators with GPU upgrade can
        # add the permission grant via a config edit.
        gpu=False,
        network_egress=[],
        # The face DB file path on disk — declared so KAI-C
        # sovereignty knows the adapter writes here. Absolute path
        # so the approval prompt is unambiguous regardless of CWD.
        host_filesystem=[_FACE_DB_PATH_ABS] if _FACE_DB_PATH_ABS else [],
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


# ── Face DB CRUD ────────────────────────────────────────────────────
#
# These routes preserve the legacy InsightFace adapter's /faces/*
# surface. The Smart Doorbell example app registers known family
# members through them. They share the same FaceDB instance the
# inference path uses, so a register call is immediately visible to
# the next /infer face_recognition call.
#
# Auth: the SDK's AuthAndCorrelationMiddleware gates every request
# except /health and the grace-window /capabilities + /hardware/evaluation
# (per its ALWAYS_OPEN_PATHS / GRACE_WINDOW_PATHS sets). That covers
# /faces/* automatically — these route handlers don't re-check the
# token. Same auth posture as the contract endpoints (multipart /infer
# etc.) — if the SDK middleware accepted the request, the operator
# already proved the bearer.


@app.post("/faces/register")
async def register_face(
    frame: UploadFile = File(..., description="JPEG/PNG face image"),
    person_id: str = Form(..., description="Stable id used in recognition responses"),
    name: str = Form(..., description="Display name"),
    category: str = Form("unknown", description="e.g. 'family' / 'friend' / 'watchlist'"),
    metadata: str = Form("{}", description="Optional JSON object string"),
) -> dict[str, Any]:
    """Register or update a face. Multipart-only (the natural shape
    for a file upload). Idempotent — re-registering the same
    ``person_id`` overwrites the embedding (useful for re-enrollment
    after a haircut, new glasses, etc.)."""
    person_id_clean = person_id.strip()
    name_clean = name.strip()
    if not person_id_clean:
        raise HTTPException(status_code=400, detail="person_id is required")
    if not name_clean:
        raise HTTPException(status_code=400, detail="name is required")

    try:
        metadata_obj = json.loads(metadata) if metadata else {}
    except json.JSONDecodeError:
        raise HTTPException(
            status_code=400,
            detail="metadata must be a valid JSON object string",
        )
    if not isinstance(metadata_obj, dict):
        raise HTTPException(
            status_code=400,
            detail="metadata must be a JSON object (not array / primitive)",
        )

    image_bytes = await frame.read()
    if not image_bytes:
        raise HTTPException(
            status_code=400,
            detail="frame upload was empty",
        )
    if len(image_bytes) > MAX_IMAGE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"frame exceeds {MAX_IMAGE_BYTES}-byte limit",
        )

    # Reuse the service's embedding path so we don't duplicate the
    # image-decode + face-detect code. Run on a worker thread so a
    # slow CPU-bound ONNX pass doesn't block the event loop while
    # other adapter routes (/capabilities polls, etc.) are waiting.
    service = _adapter_app.service
    if not service.is_ready():
        raise HTTPException(
            status_code=503,
            detail="InsightFace model not yet loaded",
        )
    try:
        embedding_response = await asyncio.to_thread(
            service.infer, {"__file__": image_bytes, "task": "face_embedding"}
        )
    except ServiceError as exc:
        # Translate the typed envelope into an HTTP error the CLI
        # subcommand can print as a one-liner — the SDK's automatic
        # ServiceError translation only runs for /infer routes.
        raise HTTPException(status_code=exc.http_status, detail=exc.message)
    embedding = embedding_response.result.get("embedding")
    if embedding is None:
        raise HTTPException(
            status_code=422,
            detail="no face detected in the supplied image",
        )
    # Refuse multi-face enrollment outright — enrolling the highest-
    # confidence face from a group photo is almost always a mistake
    # (you'd end up registering Alice but with a frame the operator
    # thinks contains Bob). The operator should crop or reshoot.
    face_count = embedding_response.result.get("face_count")
    if isinstance(face_count, int) and face_count > 1:
        raise HTTPException(
            status_code=422,
            detail=(
                f"image contains {face_count} faces; enrolment requires "
                "exactly one face — crop the image to a single subject "
                "or reshoot, then retry"
            ),
        )

    record = _SHARED_FACE_DB.register(
        person_id=person_id_clean,
        name=name_clean,
        embedding=embedding,
        category=(category or "unknown").strip() or "unknown",
        metadata=metadata_obj,
    )
    return {
        "ok": True,
        "face": record.to_public_dict(),
        "face_bbox": embedding_response.result.get("face_bbox"),
    }


@app.get("/faces")
async def list_faces(category: str | None = None) -> dict[str, Any]:
    """List registered faces. Optional ``?category=family`` filter."""
    faces = [r.to_public_dict() for r in _SHARED_FACE_DB.list_records(category=category)]
    return {"faces": faces, "count": len(faces)}


@app.get("/faces/{person_id}")
async def get_face(person_id: str) -> dict[str, Any]:
    record = _SHARED_FACE_DB.get(person_id)
    if record is None:
        raise HTTPException(
            status_code=404,
            detail=f"no registered face with person_id={person_id!r}",
        )
    return {"face": record.to_public_dict()}


@app.delete("/faces/{person_id}")
async def delete_face(person_id: str) -> dict[str, Any]:
    removed = _SHARED_FACE_DB.delete(person_id)
    if not removed:
        raise HTTPException(
            status_code=404,
            detail=f"no registered face with person_id={person_id!r}",
        )
    return {"ok": True, "person_id": person_id}


# PEP 562 module-level __getattr__ — exposes ``_service`` as a synthetic
# attribute so test fixtures that reach into ``main._service`` keep
# working post-SDK-refactor. Same idiom as the YOLOv8 / fast-plate-ocr
# adapters.
def __getattr__(name: str):
    if name == "_service":
        return _adapter_app.service
    if name == "_face_db":
        return _SHARED_FACE_DB
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
