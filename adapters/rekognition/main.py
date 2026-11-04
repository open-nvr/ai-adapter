# Copyright (c) 2026 OpenNVR
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Amazon Rekognition adapter — face_identify and a face gallery, off-site.

The same contract as the attendance InsightFace (adapters/insightface):
``/infer`` with ``task=face_identify``, and the gallery routes OpenNVR
core calls for enrolment — ``POST /faces/register``, ``DELETE
/faces/{person_id}``, ``GET /faces`` — so attendance can be pointed at
either without a change anywhere else. See service.py: this sends faces
to AWS, declares that egress, and is refused under local_only.

Run locally:
    AWS_REGION=ap-south-1 AWS_ACCESS_KEY_ID=… AWS_SECRET_ACCESS_KEY=… \\
    OPENNVR_ADAPTER_TOKEN=secret \\
    python -m uvicorn adapters.rekognition.main:app --host 0.0.0.0 --port 9019
"""
from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

from fastapi import File, Form, HTTPException, UploadFile

from adapters.rekognition.service import MAX_IMAGE_BYTES, RekognitionService
from opennvr_adapter_sdk import (
    AdapterApp,
    BodyShape,
    Cost,
    FairQueuing,
    Permissions,
    Scheduling,
)

logger = logging.getLogger(__name__)

_REGION = os.getenv("AWS_REGION", "us-east-1")

_adapter_app = AdapterApp(
    service_factory=lambda: RekognitionService(),
    name="rekognition",
    version="1.0.0",
    vendor="open-nvr",
    license="AGPL-3.0",
    model_card_url="https://docs.aws.amazon.com/rekognition/latest/dg/collections.html",
    tasks_advertised=["face_identify"],
    body_shape=BodyShape.IMAGE,
    max_body_bytes=MAX_IMAGE_BYTES,
    permissions=Permissions(
        gpu=False,
        # DECLARED, and the point of declaring: every frame and every
        # enrolment photo goes to AWS. KAI-C refuses this adapter under
        # AI_SOVEREIGNTY=local_only until an operator decides otherwise.
        network_egress=[f"https://rekognition.{_REGION}.amazonaws.com"],
        host_filesystem=[],
        shared_memory_paths=[],
        host_metadata=False,
    ),
    scheduling=Scheduling(max_inflight=4, preferred_batch_size=1,
                          fair_queuing=FairQueuing.PER_CAMERA),
    # Billed per image by AWS; one look is 1 + (faces in the frame) calls.
    cost=Cost(currency="USD"),
    supports_stream=False,
)

app = _adapter_app.fastapi_app


def _service() -> RekognitionService:
    service = _adapter_app.service
    if not service.is_ready():
        raise HTTPException(status_code=503, detail="Rekognition is not ready")
    return service


@app.post("/faces/register")
async def register_face(
    frame: UploadFile = File(...),
    person_id: str = Form(...),
    name: str = Form(""),
    category: str = Form(""),
    append: bool = Form(True),
) -> dict[str, Any]:
    """Index one photo of the person. Always appends: a Rekognition
    collection holds faces, not people, and replacing would mean deleting
    first — DELETE /faces/{person_id} does that."""
    pid = person_id.strip()
    if not pid:
        raise HTTPException(status_code=400, detail="person_id is required")
    image = await frame.read()
    if not image:
        raise HTTPException(status_code=400, detail="frame upload was empty")
    if len(image) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail=f"frame exceeds {MAX_IMAGE_BYTES}-byte limit")
    service = _service()
    try:
        samples = await asyncio.to_thread(service.register, pid, image)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None
    return {"ok": True, "face": {"person_id": pid, "name": name, "category": category or "attendance",
                                 "samples": samples}}


@app.delete("/faces/{person_id}")
async def delete_face(person_id: str) -> dict[str, Any]:
    if not await asyncio.to_thread(_service().forget, person_id):
        raise HTTPException(status_code=404, detail=f"no registered face with person_id={person_id!r}")
    return {"ok": True, "person_id": person_id}


@app.get("/faces")
async def list_faces() -> dict[str, Any]:
    people = _service().people()
    faces = [{"person_id": pid, "samples": n} for pid, n in sorted(people.items())]
    return {"faces": faces, "count": len(faces)}


def __getattr__(name: str):
    if name == "_service_instance":
        return _adapter_app.service
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
