# Copyright (c) 2026 OpenNVR
# SPDX-License-Identifier: Apache-2.0
"""Shaping the answer — the §5 result conventions.

Demonstrates: `InferCall.detection`, `DetectionResult`, `DetectionItem`,
`NormalizedBBox`, `FrameDimensions`, `ClassificationResult`,
`AsrResult`, `AsrSegment`, `LlmChatResult`, `InferResponse`.

`result` is free-form on the wire, but an app subscribing to a TASK
expects that task's convention. Follow one and your model is a drop-in
for another; invent a shape and every consumer needs special-casing.
"""
from opennvr_adapter_sdk import Adapter, InferResponse
from opennvr_adapter_sdk.contract import (
    AsrResult, AsrSegment, ClassificationItem, ClassificationResult,
    DetectionItem, DetectionResult, FrameDimensions, NormalizedBBox,
)

detector = Adapter("demo-detector", tasks=["object_detection"])


# ── §5.1 detection — boxes on a frame ──────────────────────────────


@detector.on_image()
def detect(call):
    """The easy route: a list of `call.detection(...)` items. The facade
    wraps it as `{"detections": [...]}` and clamps the coordinates."""
    return [
        call.detection("person", 0.94, 0.11, 0.22, 0.13, 0.41, track_id=7),
        call.detection("car", 0.88, 0.60, 0.55, 0.28, 0.20,
                       # Anything extra lands in `attributes`.
                       colour="white"),
    ]


def typed_detection() -> dict:
    """The precise route, when you want the frame dimensions too — the
    contract types validate the shape before it reaches the wire, so a
    box outside the frame fails here rather than in someone's UI."""
    return DetectionResult(
        detections=[DetectionItem(
            label="person", confidence=0.94,
            bbox=NormalizedBBox(x=0.11, y=0.22, w=0.13, h=0.41),
            track_id=7, attributes={"pose": "standing"},
        )],
        frame_dimensions=FrameDimensions(w=1920, h=1080),
    ).model_dump(mode="json")


# ── §5.2 classification — a label for the whole frame ──────────────


def classify() -> dict:
    return ClassificationResult(predictions=[
        ClassificationItem(label="daylight", confidence=0.97),
        ClassificationItem(label="overcast", confidence=0.61),
    ]).model_dump(mode="json")


# ── §5.3 transcription — audio in, text out ────────────────────────

transcriber = Adapter("demo-asr", tasks=["audio_transcription"],
                      modalities_out=["text"])


@transcriber.on_audio()
def transcribe(call):
    """An audio adapter gets `call.audio`. Returning a dict passes it
    through verbatim, so a convention type is the natural thing to
    return."""
    return AsrResult(
        transcript="the gate is open",
        language="en",
        segments=[AsrSegment(start_ms=0, end_ms=1400, text="the gate is open")],
    ).model_dump(mode="json")


# ── Full control ───────────────────────────────────────────────────

captioner = Adapter("demo-captioner", tasks=["image_captioning"])


@captioner.on_image()
def caption(call):
    """Return an `InferResponse` when the envelope itself matters —
    reporting a different model name per call (a router adapter), or a
    latency you measured yourself around a remote provider."""
    return InferResponse(
        model_name="blip2-opt-2.7b",
        model_version="1.0.0",
        inference_ms=412,
        result={"caption": "a delivery van at the gate"},
    )
