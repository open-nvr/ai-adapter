# Copyright (c) 2026 __ADAPTER_NAME__ authors
# SPDX-License-Identifier: __LICENSE__

"""
__ADAPTER_NAME__ — an OpenNVR AI adapter.

Scaffolded by ``opennvr-adapter new``. An adapter wraps a model and
answers the AI Adapter Contract; KAI-C polls it, sends it frames, and
routes the results to whatever app asked for them.

Everything except the model is the SDK's: the contract endpoints, auth,
correlation ids, Prometheus metrics, body parsing, the failure
envelope, the OpenAPI and AsyncAPI documents this adapter publishes
about itself, and the lifespan. What's left for YOU is the loader and
the inference handler below.

Run::

    opennvr-adapter dev                     # drive it in-process
    opennvr-adapter validate .              # the full conformance run
    uvicorn __ADAPTER_MODULE__:app --port __PORT__
"""
from __future__ import annotations

from opennvr_adapter_sdk import Adapter

adapter = Adapter(
    "__ADAPTER_ID__",
    version="1.0.0",
    # TODO: your name or organisation. Operators see it in the catalog.
    vendor="__VENDOR__",
    license="__LICENSE__",
    # TODO: the §5.x task convention(s) this adapter serves. Match an
    # existing one where you can — an app asks for a TASK, not for your
    # adapter by name, so a matching convention is what makes your model
    # a drop-in replacement for someone else's.
    # Examples: object_detection, license_plate_recognition,
    # audio_transcription, image_captioning, speech_synthesis.
    tasks=["__TASK__"],
    # TODO: what runs the model — onnxruntime, torch, tflite, openvino,
    # ultralytics, transformers, or "custom".
    framework="custom",
    # TODO: point at your weights file. The SDK hashes it for the
    # fingerprint KAI-C uses to detect a model changing underneath a
    # deployment, and reports its size in /capabilities.
    # weights="models/__ADAPTER_MODULE__.onnx",
    #
    # TODO: set gpu=True if the model needs one — KAI-C refuses to
    # register an adapter asking for more than the operator granted.
    gpu=False,
    # TODO: raise only if the model is genuinely re-entrant.
    max_inflight=1,
)


@adapter.load()
def load():
    """Load the model once, at startup.

    Import heavy ML libraries HERE, not at module top: a missing
    dependency then shows up as a red /health with the real error
    message, instead of a container that won't import.

    Whatever you return becomes ``call.model``.
    """
    # TODO: load your model.
    #   import onnxruntime as ort
    #   return ort.InferenceSession(adapter.weights)
    return None


@adapter.on_image()
def infer(call):
    """THE MODEL. Called once per frame.

    ``call.image``  the frame bytes (JPEG/PNG as the caller sent it)
    ``call.model``  whatever ``load()`` returned
    ``call.params`` the caller's knobs; ``call.param("threshold", 0.5)``
    ``call.task``   which advertised task this call is for
    ``call.camera_id``

    Return a list of ``call.detection(...)`` items for the §5.1
    detection convention, or any dict for a result shape of your own.

    Raise ``ValueError`` for input this model cannot use — the SDK turns
    it into a 400 so KAI-C does not retry a bad frame. Anything else
    becomes a 500, and ``Overloaded`` a 503 with a retry hint.
    """
    # TODO: preprocess, run the model, shape the answer. Coordinates are
    # NORMALIZED (0–1 of the frame) — divide pixel values by the frame
    # size, or every box lands in the wrong place on the operator's
    # screen.
    #
    #   boxes = call.model.run(None, {"images": preprocess(call.image)})
    #   return [call.detection(label, score, x, y, w, h)
    #           for label, score, (x, y, w, h) in boxes]
    return []


# The ASGI application. `uvicorn __ADAPTER_MODULE__:app`
#
# Every @adapter decorator must appear ABOVE this line — the app is
# built here, and anything registered afterwards would be ignored (the
# SDK raises rather than letting that pass silently).
app = adapter.app


def main() -> None:
    """Console-script entry point: serve the adapter with uvicorn.

    `__ADAPTER_MODULE__` on the PATH after `pip install -e .`, and what
    the Dockerfile could call instead of spelling out uvicorn flags.
    """
    import os

    import uvicorn

    uvicorn.run(
        app,
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "__PORT__")),
    )


if __name__ == "__main__":  # pragma: no cover
    main()
