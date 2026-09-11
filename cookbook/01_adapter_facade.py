# Copyright (c) 2026 OpenNVR
# SPDX-License-Identifier: Apache-2.0
"""`Adapter` — the facade, and the only file most adapters need.

Demonstrates: `Adapter`, `@Adapter.load`, `@Adapter.on_image`,
`@Adapter.check_hardware`, `@Adapter.on_shutdown`, `InferCall`,
`InferCall.detection`, `Overloaded`, `Adapter.app`.

An adapter wraps a model and answers the AI Adapter Contract. The
facade derives everything the contract needs except the model itself:
the fingerprint, health, the hardware verdict, the modalities, the body
shape and the error taxonomy.

Run it:

    opennvr-adapter dev            # drive it in-process
    uvicorn 01_adapter_facade:app --port 9000
"""
from opennvr_adapter_sdk import Adapter, Overloaded

adapter = Adapter(
    "fall-detection",
    version="1.0.0",
    vendor="ACME Vision",
    license="Apache-2.0",
    # An app asks for a TASK, never for your adapter by name — matching
    # an existing §5.x convention is what makes this model a drop-in
    # replacement for somebody else's.
    tasks=["object_detection"],
    framework="onnxruntime",
    # Hashed for the fingerprint KAI-C uses to notice a model changing
    # underneath a deployment, and reported as size_mb in /capabilities.
    weights="models/fall.onnx",
    gpu=False,
    max_inflight=1,
)


@adapter.load()
def load():
    """Load once, at startup. Whatever this returns is ``call.model``.

    Import heavy ML libraries HERE rather than at module top: a missing
    dependency then shows up as a red /health carrying the real error,
    instead of a container that will not import at all."""
    import onnxruntime as ort

    return ort.InferenceSession(adapter.weights,
                                providers=["CPUExecutionProvider"])


@adapter.on_image()
def detect(call):
    """THE MODEL. Called once per frame.

    Returning a list is the §5.1 detection convention; returning a dict
    is a result shape of your own."""
    if len(call.image) < 4:
        # A ValueError becomes a 400, which tells KAI-C NOT to retry.
        # Misclassifying a bad frame as a 500 produces a retry storm.
        raise ValueError("frame is empty")

    threshold = float(call.param("confidence_threshold", 0.5))
    outputs = call.model.run(None, {"images": _preprocess(call.image)})

    return [
        # Coordinates are NORMALIZED (0–1 of the frame). Divide pixel
        # values by the frame size, or every box lands in the wrong
        # place on the operator's screen.
        call.detection("fallen", score, x, y, w, h)
        for score, (x, y, w, h) in _decode(outputs)
        if score >= threshold
    ]


@adapter.check_hardware()
def check(model):
    """Optional: the real hardware requirement, tested.

    Return an `HardwareEvaluationResponse`, a `HardwareVerdict`, a bool,
    or a ``(verdict, reasoning)`` pair. Without this the facade derives
    a verdict from whether the model loaded."""
    import onnxruntime as ort

    providers = ort.get_available_providers()
    if "CUDAExecutionProvider" in providers:
        return "ok", "CUDA is available; expect ~120 fps at 640x640."
    return "warn", ("CPU only — expect ~8 fps at 640x640. Fine for one or "
                    "two cameras, not for a site.")


@adapter.on_shutdown()
def release(model):
    """Runs on the way out, with the loaded model."""
    del model


def busy_example(call):
    """Backpressure, when it is honest to shed load. A 503 with a retry
    hint is very different from a model failure: KAI-C backs off and
    comes back rather than counting an error."""
    raise Overloaded("inference queue is full", retry_after_ms=250)


def _preprocess(jpeg: bytes):
    raise NotImplementedError("your preprocessing")


def _decode(outputs):
    return []


# The ASGI application. `uvicorn 01_adapter_facade:app`
app = adapter.app
