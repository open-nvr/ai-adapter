# Copyright (c) 2026 OpenNVR
# SPDX-License-Identifier: Apache-2.0

"""
opennvr-adapter-sdk — the boilerplate-free way to write an AI Adapter
Contract v1 service.

Start with ``Adapter``
----------------------

``Adapter`` is the front door: declare the model, decorate the loader
and the inference handler, and the contract is answered for you — the
fingerprint from the weights file, health from the loader, the hardware
verdict, the modalities, the error taxonomy, the body shape, and the
OpenAPI and AsyncAPI documents the adapter publishes about itself.

.. code-block:: python

    from opennvr_adapter_sdk import Adapter

    adapter = Adapter(
        "fall-detection", version="1.0.0", vendor="ACME",
        license="Apache-2.0", tasks=["object_detection"],
        framework="onnxruntime", weights="models/fall.onnx",
    )

    @adapter.load()
    def load():
        import onnxruntime as ort
        return ort.InferenceSession(adapter.weights)

    @adapter.on_image()
    def detect(call):
        return [call.detection("fallen", score, x, y, w, h)
                for score, (x, y, w, h) in run(call.model, call.image)]

    app = adapter.app          # uvicorn my_model:app

…or the classes underneath it
-----------------------------

``Adapter`` compiles to these; use them directly when it stops fitting.

* ``AdapterService`` — the ABC every adapter implements (4 methods).
* ``AdapterApp``     — wraps the service in a FastAPI app with the
                       six mandatory endpoints, auth, correlation_id,
                       Prometheus metrics, multipart + JSON parsing.
* ``ServiceError``   — typed error envelope matching §7.

.. code-block:: python

    class MyService(AdapterService):
        def load(self): ...
        def fingerprint(self): return "sha256:..."
        def model_info(self): return ModelInfo(...)
        def hardware_evaluation(self): return HardwareEvaluationResponse(...)
        def infer(self, payload): return InferResponse(...)

    app = AdapterApp(
        service=MyService(),
        name="my-adapter", version="1.0.0", vendor="me", license="MIT",
        tasks_advertised=["my_task"],
    ).fastapi_app

Versioning: the SDK ships with the same major version as the
contract. SDK v1.x targets contract v1; a future contract v2 would
ship SDK v2.x. ``AdapterApp.supported_contract_versions`` defaults
to ``["1"]``; bump when you support both.
"""
from opennvr_adapter_sdk.adapter_app import (  # noqa: F401 — re-export; see API_TIERS
    BODY_BYTES_KEY, AdapterApp, BodyShape,
)
from opennvr_adapter_sdk.facade import (  # noqa: F401 — re-export
    Adapter, InferCall, Overloaded,
)
from opennvr_adapter_sdk.openapi import adapter_asyncapi  # noqa: F401
from opennvr_adapter_sdk.service import (  # noqa: F401 — re-export
    AdapterService, ServiceError,
)

# Re-export the most commonly-needed contract types so adapter authors
# only need one import line. Less common types (streaming messages,
# DetectionResult, AsrResult, etc.) live in ``opennvr_adapter_sdk.contract``
# — import from there when you need them.
from opennvr_adapter_sdk.contract import (  # noqa: F401 — re-exports
    Accelerator,
    AdapterInfo,
    CapabilitiesResponse,
    Cost,
    DetectorSpec,
    EndpointsInfo,
    ErrorCategory,
    ErrorDetail,
    FailureEnvelope,
    FairQueuing,
    HardwareEvaluationResponse,
    HardwareVerdict,
    HealthResponse,
    HealthStatus,
    InferEndpointInfo,
    InferResponse,
    InputSpec,
    ModelInfo,
    Permissions,
    Scheduling,
    StreamEndpointInfo,
)

__version__ = "1.3.0"

# ── The public API, in tiers ──────────────────────────────────────
#
# ``__all__`` is assembled from these, so the tiers ARE the export
# list — there is no second place to update, and the documentation
# site builds its navigation from the same tuples. A name in two
# tiers, or in none, fails tests/test_sdk_public_api.py.

#: Start here — the whole of a first adapter.
FRONT_DOOR: tuple[str, ...] = (
    "Adapter",
    "InferCall",
    "Overloaded",
    "ServiceError",
)

#: What the facade compiles to. Use them directly when a model outgrows the decorators.
CLASSES: tuple[str, ...] = (
    "AdapterService",
    "AdapterApp",
    "BodyShape",
    "BODY_BYTES_KEY",
)

#: What /capabilities and /health say about the adapter and its model.
IDENTITY: tuple[str, ...] = (
    "ModelInfo",
    "AdapterInfo",
    "CapabilitiesResponse",
    "HealthResponse",
    "HealthStatus",
    "HardwareEvaluationResponse",
    "HardwareVerdict",
    "Accelerator",
    "Permissions",
    "Scheduling",
    "Cost",
    "DetectorSpec",
    "InputSpec",
    "EndpointsInfo",
    "InferEndpointInfo",
    "StreamEndpointInfo",
)

#: Shaping a result, and failing in a way the platform can route on.
ANSWERS: tuple[str, ...] = (
    "InferResponse",
    "ErrorCategory",
    "ErrorDetail",
    "FailureEnvelope",
    "FairQueuing",
)

#: The documents the adapter publishes about itself.
SPECS: tuple[str, ...] = (
    "adapter_asyncapi",
)

#: Tier name → the names it exports, in documentation order.
API_TIERS: dict[str, tuple[str, ...]] = {
    "front-door": FRONT_DOOR,
    "classes": CLASSES,
    "identity": IDENTITY,
    "answers": ANSWERS,
    "specs": SPECS,
}

__all__ = ["API_TIERS", "__version__",
           *(name for tier in API_TIERS.values() for name in tier)]
