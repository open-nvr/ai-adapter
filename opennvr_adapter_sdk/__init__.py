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
from opennvr_adapter_sdk.adapter_app import AdapterApp, BodyShape, BODY_BYTES_KEY
from opennvr_adapter_sdk.facade import Adapter, InferCall, Overloaded
from opennvr_adapter_sdk.openapi import adapter_asyncapi
from opennvr_adapter_sdk.service import AdapterService, ServiceError

# Re-export the most commonly-needed contract types so adapter authors
# only need one import line. Less common types (streaming messages,
# DetectionResult, AsrResult, etc.) live in ``opennvr_adapter_sdk.contract``
# — import from there when you need them.
from opennvr_adapter_sdk.contract import (
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

__version__ = "1.2.0"

__all__ = [
    # ── The front door ────────────────────────────────────────────
    # Start here: Adapter + InferCall are the whole of a first
    # adapter. Everything below is what they compile down to.
    "Adapter",
    "InferCall",
    "Overloaded",
    "adapter_asyncapi",
    "AdapterApp",
    "AdapterService",
    "BodyShape",
    "BODY_BYTES_KEY",
    "ServiceError",
    "__version__",
    # contract types
    "Accelerator",
    "AdapterInfo",
    "CapabilitiesResponse",
    "Cost",
    "DetectorSpec",
    "EndpointsInfo",
    "ErrorCategory",
    "ErrorDetail",
    "FailureEnvelope",
    "FairQueuing",
    "HardwareEvaluationResponse",
    "HardwareVerdict",
    "HealthResponse",
    "HealthStatus",
    "InferEndpointInfo",
    "InferResponse",
    "InputSpec",
    "ModelInfo",
    "Permissions",
    "Scheduling",
    "StreamEndpointInfo",
]
