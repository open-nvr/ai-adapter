"""
opennvr-adapter-sdk — the boilerplate-free way to write an AI Adapter
Contract v1 service.

A minimal adapter is now ~30 lines of FastAPI, per §3.7 of the
contract design. The SDK provides:

* ``AdapterService`` — the ABC every adapter implements (4 methods).
* ``AdapterApp``     — wraps the service in a FastAPI app with the
                       six mandatory endpoints, auth, correlation_id,
                       Prometheus metrics, multipart + JSON parsing.
* ``ServiceError``   — typed error envelope matching §7.

Public API:

.. code-block:: python

    from opennvr_adapter_sdk import (
        AdapterApp, AdapterService, ServiceError, BodyShape,
    )
    from opennvr_adapter_sdk.contract_types import (
        ModelInfo, HardwareEvaluationResponse, InferResponse, ...,
    )

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
from opennvr_adapter_sdk.adapter_app import AdapterApp, BodyShape
from opennvr_adapter_sdk.service import AdapterService, ServiceError

# Re-export contract types from the central source-of-truth so adapter
# authors only need one import line.
from app.interfaces.contract import (
    AdapterInfo,
    CapabilitiesResponse,
    Cost,
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
    ModelInfo,
    Permissions,
    Scheduling,
    StreamEndpointInfo,
)

__version__ = "1.0.0"

__all__ = [
    "AdapterApp",
    "AdapterService",
    "BodyShape",
    "ServiceError",
    "__version__",
    # contract types
    "AdapterInfo",
    "CapabilitiesResponse",
    "Cost",
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
    "ModelInfo",
    "Permissions",
    "Scheduling",
    "StreamEndpointInfo",
]
