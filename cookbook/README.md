# The adapter SDK cookbook

One runnable file per class — how it is constructed or subclassed, and
which APIs it uses. Every file is imported and exercised by
`tests/test_cookbook.py`, so an example that names something the SDK no
longer exports breaks in CI rather than misleading you months later.

## Write an adapter

| File | Covers | When you reach for it |
|---|---|---|
| [01_adapter_facade.py](01_adapter_facade.py) | `Adapter`, `InferCall`, `Overloaded` | **Start here.** Declare the model, decorate the loader and the handler. Most adapters need nothing else. |
| [02_adapter_service.py](02_adapter_service.py) | `AdapterService`, `AdapterApp`, `BodyShape` | The classes the facade compiles to, for a model that outgrows the decorators. |

## Answer correctly

| File | Covers |
|---|---|
| [03_result_conventions.py](03_result_conventions.py) | The §5 conventions — detection, classification, transcription — and why following one makes your model a drop-in for somebody else's. |
| [04_errors_and_backpressure.py](04_errors_and_backpressure.py) | `ServiceError`, `ErrorCategory`, `Overloaded`. The category decides whether KAI-C retries, so it is not paperwork. |
| [05_streaming.py](05_streaming.py) | The §6 WebSocket protocol: one session, one warm model, and backpressure the caller can see. |

## Be operable

| File | Covers |
|---|---|
| [06_capabilities_and_metrics.py](06_capabilities_and_metrics.py) | What KAI-C reads and what the operator's Prometheus scrapes, including your own counters. |
| [08_weights_and_packaging.py](08_weights_and_packaging.py) | `ensure_model_file`, baked-in versus fetched weights, and the sovereignty rule that decides which is allowed. |

## Prove it, then ship it

| File | Covers |
|---|---|
| [07_testing_and_conformance.py](07_testing_and_conformance.py) | `ConformanceRunner` in-process, and the tests worth writing about your own model. |
| [09_specs_and_publishing.py](09_specs_and_publishing.py) | The OpenAPI and AsyncAPI documents your adapter publishes, and the listing that makes it installable. |

## Try one

```bash
pip install opennvr-adapter-sdk
opennvr-adapter new my-model     # a runnable adapter + tests
cd my-model
opennvr-adapter dev              # drive it in-process
opennvr-adapter validate .       # the conformance run
```
