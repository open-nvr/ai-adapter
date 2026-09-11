# OpenNVR Adapter SDK

Publish a model on [OpenNVR](https://opennvr.org). An adapter wraps a
model and answers the AI Adapter Contract, so any deployment can route
work to it and any app can ask for its task by name.

```bash
pip install opennvr-adapter-sdk
```

**Apache-2.0.** Your adapter is yours, under any licence you choose —
the SDK talks to the platform over HTTP and nothing links.

## A whole adapter

```python
from opennvr_adapter_sdk import Adapter

adapter = Adapter(
    "fall-detection",
    version="1.0.0",
    vendor="ACME",
    license="Apache-2.0",
    tasks=["object_detection"],
    framework="onnxruntime",
    weights="models/fall.onnx",
)

@adapter.load()
def load():
    import onnxruntime as ort
    return ort.InferenceSession(adapter.weights)

@adapter.on_image()
def detect(call):
    boxes = call.model.run(None, {"images": preprocess(call.image)})
    return [call.detection("fallen", score, x, y, w, h)
            for score, (x, y, w, h) in boxes]

app = adapter.app          # uvicorn my_model:app
```

That is conformant. The SDK derives what the contract needs and you
would otherwise hand-write: the **fingerprint** from the weights file,
**health** from the loader, the **hardware verdict**, the **modalities**
and **body shape** from the handler you registered, and the **error
taxonomy**.

## Where to go next

<div class="grid cards" markdown>

- **[Quickstart](quickstart.md)** — a conformant adapter in ten minutes,
  with no Docker and no stack.
- **[Concepts](concepts.md)** — the task bargain, and why it decides
  whether your model ever receives work.
- **[Cookbook](cookbook.md)** — one runnable example per class.
- **[API reference](reference/front-door.md)** — every export, in tiers.
- **[Publishing](guides/publishing.md)** — getting listed.

</div>

## The deal

OpenNVR takes **no fee**. Your adapter is yours, at your licence and
your price; the catalog is discovery, not a gate — an adapter is a
container that answers HTTP, and an operator can run yours without
anyone's permission. See
[CONTRIBUTING_ADAPTERS.md](https://github.com/open-nvr/open-nvr/blob/main/docs/CONTRIBUTING_ADAPTERS.md).
