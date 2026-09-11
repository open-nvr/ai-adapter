# Quickstart

Ten minutes, no Docker, no stack, no camera.

## 1. Scaffold

```bash
pip install opennvr-adapter-sdk
opennvr-adapter new my-model --task object_detection
cd my-model
uv sync --extra dev            # or: pip install -e '.[dev]'
```

You get a runnable adapter, a Dockerfile with a healthcheck on the
contract's own endpoint, a README, and smoke tests that already pass.

## 2. Drive it

```bash
opennvr-adapter dev
```

```
opennvr-adapter dev — my-model 1.0.0
  loading the model…                          ok (0.3s)

  GET  /health                  ok      status=ok
  GET  /capabilities            ok      tasks: object_detection
  GET  /hardware/evaluation     ok      ok — The model loaded on this host.
  GET  /metrics                 ok      40 lines
  POST /infer (built-in 1x1 JPEG) ok    12ms
        detections: 1
          fallen          0.91  bbox 0.10,0.20 0.30x0.40

  All green.
```

Everything goes through the real ASGI app — the same routes, the same
body parsing, the same failure envelope — so what passes here is what
KAI-C will see. `--image photo.jpg` sends a real file, `--param
threshold=0.8` adds caller params, `--repeat 20` warms the model up.

## 3. Write the model

Two functions:

```python
@adapter.load()
def load():
    import onnxruntime as ort
    return ort.InferenceSession(adapter.weights)

@adapter.on_image()
def detect(call):
    return [call.detection("fallen", 0.91, 0.1, 0.2, 0.3, 0.4)]
```

`call` carries `image` / `audio` / `text` / `data`, `params`,
`param(name, default)`, `task`, `camera_id`, and `model`. Return a list
for the §5.1 detection convention, a dict for a shape of your own, or an
`InferResponse` for full control.

!!! warning "Coordinates are normalized"
    `0–1` of the frame, not pixels. A pixel box passes every test you
    write and lands in the wrong place on the operator's screen. Divide
    by the frame size.

Errors classify themselves: a `ValueError` becomes a 400 that KAI-C
does not retry, anything else a 500, and `Overloaded` a 503 with a
retry hint. Raise `ServiceError` directly when you want to be precise.

## 4. Check it

```bash
uv run pytest -q
opennvr-adapter validate .
```

```
opennvr-adapter validate — my-model 1.0.0
  ✓ health / capabilities / hardware_evaluation / metrics / infer
  OK — 6 passed, 1 warning(s). KAI-C will accept this adapter.
```

`validate` runs the same checks KAI-C runs, in-process. Put it in your
CI: it is the only assurance you can get without a deployment to try it
in.

## 5. Publish it

```bash
docker build -t ghcr.io/you/my-model:1.0.0 .
opennvr-adapter listing . --image ghcr.io/you/my-model:1.0.0
```

The listing is generated from your adapter's own `/capabilities`. Fill
in the `TODO`s and open a pull request against
[`adapters_index.yml`](https://github.com/open-nvr/open-nvr/blob/main/server/config/adapters_index.yml).
See [Publishing](guides/publishing.md).
