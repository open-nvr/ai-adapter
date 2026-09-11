# __ADAPTER_NAME__

An [OpenNVR](https://opennvr.org) AI adapter: it wraps a model and
answers the AI Adapter Contract, so any OpenNVR deployment can route
work to it and any app can ask for its task by name.

Scaffolded by `opennvr-adapter new`.

## Quick start

```bash
uv sync                       # or: pip install -e ".[dev]"
opennvr-adapter dev           # drive it in-process — no Docker, no stack
opennvr-adapter validate .    # the full conformance run
uv run pytest -q
```

`dev` loads the model, calls every contract endpoint and sends one
frame, printing what the model answered. `validate` runs the same
checks KAI-C will: a green run means the deployment will accept this
adapter.

## What it does

TODO: one paragraph. What does this adapter detect, transcribe or
generate? What model does it wrap, and where do the weights come from?
An operator reads this before installing it near their cameras.

## Why this model

TODO: accuracy, licence, provenance, size, and what it is bad at. Be
honest about the last one — an operator who finds out later removes the
adapter.

## Configuration

| Environment variable | Meaning |
|---|---|
| `OPENNVR_ADAPTER_TOKEN` | Bearer token KAI-C must present. Unset = dev mode, auth off. |

TODO: add your own (weights path, device, thresholds).

## The contract surface

| Endpoint | What it answers |
|---|---|
| `GET /health` | Liveness and model-load state. |
| `GET /capabilities` | Identity, model info, fingerprint, tasks, permissions. |
| `GET /hardware/evaluation` | Whether this host can run the model well. |
| `GET /metrics` | Prometheus exposition. |
| `POST /infer` | One inference. |
| `GET /openapi.json` | **OpenAPI 3.1** — every response typed, Swagger UI at `/docs`. |
| `GET /asyncapi.json` | **AsyncAPI 3.0** — the streaming protocol. |

```bash
opennvr-adapter spec > openapi.json          # without running it
opennvr-adapter spec --format asyncapi --yaml
```

## Publishing it

Build the image, push it where operators can pull it, and list the
adapter so deployments can find it — see
[the adapter docs](https://github.com/open-nvr/ai-adapter).

Your adapter is yours, under any licence: `opennvr-adapter-sdk` is
Apache-2.0 and talks to the platform over HTTP.
