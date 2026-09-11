# Concepts

## An app asks for a task, never for your adapter

An app's manifest says `requires_tasks: [license_plate_recognition]`.
It does not name an adapter, and it cannot. **Any adapter advertising
that task satisfies it** — which is what makes a better plate reader a
drop-in for the shipped one, swappable by the operator without touching
a single app.

The consequence runs both ways, and it is the single most important
thing to get right:

| You do | What happens |
|---|---|
| Advertise an existing convention (`server/config/tasks.yml`) | Every app wanting that capability can use your model on day one |
| Invent a task name | Nothing asks for it. Your adapter installs, reports healthy, and receives no work — silently |

`opennvr-adapter validate` and the index validator both check this,
because the failure has no other symptom.

## The six endpoints

| | Read by | For |
|---|---|---|
| `GET /health` | KAI-C, the orchestrator | May this adapter take traffic? Unauthenticated, so a failing adapter can still be scraped. |
| `GET /capabilities` | KAI-C | What it is, what it can do, what it needs. Re-read every 60s. |
| `GET /hardware/evaluation` | the operator | Can this host run it well? Rendered verbatim, so make the reasoning actionable. |
| `GET /metrics` | Prometheus | Whether the model is keeping up. |
| `POST /infer` | KAI-C | One inference. |
| `WS /infer/stream` | KAI-C | Many, down one warm session. Optional. |

Plus two the SDK adds: `GET /openapi.json` and `GET /asyncapi.json`.

## The fingerprint is a safety feature

`model.fingerprint` is a content hash of the weights. KAI-C records it
at registration and on every poll, and a change is a tamper signal that
raises an audit event.

**A null fingerprint is skipped**, so an adapter without one is silently
exempt from the check that protects the operator. `Adapter(weights=...)`
hashes the file for you; without a weights file the facade still
derives a deterministic value rather than returning nothing.

## Errors are routing information

| Category | Status | Retried? | Use it when |
|---|---|---|---|
| `transport_error` | 400 | no | the request is malformed |
| `permission_denied` | 403 | no | policy refused this call |
| `not_supported` | 501 | no | this adapter does not do that |
| `model_error` | 500 | no | valid input, the model failed |
| `provider_error` | 502 | yes | an upstream dependency failed |
| `overloaded` | 503 | yes | backpressure — come back later |

Classifying a bad frame as a `model_error` turns one dropped frame into
a retry storm. Classifying a genuine overload as a `model_error` gets
your adapter reported as broken.

## Permissions are a contract with the operator

`Permissions` in `/capabilities` declares what the adapter needs: a GPU,
egress to named hosts, host filesystem paths. KAI-C **refuses to
register an adapter asking for more than the operator granted** — so
over-declaring blocks the install, and an undeclared egress host turning
up in an audit log gets the adapter removed.

Declare exactly what you use, and say so in the listing summary.

## Body shapes

Which decorator you use determines how `/infer` parses a request, and
therefore what the published OpenAPI document tells a caller to send.

| Decorator | Body shape | Multipart field | JSON field |
|---|---|---|---|
| `@adapter.on_image` | `IMAGE` | `frame` | `frame_b64` |
| `@adapter.on_audio` | `AUDIO` | `audio` | `audio_b64` |
| `@adapter.on_data` | `GENERIC` | `data` | `data_b64` |
| `@adapter.on_text` | `TEXT` | — | the JSON body itself |

KAI-C sends multipart; the base64 route exists so a shell, a test or a
quick curl can reach the same endpoint.
