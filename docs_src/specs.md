# Specs your adapter publishes

Both documents are generated from the contract types the adapter
actually returns, so neither can drift from the implementation.

```bash
opennvr-adapter spec                          # OpenAPI 3.1, JSON
opennvr-adapter spec --format asyncapi --yaml
curl http://localhost:9000/openapi.json       # …or ask a running adapter
```

## OpenAPI 3.1 — `/openapi.json`

Every response typed from the contract models, the `/infer` request body
described for **this adapter's** body shape (both the multipart route
KAI-C uses and the base64 JSON one, with the size limit), the §7 failure
envelope on every error status, `/metrics` declared as Prometheus text,
and bearer auth declared exactly where the middleware enforces it —
`/health` and `/metrics` stay open so an operator can scrape an adapter
that is failing to load.

Swagger UI comes with it, at `/docs`.

## AsyncAPI 3.0 — `/asyncapi.json`

OpenAPI stops at the door of a WebSocket. The `/infer/stream` protocol
is published here instead: all ten §6 message types, generated from the
contract models the session exchanges, with the directions right.

An adapter that does not stream publishes the document with no channels
rather than advertising a protocol it will answer 501 to.

::: opennvr_adapter_sdk.openapi
    options:
      members: false
