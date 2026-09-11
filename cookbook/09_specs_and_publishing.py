# Copyright (c) 2026 OpenNVR
# SPDX-License-Identifier: Apache-2.0
"""What the adapter says about itself — and how to ship it.

Demonstrates: `adapter_asyncapi`, `contract_openapi_extras`,
`error_responses`, `infer_request_body`, `CONTRACT_VERSION`, and the
`opennvr-adapter spec` command.

Every adapter publishes two machine-readable documents, both generated
from the contract types it actually returns, so neither can drift from
the implementation:

* **OpenAPI 3.1** at `/openapi.json` — the HTTP surface, with Swagger
  UI at `/docs`. Point a client generator at a running adapter and it
  works.
* **AsyncAPI 3.0** at `/asyncapi.json` — the `/infer/stream` protocol,
  which OpenAPI cannot express.
"""
import json

from opennvr_adapter_sdk import Adapter, adapter_asyncapi
from opennvr_adapter_sdk.openapi import CONTRACT_VERSION

adapter = Adapter("demo-specs", version="2.0.0", vendor="ACME",
                  license="Apache-2.0", tasks=["object_detection"])
adapter.on_image()(lambda call: [])


def the_http_surface() -> dict:
    """FastAPI generates it; the SDK supplies what FastAPI cannot infer
    — the response models, the `/infer` body for this adapter's own
    body shape, the §7 envelope on every error status, and the bearer
    scheme where the middleware enforces it."""
    document = adapter.app.openapi()
    assert document["info"]["x-opennvr-contract-version"] == CONTRACT_VERSION
    assert document["info"]["x-opennvr-tasks"] == ["object_detection"]
    return document


def the_streaming_surface() -> dict:
    """Generated from the §6 message types, so every field a session
    exchanges is described. An adapter that does not stream publishes
    the document with no channels rather than advertising a protocol it
    will answer 501 to."""
    return adapter_asyncapi("demo-specs", "2.0.0", supports_stream=False,
                            tasks=("object_detection",))


def write_them_for_the_docs_site() -> None:
    """Or from the command line, without running the adapter at all::

        opennvr-adapter spec > openapi.json
        opennvr-adapter spec --format asyncapi --yaml > asyncapi.yaml

    Which is what lets them go into CI, a client generator, or a
    published API reference."""
    with open("openapi.json", "w") as handle:
        json.dump(the_http_surface(), handle, indent=2)


# ── Publishing the adapter ─────────────────────────────────────────
#
# 1. `opennvr-adapter validate .` — green means KAI-C will accept it.
# 2. Build and push the image where operators can pull it.
# 3. Write the listing so deployments can find it:
#
#        opennvr-adapter listing . --image ghcr.io/you/demo-specs:1.0.0
#
#    That prints an adapters-index entry — identity, the task it
#    advertises, its permissions, the image digest, and the model card.
#    Open a PR adding it to `server/config/adapters_index.yml` in
#    open-nvr and every deployment can install it.
#
# Your adapter stays yours: the SDK is Apache-2.0 and talks to the
# platform over HTTP, so the licence is your choice.

app = adapter.app
