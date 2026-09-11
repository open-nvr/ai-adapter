# Copyright (c) 2026 OpenNVR
# SPDX-License-Identifier: Apache-2.0
"""Getting the weights there — `ensure_model_file`, and what to bake in.

Demonstrates: `ensure_model_file`, `Adapter(weights=...)`, the
fingerprint's relationship to the weights, and the sovereignty rule
that decides whether a download may happen at all.

Two honest options, and the choice is the operator's as much as yours:

* **Bake the weights into the image.** Reproducible, works air-gapped,
  and the fingerprint is fixed at build time. Costs image size.
* **Fetch on first load.** A small image and shared weights across
  replicas, but it needs egress — which a deployment running
  `sovereignty=local_only` will not grant. `ensure_model_file` handles
  both: a file already present always wins and nothing is downloaded,
  so an operator can pre-populate the volume and the same image works
  air-gapped.
"""
import logging
import os

from opennvr_adapter_sdk import Adapter
from opennvr_adapter_sdk.model_fetch import ensure_model_file

logger = logging.getLogger("demo-weights")

WEIGHTS_DIR = os.getenv("MODEL_DIR", "/models")
WEIGHTS_PATH = f"{WEIGHTS_DIR}/fall-detection.onnx"

adapter = Adapter(
    "demo-weights",
    version="1.0.0",
    tasks=["object_detection"],
    framework="onnxruntime",
    # The SDK hashes this file for the fingerprint. Point it at the
    # final location, not a temporary one — the hash must be stable
    # across restarts or KAI-C reads every boot as a model change.
    weights=WEIGHTS_PATH,
    # Declare the egress. An operator who sees an undeclared host in
    # their audit log removes the adapter.
    network_egress=["huggingface.co"],
)


@adapter.load()
def load():
    """Fetch once, then never touch the network again.

    `ensure_model_file` is stdlib-only, so it adds no dependency to the
    image, and it is a no-op when the file is already there."""
    ensure_model_file(
        WEIGHTS_PATH,
        "https://huggingface.co/acme/fall-detection/resolve/main/model.onnx",
        label="fall-detection weights",
        logger=logger,
    )
    import onnxruntime as ort

    return ort.InferenceSession(WEIGHTS_PATH)


@adapter.on_image()
def detect(call):
    return []


# ── What the Dockerfile does with this ─────────────────────────────
#
# Baked in — reproducible, air-gap-ready, larger image:
#
#     COPY models/fall-detection.onnx /models/
#
# Fetched on first boot — small image, needs egress and a volume that
# survives a restart, or every restart re-downloads:
#
#     VOLUME /models
#
# Either way the adapter code above is identical, which is the point:
# the operator chooses at deploy time, not you at build time.

app = adapter.app
