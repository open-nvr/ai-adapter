# Copyright (c) 2026 OpenNVR
# SPDX-License-Identifier: Apache-2.0

"""
``opennvr-adapter listing`` — the entry that makes an adapter installable.

Building a conformant adapter is not the same as publishing one. Apps
have had a catalog since the beginning; adapters had none, so a perfect
third-party adapter had nowhere to be listed and no way for an operator
to find it. ``server/config/adapters_index.yml`` in open-nvr is that
catalog, and this command writes an entry for it FROM THE ADAPTER
ITSELF — identity, version, tasks, permissions and model info all come
from ``/capabilities``, so the listing cannot claim something the
adapter does not do.

    opennvr-adapter listing . --image ghcr.io/you/fall-detection:1.0.0

Print it, check it, and open a pull request adding it to the index.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from opennvr_adapter_sdk.discovery import NotAnAdapter, asgi_app, identity, load

#: Fields the reviewer fills in that the adapter cannot know.
PLACEHOLDER = "TODO"


def build_entry(capabilities: dict[str, Any], *, image: str,
                source: str = "", docs_url: str = "",
                contact: str = "") -> dict[str, Any]:
    """One adapters-index entry, from a live ``/capabilities`` payload."""
    adapter = capabilities.get("adapter") or {}
    model = capabilities.get("model") or {}
    permissions = capabilities.get("permissions") or {}
    scheduling = capabilities.get("scheduling") or {}
    endpoints = capabilities.get("endpoints") or {}

    name = adapter.get("name") or "adapter"
    return {
        "id": name,
        "name": _title(name),
        # The reviewer's job: one sentence an operator can decide on.
        "summary": PLACEHOLDER,
        "version": adapter.get("version") or "0.0.0",
        "image": image,
        # What an app asks for. An adapter advertising nothing gets no
        # work, so this is the load-bearing field.
        "tasks_advertised": list(capabilities.get("tasks_advertised") or []),
        "model": {
            "framework": model.get("framework") or PLACEHOLDER,
            "size_mb": model.get("size_mb"),
            "modalities_in": list(model.get("modalities_in") or []),
            "modalities_out": list(model.get("modalities_out") or []),
            # Recorded so an operator can see the adapter identifies its
            # weights at all — KAI-C skips drift detection without one.
            "fingerprinted": bool(model.get("fingerprint")),
        },
        "permissions": {
            "gpu": bool(permissions.get("gpu")),
            "network_egress": list(permissions.get("network_egress") or []),
            "host_filesystem": list(permissions.get("host_filesystem") or []),
        },
        "scheduling": {
            "max_inflight": scheduling.get("max_inflight", 1),
            "fair_queuing": scheduling.get("fair_queuing", "per_camera"),
        },
        "supports_stream": bool(
            (endpoints.get("infer_stream") or {}).get("supported")),
        "license": adapter.get("license") or PLACEHOLDER,
        "vendor": adapter.get("vendor") or PLACEHOLDER,
        "model_card_url": adapter.get("model_card_url") or PLACEHOLDER,
        "docs_url": docs_url or PLACEHOLDER,
        "source": source or PLACEHOLDER,
        "contact": contact or PLACEHOLDER,
    }


def run_listing(directory: Path, *, image: str, source: str = "",
                docs_url: str = "", contact: str = "",
                output: str | None = None) -> int:
    """Render the adapters-index entry for the adapter in ``directory``."""
    try:
        module, _name = load(Path(directory))
        app = asgi_app(module)
    except NotAnAdapter as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    try:
        import yaml
        from fastapi.testclient import TestClient
    except ImportError as exc:
        print(f"error: {exc.name} is required for `listing` "
              f"(pip install 'opennvr-adapter-sdk[yaml]')", file=sys.stderr)
        return 2

    with TestClient(app) as client:
        response = client.get("/capabilities")
        if response.status_code != 200:
            print(f"error: the adapter's /capabilities returned "
                  f"HTTP {response.status_code}; run `opennvr-adapter validate .` "
                  f"first", file=sys.stderr)
            return 1
        capabilities = response.json()

    name, _version, tasks = identity(module)
    entry = build_entry(capabilities, image=image, source=source,
                        docs_url=docs_url, contact=contact)

    warnings: list[str] = []
    if not tasks and not entry["tasks_advertised"]:
        warnings.append(
            "this adapter advertises no task — KAI-C will never route work to "
            "it, and the listing would be undiscoverable")
    if not entry["model"]["fingerprinted"]:
        warnings.append(
            "model.fingerprint is null — KAI-C skips drift detection, so a "
            "model swapped underneath a deployment goes unnoticed")
    if entry["permissions"]["network_egress"]:
        warnings.append(
            "this adapter declares network egress; say in `summary` what it "
            "talks to, because an operator will be asked to allow it")

    text = yaml.safe_dump([entry], sort_keys=False, allow_unicode=True)
    header = (
        f"# Adapters-index entry for {name}.\n"
        f"#\n"
        f"# Fill in every {PLACEHOLDER}, then open a pull request adding this\n"
        f"# to server/config/adapters_index.yml in open-nvr. Everything else\n"
        f"# was read from the adapter's own /capabilities, so the listing\n"
        f"# cannot claim something the adapter does not do.\n"
    )
    document = header + text

    if output:
        path = Path(output).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(document, encoding="utf-8")
        print(f"wrote the listing entry to {path}", file=sys.stderr)
    else:
        sys.stdout.write(document)

    for warning in warnings:
        print(f"warning: {warning}", file=sys.stderr)
    return 0


def _title(adapter_id: str) -> str:
    return " ".join(part.capitalize() for part in adapter_id.split("-"))


__all__ = ["run_listing", "build_entry", "PLACEHOLDER"]
