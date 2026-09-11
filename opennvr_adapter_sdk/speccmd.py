# Copyright (c) 2026 OpenNVR
# SPDX-License-Identifier: Apache-2.0

"""
``opennvr-adapter spec`` — emit an adapter's specs without running it.

The same documents the adapter serves at ``/openapi.json`` and
``/asyncapi.json``, produced from the module alone. That is what lets
them go into CI, into a client generator, or into a published API
reference::

    opennvr-adapter spec                          # OpenAPI 3.1, JSON
    opennvr-adapter spec --format asyncapi --yaml
    opennvr-adapter spec -o openapi.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from opennvr_adapter_sdk.discovery import (
    NotAnAdapter, adapter_app_of, asgi_app, facade_of, identity, load,
)


def run_spec(directory: Path, *, fmt: str = "openapi", as_yaml: bool = False,
             output: str | None = None) -> int:
    """Render the requested spec for the adapter in ``directory``."""
    try:
        module, _name = load(Path(directory))
    except NotAnAdapter as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if fmt == "asyncapi":
        from opennvr_adapter_sdk.openapi import adapter_asyncapi

        name, version, tasks = identity(module)
        facade = facade_of(module)
        wrapper = adapter_app_of(module)
        supports_stream = bool(
            getattr(facade, "_stream_fn", None) if facade is not None
            else getattr(wrapper, "_supports_stream", False))
        document = adapter_asyncapi(
            name, version, supports_stream=supports_stream, tasks=tasks,
            license_name=(getattr(facade, "license", "") if facade else ""))
    else:
        try:
            document = asgi_app(module).openapi()
        except NotAnAdapter as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2

    if as_yaml:
        try:
            import yaml
        except ImportError:
            print("error: PyYAML is required for --yaml "
                  "(pip install pyyaml)", file=sys.stderr)
            return 2
        text = yaml.safe_dump(document, sort_keys=False, allow_unicode=True)
    else:
        text = json.dumps(document, indent=2, ensure_ascii=False) + "\n"

    if output:
        path = Path(output).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        print(f"wrote {fmt} ({len(text)} bytes) to {path}", file=sys.stderr)
    else:
        sys.stdout.write(text)
    return 0


__all__ = ["run_spec"]
