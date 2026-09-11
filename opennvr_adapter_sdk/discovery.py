# Copyright (c) 2026 OpenNVR
# SPDX-License-Identifier: Apache-2.0

"""
Finding the adapter in a directory.

Shared by ``opennvr-adapter dev``, ``validate`` and ``spec`` so the
three commands can never disagree about what counts as an adapter —
which is its own class of bug: a tool that says "no adapter here" about
a directory another tool happily runs teaches the developer to distrust
both.

An adapter is a top-level module that builds either an
:class:`~.facade.Adapter` or an :class:`~.adapter_app.AdapterApp`. The
module named by ``[project.scripts]`` wins; otherwise the one ``*.py``
that mentions one of them.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

try:  # 3.11+
    import tomllib
except ModuleNotFoundError:  # pragma: no cover — 3.10
    import tomli as tomllib  # type: ignore[no-redef]

_MARKERS = ("Adapter(", "AdapterApp(")


class NotAnAdapter(RuntimeError):
    """Raised with an operator-readable message, not a traceback."""


def find_module(directory: Path) -> str | None:
    """The module name that builds the adapter, or ``None``."""
    pyproject = directory / "pyproject.toml"
    if pyproject.is_file():
        try:
            data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
        except tomllib.TOMLDecodeError:
            data = {}
        for target in (data.get("project", {}).get("scripts") or {}).values():
            module = str(target).split(":", 1)[0].strip()
            if module and (directory / f"{module}.py").is_file():
                return module
    candidates = [
        path.stem for path in sorted(directory.glob("*.py"))
        if any(marker in path.read_text(encoding="utf-8", errors="ignore")
               for marker in _MARKERS)
    ]
    return candidates[0] if candidates else None


def load(directory: Path) -> tuple[Any, str]:
    """Import the adapter module. Returns ``(module, module_name)``."""
    directory = directory.expanduser().resolve()
    if not directory.is_dir():
        raise NotAnAdapter(f"{directory} is not a directory")
    name = find_module(directory)
    if name is None:
        raise NotAnAdapter(
            f"no adapter module found in {directory} — expected a "
            f"[project.scripts] entry in pyproject.toml, or one top-level "
            f"*.py that builds an Adapter(...) or AdapterApp(...)"
        )
    sys.path.insert(0, str(directory))
    try:
        sys.modules.pop(name, None)
        spec = importlib.util.spec_from_file_location(name, directory / f"{name}.py")
        module = importlib.util.module_from_spec(spec)      # type: ignore[arg-type]
        sys.modules[name] = module
        spec.loader.exec_module(module)                     # type: ignore[union-attr]
    except Exception as exc:  # noqa: BLE001 — any import error is the user's
        raise NotAnAdapter(
            f"importing {name!r} failed: {type(exc).__name__}: {exc}") from exc
    finally:
        try:
            sys.path.remove(str(directory))
        except ValueError:
            pass
    return module, name


def facade_of(module: Any) -> Any | None:
    """The module-level :class:`~.facade.Adapter`, if the adapter is
    written against the facade."""
    from opennvr_adapter_sdk.facade import Adapter

    for _name, value in vars(module).items():
        if isinstance(value, Adapter):
            return value
    return None


def adapter_app_of(module: Any) -> Any | None:
    """The module-level :class:`~.adapter_app.AdapterApp`, for an adapter
    written against the classes."""
    from opennvr_adapter_sdk.adapter_app import AdapterApp

    for _name, value in vars(module).items():
        if isinstance(value, AdapterApp):
            return value
    return None


def asgi_app(module: Any) -> Any:
    """The ASGI application to drive, whichever way the adapter is
    written. Raises :class:`NotAnAdapter` with a fixable message."""
    from fastapi import FastAPI

    facade = facade_of(module)
    if facade is not None:
        return facade.app
    wrapper = adapter_app_of(module)
    if wrapper is not None:
        return wrapper.fastapi_app
    for _name, value in vars(module).items():
        if isinstance(value, FastAPI):
            return value
    raise NotAnAdapter(
        f"{module.__name__!r} builds no adapter — expose an Adapter, an "
        f"AdapterApp, or the FastAPI app it produces at module level "
        f"(`app = adapter.app`)"
    )


def identity(module: Any) -> tuple[str, str, tuple[str, ...]]:
    """``(name, version, tasks)`` for whatever the module builds."""
    facade = facade_of(module)
    if facade is not None:
        return facade.id, facade.version, tuple(facade.tasks)
    wrapper = adapter_app_of(module)
    if wrapper is not None:
        return (getattr(wrapper, "_name", "adapter"),
                getattr(wrapper, "_version", "0.0.0"),
                tuple(getattr(wrapper, "_tasks_advertised", ()) or ()))
    return "adapter", "0.0.0", ()


__all__ = ["NotAnAdapter", "find_module", "load", "facade_of",
           "adapter_app_of", "asgi_app", "identity"]
