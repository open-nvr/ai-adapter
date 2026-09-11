# Copyright (c) 2026 OpenNVR
# SPDX-License-Identifier: Apache-2.0
"""The public API has a front door, and the front door is enforced.

Thirty exports with no ordering is a wall: a model developer cannot tell
what to learn first from what exists for the one adapter that needs it.
`API_TIERS` is that ordering, and because `__all__` is assembled from
it, the tiers cannot fall out of step with the exports — nor can the
documentation site, which builds its navigation from the same tuples.
"""
from __future__ import annotations

import importlib
import inspect
import pkgutil

import pytest

import opennvr_adapter_sdk as sdk

#: Reachable from the package but deliberately not exported.
NOT_PUBLIC = {"annotations", "Metrics",
              *(tier.upper().replace("-", "_") for tier in sdk.API_TIERS)}

#: Submodules whose docstring lives on the package they represent.
SKIP_MODULES = {"templates"}


def all_tiered() -> list[str]:
    return [name for tier in sdk.API_TIERS.values() for name in tier]


def test_every_tiered_name_exists():
    missing = [name for name in all_tiered() if not hasattr(sdk, name)]
    assert missing == [], f"tiered but not importable: {missing}"


def test_no_name_is_in_two_tiers():
    names = all_tiered()
    duplicates = sorted({n for n in names if names.count(n) > 1})
    assert duplicates == [], f"exported from more than one tier: {duplicates}"


def test_all_is_assembled_from_the_tiers():
    assert sdk.__all__ == ["API_TIERS", "__version__", *all_tiered()]


def test_the_front_door_stays_small():
    """The first tier is what a model developer reads before writing
    anything. Past a handful of names it has stopped being a front
    door."""
    assert len(sdk.API_TIERS["front-door"]) <= 6


def test_every_public_name_is_tiered():
    exported = set(sdk.__all__)
    submodules = {m.name for m in pkgutil.iter_modules(sdk.__path__)}
    stray = sorted(
        name for name in vars(sdk)
        if not name.startswith("_")
        and name not in exported
        and name not in submodules
        and name not in NOT_PUBLIC
    )
    assert stray == [], f"reachable but in no tier: {stray}"


@pytest.mark.parametrize("name", all_tiered())
def test_every_export_is_documented(name):
    """The documentation site is generated from these docstrings, so a
    missing one is a blank page."""
    obj = getattr(sdk, name)
    if not (inspect.isclass(obj) or inspect.isfunction(obj)):
        return
    assert (obj.__doc__ or "").strip(), f"{name} has no docstring"


def test_every_submodule_has_a_module_docstring():
    for module in pkgutil.iter_modules(sdk.__path__):
        if module.name.startswith("_") or module.name in SKIP_MODULES:
            continue
        imported = importlib.import_module(f"opennvr_adapter_sdk.{module.name}")
        assert (imported.__doc__ or "").strip(), \
            f"opennvr_adapter_sdk.{module.name} has no module docstring"


def test_the_version_tracks_the_contract_major():
    """SDK 1.x targets contract v1. A contract v2 ships SDK 2.x, which
    is what the scaffold's dependency pin relies on."""
    from opennvr_adapter_sdk.openapi import CONTRACT_VERSION

    assert sdk.__version__.split(".")[0] == CONTRACT_VERSION
