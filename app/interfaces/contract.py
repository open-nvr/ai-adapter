# Copyright (c) 2026 OpenNVR
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Backward-compatibility shim — the AI Adapter Contract v1 Pydantic
types now live in ``opennvr_adapter_sdk.contract``.

The move (A2.3d) was made so the SDK can be published to PyPI without
pulling in the rest of the ai-adapter app as a dependency. Imports
from ``app.interfaces.contract`` continue to work via this re-export
so the rest of the codebase doesn't have to update in lockstep, but
new code should prefer ``from opennvr_adapter_sdk.contract import ...``
(or ``from opennvr_adapter_sdk import ...`` for the common subset
re-exported at the package root).

This shim itself is licensed AGPL-3.0 (matching the rest of
``app/``); the SDK module it re-exports is Apache-2.0 to keep
third-party adapter authors out of copyleft scope.
"""
from opennvr_adapter_sdk.contract import *  # noqa: F401, F403
from opennvr_adapter_sdk.contract import __all__  # noqa: F401
