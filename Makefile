# Copyright (c) 2026 OpenNVR
# SPDX-License-Identifier: Apache-2.0
#
# Developer targets for the ai-adapter repository and the SDK it ships.

PY ?= python3

.PHONY: help test sdk-test sdk-site sdk-site-serve conform

help:
	@echo "OpenNVR ai-adapter Makefile targets:"
	@echo "  make test              Run the repository test suite."
	@echo "  make sdk-test          Run just the SDK's own suites."
	@echo "  make sdk-site          Build the published adapter reference to site/."
	@echo "  make sdk-site-serve    Serve it with live reload on :8002."

test:
	@$(PY) -m pytest tests/ -q

sdk-test:
	@$(PY) -m pytest tests/test_sdk_*.py -q

# --------------------------------------------------------------------------
# make sdk-site / sdk-site-serve
# --------------------------------------------------------------------------
# The published adapter reference (opennvr.org/adapters): mkdocs-material +
# mkdocstrings, so the site IS the docstrings. Navigation and the reference
# pages come from opennvr_adapter_sdk.API_TIERS via scripts/gen_reference.py
# — tests/test_sdk_docs_site.py fails if the generated pages are stale.
#
# Both steps run in the SAME uv environment: gen_reference.py imports the SDK
# to read API_TIERS, so a bare system python lacks fastapi and fails. mkdocs
# is pinned below 2.0, which removes the plugin system this site needs.
# Generated HTML is not committed.
SDK_DOCS_DEPS = --with "mkdocs<2" --with mkdocs-material --with mkdocstrings-python \
	--with pymdown-extensions --with ruff --with fastapi --with pydantic --python 3.12

sdk-site:
	@uv run $(SDK_DOCS_DEPS) $(PY) scripts/gen_reference.py \
		&& uv run $(SDK_DOCS_DEPS) mkdocs build --strict
	@echo "→ site/index.html"

sdk-site-serve:
	@uv run $(SDK_DOCS_DEPS) $(PY) scripts/gen_reference.py \
		&& uv run $(SDK_DOCS_DEPS) mkdocs serve -a 127.0.0.1:8002

conform:
	@$(PY) -m opennvr_adapter_sdk.scaffold conform $(URL)
