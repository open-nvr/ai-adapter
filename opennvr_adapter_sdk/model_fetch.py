# Copyright (c) 2026 OpenNVR
# SPDX-License-Identifier: Apache-2.0

"""First-boot model download for adapters whose weights ship outside the image.

The pattern (established by the whisper adapter, where faster-whisper does the
same internally): the image ships WITHOUT weights, and on first ``load()`` the
adapter fetches them into its mounted weights volume. Every later boot finds
the files already present and never touches the network. Operators running
``sovereignty=local_only`` pre-populate the weights dir instead — a present
file always wins and no download is attempted.

Stdlib-only (urllib) so it adds no dependency to any adapter image.
"""
from __future__ import annotations

import logging
import os
import shutil
import urllib.request

_DEFAULT_TIMEOUT_S = 60  # per-read timeout, not whole-file — GGUFs are gigabytes


def ensure_model_file(
    path: str,
    url: str,
    *,
    label: str = "model",
    logger: logging.Logger | None = None,
) -> str:
    """Make sure ``path`` exists, downloading it from ``url`` if missing.

    Returns ``path``. Raises ``FileNotFoundError`` when the file is missing
    and no URL is configured (the operator explicitly disabled fetching by
    setting the ``*_MODEL_URL`` env var to an empty string).

    Downloads stream to ``<path>.part`` and are renamed into place only on
    success, so a killed container never leaves a truncated file that a later
    boot mistakes for real weights.
    """
    log = logger or logging.getLogger(__name__)
    if os.path.isfile(path) and os.path.getsize(path) > 0:
        return path
    if not url:
        raise FileNotFoundError(
            f"{label} not found at '{path}' and no download URL is configured. "
            "Mount the file there, or set the corresponding *_MODEL_URL."
        )
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    part = f"{path}.part"
    log.info("downloading %s from %s -> %s", label, url, path)
    try:
        with urllib.request.urlopen(url, timeout=_DEFAULT_TIMEOUT_S) as resp:
            if getattr(resp, "status", 200) not in (None, 200):
                raise OSError(f"HTTP {resp.status}")
            with open(part, "wb") as out:
                shutil.copyfileobj(resp, out, length=8 * 1024 * 1024)
    except Exception:
        try:
            os.remove(part)
        except OSError:
            pass
        raise
    os.replace(part, path)
    log.info("%s ready (%.1f MB)", label, os.path.getsize(path) / 1e6)
    return path
