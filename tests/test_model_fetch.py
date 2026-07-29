# Copyright (c) 2026 OpenNVR
# SPDX-License-Identifier: Apache-2.0
"""ensure_model_file — first-boot model download helper (SDK 1.1.0)."""
import pytest

from opennvr_adapter_sdk.model_fetch import ensure_model_file


def test_existing_file_wins_no_network(tmp_path):
    target = tmp_path / "model.gguf"
    target.write_bytes(b"WEIGHTS")
    # URL is garbage on purpose — a present file must short-circuit before
    # any network access (the sovereignty guarantee).
    out = ensure_model_file(str(target), "http://256.0.0.1/nope")
    assert out == str(target)
    assert target.read_bytes() == b"WEIGHTS"


def test_missing_file_no_url_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="no download URL"):
        ensure_model_file(str(tmp_path / "model.gguf"), "")


def test_download_from_url(tmp_path):
    src = tmp_path / "src.bin"
    src.write_bytes(b"MODELBYTES" * 100)
    target = tmp_path / "weights" / "model.bin"   # parent dir gets created
    out = ensure_model_file(str(target), src.as_uri())
    assert out == str(target)
    assert target.read_bytes() == src.read_bytes()
    assert not target.with_suffix(".bin.part").exists()


def test_failed_download_leaves_no_partial(tmp_path):
    target = tmp_path / "model.bin"
    with pytest.raises(Exception):
        ensure_model_file(str(target), (tmp_path / "missing-src.bin").as_uri())
    assert not target.exists()
    assert not (tmp_path / "model.bin.part").exists()


def test_empty_existing_file_is_refetched(tmp_path):
    src = tmp_path / "src.bin"
    src.write_bytes(b"REAL")
    target = tmp_path / "model.bin"
    target.write_bytes(b"")   # truncated leftover must not count as present
    ensure_model_file(str(target), src.as_uri())
    assert target.read_bytes() == b"REAL"
