# Copyright (c) 2026 OpenNVR
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Unit tests for the InsightFace face DB.

The service-level integration is covered in
``test_insightface_service.py``; these tests pin behaviours that are
properties of the DB itself (tiebreak rules, normalisation, JSON
round-trips, threshold semantics).
"""
from __future__ import annotations

import json
import os
import tempfile

import pytest

from adapters.insightface.face_db import FaceDB


def _make_unit_embedding(seed: int) -> list[float]:
    """A trivial 4-d L2-normalised vector with a stable shape per seed."""
    base = [float((seed + i) % 7) + 0.1 for i in range(4)]
    norm = sum(v * v for v in base) ** 0.5
    return [v / norm for v in base]


# ── best_match tiebreak ────────────────────────────────────────────


def test_best_match_tiebreak_prefers_smaller_person_id() -> None:
    """When two faces score the same similarity, ``person_id`` sort
    order — not dict insertion order — picks the winner. This makes
    the recognition outcome deterministic across processes that
    re-load the DB in a different sequence."""
    db = FaceDB()
    # Register in "wrong" order: zelda first so dict insertion order
    # would prefer her if we weren't doing a proper tiebreak.
    db.register(
        person_id="zelda",
        name="Zelda",
        embedding=_make_unit_embedding(1),
    )
    db.register(
        person_id="alice",
        name="Alice",
        embedding=_make_unit_embedding(1),  # identical embedding → tie
    )

    match = db.best_match(_make_unit_embedding(1), threshold=0.5)
    assert match is not None
    assert match["person_id"] == "alice"


def test_best_match_higher_similarity_still_wins_over_smaller_id() -> None:
    """Tiebreak only triggers on exact equality — higher similarity
    always wins regardless of ``person_id`` ordering."""
    db = FaceDB()
    db.register(
        person_id="alice",
        name="Alice",
        embedding=_make_unit_embedding(1),
    )
    db.register(
        person_id="zelda",
        name="Zelda",
        embedding=_make_unit_embedding(2),  # different embedding
    )

    # Query is closer to zelda's embedding than alice's.
    match = db.best_match(_make_unit_embedding(2), threshold=0.5)
    assert match is not None
    assert match["person_id"] == "zelda"


def test_best_match_below_threshold_returns_none() -> None:
    db = FaceDB()
    db.register(
        person_id="alice",
        name="Alice",
        embedding=_make_unit_embedding(1),
    )
    # A query orthogonal-ish to alice should score below a high threshold.
    other = _make_unit_embedding(99)
    assert db.best_match(other, threshold=0.99) is None


# ── persistence round-trip ─────────────────────────────────────────


def test_persistence_round_trip_atomic() -> None:
    """Register → reopen → records are intact, and the on-disk file
    isn't left in a half-written ``.tmp`` state."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "faces.json")
        db1 = FaceDB(storage_path=path)
        db1.register(
            person_id="alice",
            name="Alice",
            embedding=_make_unit_embedding(1),
            category="family",
            metadata={"role": "owner"},
        )
        assert os.path.exists(path), "register should persist immediately"
        assert not os.path.exists(path + ".tmp"), "tmp file should be renamed"

        raw = json.loads(open(path).read())
        assert raw["schema_version"] == 1
        assert raw["records"][0]["person_id"] == "alice"
        assert raw["records"][0]["category"] == "family"

        # Reload into a fresh instance and verify match still finds alice.
        db2 = FaceDB(storage_path=path)
        match = db2.best_match(_make_unit_embedding(1), threshold=0.5)
        assert match is not None
        assert match["person_id"] == "alice"
        assert match["category"] == "family"


def test_delete_removes_from_disk() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "faces.json")
        db = FaceDB(storage_path=path)
        db.register(
            person_id="bob",
            name="Bob",
            embedding=_make_unit_embedding(3),
        )
        assert db.delete("bob") is True
        # Re-open from disk; bob must be gone.
        db2 = FaceDB(storage_path=path)
        assert db2.get("bob") is None


def test_delete_missing_returns_false() -> None:
    db = FaceDB()
    assert db.delete("nobody") is False


# ── validation ─────────────────────────────────────────────────────


def test_register_rejects_empty_person_id() -> None:
    db = FaceDB()
    with pytest.raises(ValueError, match="person_id is required"):
        db.register(person_id="", name="x", embedding=_make_unit_embedding(1))


def test_register_rejects_empty_name() -> None:
    db = FaceDB()
    with pytest.raises(ValueError, match="name is required"):
        db.register(person_id="x", name="   ", embedding=_make_unit_embedding(1))


# ── category filtering ─────────────────────────────────────────────


def test_best_match_category_filter() -> None:
    db = FaceDB()
    db.register(
        person_id="alice",
        name="Alice",
        embedding=_make_unit_embedding(1),
        category="family",
    )
    db.register(
        person_id="bob",
        name="Bob",
        embedding=_make_unit_embedding(1),  # identical embedding
        category="watchlist",
    )

    # Without filter: alice wins on tiebreak ('alice' < 'bob').
    match = db.best_match(_make_unit_embedding(1), threshold=0.5)
    assert match is not None and match["person_id"] == "alice"

    # With watchlist filter: bob is the only candidate.
    match = db.best_match(
        _make_unit_embedding(1), threshold=0.5, category="watchlist"
    )
    assert match is not None and match["person_id"] == "bob"
