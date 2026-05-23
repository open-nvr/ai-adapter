# Copyright (c) 2026 OpenNVR
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
In-memory face embedding database with optional JSON-file persistence.

This is a minimal store for the SDK-based InsightFace adapter — the
legacy in-tree ``app/db/face_db.py`` was commented-out skeleton for a
future MySQL backend. For v0.1 we ship a JSON file on disk so the
homelab use case (small registered-face list, single-host install)
works without standing up a database. For larger deployments swap in
a pgvector backend behind the same ``FaceDB`` interface.

Embeddings are 512-d float32 vectors from InsightFace's ArcFace
backbone, L2-normalised on insert. Cosine similarity reduces to a
dot product on normalised vectors so similarity is one matmul.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

logger = logging.getLogger(__name__)


@dataclass
class FaceRecord:
    """One registered face."""
    person_id: str
    name: str
    embedding: list[float]  # L2-normalised, 512-d
    category: str = "unknown"
    metadata: dict[str, Any] = field(default_factory=dict)
    registered_at: float = field(default_factory=time.time)

    def to_public_dict(self) -> dict[str, Any]:
        """Public-facing dict — drops the embedding (clients don't need
        the vector, it's an internal detail)."""
        return {
            "person_id": self.person_id,
            "name": self.name,
            "category": self.category,
            "metadata": dict(self.metadata),
            "registered_at": self.registered_at,
        }


class FaceDB:
    """Thread-safe in-memory face database with JSON persistence."""

    def __init__(self, storage_path: str | None = None) -> None:
        self._records: dict[str, FaceRecord] = {}
        self._storage_path = storage_path
        self._lock = threading.RLock()
        if storage_path:
            self._load()

    # ── CRUD ───────────────────────────────────────────────────────

    def register(
        self,
        *,
        person_id: str,
        name: str,
        embedding: list[float] | Any,
        category: str = "unknown",
        metadata: dict[str, Any] | None = None,
    ) -> FaceRecord:
        """Register or update a person. Idempotent — re-registering the
        same ``person_id`` overwrites the embedding (useful for
        re-enrolling after a haircut, new glasses, etc.)."""
        if not person_id or not person_id.strip():
            raise ValueError("person_id is required")
        if not name or not name.strip():
            raise ValueError("name is required")

        embedding_list = _ensure_list(embedding)
        normalised = _l2_normalise(embedding_list)

        record = FaceRecord(
            person_id=person_id.strip(),
            name=name.strip(),
            embedding=normalised,
            category=(category or "unknown").strip() or "unknown",
            metadata=dict(metadata or {}),
        )
        with self._lock:
            self._records[record.person_id] = record
            self._save()
        return record

    def get(self, person_id: str) -> FaceRecord | None:
        with self._lock:
            return self._records.get(person_id)

    def delete(self, person_id: str) -> bool:
        with self._lock:
            removed = self._records.pop(person_id, None) is not None
            if removed:
                self._save()
            return removed

    def list_records(self, category: str | None = None) -> list[FaceRecord]:
        with self._lock:
            if category is None:
                return list(self._records.values())
            return [r for r in self._records.values() if r.category == category]

    def __len__(self) -> int:
        with self._lock:
            return len(self._records)

    # ── Recognition ────────────────────────────────────────────────

    def best_match(
        self,
        query_embedding: list[float] | Any,
        *,
        threshold: float = 0.5,
        category: str | None = None,
    ) -> dict[str, Any] | None:
        """Find the registered face with the highest cosine similarity
        above ``threshold``. Returns ``None`` if no match.

        Cosine similarity on L2-normalised vectors == dot product.
        Embeddings are normalised at register-time and we normalise
        the query here, so the math is a single pass over records.

        Tiebreak: when two registered faces score exactly the same
        similarity (rare with float32 embeddings but possible — e.g.
        synthetic test data, or genuine identical-twin enrollment),
        the record whose ``person_id`` sorts earlier lexicographically
        wins. Deterministic across processes regardless of dict
        insertion order or persistence-replay sequence.
        """
        query = _l2_normalise(_ensure_list(query_embedding))
        with self._lock:
            best: tuple[float, FaceRecord] | None = None
            for record in self._records.values():
                if category is not None and record.category != category:
                    continue
                sim = _dot(query, record.embedding)
                if sim < threshold:
                    continue
                if best is None:
                    best = (sim, record)
                    continue
                # Higher similarity wins; on exact tie, smaller
                # person_id wins (deterministic, doesn't depend on
                # iteration order or which row was registered first).
                if sim > best[0] or (
                    sim == best[0] and record.person_id < best[1].person_id
                ):
                    best = (sim, record)
        if best is None:
            return None
        sim, record = best
        return {
            "person_id": record.person_id,
            "name": record.name,
            "category": record.category,
            "similarity": round(sim, 4),
            "metadata": dict(record.metadata),
        }

    def search_similar(
        self,
        query_embedding: list[float] | Any,
        *,
        threshold: float = 0.5,
        category: str | None = None,
    ) -> list[dict[str, Any]]:
        """All registered faces above the threshold, sorted by
        descending similarity. Used by the watchlist-check task."""
        query = _l2_normalise(_ensure_list(query_embedding))
        with self._lock:
            hits: list[tuple[float, FaceRecord]] = []
            for record in self._records.values():
                if category is not None and record.category != category:
                    continue
                sim = _dot(query, record.embedding)
                if sim >= threshold:
                    hits.append((sim, record))
        hits.sort(key=lambda pair: pair[0], reverse=True)
        return [
            {
                "person_id": record.person_id,
                "name": record.name,
                "category": record.category,
                "similarity": round(sim, 4),
            }
            for sim, record in hits
        ]

    # ── Persistence ────────────────────────────────────────────────

    def _load(self) -> None:
        path = Path(self._storage_path) if self._storage_path else None
        if path is None or not path.is_file():
            return
        try:
            raw = json.loads(path.read_text())
        except Exception:
            logger.exception("face DB: failed to load %s; starting empty", path)
            return
        records = raw.get("records") if isinstance(raw, dict) else None
        if not isinstance(records, list):
            return
        for entry in records:
            if not isinstance(entry, dict):
                continue
            try:
                self._records[entry["person_id"]] = FaceRecord(
                    person_id=str(entry["person_id"]),
                    name=str(entry["name"]),
                    embedding=[float(v) for v in entry["embedding"]],
                    category=str(entry.get("category", "unknown")),
                    metadata=dict(entry.get("metadata", {})),
                    registered_at=float(entry.get("registered_at", time.time())),
                )
            except Exception:
                logger.exception("face DB: skipping malformed entry %r", entry)

    def _save(self) -> None:
        if not self._storage_path:
            return
        path = Path(self._storage_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Write to a tmp file then rename — keeps the on-disk DB
        # consistent if the process dies mid-write.
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        payload = {
            "schema_version": 1,
            "records": [asdict(r) for r in self._records.values()],
        }
        tmp_path.write_text(json.dumps(payload, indent=2))
        os.replace(tmp_path, path)


# ── Helpers ────────────────────────────────────────────────────────


def _ensure_list(embedding: Any) -> list[float]:
    """Accept either a Python list, a tuple, or any numpy-like
    array exposing tolist(); return a plain list[float]."""
    if hasattr(embedding, "tolist"):
        embedding = embedding.tolist()
    if not isinstance(embedding, (list, tuple)):
        raise TypeError(
            f"embedding must be a sequence of floats, got {type(embedding).__name__}"
        )
    return [float(v) for v in embedding]


def _l2_normalise(vector: list[float]) -> list[float]:
    norm = sum(v * v for v in vector) ** 0.5
    if norm < 1e-12:
        # Degenerate — return as-is. The caller will see similarity 0
        # against every other normalised embedding, which is exactly
        # the right answer for an all-zero face vector.
        return list(vector)
    return [v / norm for v in vector]


def _dot(a: Iterable[float], b: Iterable[float]) -> float:
    return float(sum(x * y for x, y in zip(a, b)))
