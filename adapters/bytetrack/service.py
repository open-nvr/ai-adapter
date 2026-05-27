# Copyright (c) 2026 OpenNVR
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
ByteTrackService — ByteTrack-based multi-object tracking adapter.

Takes a frame's detections in (JSON, BodyShape.TEXT) and returns the
same detections with persistent ``track_id`` fields populated. State
is maintained PER-CAMERA so tracks from camera A don't bleed into
camera B's track IDs.

Design choices worth pinning:

* **No per-frame image.** Tracking is a post-processing step on
  detections — we don't need pixels. The upstream detector
  (YOLOv8 / YOLOv11 / fast-plate-ocr / anything) produces boxes;
  ByteTrack assigns IDs across frames. Keeping the adapter
  pixel-free means it stays cheap (no torch, no opencv-decode) and
  composable.

* **Stateful per camera, garbage-collected.** Each ``camera_id`` gets
  its own ``supervision.ByteTrack`` instance. Idle cameras
  (no inference call for ``TRACKER_IDLE_TTL_SECONDS``) get pruned to
  keep memory bounded. Worst case is a tracker stays warm for
  a few minutes longer than necessary; correctness is unaffected.

* **Configuration per-call, not adapter-wide.** ``track_activation_
  threshold``, ``lost_track_buffer``, ``minimum_matching_threshold``,
  and ``frame_rate`` can be overridden on each /infer payload.
  Different cameras have different scene dynamics (a static
  doorbell vs a busy parking lot) and a single global tuning is
  always wrong for somebody. Per-call defaults match supervision's
  ByteTrack defaults.

* **Idempotent on empty input.** A frame with no detections still
  ticks the tracker forward (so lost-track timers age correctly).
  Returns ``detections: []`` plus the standard envelope.
"""
from __future__ import annotations

import logging
import os
import platform
import threading
import time
from datetime import datetime, timezone
from typing import Any

from opennvr_adapter_sdk import AdapterService, ServiceError
from opennvr_adapter_sdk.contract import (
    DetectionItem,
    DetectionResult,
    ErrorCategory,
    FrameDimensions,
    HardwareEvaluationResponse,
    HardwareVerdict,
    HealthStatus,
    InferResponse,
    ModelInfo,
    NormalizedBBox,
)

logger = logging.getLogger(__name__)


MODEL_FRAMEWORK: str = "supervision-bytetrack"

# Default supervision.ByteTrack constructor args. Operators can
# override any of these per-call via the /infer payload.
DEFAULT_TRACK_ACTIVATION_THRESHOLD: float = 0.25
DEFAULT_LOST_TRACK_BUFFER: int = 30
DEFAULT_MINIMUM_MATCHING_THRESHOLD: float = 0.8
DEFAULT_FRAME_RATE: int = 30

# Per-camera tracker eviction. A camera that hasn't seen an inference
# call for this long is considered inactive and its tracker is GC'd.
# Five minutes is the sweet spot: long enough that ordinary camera
# stalls / KAI-C restarts don't lose track continuity, short enough
# that cameras the operator actually deleted don't sit in memory
# until the next adapter restart. Configurable via env var below for
# operators with unusual workloads.
TRACKER_IDLE_TTL_SECONDS: float = float(
    os.environ.get("BYTETRACK_IDLE_TTL_SECONDS", "300")
)

# Maximum size of one /infer payload. Detections are JSON — a
# realistic upper bound is ~1000 detections at ~300 bytes each ≈
# 300 KB. 1 MiB gives generous headroom without inviting abuse.
MAX_PAYLOAD_BYTES: int = 1 * 1024 * 1024


class _TrackerEntry:
    """One per camera. Holds the supervision.ByteTrack instance + the
    last time we touched it for the GC sweep."""

    __slots__ = ("tracker", "last_used_at", "config_signature")

    def __init__(self, tracker: Any, config_signature: tuple) -> None:
        self.tracker = tracker
        self.last_used_at: float = time.monotonic()
        # supervision.ByteTrack doesn't expose its constructor args
        # after the fact; we record what we built it with so an /infer
        # call that requests different tuning rebuilds it fresh.
        self.config_signature: tuple = config_signature


class ByteTrackService(AdapterService):
    """Stateful façade around supervision.ByteTrack with per-camera
    state."""

    def __init__(self) -> None:
        self._trackers: dict[str, _TrackerEntry] = {}
        # Single lock around _trackers — the dict isn't safe for
        # concurrent mutation. Inference is short and CPU-bound so
        # the lock contention is negligible.
        self._lock = threading.Lock()
        self._load_state: HealthStatus = HealthStatus.LOADING
        self._load_error: str | None = None
        self._supervision_version: str | None = None

    # ── AdapterService impl ────────────────────────────────────────

    def load(self) -> None:
        """Import supervision once at startup so the first /infer call
        doesn't pay the ~200 ms import cost. supervision pulls numpy
        + scipy + opencv-python-headless transitively; all that work
        happens here, not on the request path."""
        if self._load_state == HealthStatus.OK:
            return
        try:
            import supervision as sv  # type: ignore

            # Sanity-check that ByteTrack is present (some pre-0.17
            # versions of supervision didn't ship it). Future-proofs
            # against an accidental downgrade in pyproject.toml.
            assert hasattr(sv, "ByteTrack"), (
                "supervision.ByteTrack missing — bump supervision to >=0.21"
            )
            self._supervision_version = getattr(sv, "__version__", "unknown")
            self._load_state = HealthStatus.OK
            self._load_error = None
            logger.info(
                "ByteTrackService ready (supervision=%s)",
                self._supervision_version,
            )
        except Exception as exc:
            self._load_state = HealthStatus.ERROR
            self._load_error = str(exc)
            logger.exception("ByteTrackService failed to load supervision")

    def is_ready(self) -> bool:
        return self._load_state == HealthStatus.OK

    def fingerprint(self) -> str | None:
        # The "model" here is the supervision package version. No
        # weights file to sha256 — what differs across deployments is
        # the supervision pin, which we surface via the prefix below.
        # Returning ``None`` would also be honest, but the contract's
        # drift-detection layer prefers a non-None value, so we hand
        # back a deterministic synthetic fingerprint that drifts only
        # when the underlying package version changes.
        if not self._supervision_version:
            return None
        return f"supervision:{self._supervision_version}"

    def model_info(self) -> ModelInfo:
        return ModelInfo(
            name="bytetrack",
            version=self._supervision_version or "unknown",
            framework=MODEL_FRAMEWORK,
            size_mb=None,  # no weights file
            modalities_in=["text"],     # JSON detections in
            modalities_out=["text"],    # JSON detections out
            fingerprint=self.fingerprint(),
        )

    def hardware_evaluation(self) -> HardwareEvaluationResponse:
        if self._load_state == HealthStatus.OK:
            verdict = HardwareVerdict.OK
            reasoning = "ByteTrack is CPU-only; no GPU required."
        elif self._load_state == HealthStatus.LOADING:
            verdict = HardwareVerdict.WARN
            reasoning = "supervision import still in progress."
        else:
            verdict = HardwareVerdict.BLOCKED
            reasoning = f"supervision import failed: {self._load_error}"

        # active_trackers reads under the lock — dict mutation from
        # an in-flight infer() would otherwise race the snapshot.
        # The lock is cheap (single-digit-microsecond hold) and
        # makes the field meaningful under load.
        with self._lock:
            active_trackers = len(self._trackers)

        return HardwareEvaluationResponse(
            verdict=verdict,
            reasoning=reasoning,
            checked_at=datetime.now(timezone.utc),
            details={
                "gpu_required": False,
                "supervision_version": self._supervision_version,
                "cpu_count": os.cpu_count() or 0,
                "platform": platform.platform(),
                "python_version": platform.python_version(),
                "active_trackers": active_trackers,
            },
        )

    # ── §3.5 inference entry point ─────────────────────────────────

    def infer(self, payload: dict[str, Any]) -> InferResponse:
        """Run one tracking update for ``camera_id``'s detections.

        Payload shape (BodyShape.TEXT — JSON only):

            {
              "camera_id": "front-door",        # required, non-empty
              "detections": [                    # required, may be []
                {"label": "person", "confidence": 0.92,
                 "bbox": {"x": 0.1, "y": 0.2, "w": 0.3, "h": 0.5}},
                ...
              ],
              "frame_dimensions": {"w": 1920, "h": 1080},   # optional
              "tracker_config": {                # optional, per-call tuning
                "track_activation_threshold": 0.25,
                "lost_track_buffer": 30,
                "minimum_matching_threshold": 0.8,
                "frame_rate": 30
              }
            }

        Response: ``DetectionResult`` with each detection's
        ``track_id`` populated (or ``None`` if the tracker didn't
        assign one this frame — that's how ByteTrack signals "I see
        this box but haven't decided which existing track it belongs
        to yet"). Order of the output detections matches the input.
        """
        if self._load_state != HealthStatus.OK:
            raise ServiceError(
                ErrorCategory.MODEL_ERROR,
                code=(
                    "supervision_missing"
                    if self._load_state == HealthStatus.ERROR
                    else "bytetrack.model_loading"
                ),
                message=self._load_error or "supervision still loading.",
                transient=(self._load_state == HealthStatus.LOADING),
                http_status=503,
                retry_after_ms=2000 if self._load_state == HealthStatus.LOADING else None,
            )

        start = time.monotonic()

        camera_id = payload.get("camera_id")
        if not isinstance(camera_id, str) or not camera_id:
            raise ServiceError(
                ErrorCategory.TRANSPORT_ERROR,
                code="malformed_input",
                message="camera_id must be a non-empty string.",
                transient=False,
                http_status=400,
            )

        raw_detections = payload.get("detections")
        if not isinstance(raw_detections, list):
            raise ServiceError(
                ErrorCategory.TRANSPORT_ERROR,
                code="malformed_input",
                message="detections must be a list.",
                transient=False,
                http_status=400,
            )

        try:
            config = _parse_tracker_config(payload.get("tracker_config") or {})
        except _DetectionParseError as exc:
            raise ServiceError(
                ErrorCategory.TRANSPORT_ERROR,
                code="malformed_input",
                message=str(exc),
                transient=False,
                http_status=400,
            ) from exc

        # Translate the operator's normalized bbox payload into the
        # xyxy pixel-coordinate shape supervision expects. The frame
        # dimensions are optional, but if present must be complete —
        # half-specified dimensions ({"w": 1920} with no h) would
        # silently default the missing axis to 1, producing a
        # tracker that treats every bbox as a 1-pixel-tall slice and
        # ruining IoU matching.
        frame_dims_raw = payload.get("frame_dimensions")
        if frame_dims_raw is None:
            frame_dims = None
            frame_w, frame_h = 1, 1
        elif isinstance(frame_dims_raw, dict) and not frame_dims_raw:
            # Empty dict treated as "not provided" — same as None.
            frame_dims = None
            frame_w, frame_h = 1, 1
        elif (
            isinstance(frame_dims_raw, dict)
            and isinstance(frame_dims_raw.get("w"), (int, float))
            and isinstance(frame_dims_raw.get("h"), (int, float))
            and int(frame_dims_raw["w"]) > 0
            and int(frame_dims_raw["h"]) > 0
        ):
            frame_dims = frame_dims_raw
            frame_w = int(frame_dims_raw["w"])
            frame_h = int(frame_dims_raw["h"])
        else:
            raise ServiceError(
                ErrorCategory.TRANSPORT_ERROR,
                code="malformed_input",
                message=(
                    "frame_dimensions must be omitted or include both "
                    "positive integer w and h."
                ),
                transient=False,
                http_status=400,
            )
        try:
            xyxy, confidences, class_ids, source_index_to_input = _to_supervision_arrays(
                raw_detections, frame_w, frame_h
            )
        except _DetectionParseError as exc:
            raise ServiceError(
                ErrorCategory.TRANSPORT_ERROR,
                code="malformed_input",
                message=str(exc),
                transient=False,
                http_status=400,
            ) from exc

        # Hold the lock for the entire tracker call — not just the
        # lookup. supervision.ByteTrack.update_with_detections mutates
        # the tracker's internal Kalman state, so two concurrent calls
        # for the SAME camera_id would race. KAI-C's
        # scheduling.fair_queuing=PER_CAMERA serialises per camera_id
        # WHEN the call comes through KAI-C, but direct callers
        # (conformance suite, operator curl, custom apps that hit the
        # adapter HTTP surface) bypass that. The global lock here
        # makes the adapter safe against any caller mix.
        #
        # The contention cost is small: tracking takes ~3 ms per call
        # so even at 30 cameras × 30 FPS the lock is held ~3 % of the
        # time. If profiling ever shows this as a hot spot, switch to
        # a per-camera lock dict — but that's a v0.3 concern.
        with self._lock:
            entry = self._get_or_create_tracker(camera_id, config)
            entry.last_used_at = time.monotonic()
            # Evict idle trackers periodically. Doing it inline on
            # every call is cheap (one dict scan, ~O(N) where N is
            # active cameras — for any realistic homelab N is small).
            self._gc_idle_trackers_locked()

            track_ids = _run_tracker(
                entry.tracker, xyxy, confidences, class_ids, frame_w, frame_h
            )

        # Stitch results back into the original input order. The
        # tracker may drop "low-confidence" detections that fall below
        # its activation threshold; those keep their input bbox but
        # get ``track_id=None`` so downstream consumers can still see
        # the detection (and know it wasn't tracked).
        items: list[DetectionItem] = []
        for input_idx, det in enumerate(raw_detections):
            tracker_idx = source_index_to_input.get(input_idx)
            track_id = (
                int(track_ids[tracker_idx])
                if tracker_idx is not None and track_ids[tracker_idx] is not None
                else None
            )
            items.append(
                DetectionItem(
                    label=det.get("label", ""),
                    confidence=float(det.get("confidence", 0.0)),
                    bbox=NormalizedBBox(**det["bbox"]),
                    track_id=track_id,
                    attributes=det.get("attributes") or {},
                )
            )

        result = DetectionResult(
            detections=items,
            frame_dimensions=FrameDimensions(w=frame_w, h=frame_h)
            if frame_dims
            else None,
        )

        inference_ms = int((time.monotonic() - start) * 1000)
        return InferResponse(
            model_name="bytetrack",
            model_version=self._supervision_version or "unknown",
            inference_ms=inference_ms,
            result=result.model_dump(mode="json"),
        )

    # ── Per-camera tracker management ──────────────────────────────

    def _get_or_create_tracker(
        self, camera_id: str, config: "_TrackerConfig"
    ) -> _TrackerEntry:
        """Return the tracker for ``camera_id``. If none exists, OR if
        the requested config differs from what we built before,
        rebuild it. Rebuilding loses track continuity for that camera
        but is the only honest path when an operator changes tuning —
        the alternative is to silently ignore the new config.
        """
        signature = config.signature()
        entry = self._trackers.get(camera_id)
        if entry is not None and entry.config_signature == signature:
            return entry

        if entry is not None:
            logger.info(
                "ByteTrack: rebuilding tracker for camera_id=%s (config changed)",
                camera_id,
            )

        import supervision as sv  # type: ignore

        # supervision >=0.21 ByteTrack constructor keyword args.
        tracker = sv.ByteTrack(
            track_activation_threshold=config.track_activation_threshold,
            lost_track_buffer=config.lost_track_buffer,
            minimum_matching_threshold=config.minimum_matching_threshold,
            frame_rate=config.frame_rate,
        )
        new_entry = _TrackerEntry(tracker, signature)
        self._trackers[camera_id] = new_entry
        return new_entry

    def _gc_idle_trackers_locked(self) -> None:
        """Drop trackers we haven't touched in TRACKER_IDLE_TTL_SECONDS.
        Caller MUST hold ``self._lock``."""
        if not self._trackers:
            return
        now = time.monotonic()
        deadline = now - TRACKER_IDLE_TTL_SECONDS
        # Snapshot the dict keys before mutating it during iteration.
        stale = [cid for cid, e in self._trackers.items() if e.last_used_at < deadline]
        for cid in stale:
            logger.info(
                "ByteTrack: evicting idle tracker for camera_id=%s "
                "(idle for %.0fs)",
                cid,
                now - self._trackers[cid].last_used_at,
            )
            del self._trackers[cid]


# ────────────────────────────────────────────────────────────────────
# Helpers — parsing / supervision adapter
# ────────────────────────────────────────────────────────────────────


class _DetectionParseError(ValueError):
    """Raised when a detection dict doesn't have the fields we need."""


class _TrackerConfig:
    """Validated per-call tuning, with a signature suitable for cache
    invalidation in _get_or_create_tracker."""

    __slots__ = (
        "track_activation_threshold",
        "lost_track_buffer",
        "minimum_matching_threshold",
        "frame_rate",
    )

    def __init__(
        self,
        track_activation_threshold: float,
        lost_track_buffer: int,
        minimum_matching_threshold: float,
        frame_rate: int,
    ) -> None:
        self.track_activation_threshold = track_activation_threshold
        self.lost_track_buffer = lost_track_buffer
        self.minimum_matching_threshold = minimum_matching_threshold
        self.frame_rate = frame_rate

    def signature(self) -> tuple:
        return (
            self.track_activation_threshold,
            self.lost_track_buffer,
            self.minimum_matching_threshold,
            self.frame_rate,
        )


def _parse_tracker_config(raw: dict[str, Any]) -> _TrackerConfig:
    """Parse + validate the optional ``tracker_config`` block.

    Defaults match supervision.ByteTrack's own defaults. Out-of-range
    values raise a TRANSPORT_ERROR via the caller (we just raise
    ValueError here; the caller wraps).
    """
    def _f(key: str, default: float, low: float, high: float) -> float:
        val = raw.get(key, default)
        try:
            f = float(val)
        except (TypeError, ValueError):
            raise _DetectionParseError(
                f"tracker_config.{key} must be a number, got {val!r}"
            )
        if not (low <= f <= high):
            raise _DetectionParseError(
                f"tracker_config.{key} must be in [{low}, {high}], got {f}"
            )
        return f

    def _i(key: str, default: int, low: int, high: int) -> int:
        val = raw.get(key, default)
        try:
            i = int(val)
        except (TypeError, ValueError):
            raise _DetectionParseError(
                f"tracker_config.{key} must be an integer, got {val!r}"
            )
        if not (low <= i <= high):
            raise _DetectionParseError(
                f"tracker_config.{key} must be in [{low}, {high}], got {i}"
            )
        return i

    return _TrackerConfig(
        track_activation_threshold=_f(
            "track_activation_threshold",
            DEFAULT_TRACK_ACTIVATION_THRESHOLD,
            0.0,
            1.0,
        ),
        lost_track_buffer=_i(
            "lost_track_buffer", DEFAULT_LOST_TRACK_BUFFER, 1, 10000
        ),
        minimum_matching_threshold=_f(
            "minimum_matching_threshold",
            DEFAULT_MINIMUM_MATCHING_THRESHOLD,
            0.0,
            1.0,
        ),
        frame_rate=_i("frame_rate", DEFAULT_FRAME_RATE, 1, 240),
    )


def _to_supervision_arrays(
    raw_detections: list[Any], frame_w: int, frame_h: int
) -> tuple[Any, Any, Any, dict[int, int]]:
    """Translate the operator's normalized-bbox detection list into the
    numpy arrays supervision.Detections expects.

    Returns ``(xyxy, confidence, class_id, input_idx → tracker_idx)``.
    The index map exists because we may filter out malformed entries
    (currently we raise on those, but a future relaxation might let
    us skip-and-continue, and the call site already handles missing
    indices).
    """
    import numpy as np

    if not raw_detections:
        return (
            np.empty((0, 4), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
            {},
        )

    xyxy_rows: list[list[float]] = []
    confs: list[float] = []
    classes: list[int] = []
    index_map: dict[int, int] = {}

    # supervision needs an integer class_id per detection. Labels are
    # strings in the contract, so we synthesise a small per-call
    # label→id map (consistent within one call, not across calls).
    # ByteTrack only uses class_id to ensure tracks don't switch
    # class — same scheme that downstream consumers care about for
    # "did the person become a dog?" prevention.
    label_to_id: dict[str, int] = {}

    for input_idx, det in enumerate(raw_detections):
        if not isinstance(det, dict):
            raise _DetectionParseError(
                f"detections[{input_idx}] must be an object, got {type(det).__name__}"
            )
        bbox = det.get("bbox")
        if not isinstance(bbox, dict):
            raise _DetectionParseError(
                f"detections[{input_idx}].bbox must be an object"
            )
        try:
            x = float(bbox["x"])
            y = float(bbox["y"])
            w = float(bbox["w"])
            h = float(bbox["h"])
        except (KeyError, TypeError, ValueError) as exc:
            raise _DetectionParseError(
                f"detections[{input_idx}].bbox must have numeric x, y, w, h"
            ) from exc

        # Convert normalized (x, y, w, h) → pixel (x1, y1, x2, y2).
        # supervision works in pixel coords; we keep the operator's
        # normalized coords in the output but feed pixels in.
        x1 = x * frame_w
        y1 = y * frame_h
        x2 = (x + w) * frame_w
        y2 = (y + h) * frame_h

        label = det.get("label", "")
        if label not in label_to_id:
            label_to_id[label] = len(label_to_id)
        class_id = label_to_id[label]

        try:
            confidence = float(det.get("confidence", 0.0))
        except (TypeError, ValueError):
            raise _DetectionParseError(
                f"detections[{input_idx}].confidence must be a number"
            )

        index_map[input_idx] = len(xyxy_rows)
        xyxy_rows.append([x1, y1, x2, y2])
        confs.append(confidence)
        classes.append(class_id)

    return (
        np.asarray(xyxy_rows, dtype=np.float32),
        np.asarray(confs, dtype=np.float32),
        np.asarray(classes, dtype=np.int64),
        index_map,
    )


def _run_tracker(
    tracker: Any,
    xyxy: Any,
    confidences: Any,
    class_ids: Any,
    frame_w: int,
    frame_h: int,
) -> list[int | None]:
    """Drive one ByteTrack update and return the per-detection
    track_ids in the same order as the input arrays. ``None`` at a
    position means ByteTrack didn't assign an ID to that detection
    this frame (low confidence / not yet activated as a track).

    Reverse mapping: supervision.update_with_detections may return
    fewer rows than the input (low-confidence detections get filtered)
    AND may return them in a different order. Earlier versions of
    this function keyed the reverse map on rounded bbox tuples — that
    silently collapsed any two input detections with identical
    bboxes onto one track_id, corrupting downstream dedup. The
    correct round-trip is via the ``data`` dict supervision preserves
    through tracking: we attach the original input index as
    ``data["input_idx"]`` on the way in, read it back on the way out.
    """
    import numpy as np
    import supervision as sv  # type: ignore

    n = len(xyxy)
    if n == 0:
        # supervision >=0.21 ByteTrack.update_with_detections requires
        # a Detections object even when empty — to keep lost-track
        # buffers ageing forward, we still call update.
        empty = sv.Detections.empty()
        tracker.update_with_detections(empty)
        return []

    detections = sv.Detections(
        xyxy=xyxy,
        confidence=confidences,
        class_id=class_ids,
        # Round-trip the input index through supervision's data dict.
        # supervision.Detections preserves ``data`` arrays through
        # update_with_detections (it filters/reorders them in lockstep
        # with the surviving rows), so reading data["input_idx"] back
        # gives us the original input position for each tracked row —
        # even when two detections have identical bboxes.
        data={"input_idx": np.arange(n, dtype=np.int64)},
    )
    tracked = tracker.update_with_detections(detections)

    result: list[int | None] = [None] * n
    if tracked.tracker_id is None:
        return result

    tracked_input_idx = tracked.data.get("input_idx") if tracked.data else None
    if tracked_input_idx is None or len(tracked_input_idx) != len(tracked.tracker_id):
        # Defensive: if supervision changed its ``data`` semantics in
        # a future release and stopped propagating our index array
        # through filtering, fall back to "no assignment". Better to
        # under-attribute tracks than to misattribute them.
        logger.warning(
            "ByteTrack: supervision did not propagate input_idx — "
            "skipping track-id assignment for this frame"
        )
        return result

    for out_pos in range(len(tracked.tracker_id)):
        tid = tracked.tracker_id[out_pos]
        # supervision's tracker_id is int64. The pre-cast None check
        # earlier was dead code (an int64 element is never None and
        # never a Python float); -1 / negative IDs would be the real
        # "saw the box but didn't activate" signal in supervision
        # forks that use that convention. The current 0.21–0.29
        # mainline only emits positive IDs for matched tracks, so a
        # value-is-positive check is the honest filter.
        if tid is None or int(tid) < 0:
            continue
        input_pos = int(tracked_input_idx[out_pos])
        if 0 <= input_pos < n:
            result[input_pos] = int(tid)
    return result
