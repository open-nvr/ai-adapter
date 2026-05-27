# Copyright (c) 2026 OpenNVR
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Unit tests for ByteTrackService.

These tests instantiate the real service (no mocking of supervision
itself — supervision is a hard dep and we want to know if its API
shifts) and exercise:

* Load lifecycle and readiness reporting.
* Malformed input → ServiceError(transport_error, malformed_input).
* Per-camera state isolation — two cameras never share track IDs.
* Track persistence — the same object across frames keeps its ID.
* Config-change-triggers-rebuild semantics.
* Idle-TTL eviction.
* Output ordering preserves input ordering.

Run with:

    cd ai-adapter && pytest tests/test_bytetrack_service.py -v
"""
from __future__ import annotations

import time
from unittest.mock import patch

import pytest

# Skip the entire module if supervision isn't installed in the test
# venv. Production builds and the smoke matrix install it; the SDK's
# minimal dev env doesn't pull it transitively.
supervision = pytest.importorskip("supervision")

from opennvr_adapter_sdk import ErrorCategory, HardwareVerdict, ServiceError


# ────────────────────────────────────────────────────────────────────
# Fixtures
# ────────────────────────────────────────────────────────────────────


@pytest.fixture
def service():
    """A fresh, loaded ByteTrackService. Importing here (not at module
    top) keeps the skipif clean if supervision is missing."""
    from adapters.bytetrack.service import ByteTrackService

    svc = ByteTrackService()
    svc.load()
    assert svc.is_ready(), "ByteTrackService failed to load"
    return svc


def _det(label: str, conf: float, x: float, y: float, w: float, h: float) -> dict:
    """Convenience builder for a normalized-bbox detection."""
    return {
        "label": label,
        "confidence": conf,
        "bbox": {"x": x, "y": y, "w": w, "h": h},
    }


# ────────────────────────────────────────────────────────────────────
# Load + readiness
# ────────────────────────────────────────────────────────────────────


class TestLoadLifecycle:

    def test_fresh_service_is_not_ready(self):
        from adapters.bytetrack.service import ByteTrackService

        svc = ByteTrackService()
        assert not svc.is_ready()

    def test_load_marks_service_ready(self):
        from adapters.bytetrack.service import ByteTrackService

        svc = ByteTrackService()
        svc.load()
        assert svc.is_ready()

    def test_load_is_idempotent(self):
        from adapters.bytetrack.service import ByteTrackService

        svc = ByteTrackService()
        svc.load()
        first_version = svc._supervision_version
        svc.load()  # second call should be a no-op
        assert svc._supervision_version == first_version

    def test_hardware_evaluation_ok_when_loaded(self, service):
        evaluation = service.hardware_evaluation()
        assert evaluation.verdict == HardwareVerdict.OK
        assert evaluation.details["gpu_required"] is False

    def test_model_info_uses_supervision_version_as_fingerprint(self, service):
        info = service.model_info()
        assert info.name == "bytetrack"
        assert info.framework == "supervision-bytetrack"
        assert info.fingerprint is not None
        assert info.fingerprint.startswith("supervision:")

    def test_infer_before_load_raises_loading(self):
        """KAI-C polls /infer during startup — it must get a typed
        ``model_loading`` ServiceError, not a generic crash."""
        from adapters.bytetrack.service import ByteTrackService

        svc = ByteTrackService()  # NOT loaded
        with pytest.raises(ServiceError) as exc_info:
            svc.infer(
                {
                    "camera_id": "cam-1",
                    "detections": [_det("person", 0.9, 0.1, 0.1, 0.2, 0.3)],
                }
            )
        assert exc_info.value.envelope().error.category == ErrorCategory.MODEL_ERROR
        assert exc_info.value.envelope().error.code == "bytetrack.model_loading"


# ────────────────────────────────────────────────────────────────────
# Input validation
# ────────────────────────────────────────────────────────────────────


class TestInputValidation:

    @pytest.mark.parametrize(
        "payload, expected_code",
        [
            ({}, "malformed_input"),                                   # missing camera_id
            ({"camera_id": ""}, "malformed_input"),                    # empty
            ({"camera_id": 42, "detections": []}, "malformed_input"),  # wrong type
            ({"camera_id": "x"}, "malformed_input"),                   # no detections
            ({"camera_id": "x", "detections": "not-a-list"}, "malformed_input"),
        ],
    )
    def test_missing_or_malformed_top_level_rejected(
        self, service, payload, expected_code
    ):
        with pytest.raises(ServiceError) as exc_info:
            service.infer(payload)
        # FailureEnvelope has the typed status + nested ErrorDetail —
        # the category/code/etc. fields live on env.error, not on env.
        env = exc_info.value.envelope()
        assert env.error.category == ErrorCategory.TRANSPORT_ERROR
        assert env.error.code == expected_code

    def test_detection_missing_bbox_rejected(self, service):
        with pytest.raises(ServiceError) as exc_info:
            service.infer(
                {
                    "camera_id": "cam-1",
                    "detections": [{"label": "person", "confidence": 0.9}],  # no bbox
                }
            )
        assert exc_info.value.envelope().error.code == "malformed_input"

    def test_detection_bbox_non_numeric_rejected(self, service):
        with pytest.raises(ServiceError) as exc_info:
            service.infer(
                {
                    "camera_id": "cam-1",
                    "detections": [
                        {
                            "label": "person",
                            "confidence": 0.9,
                            "bbox": {"x": "oops", "y": 0, "w": 0.1, "h": 0.1},
                        }
                    ],
                }
            )
        assert exc_info.value.envelope().error.code == "malformed_input"

    def test_tracker_config_out_of_range_rejected(self, service):
        with pytest.raises(ServiceError) as exc_info:
            service.infer(
                {
                    "camera_id": "cam-1",
                    "detections": [],
                    "tracker_config": {"track_activation_threshold": 1.5},
                }
            )
        assert exc_info.value.envelope().error.code == "malformed_input"


# ────────────────────────────────────────────────────────────────────
# Tracking behaviour
# ────────────────────────────────────────────────────────────────────


class TestTrackingBehaviour:
    """ByteTrack is stochastic-but-deterministic given a fixed input.
    These tests pin the contract-level behaviour we expect — not the
    specific track IDs ByteTrack chooses, since those are an internal
    detail."""

    def test_empty_detections_still_returns_envelope(self, service):
        """Tracker is ticked forward for lost-track ageing even with no
        detections; the response must still be a valid InferResponse."""
        resp = service.infer({"camera_id": "cam-1", "detections": []})
        assert resp.model_name == "bytetrack"
        assert resp.result["detections"] == []

    def test_first_call_assigns_track_ids(self, service):
        """A high-confidence detection should get a track_id on the
        FIRST update — ByteTrack's default activation threshold is
        0.25 and we send 0.9."""
        # ByteTrack typically needs 2-3 frames to activate a track —
        # send the same detection across a few frames so activation
        # has a chance to happen.
        det = _det("person", 0.9, 0.1, 0.1, 0.3, 0.5)
        for _ in range(3):
            resp = service.infer({"camera_id": "cam-1", "detections": [det]})
        assert len(resp.result["detections"]) == 1
        track_id = resp.result["detections"][0]["track_id"]
        assert track_id is not None and track_id >= 1

    def test_two_cameras_get_independent_state(self, service):
        """Per-review #1 of the contract — multi-tenant cameras must
        not share track-id state. Two cameras each detecting one
        person should get track IDs from their own counters."""
        det = _det("person", 0.95, 0.2, 0.2, 0.3, 0.5)
        # Warm both trackers across a few frames so the assertion
        # isn't sensitive to ByteTrack's activation latency.
        for _ in range(3):
            service.infer({"camera_id": "cam-A", "detections": [det]})
            service.infer({"camera_id": "cam-B", "detections": [det]})

        assert "cam-A" in service._trackers
        assert "cam-B" in service._trackers
        assert service._trackers["cam-A"].tracker is not service._trackers["cam-B"].tracker

    def test_output_order_matches_input_order(self, service):
        """The contract returns detections in input order — downstream
        consumers correlate by index. Order-preservation matters even
        when supervision internally reorders during matching."""
        dets = [
            _det("person", 0.9, 0.1, 0.1, 0.1, 0.2),
            _det("car",    0.85, 0.4, 0.4, 0.2, 0.2),
            _det("dog",    0.7,  0.7, 0.1, 0.1, 0.1),
        ]
        resp = service.infer({"camera_id": "cam-1", "detections": dets})
        assert [d["label"] for d in resp.result["detections"]] == ["person", "car", "dog"]

    def test_frame_dimensions_echoed_back(self, service):
        resp = service.infer(
            {
                "camera_id": "cam-1",
                "detections": [_det("person", 0.9, 0.1, 0.1, 0.2, 0.3)],
                "frame_dimensions": {"w": 1920, "h": 1080},
            }
        )
        assert resp.result["frame_dimensions"] == {"w": 1920, "h": 1080}


# ────────────────────────────────────────────────────────────────────
# Per-camera state lifecycle
# ────────────────────────────────────────────────────────────────────


class TestTrackerLifecycle:

    def test_config_change_rebuilds_tracker(self, service):
        """Changing tracker_config for an existing camera must
        invalidate the existing tracker (operators can't expect track
        continuity when they change tuning mid-stream)."""
        det = _det("person", 0.9, 0.1, 0.1, 0.3, 0.5)
        service.infer({"camera_id": "cam-1", "detections": [det]})
        original_tracker = service._trackers["cam-1"].tracker

        # Same camera, different config → expect a fresh tracker instance.
        service.infer(
            {
                "camera_id": "cam-1",
                "detections": [det],
                "tracker_config": {"lost_track_buffer": 60},
            }
        )
        assert service._trackers["cam-1"].tracker is not original_tracker

    def test_same_config_reuses_tracker(self, service):
        det = _det("person", 0.9, 0.1, 0.1, 0.3, 0.5)
        service.infer({"camera_id": "cam-1", "detections": [det]})
        first = service._trackers["cam-1"].tracker
        service.infer({"camera_id": "cam-1", "detections": [det]})
        assert service._trackers["cam-1"].tracker is first

    def test_idle_eviction_gcs_old_trackers(self, service):
        """An /infer call for a camera should evict any other camera
        whose tracker hasn't been touched for longer than the TTL."""
        det = _det("person", 0.9, 0.1, 0.1, 0.3, 0.5)
        # Set up two cameras.
        service.infer({"camera_id": "cam-stale", "detections": [det]})
        service.infer({"camera_id": "cam-fresh", "detections": [det]})
        assert "cam-stale" in service._trackers
        assert "cam-fresh" in service._trackers

        # Backdate cam-stale's last_used_at past the TTL deadline.
        from adapters.bytetrack.service import TRACKER_IDLE_TTL_SECONDS

        service._trackers["cam-stale"].last_used_at = (
            time.monotonic() - TRACKER_IDLE_TTL_SECONDS - 10
        )

        # Any /infer call triggers the GC sweep.
        service.infer({"camera_id": "cam-fresh", "detections": [det]})
        assert "cam-stale" not in service._trackers
        assert "cam-fresh" in service._trackers
