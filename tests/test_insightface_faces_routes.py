# Copyright (c) 2026 OpenNVR
# SPDX-License-Identifier: AGPL-3.0-or-later

"""HTTP-level tests for the InsightFace adapter's /faces/* CRUD routes.

The CRUD surface is bolted onto the SDK FastAPI app via
``adapters.insightface.main``. These tests boot the full app via
``TestClient`` (so the SDK middleware + body parser run) and verify
the route behaviour end-to-end: register, list, get, delete, the
multi-face refusal, and the no-face refusal.
"""
from __future__ import annotations

from tests._insightface_service_fixtures import (  # noqa: F401
    insightface_app,
    insightface_environment,
)


def _register_payload(person_id: str = "alice", name: str = "Alice Smith"):
    return {
        "person_id": person_id,
        "name": name,
        "category": "family",
        "metadata": '{"role":"owner"}',
    }


def test_register_succeeds_for_single_face_image(insightface_app, insightface_environment):
    files = {"frame": ("alice.jpg", insightface_environment["sample_jpeg"], "image/jpeg")}
    resp = insightface_app.post(
        "/faces/register", data=_register_payload(), files=files
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["ok"] is True
    assert body["face"]["person_id"] == "alice"
    assert body["face"]["category"] == "family"


def test_register_rejects_multi_face_image(
    insightface_app, insightface_environment, monkeypatch
):
    """An image with more than one face must be rejected with 422 —
    otherwise the operator silently enrolls Alice using a frame they
    thought contained Bob (or vice versa)."""
    # Reach into the running service and swap its FaceAnalysis.get to
    # return two faces for this request.
    import adapters.insightface.main as main_module

    svc = main_module._adapter_app.service  # type: ignore[attr-defined]
    FakeFace = insightface_environment["FakeFace"]
    monkeypatch.setattr(
        svc._face_app,
        "get",
        lambda img: [
            FakeFace(det_score=0.95, embedding=insightface_environment["embedding_alice"]),
            FakeFace(det_score=0.92, embedding=insightface_environment["embedding_bob"]),
        ],
    )

    files = {"frame": ("group.jpg", insightface_environment["sample_jpeg"], "image/jpeg")}
    resp = insightface_app.post(
        "/faces/register", data=_register_payload(), files=files
    )
    assert resp.status_code == 422, resp.text
    detail = resp.json().get("detail", "")
    assert "2 faces" in detail
    assert "exactly one face" in detail


def test_register_rejects_no_face_image(
    insightface_app, insightface_environment, monkeypatch
):
    import adapters.insightface.main as main_module

    svc = main_module._adapter_app.service  # type: ignore[attr-defined]
    monkeypatch.setattr(svc._face_app, "get", lambda img: [])

    files = {"frame": ("blank.jpg", insightface_environment["sample_jpeg"], "image/jpeg")}
    resp = insightface_app.post(
        "/faces/register", data=_register_payload(), files=files
    )
    assert resp.status_code == 422, resp.text
    assert "no face detected" in resp.json().get("detail", "")


def test_register_rejects_empty_person_id(insightface_app, insightface_environment):
    files = {"frame": ("alice.jpg", insightface_environment["sample_jpeg"], "image/jpeg")}
    bad = _register_payload(person_id="   ")
    resp = insightface_app.post("/faces/register", data=bad, files=files)
    assert resp.status_code == 400
    assert "person_id is required" in resp.json().get("detail", "")


def test_register_rejects_non_object_metadata(insightface_app, insightface_environment):
    files = {"frame": ("alice.jpg", insightface_environment["sample_jpeg"], "image/jpeg")}
    bad = _register_payload()
    bad["metadata"] = "[1,2,3]"  # JSON array, not object
    resp = insightface_app.post("/faces/register", data=bad, files=files)
    assert resp.status_code == 400


def test_list_and_get_round_trip(insightface_app, insightface_environment):
    files = {"frame": ("alice.jpg", insightface_environment["sample_jpeg"], "image/jpeg")}
    insightface_app.post("/faces/register", data=_register_payload(), files=files)

    list_resp = insightface_app.get("/faces")
    assert list_resp.status_code == 200
    body = list_resp.json()
    assert body["count"] >= 1
    ids = [r["person_id"] for r in body["faces"]]
    assert "alice" in ids

    get_resp = insightface_app.get("/faces/alice")
    assert get_resp.status_code == 200
    assert get_resp.json()["face"]["name"] == "Alice Smith"


def test_get_missing_returns_404(insightface_app):
    resp = insightface_app.get("/faces/nobody")
    assert resp.status_code == 404


def test_delete_round_trip(insightface_app, insightface_environment):
    files = {"frame": ("alice.jpg", insightface_environment["sample_jpeg"], "image/jpeg")}
    insightface_app.post("/faces/register", data=_register_payload(), files=files)

    del_resp = insightface_app.delete("/faces/alice")
    assert del_resp.status_code == 200
    assert del_resp.json()["ok"] is True

    # Second delete is now a 404.
    again = insightface_app.delete("/faces/alice")
    assert again.status_code == 404


def test_patch_edits_name_category_and_merges_metadata(insightface_app, insightface_environment):
    files = {"frame": ("alice.jpg", insightface_environment["sample_jpeg"], "image/jpeg")}
    insightface_app.post("/faces/register", data=_register_payload(), files=files)

    resp = insightface_app.patch("/faces/alice", json={
        "category": "watchlist",
        "metadata": {"notes": "Flat 4B", "valid_until": "2026-12-31"},
    })
    assert resp.status_code == 200, resp.text
    face = resp.json()["face"]
    assert face["category"] == "watchlist"
    assert face["name"] == "Alice Smith"            # untouched
    assert face["metadata"]["notes"] == "Flat 4B"

    # A second patch merges: the note survives, the expiry is removed.
    resp = insightface_app.patch("/faces/alice", json={"metadata": {"valid_until": None}})
    face = resp.json()["face"]
    assert face["metadata"]["notes"] == "Flat 4B"
    assert "valid_until" not in face["metadata"]
    assert face["metadata"]["role"] == "owner"       # what register set, still there

    # The embedding is intact: recognition still works is covered by the
    # service tests; here, the record is still listed once.
    assert insightface_app.get("/faces").json()["count"] == 1


def test_patch_rejects_unknown_fields_blank_names_and_missing_people(insightface_app, insightface_environment):
    files = {"frame": ("alice.jpg", insightface_environment["sample_jpeg"], "image/jpeg")}
    insightface_app.post("/faces/register", data=_register_payload(), files=files)
    assert insightface_app.patch("/faces/alice", json={"embedding": [1, 2]}).status_code == 422
    assert insightface_app.patch("/faces/alice", json={"name": "  "}).status_code == 422
    assert insightface_app.patch("/faces/alice", json={"metadata": "x"}).status_code == 422
    assert insightface_app.patch("/faces/nobody", json={"name": "N"}).status_code == 404


def test_append_adds_a_sample_and_matching_uses_the_best_one(insightface_app, insightface_environment):
    """Two photos of Alice; a query near the second must still be Alice,
    with the similarity of the closer sample — not a blend."""
    files = {"frame": ("alice.jpg", insightface_environment["sample_jpeg"], "image/jpeg")}
    first = insightface_app.post("/faces/register", data=_register_payload(), files=files).json()["face"]
    assert first["samples"] == 1

    second = insightface_app.post(
        "/faces/register", files=files,
        data={"person_id": "alice", "name": "", "category": "", "append": "true"},
    )
    assert second.status_code == 200, second.text
    face = second.json()["face"]
    assert face["samples"] == 2
    assert face["name"] == "Alice Smith" and face["category"] == "family"   # kept
    assert face["metadata"]["role"] == "owner"                              # kept

    # Without append, the same call replaces: back to one sample, and a
    # blank name is now an error again.
    replaced = insightface_app.post("/faces/register", data=_register_payload(name="Alice S."), files=files)
    assert replaced.json()["face"]["samples"] == 1
    assert replaced.json()["face"]["name"] == "Alice S."
    assert insightface_app.post("/faces/register", files=files,
                                data={"person_id": "nobody", "name": "", "append": "true"}).status_code == 400


def test_remove_sample_keeps_at_least_one(insightface_app, insightface_environment):
    files = {"frame": ("alice.jpg", insightface_environment["sample_jpeg"], "image/jpeg")}
    insightface_app.post("/faces/register", data=_register_payload(), files=files)
    insightface_app.post("/faces/register", files=files, data={"person_id": "alice", "name": "", "append": "true"})
    assert insightface_app.get("/faces/alice").json()["face"]["samples"] == 2
    assert insightface_app.delete("/faces/alice/samples/7").status_code == 404   # no such sample
    assert insightface_app.delete("/faces/nobody/samples/0").status_code == 404
    ok = insightface_app.delete("/faces/alice/samples/0")
    assert ok.status_code == 200 and ok.json()["face"]["samples"] == 1
    assert insightface_app.delete("/faces/alice/samples/0").status_code == 409   # the last one
