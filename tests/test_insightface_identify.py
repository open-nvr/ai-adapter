# Copyright (c) 2026 OpenNVR
# SPDX-License-Identifier: AGPL-3.0-or-later

"""face_identify — every face in the frame looked up — and the
detector-size setting it needs for rooms."""
from __future__ import annotations

import importlib
import sys

import pytest

from opennvr_adapter_sdk import ServiceError
from tests._insightface_service_fixtures import (  # noqa: F401
    insightface_environment,
    sample_jpeg,
)


def _build_service(env):
    if "adapters.insightface.service" in sys.modules:
        importlib.reload(sys.modules["adapters.insightface.service"])
    from adapters.insightface.face_db import FaceDB
    from adapters.insightface.service import InsightFaceService

    return InsightFaceService(face_db=FaceDB(storage_path=str(env["db_path"])))


def _vec(*hot: int) -> list[float]:
    """A 512-d vector with 1.0 at the given positions."""
    v = [0.0] * 512
    for i in hot:
        v[i] = 1.0
    return v


def _room(env, monkeypatch, svc, faces):
    FakeFace = env["FakeFace"]
    monkeypatch.setattr(svc._face_app, "get", lambda img: [
        FakeFace(bbox=bbox, det_score=score, embedding=emb) for bbox, score, emb in faces
    ])


def _identify(svc, jpeg, **params):
    return svc.infer({"__file__": jpeg, "task": "face_identify", **params}).result


# ── The task ───────────────────────────────────────────────────────


def test_identify_looks_up_every_face_not_just_the_best(
    insightface_environment, sample_jpeg, monkeypatch
):
    env = insightface_environment
    svc = _build_service(env)
    svc.load()
    svc.face_db.register(person_id="alice", name="Alice", embedding=env["embedding_alice"])
    svc.face_db.register(person_id="bob", name="Bob", embedding=env["embedding_bob"])
    _room(env, monkeypatch, svc, [
        ((10, 10, 60, 70), 0.95, env["embedding_alice"]),
        ((100, 10, 150, 70), 0.90, env["embedding_bob"]),
        ((200, 10, 250, 70), 0.85, _vec(0)),   # a stranger: 0.0625 to Alice, 0 to Bob
    ])

    result = _identify(svc, sample_jpeg, threshold=0.9)

    assert result["task"] == "face_identify"
    assert result["face_count"] == 3
    assert result["recognized_count"] == 2
    assert [f["person_id"] for f in result["faces"]] == ["alice", "bob", None]
    assert result["faces"][0]["similarity"] >= 0.99
    assert result["frame"] == {"w": 320, "h": 240}
    assert result["registered_faces"] == 2


def test_identify_miss_says_how_near_but_not_who(
    insightface_environment, sample_jpeg, monkeypatch
):
    env = insightface_environment
    svc = _build_service(env)
    svc.load()
    svc.face_db.register(person_id="alice", name="Alice", embedding=env["embedding_alice"])
    # Bob's vector is orthogonal to Alice's: a stranger to this DB.
    _room(env, monkeypatch, svc, [((10, 10, 60, 70), 0.95, env["embedding_bob"])])
    face = _identify(svc, sample_jpeg, threshold=0.5)["faces"][0]

    assert face["recognized"] is False
    assert face["person_id"] is None and face["name"] is None
    assert face["similarity"] == pytest.approx(0.0, abs=1e-4)


def test_identify_reports_the_runner_up_for_a_margin_check(
    insightface_environment, sample_jpeg, monkeypatch
):
    env = insightface_environment
    svc = _build_service(env)
    svc.load()
    svc.face_db.register(person_id="p1", name="One", embedding=_vec(0))
    svc.face_db.register(person_id="p2", name="Two", embedding=_vec(0, 1))
    _room(env, monkeypatch, svc, [((10, 10, 60, 70), 0.95, _vec(0))])

    face = _identify(svc, sample_jpeg)["faces"][0]

    assert face["person_id"] == "p1"
    assert face["similarity"] == pytest.approx(1.0, abs=1e-4)
    assert face["second_similarity"] == pytest.approx(2 ** -0.5, abs=1e-4)


def test_identify_with_empty_db_detects_but_matches_nobody(
    insightface_environment, sample_jpeg, monkeypatch
):
    env = insightface_environment
    svc = _build_service(env)
    svc.load()
    _room(env, monkeypatch, svc, [((10, 10, 60, 70), 0.95, env["embedding_alice"])])

    result = _identify(svc, sample_jpeg)

    assert result["face_count"] == 1
    face = result["faces"][0]
    assert face["recognized"] is False
    assert face["similarity"] is None and face["second_similarity"] is None


def test_identify_with_no_faces_is_an_empty_list(
    insightface_environment, sample_jpeg, monkeypatch
):
    env = insightface_environment
    svc = _build_service(env)
    svc.load()
    _room(env, monkeypatch, svc, [])

    result = _identify(svc, sample_jpeg)

    assert result["faces"] == [] and result["face_count"] == 0
    assert result["dropped"] == {"too_small": 0, "over_limit": 0}


def test_identify_drops_and_counts_faces_below_min_face_px(
    insightface_environment, sample_jpeg, monkeypatch
):
    env = insightface_environment
    svc = _build_service(env)
    svc.load()
    _room(env, monkeypatch, svc, [
        ((10, 10, 60, 70), 0.95, env["embedding_alice"]),   # 50 x 60
        ((100, 10, 120, 70), 0.95, env["embedding_bob"]),   # 20 wide
    ])

    result = _identify(svc, sample_jpeg, min_face_px=40)

    assert result["face_count"] == 1
    assert result["detected_count"] == 2
    assert result["dropped"]["too_small"] == 1


def test_identify_keeps_the_most_confident_max_faces(
    insightface_environment, sample_jpeg, monkeypatch
):
    env = insightface_environment
    svc = _build_service(env)
    svc.load()
    _room(env, monkeypatch, svc, [
        ((10, 10, 60, 70), 0.60, env["embedding_alice"]),
        ((100, 10, 150, 70), 0.99, env["embedding_bob"]),
        ((200, 10, 250, 70), 0.80, env["embedding_alice"]),
    ])

    result = _identify(svc, sample_jpeg, max_faces=2)

    assert [f["confidence"] for f in result["faces"]] == [0.99, 0.8]
    assert result["dropped"]["over_limit"] == 1


def test_identify_low_confidence_faces_never_reach_matching(
    insightface_environment, sample_jpeg, monkeypatch
):
    """The service-wide confidence floor (0.5) applies before
    face_identify, as it does for the other tasks."""
    env = insightface_environment
    svc = _build_service(env)
    svc.load()
    svc.face_db.register(person_id="alice", name="Alice", embedding=env["embedding_alice"])
    _room(env, monkeypatch, svc, [((10, 10, 60, 70), 0.3, env["embedding_alice"])])

    assert _identify(svc, sample_jpeg)["faces"] == []


def test_identify_category_limits_who_can_match(
    insightface_environment, sample_jpeg, monkeypatch
):
    env = insightface_environment
    svc = _build_service(env)
    svc.load()
    svc.face_db.register(person_id="alice", name="Alice",
                         embedding=env["embedding_alice"], category="family")
    _room(env, monkeypatch, svc, [((10, 10, 60, 70), 0.95, env["embedding_alice"])])

    assert _identify(svc, sample_jpeg, category="staff")["faces"][0]["recognized"] is False
    assert _identify(svc, sample_jpeg, category="family")["faces"][0]["recognized"] is True


@pytest.mark.parametrize("params", [
    {"max_faces": 0}, {"max_faces": 501}, {"max_faces": "many"},
    {"min_face_px": -1}, {"min_face_px": True},
])
def test_identify_rejects_bad_bounds(insightface_environment, sample_jpeg, params):
    svc = _build_service(insightface_environment)
    svc.load()
    with pytest.raises(ServiceError) as exc:
        _identify(svc, sample_jpeg, **params)
    assert exc.value.http_status == 400


def test_identify_is_advertised(insightface_environment):
    from adapters.insightface.service import _SUPPORTED_TASKS

    assert "face_identify" in _SUPPORTED_TASKS


# ── FaceDB.match_many ──────────────────────────────────────────────


def _db():
    from adapters.insightface.face_db import FaceDB
    return FaceDB()


def test_match_many_agrees_with_best_match():
    db = _db()
    db.register(person_id="a", name="A", embedding=_vec(0))
    db.register(person_id="b", name="B", embedding=_vec(1))
    db.register(person_id="b", name="B", embedding=_vec(2), append=True)
    queries = [_vec(0), _vec(2), _vec(0, 2), _vec(3)]

    many = db.match_many(queries, threshold=0.5)

    for query, answer in zip(queries, many):
        single = db.best_match(query, threshold=0.5)
        if single is None:
            assert answer["match"] is None
        else:
            assert answer["match"]["person_id"] == single["person_id"]
            assert answer["match"]["similarity"] == single["similarity"]


def test_match_many_takes_the_best_sample_per_person():
    db = _db()
    db.register(person_id="b", name="B", embedding=_vec(1))
    db.register(person_id="b", name="B", embedding=_vec(2), append=True)

    [answer] = db.match_many([_vec(2)])

    assert answer["match"]["person_id"] == "b"
    assert answer["match"]["similarity"] == pytest.approx(1.0)
    assert answer["second_similarity"] is None  # one person, no runner-up


def test_match_many_tie_goes_to_the_smaller_person_id():
    db = _db()
    db.register(person_id="zed", name="Z", embedding=_vec(0))
    db.register(person_id="amy", name="A", embedding=_vec(0))

    [answer] = db.match_many([_vec(0)])

    assert answer["match"]["person_id"] == "amy"


def test_match_many_sees_registrations_and_deletions():
    """The cached matrix is dropped on every mutation."""
    db = _db()
    db.register(person_id="a", name="A", embedding=_vec(0))
    assert db.match_many([_vec(1)])[0]["match"] is None

    db.register(person_id="b", name="B", embedding=_vec(1))
    assert db.match_many([_vec(1)])[0]["match"]["person_id"] == "b"

    db.delete("b")
    assert db.match_many([_vec(1)])[0]["match"] is None

    db.update("a", category="staff")
    assert db.match_many([_vec(0)], category="staff")[0]["match"]["person_id"] == "a"


def test_match_many_with_no_queries_or_no_people():
    db = _db()
    assert db.match_many([]) == []
    assert db.match_many([_vec(0)]) == [
        {"match": None, "nearest_similarity": None, "second_similarity": None}]


def test_match_many_normalises_the_query():
    db = _db()
    db.register(person_id="a", name="A", embedding=_vec(0))
    [answer] = db.match_many([[5.0] + [0.0] * 511])
    assert answer["match"]["similarity"] == pytest.approx(1.0)


# ── Detector size ──────────────────────────────────────────────────


def test_det_size_defaults_to_640(insightface_environment, monkeypatch):
    monkeypatch.delenv("OPENNVR_INSIGHTFACE_DET_SIZE", raising=False)
    svc = _build_service(insightface_environment)
    svc.load()
    assert svc._face_app.prepared_with["det_size"] == (640, 640)


def test_det_size_from_env_reaches_prepare(insightface_environment, monkeypatch):
    monkeypatch.setenv("OPENNVR_INSIGHTFACE_DET_SIZE", "1280")
    svc = _build_service(insightface_environment)
    svc.load()
    assert svc.is_ready()
    assert svc._face_app.prepared_with["det_size"] == (1280, 1280)
    assert svc.hardware_evaluation().details["det_size"] == 1280


@pytest.mark.parametrize("bad", ["big", "1000", "96", "4096"])
def test_bad_det_size_fails_the_load_with_the_reason(
    insightface_environment, monkeypatch, bad
):
    monkeypatch.setenv("OPENNVR_INSIGHTFACE_DET_SIZE", bad)
    svc = _build_service(insightface_environment)
    svc.load()
    assert not svc.is_ready()
    assert "OPENNVR_INSIGHTFACE_DET_SIZE" in svc._load_error
