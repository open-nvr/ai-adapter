# Copyright (c) 2026 OpenNVR
# SPDX-License-Identifier: AGPL-3.0-or-later

"""The Rekognition adapter against a fake Rekognition: no AWS, no network."""
from __future__ import annotations

import importlib
import io
import sys

import pytest
from PIL import Image

from opennvr_adapter_sdk import ServiceError


def _jpeg(w=400, h=300) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (w, h), (120, 120, 120)).save(buf, format="JPEG")
    return buf.getvalue()


class NotFound(Exception):
    pass


NotFound.__name__ = "ResourceNotFoundException"


class InvalidParameter(Exception):
    pass


InvalidParameter.__name__ = "InvalidParameterException"


class FakeRekognition:
    """One collection in memory. ``faces_in`` scripts what DetectFaces
    sees; ``who`` scripts which enrolled person a crop matches."""

    def __init__(self):
        self.collections: dict[str, list[dict]] = {}
        self.faces_in: list[dict] = []
        self.who: list[list[tuple[str, float]]] = []
        self.calls: list[str] = []
        self._n = 0

    def describe_collection(self, CollectionId):
        if CollectionId not in self.collections:
            raise NotFound("ResourceNotFoundException: no such collection")
        return {"FaceCount": len(self.collections[CollectionId])}

    def create_collection(self, CollectionId):
        self.collections[CollectionId] = []

    def list_faces(self, CollectionId, MaxResults, NextToken=None):
        return {"Faces": list(self.collections[CollectionId])}

    def detect_faces(self, Image):
        self.calls.append("detect")
        return {"FaceDetails": list(self.faces_in)}

    def search_faces_by_image(self, CollectionId, Image, FaceMatchThreshold, MaxFaces):
        self.calls.append(f"search@{FaceMatchThreshold}")
        script = self.who.pop(0) if self.who else []
        if script == "noface":
            raise InvalidParameter("InvalidParameterException: There are no faces in the image")
        return {"FaceMatches": [{"Similarity": s, "Face": {"ExternalImageId": pid}}
                                for pid, s in script if s >= FaceMatchThreshold]}

    def index_faces(self, CollectionId, Image, ExternalImageId, **_):
        self._n += 1
        face = {"FaceId": f"f{self._n}", "ExternalImageId": ExternalImageId}
        self.collections[CollectionId].append(face)
        return {"FaceRecords": [{"Face": face}]}

    def delete_faces(self, CollectionId, FaceIds):
        self.collections[CollectionId] = [f for f in self.collections[CollectionId]
                                          if f["FaceId"] not in FaceIds]


def _face(left, top, w=0.1, h=0.15, conf=99.0):
    return {"BoundingBox": {"Left": left, "Top": top, "Width": w, "Height": h}, "Confidence": conf}


@pytest.fixture
def svc(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENNVR_REKOGNITION_MIN_SIMILARITY", raising=False)
    from adapters.rekognition.service import RekognitionService

    fake = FakeRekognition()
    s = RekognitionService(client=fake, collection="att", index_path=str(tmp_path / "idx.json"))
    s.load()
    return s, fake


def test_load_creates_the_collection(svc):
    s, fake = svc
    assert s.is_ready() and "att" in fake.collections


def test_face_identify_looks_up_every_face_in_the_frame(svc):
    s, fake = svc
    fake.faces_in = [_face(0.1, 0.1), _face(0.5, 0.1, conf=80.0), _face(0.8, 0.6)]
    fake.who = [[("asha", 99.2), ("bilal", 91.0)], [], [("bilal", 97.0)]]
    r = s.infer({"__file__": _jpeg(), "task": "face_identify", "threshold": 0.5}).result
    assert r["face_count"] == 3 and r["detected_count"] == 3 and r["recognized_count"] == 2
    # Most confident first (99, 99, 80): the scripted matches land in that order.
    assert [f["person_id"] for f in r["faces"]] == ["asha", None, "bilal"]
    asha = r["faces"][0]
    assert asha["similarity"] == pytest.approx(0.992) and asha["second_similarity"] == pytest.approx(0.91)
    assert r["frame"] == {"w": 400, "h": 300}
    assert asha["bbox"] == [40, 30, 80, 75]


def test_a_low_threshold_never_searches_below_the_floor(svc):
    """Rekognition is not cosine: 0.5 from an attendance policy must not
    become a 50 % match."""
    s, fake = svc
    fake.faces_in = [_face(0.1, 0.1)]
    fake.who = [[("asha", 70.0)]]
    r = s.infer({"__file__": _jpeg(), "task": "face_identify", "threshold": 0.5}).result
    assert r["faces"][0]["recognized"] is False
    assert "search@90.0" in fake.calls
    fake.faces_in = [_face(0.1, 0.1)]
    fake.who = [[("asha", 93.0)]]
    s.infer({"__file__": _jpeg(), "task": "face_identify", "threshold": 0.95})
    assert "search@95.0" in fake.calls


def test_a_crop_rekognition_finds_no_face_in_is_a_miss(svc):
    s, fake = svc
    fake.faces_in = [_face(0.1, 0.1)]
    fake.who = ["noface"]
    r = s.infer({"__file__": _jpeg(), "task": "face_identify"}).result
    assert r["faces"][0]["recognized"] is False


def test_small_faces_and_the_limit_are_dropped_and_counted(svc):
    s, fake = svc
    fake.faces_in = [_face(0.1, 0.1), _face(0.5, 0.5, w=0.02, h=0.02)]
    r = s.infer({"__file__": _jpeg(), "task": "face_identify", "min_face_px": 20}).result
    assert r["face_count"] == 1 and r["dropped"]["too_small"] == 1


def test_bad_requests_are_400(svc):
    s, _ = svc
    for params in ({"task": "face_recognition"}, {"threshold": 2}, {"max_faces": 0}):
        with pytest.raises(ServiceError) as e:
            s.infer({"__file__": _jpeg(), **params})
        assert e.value.http_status == 400
    with pytest.raises(ServiceError):
        s.infer({"__file__": b"not an image", "task": "face_identify"})


def test_enrol_one_face_count_rebuild_and_forget(svc, tmp_path):
    s, fake = svc
    fake.faces_in = [_face(0.1, 0.1)]
    assert s.register("asha", _jpeg()) == 1
    assert s.register("asha", _jpeg()) == 2
    fake.faces_in = [_face(0.1, 0.1), _face(0.5, 0.5)]
    with pytest.raises(ValueError, match="2 faces"):
        s.register("asha", _jpeg())
    fake.faces_in = []
    with pytest.raises(ValueError, match="no face"):
        s.register("asha", _jpeg())

    # A restart rebuilds who has which faces from the collection itself.
    from adapters.rekognition.service import RekognitionService
    again = RekognitionService(client=fake, collection="att")
    again.load()
    assert again.people() == {"asha": 2}

    assert again.forget("asha") is True and fake.collections["att"] == []
    assert again.forget("asha") is False


def test_routes_and_declared_egress(monkeypatch, tmp_path):
    monkeypatch.setenv("AWS_REGION", "ap-south-1")
    monkeypatch.delenv("OPENNVR_ADAPTER_TOKEN", raising=False)
    fake = FakeRekognition()
    import adapters.rekognition.service as svc_mod
    monkeypatch.setattr(svc_mod, "_client_from_env", lambda: fake)
    if "adapters.rekognition.main" in sys.modules:
        importlib.reload(sys.modules["adapters.rekognition.main"])
    import adapters.rekognition.main as main
    from fastapi.testclient import TestClient

    with TestClient(main.app) as client:
        caps = client.get("/capabilities").json()
        assert caps["tasks_advertised"] == ["face_identify"]
        assert caps["permissions"]["network_egress"] == ["https://rekognition.ap-south-1.amazonaws.com"]
        fake.faces_in = [_face(0.1, 0.1)]
        r = client.post("/faces/register", files={"frame": ("a.jpg", _jpeg(), "image/jpeg")},
                        data={"person_id": "u-1", "name": "Asha", "append": "true"})
        assert r.status_code == 200 and r.json()["face"]["samples"] == 1
        fake.faces_in = []
        r = client.post("/faces/register", files={"frame": ("a.jpg", _jpeg(), "image/jpeg")},
                        data={"person_id": "u-1"})
        assert r.status_code == 422
        assert client.get("/faces").json()["faces"] == [{"person_id": "u-1", "samples": 1}]
        # The only photo cannot go one by one; a second one can.
        assert client.delete("/faces/u-1/samples/0").status_code == 409
        fake.faces_in = [_face(0.1, 0.1)]
        client.post("/faces/register", files={"frame": ("b.jpg", _jpeg(), "image/jpeg")},
                    data={"person_id": "u-1"})
        r = client.delete("/faces/u-1/samples/0")
        assert r.status_code == 200 and r.json()["face"]["samples"] == 1
        assert client.delete("/faces/u-1/samples/7").status_code == 404
        assert client.delete("/faces/nobody/samples/0").status_code == 404
        assert client.delete("/faces/u-1").status_code == 200
        assert client.delete("/faces/u-1").status_code == 404


def _with_points(face):
    face["Landmarks"] = [{"Type": t, "X": x, "Y": 0.2} for t, x in
                         (("eyeLeft", 0.12), ("eyeRight", 0.18), ("nose", 0.15),
                          ("mouthLeft", 0.13), ("mouthRight", 0.17))]
    return face


def test_without_match_only_detect_faces_is_paid_for(svc):
    s, fake = svc
    fake.faces_in = [_with_points(_face(0.1, 0.1))]
    r = s.infer({"__file__": _jpeg(), "task": "face_identify", "match": "false"}).result
    assert fake.calls == ["detect"]
    [face] = r["faces"]
    assert face["person_id"] is None and face["recognized"] is False
    # Pixels, in InsightFace's five-point order.
    assert face["landmarks"] == [[48, 60], [72, 60], [60, 60], [52, 60], [68, 60]]


def test_one_photo_can_be_forgotten_but_not_the_last(svc):
    s, fake = svc
    fake.faces_in = [_face(0.1, 0.1)]
    for _ in range(3):
        s.register("asha", _jpeg())
    first, second, third = s._index["asha"]
    assert s.forget_sample("asha", 1) == 2
    assert s._index["asha"] == [first, third]
    assert [f["FaceId"] for f in fake.collections["att"]] == [first, third]
    with pytest.raises(IndexError):
        s.forget_sample("asha", 5)
    s.forget_sample("asha", 0)
    with pytest.raises(ValueError):
        s.forget_sample("asha", 0)
    with pytest.raises(KeyError):
        s.forget_sample("nobody", 0)


def test_a_restart_keeps_the_order_photos_are_addressed_by(svc, tmp_path):
    from adapters.rekognition.service import RekognitionService

    s, fake = svc
    fake.faces_in = [_face(0.1, 0.1)]
    for _ in range(3):
        s.register("asha", _jpeg())
    order = list(s._index["asha"])
    fake.collections["att"].reverse()           # ListFaces promises no order
    again = RekognitionService(client=fake, collection="att", index_path=str(tmp_path / "idx.json"))
    again.load()
    assert again._index["asha"] == order
