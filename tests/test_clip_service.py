# Copyright (c) 2026 OpenNVR
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
The CLIP adapter's behaviour, against a stubbed model.

What is pinned here is the CONTRACT this adapter keeps with its
consumer, not whether CLIP is any good — that is a property of the
weights and is checked in ``test_clip_real_model.py``.

Three promises carry the whole feature:

* **It answers a request with no image.** Every other vision adapter in
  this repo refuses one, and the SDK's IMAGE body shape refuses it
  before the service is reached. Half of semantic search is the
  operator's typed query, so a text-only call has to work — and if it
  ever stops working, search silently loses its second ranking arm and
  nothing anywhere says so.
* **Vectors come back unit length.** The consumer compares them with a
  dot product and is told ``normalized: true``. A raw vector would make
  every similarity wrong by a factor nobody would think to look for.
* **The dimension and the model are reported, not implied.** Comparing
  vectors from two different models produces a confident ordering of
  noise. These fields are the only way a store can refuse to.
"""
from __future__ import annotations

import base64
import math

import pytest

from tests._clip_service_fixtures import (  # noqa: F401
    FAKE_DIM,
    clip_app,
    clip_app_with_auth,
    clip_environment,
    service,
    small_jpeg,
)


def _unit_length(vector) -> float:
    return math.sqrt(sum(v * v for v in vector))


# ── the request that carries no image ────────────────────────────────


def test_a_text_query_is_embedded(service):
    """THE case the IMAGE body shape would have refused."""
    out = service.infer({"text": "a red truck at the loading bay"})

    assert out.result["modality"] == "text"
    assert len(out.result["embedding"]) == FAKE_DIM


def test_the_http_route_accepts_a_body_with_no_frame(clip_app):
    """Pinned at the TRANSPORT level, because this is where it would
    break: the SDK's IMAGE parser rejects a frameless JSON body with a
    400 before ``infer()`` is called, and switching this adapter's body
    shape back would be an easy, silent mistake."""
    resp = clip_app.post("/infer", json={"task": "embed", "text": "a lorry"})

    assert resp.status_code == 200, resp.text
    assert resp.json()["result"]["modality"] == "text"


def test_an_image_is_embedded(service, small_jpeg):
    out = service.infer({"frame_b64": base64.b64encode(small_jpeg).decode()})

    assert out.result["modality"] == "image"
    assert len(out.result["embedding"]) == FAKE_DIM


def test_text_and_image_do_not_produce_the_same_vector(service, small_jpeg):
    """The two modalities have to actually depend on their input. A
    service that ignored one of them would still pass every shape
    assertion above."""
    text = service.infer({"text": "a lorry"}).result["embedding"]
    image = service.infer(
        {"frame_b64": base64.b64encode(small_jpeg).decode()}).result["embedding"]

    assert text != image


def test_a_multipart_file_upload_is_refused_with_a_reason(clip_app, small_jpeg):
    """The cost of the TEXT body shape, pinned so it stays honest.

    TEXT is what lets a frameless query through the door, and the price
    is that the SDK's TEXT parser collects only string form fields — a
    multipart FILE part never reaches the service. Somebody hand-testing
    with curl will hit this, and the refusal has to say WHY rather than
    telling them to send the image they just sent.
    """
    resp = clip_app.post(
        "/infer",
        files={"frame": ("f.jpg", small_jpeg, "image/jpeg")},
        data={"params": '{"task": "embed"}'},
    )

    assert resp.status_code == 400
    body = resp.text.lower()
    assert "frame_b64" in body
    assert "multipart" in body, (
        "the refusal does not mention multipart, so a caller who "
        "uploaded a file is told to send an image with no hint that "
        "the upload was the problem")


# ── the promises about the vector ────────────────────────────────────


def test_vectors_come_back_unit_length(service, small_jpeg):
    """The fixture's fake encoders return 3-4-0… and 0-0-6-8, neither
    of which is unit length — so this fails if normalisation is
    dropped, rather than passing on a lucky fixture."""
    for payload in ({"text": "a lorry"},
                    {"frame_b64": base64.b64encode(small_jpeg).decode()}):
        out = service.infer(payload)

        assert _unit_length(out.result["embedding"]) == pytest.approx(1.0, abs=1e-5)
        assert out.result["normalized"] is True


def test_the_dimension_is_reported_and_matches_the_vector(service):
    out = service.infer({"text": "a lorry"})

    assert out.result["dim"] == len(out.result["embedding"]) == FAKE_DIM


def test_the_model_identity_travels_with_the_vector(service):
    """A store holding vectors from two models ranks incoherently, and
    the only way to notice is for each vector to say where it came
    from."""
    out = service.infer({"text": "a lorry"})

    assert out.result["model"].startswith("open_clip/")


def test_the_dimension_is_read_from_the_model_not_assumed(service):
    """FAKE_DIM is 8, not 512. If the service hard-coded ViT-B-32's
    width, this is where it shows — and the consequence in production
    would be ``CLIP_MODEL=ViT-L-14`` writing 768-wide vectors into a
    store told they were 512 wide."""
    assert service._dim == FAKE_DIM


# ── refusals ─────────────────────────────────────────────────────────


def test_neither_text_nor_image_is_refused(service):
    from opennvr_adapter_sdk import ServiceError

    with pytest.raises(ServiceError) as exc:
        service.infer({"task": "embed"})

    assert exc.value.envelope().error.code == "malformed_input"


def test_both_text_and_image_is_refused_rather_than_guessed(service, small_jpeg):
    """Picking one silently would turn a caller's bug into a bad
    ranking discovered weeks later with nothing to trace it to."""
    from opennvr_adapter_sdk import ServiceError

    with pytest.raises(ServiceError) as exc:
        service.infer({
            "text": "a lorry",
            "frame_b64": base64.b64encode(small_jpeg).decode(),
        })

    assert exc.value.envelope().error.code == "ambiguous_input"


def test_a_corrupt_image_is_a_typed_refusal_not_a_crash(service):
    from opennvr_adapter_sdk import ServiceError

    with pytest.raises(ServiceError) as exc:
        service.infer({"frame_b64": base64.b64encode(b"not an image").decode()})

    assert exc.value.envelope().error.code == "malformed_input"
    assert exc.value.envelope().error.transient is False


def test_invalid_base64_is_a_typed_refusal(service):
    from opennvr_adapter_sdk import ServiceError

    with pytest.raises(ServiceError) as exc:
        service.infer({"frame_b64": "!!!not base64!!!"})

    assert exc.value.envelope().error.code == "malformed_input"


def test_an_oversized_image_is_refused_before_it_is_decoded(service):
    """Checked on the ENCODED length. Decoding first to discover the
    body is too big is how a size guard becomes a memory amplifier."""
    import adapters.clip.service as mod
    from opennvr_adapter_sdk import ServiceError

    with pytest.raises(ServiceError) as exc:
        service.infer({"frame_b64": "A" * (mod.MAX_IMAGE_BYTES + 1)})

    assert exc.value.envelope().error.code == "body_too_large"


def test_an_oversized_query_is_refused(service):
    import adapters.clip.service as mod
    from opennvr_adapter_sdk import ServiceError

    with pytest.raises(ServiceError) as exc:
        service.infer({"text": "x" * (mod.MAX_TEXT_CHARS + 1)})

    assert exc.value.envelope().error.code == "text_too_long"


def test_a_service_that_failed_to_load_says_so_transiently_or_not(clip_environment):
    """`still loading` is transient and worth a retry; `weights missing`
    is not, and a caller that retries it forever is a caller nobody
    told."""
    import importlib

    import adapters.clip.service as mod
    from opennvr_adapter_sdk import ServiceError
    from opennvr_adapter_sdk.contract import HealthStatus

    importlib.reload(mod)
    svc = mod.ClipEmbeddingService()          # never loaded
    with pytest.raises(ServiceError) as exc:
        svc.infer({"text": "a lorry"})
    assert exc.value.envelope().error.transient is True

    svc._load_state = HealthStatus.ERROR
    svc._load_error = "no weights"
    with pytest.raises(ServiceError) as exc:
        svc.infer({"text": "a lorry"})
    assert exc.value.envelope().error.transient is False


# ── what the adapter declares about itself ───────────────────────────


def test_it_advertises_embed_and_only_embed(clip_app):
    caps = clip_app.get("/capabilities").json()

    assert caps["tasks_advertised"] == ["embed"]


def test_it_declares_both_input_modalities(clip_app):
    """The machine-readable form of "this one takes words too".

    Under /capabilities, where ModelInfo actually lives — there is no
    /model route on the contract.
    """
    info = clip_app.get("/capabilities").json()["model"]

    assert set(info["modalities_in"]) == {"image", "text"}
    assert info["modalities_out"] == ["embedding"]


def test_a_populated_cache_declares_no_egress(monkeypatch, tmp_path):
    """The posture an air-gapped site checks. Weights already present
    means the adapter never dials anything, and it must SAY so."""
    import importlib

    cache = tmp_path / "weights"
    cache.mkdir()
    (cache / "open_clip_pytorch_model.bin").write_bytes(b"x")
    monkeypatch.setenv("CLIP_CACHE_DIR", str(cache))
    monkeypatch.delenv("CLIP_OFFLINE", raising=False)

    import adapters.clip.main as main_mod
    importlib.reload(main_mod)

    assert main_mod._weights_fetch_egress() == []


def test_an_empty_cache_declares_the_one_host_it_will_dial(monkeypatch, tmp_path):
    """The other direction, and the one that matters for §8: an
    undeclared egress host in an audit log is what gets an adapter
    removed."""
    import importlib

    monkeypatch.setenv("CLIP_CACHE_DIR", str(tmp_path / "empty"))
    monkeypatch.delenv("CLIP_OFFLINE", raising=False)

    import adapters.clip.main as main_mod
    importlib.reload(main_mod)

    assert main_mod._weights_fetch_egress() == ["huggingface.co"]


def test_offline_is_an_assertion_stronger_than_an_empty_cache(monkeypatch, tmp_path):
    """A cache that happens to be full today is not a promise. Setting
    CLIP_OFFLINE is."""
    import importlib

    monkeypatch.setenv("CLIP_CACHE_DIR", str(tmp_path / "empty"))
    monkeypatch.setenv("CLIP_OFFLINE", "1")

    import adapters.clip.main as main_mod
    importlib.reload(main_mod)

    assert main_mod._weights_fetch_egress() == []
