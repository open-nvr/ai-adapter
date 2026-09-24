# Copyright (c) 2026 OpenNVR
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
The real weights, and the one claim the stub cannot make.

Everything in ``test_clip_service.py`` runs against a fake encoder,
which is right: it pins the CONTRACT — normalisation, the dimension
travelling with the vector, the refusals — and does it in a second and
a half with no download.

None of it can tell you whether the thing works. Whether "a red truck"
lands nearer a picture of a red truck than a picture of a blue circle
is a property of the WEIGHTS, and asserting it against a stub that
returns a fixed array would be theatre — the test would pass with the
model wired backwards.

So this file runs the real ViT-B-32 and asserts the property the whole
feature rests on: **text and images share a space, and the geometry of
that space is meaningful.** If this fails, semantic search ranks by
noise and every test in the other file still passes.

SKIPPED unless the weights are already cached, because a 600 MB
download does not belong in a routine test run:

    CLIP_TEST_WEIGHTS=1 CLIP_CACHE_DIR=/path/to/cache pytest tests/test_clip_real_model.py

The shapes below are drawn with PIL rather than shipped as fixtures —
a repo should not carry test images it can generate, and a crude red
truck is enough to separate from a blue circle by a wide margin.
"""
from __future__ import annotations

import io
import os

import pytest

pytestmark = pytest.mark.skipif(
    not os.getenv("CLIP_TEST_WEIGHTS"),
    reason="set CLIP_TEST_WEIGHTS=1 (and CLIP_CACHE_DIR) to run against "
           "the real ~600 MB weights",
)


@pytest.fixture(scope="module")
def real_service():
    import adapters.clip.service as mod

    svc = mod.ClipEmbeddingService()
    svc.load()
    if not svc.is_ready():
        pytest.skip(f"weights unavailable: {svc._load_error}")
    return svc


def _png(draw_fn) -> bytes:
    from PIL import Image, ImageDraw

    im = Image.new("RGB", (224, 224), "white")
    draw_fn(ImageDraw.Draw(im))
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()


def _red_truck() -> bytes:
    def draw(d):
        d.rectangle([40, 110, 190, 170], fill="red")     # body
        d.rectangle([40, 80, 110, 120], fill="red")      # cab
        d.ellipse([55, 160, 85, 190], fill="black")      # wheels
        d.ellipse([150, 160, 180, 190], fill="black")
    return _png(draw)


def _blue_circle() -> bytes:
    return _png(lambda d: d.ellipse([50, 50, 180, 180], fill="blue"))


def _dot(a, b) -> float:
    return sum(x * y for x, y in zip(a, b))


def _embed_image(svc, blob):
    import base64
    return svc.infer({"frame_b64": base64.b64encode(blob).decode()}).result["embedding"]


def _embed_text(svc, text):
    return svc.infer({"text": text}).result["embedding"]


def test_the_model_is_512_dimensional(real_service):
    """The dimension OpenNVR's store will be built around. If a weights
    change moves it, a store full of 512-wide vectors stops being
    comparable and this is where that is caught."""
    assert real_service._dim == 512
    assert len(_embed_text(real_service, "a truck")) == 512


def test_words_and_pictures_land_in_the_same_space(real_service):
    """THE claim the whole feature rests on.

    'a red truck' must be nearer the red truck than the blue circle,
    and 'a blue circle' the other way round. Both directions, because
    one alone would pass on a model that simply likes red.
    """
    truck = _embed_image(real_service, _red_truck())
    circle = _embed_image(real_service, _blue_circle())

    truck_words = _embed_text(real_service, "a red truck")
    circle_words = _embed_text(real_service, "a blue circle")

    assert _dot(truck, truck_words) > _dot(circle, truck_words), (
        "'a red truck' was nearer a blue circle than a red truck — the "
        "text and image towers are not sharing a space")
    assert _dot(circle, circle_words) > _dot(truck, circle_words)


def test_the_margin_is_wide_enough_to_rank_with(real_service):
    """Ordering correctly is not enough — fusion needs the right answer
    to win by something. A margin near zero would rank correctly on
    these two toys and arbitrarily on real footage."""
    truck = _embed_image(real_service, _red_truck())
    circle = _embed_image(real_service, _blue_circle())
    words = _embed_text(real_service, "a red truck")

    margin = _dot(truck, words) - _dot(circle, words)
    assert margin > 0.05, f"margin only {margin:.3f}"


def test_vectors_from_the_real_model_are_unit_length(real_service):
    """The stub proves _unit() is called; this proves it is called on
    something real, where the raw vectors are emphatically not unit
    length to begin with."""
    import math

    for vector in (_embed_text(real_service, "a lorry"),
                   _embed_image(real_service, _red_truck())):
        assert math.sqrt(sum(v * v for v in vector)) == pytest.approx(1.0, abs=1e-5)


def test_the_same_input_embeds_identically_twice(real_service):
    """Determinism. A model left in training mode (dropout live) would
    return a slightly different vector each call, which would make a
    re-enriched store disagree with itself for no visible reason."""
    first = _embed_text(real_service, "a red truck at the loading bay")
    second = _embed_text(real_service, "a red truck at the loading bay")

    assert _dot(first, second) == pytest.approx(1.0, abs=1e-6)


def test_the_fingerprint_is_stable_across_loads(real_service):
    """Drift detection compares this between boots. A fingerprint that
    changed on every load would cry drift forever; one that never
    changed would never notice a swapped checkpoint."""
    import adapters.clip.service as mod

    again = mod.ClipEmbeddingService()
    again.load()
    if not again.is_ready():
        pytest.skip("second load failed")

    assert again.fingerprint() == real_service.fingerprint()
    assert real_service.fingerprint().startswith("sha256:")


def test_a_paraphrase_beats_an_unrelated_phrase(real_service):
    """The reason this exists at all: the operator types a word the
    captioner never wrote. 'lorry' has to beat 'bicycle' on a picture
    of a truck, or the second ranking arm adds nothing a keyword search
    could not already do."""
    truck = _embed_image(real_service, _red_truck())

    lorry = _dot(truck, _embed_text(real_service, "a lorry"))
    bicycle = _dot(truck, _embed_text(real_service, "a bicycle"))

    assert lorry > bicycle, f"lorry {lorry:.3f} did not beat bicycle {bicycle:.3f}"
