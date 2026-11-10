"""The text-embed adapter without its model: tokenisation and pooling, the
engine, the service's contract and refusals, and the HTTP surface."""
from __future__ import annotations

import numpy as np
import pytest

from opennvr_adapter_sdk.contract import InferResponse
from tests._text_embed_fixtures import (  # noqa: F401
    FAKE_DIM,
    FakeSession,
    FakeTokenizer,
    text_embed_app,
    text_embed_environment,
    write_model_dir,
)

# ── features ────────────────────────────────────────────────────────

def test_tokenize_pads_a_batch_to_its_longest_and_truncates_at_max_tokens():
    from adapters.text_embed.features import tokenize

    f = tokenize(["a red truck", "a person"], FakeTokenizer())
    assert f["input_ids"].shape == (2, 5) and f["input_ids"].dtype == np.int64
    assert f["attention_mask"].tolist() == [[1, 1, 1, 1, 1], [1, 1, 1, 1, 0]]
    assert f["token_type_ids"].tolist() == [[0] * 5, [0] * 5]
    assert f["input_ids"][1, 4] == 0
    long = tokenize([" ".join(["w"] * 300)], FakeTokenizer(), max_tokens=16)
    assert long["input_ids"].shape == (1, 16)


def test_pool_takes_cls_or_the_masked_mean():
    from adapters.text_embed.features import pool

    hidden = np.array([[[1.0, 0.0], [3.0, 0.0], [100.0, 100.0]]], dtype=np.float32)
    mask = np.array([[1, 1, 0]])
    assert pool(hidden, mask, "cls").tolist() == [[1.0, 0.0]]
    assert pool(hidden, mask, "mean").tolist() == [[2.0, 0.0]]       # the padded row is ignored


# ── engine ──────────────────────────────────────────────────────────

def test_engine_returns_unit_vectors_and_only_the_inputs_the_graph_has():
    from adapters.text_embed.engine import TextEmbedEngine

    sess = FakeSession(inputs=("input_ids", "attention_mask"))       # no token_type_ids
    eng = TextEmbedEngine(sess, FakeTokenizer())
    outs = eng.embed(["a red truck", "a person in a blue shirt"])
    assert len(outs) == 2 and eng.dim == FAKE_DIM
    for o in outs:
        assert len(o.vector) == FAKE_DIM and abs(np.linalg.norm(o.vector) - 1) < 1e-5
    assert set(sess.feeds[0]) == {"input_ids", "attention_mask"}
    # the same sentence embeds the same, a different one differently
    again = eng.embed(["a red truck"])[0].vector
    assert again == pytest.approx(outs[0].vector)
    assert np.dot(outs[0].vector, outs[1].vector) < 0.99


# ── service ─────────────────────────────────────────────────────────

def _service(tmp_path, meta=None, **env):
    from adapters.text_embed.service import TextEmbedService

    d = tmp_path / "m"
    write_model_dir(d, meta)
    built = {}

    def factory(path, requested, threads):
        built.update(path=path, requested=requested, threads=threads)
        built["session"] = FakeSession()
        return built["session"]

    svc = TextEmbedService(str(d), session_factory=factory, tokenizer_factory=lambda p: FakeTokenizer())
    svc.load()
    return svc, built


def test_service_loads_on_the_cpu_with_two_threads_and_embeds_one_or_many(tmp_path):
    svc, built = _service(tmp_path)
    assert svc.is_ready() and built["requested"] == "CPUExecutionProvider" and built["threads"] == 2
    r = svc.infer({"task": "text_embed", "text": "a red truck", "kind": "query"})
    assert isinstance(r, InferResponse) and r.status == "ok"
    assert r.result["dim"] == FAKE_DIM and r.result["modality"] == "text" and r.result["normalized"] is True
    assert r.result["model"] == "BAAI/bge-small-en-v1.5/cls-onnx" and r.result["kind"] == "query"
    assert "embeddings" not in r.result
    r2 = svc.infer({"task": "text_embed", "texts": ["a", "b", "c"]})
    assert len(r2.result["embeddings"]) == 3 and "embedding" not in r2.result
    assert r2.result["kind"] == "document"
    info = svc.model_info()
    assert info.name == "BAAI/bge-small-en-v1.5" and info.modalities_in == ["text"]
    assert info.modalities_out == ["embedding"] and svc.fingerprint().startswith("sha256:")
    assert svc.hardware_evaluation().details["embedding_dim"] == FAKE_DIM


def test_a_query_prefix_from_model_json_is_prepended_for_queries_only(tmp_path):
    meta = {"model_id": "x/y", "pooling": "cls", "query_prefix": "Represent this sentence: "}
    svc, built = _service(tmp_path, meta)
    tok = FakeTokenizer()
    q = svc.infer({"task": "text_embed", "text": "red truck", "kind": "query"}).result["embedding"]
    d = svc.infer({"task": "text_embed", "text": "red truck", "kind": "document"}).result["embedding"]
    assert q != pytest.approx(d)
    # the document embedding is the bare text's
    fed = built["session"].feeds[-1]["input_ids"][0].tolist()
    assert fed == tok.encode_batch(["red truck"])[0].ids
    assert svc.model_label == "x/y/cls-onnx"


def test_mean_pooling_is_honoured_and_an_unknown_pooling_refuses_to_load(tmp_path):
    svc, _ = _service(tmp_path, {"model_id": "m", "pooling": "mean"})
    assert svc.is_ready() and svc.model_label == "m/mean-onnx"
    bad, _ = _service(tmp_path, {"model_id": "m", "pooling": "max"})
    assert not bad.is_ready() and "pooling" in (bad.hardware_evaluation().reasoning or "")


def test_service_refuses_bad_requests(tmp_path):
    from opennvr_adapter_sdk import ServiceError

    svc, _ = _service(tmp_path)
    for payload, code, status in (
        ({"task": "text_embed"}, "malformed_input", 400),
        ({"task": "text_embed", "text": ""}, "malformed_input", 400),
        ({"task": "text_embed", "text": "a", "texts": ["b"]}, "malformed_input", 400),
        ({"task": "text_embed", "texts": []}, "malformed_input", 400),
        ({"task": "text_embed", "texts": ["a", 3]}, "malformed_input", 400),
        ({"task": "text_embed", "texts": ["a"] * 65}, "malformed_input", 413),
        ({"task": "text_embed", "text": "x" * 5000}, "text_too_long", 413),
        ({"task": "text_embed", "text": "a", "kind": "image"}, "malformed_input", 400),
        ({"task": "embed", "text": "a"}, "malformed_input", 400),
    ):
        with pytest.raises(ServiceError) as e:
            svc.infer(payload)
        assert (e.value.code, e.value.http_status) == (code, status), payload


def test_missing_files_leave_the_service_in_error_not_crashed(tmp_path):
    from adapters.text_embed.service import TextEmbedService

    svc = TextEmbedService(str(tmp_path / "nowhere"), session_factory=lambda *a: FakeSession(),
                           tokenizer_factory=lambda p: FakeTokenizer())
    svc.load()
    assert not svc.is_ready()
    from opennvr_adapter_sdk import ServiceError
    with pytest.raises(ServiceError) as e:
        svc.infer({"task": "text_embed", "text": "a"})
    assert e.value.code == "weights_missing" and e.value.http_status == 503


# ── HTTP ────────────────────────────────────────────────────────────

def test_capabilities_advertise_text_embed_and_infer_answers_over_http(text_embed_app):
    caps = text_embed_app.get("/capabilities").json()
    assert caps["tasks_advertised"] == ["text_embed"]
    assert caps["model"]["name"] == "BAAI/bge-small-en-v1.5"
    r = text_embed_app.post("/infer", json={"task": "text_embed", "text": "a red truck", "kind": "query"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["result"]["dim"] == FAKE_DIM and len(body["result"]["embedding"]) == FAKE_DIM
    r = text_embed_app.post("/infer", json={"task": "text_embed", "texts": ["a", "b"]})
    assert r.status_code == 200 and len(r.json()["result"]["embeddings"]) == 2
