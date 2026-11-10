"""Shared fixtures for the text-embed adapter: a model directory with the
three files, a fake ONNX session that answers a deterministic (not unit
length) hidden state from the token ids, and a fake tokenizer — so the
engine, the service, the SDK plumbing and the conformance kit run without
the 133 MB model."""
from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

FAKE_DIM = 8


class _In:
    def __init__(self, name):
        self.name = name


class FakeSession:
    """``last_hidden_state`` [batch, seq, 8]: row t of sequence b is a
    function of the token id at that position AND of the whole sequence
    (as attention would make it — so the [CLS] row changes when any
    word does), scaled by 3 so the service's normalisation is visible."""

    def __init__(self, inputs=("input_ids", "attention_mask", "token_type_ids")):
        self._inputs = list(inputs)
        self.feeds: list[dict] = []

    def get_inputs(self):
        return [_In(n) for n in self._inputs]

    def run(self, _outputs, feeds):
        self.feeds.append({k: np.array(v, copy=True) for k, v in feeds.items()})
        ids = feeds["input_ids"]
        b, s = ids.shape
        out = np.zeros((b, s, FAKE_DIM), dtype=np.float32)
        for i in range(b):
            whole = int(sum(int(x) for x in ids[i])) * 7919
            for t in range(s):
                rng = np.random.default_rng(whole + int(ids[i, t]) + 1)
                out[i, t] = rng.standard_normal(FAKE_DIM) * 3
        return [out]


class FakeTokenizer:
    """Whitespace words → ids by hash; [CLS]=101 first, [SEP]=102 last."""

    class _Enc:
        def __init__(self, ids):
            self.ids = ids
            self.attention_mask = [1] * len(ids)

    def encode_batch(self, texts):
        out = []
        for t in texts:
            ids = [101] + [1000 + (abs(hash(w)) % 5000) for w in t.split()] + [102]
            out.append(self._Enc(ids))
        return out


def write_model_dir(dir_: Path, meta: dict | None = None) -> dict:
    dir_.mkdir(parents=True, exist_ok=True)
    meta = meta if meta is not None else {"model_id": "BAAI/bge-small-en-v1.5", "pooling": "cls",
                                          "max_tokens": 128, "query_prefix": ""}
    (dir_ / "model.onnx").write_bytes(b"\x00" * 64)
    (dir_ / "tokenizer.json").write_bytes(b"{}")
    (dir_ / "model.json").write_text(json.dumps(meta))
    return meta


@pytest.fixture
def text_embed_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    model_dir = tmp_path / "text-embed"
    meta = write_model_dir(model_dir)
    monkeypatch.setenv("TEXT_EMBED_MODEL_DIR", str(model_dir))
    monkeypatch.delenv("ADAPTER_ONNX_PROVIDERS", raising=False)
    import adapters.text_embed.service as svc

    made: dict[str, object] = {}

    def fake_session(path, requested, threads):
        made["requested"] = requested
        made["threads"] = threads
        made["session"] = FakeSession()
        return made["session"]

    monkeypatch.setattr(svc, "build_session", fake_session)
    monkeypatch.setattr(svc, "load_tokenizer", lambda path: FakeTokenizer())
    if "adapters.text_embed.main" in sys.modules:
        importlib.reload(sys.modules["adapters.text_embed.main"])
    return {"model_dir": model_dir, "metadata": meta, "made": made}


@pytest.fixture
def text_embed_app(text_embed_environment, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("OPENNVR_ADAPTER_TOKEN", raising=False)
    from fastapi.testclient import TestClient

    import adapters.text_embed.main as main_module
    importlib.reload(main_module)
    with TestClient(main_module.app) as client:
        yield client
