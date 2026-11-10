"""One ONNX session, batches of sentences in, unit vectors out."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from .features import CLS, MAX_TOKENS, l2_normalize, pool, tokenize


@dataclass
class Embedding:
    vector: list[float]
    inference_ms: float


class TextEmbedEngine:
    """``session`` needs ``get_inputs()`` (names) and ``run(None, feeds)``
    returning ``[last_hidden_state]``; ``tokenizer`` needs ``encode_batch``."""

    def __init__(self, session: Any, tokenizer: Any, *, pooling: str = CLS,
                 max_tokens: int = MAX_TOKENS) -> None:
        self._session = session
        self._tokenizer = tokenizer
        self._pooling = pooling
        self._max_tokens = max_tokens
        self._inputs = {i.name for i in session.get_inputs()}
        self.dim: int | None = None

    def embed(self, texts: list[str]) -> list[Embedding]:
        feeds = tokenize(texts, self._tokenizer, max_tokens=self._max_tokens)
        feeds = {k: v for k, v in feeds.items() if k in self._inputs}
        t0 = time.perf_counter()
        hidden = self._session.run(None, feeds)[0]
        ms = (time.perf_counter() - t0) * 1000
        vecs = pool(np.asarray(hidden, dtype=np.float32), feeds["attention_mask"], self._pooling)
        self.dim = int(vecs.shape[1])
        per = ms / max(1, len(texts))
        return [Embedding([float(x) for x in l2_normalize(v)], per) for v in vecs]
