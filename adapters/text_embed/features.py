"""Tokenisation and pooling for a BERT-style sentence encoder (BGE, MiniLM,
E5, GTE: ``input_ids``/``attention_mask``/``token_type_ids`` in,
``last_hidden_state`` out). The vector is the [CLS] position, L2-normalised
— what the BGE family is trained for; ``POOLING=mean`` for the MiniLM
family, which was trained on the mean instead."""
from __future__ import annotations

from typing import Any

import numpy as np

#: Longest input in tokens. Captions are a sentence or two and a query is
#: a few words; 128 covers both with room, and attention is quadratic.
MAX_TOKENS = 128
CLS = "cls"
MEAN = "mean"


def l2_normalize(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


def tokenize(texts: list[str], tokenizer: Any, *, max_tokens: int = MAX_TOKENS) -> dict[str, np.ndarray]:
    """One padded batch: ``input_ids``, ``attention_mask`` and
    ``token_type_ids`` as int64 [batch, longest]. ``tokenizer`` is a
    HuggingFace ``tokenizers.Tokenizer`` (``encode_batch`` → ``.ids`` and
    ``.attention_mask``); truncation to ``max_tokens`` is enforced here so
    a saved tokenizer's own settings cannot widen it."""
    encs = tokenizer.encode_batch([t.strip() for t in texts])
    ids = [list(e.ids)[:max_tokens] for e in encs]
    masks = [list(e.attention_mask)[:max_tokens] for e in encs]
    width = max(1, max(len(i) for i in ids))
    ids_arr = np.zeros((len(ids), width), dtype=np.int64)
    mask_arr = np.zeros((len(ids), width), dtype=np.int64)
    for row, (i, m) in enumerate(zip(ids, masks)):
        ids_arr[row, :len(i)] = i
        mask_arr[row, :len(m)] = m
    return {"input_ids": ids_arr, "attention_mask": mask_arr,
            "token_type_ids": np.zeros_like(ids_arr)}


def pool(hidden: np.ndarray, mask: np.ndarray, method: str = CLS) -> np.ndarray:
    """[batch, seq, dim] → [batch, dim]: the [CLS] row, or the mask-weighted
    mean of the real tokens."""
    if method == MEAN:
        m = mask[..., None].astype(np.float32)
        summed = (hidden * m).sum(axis=1)
        return summed / np.clip(m.sum(axis=1), 1e-9, None)
    return hidden[:, 0, :]
