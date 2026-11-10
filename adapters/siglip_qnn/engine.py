"""Two sessions, one shared space: the image encoder and the text encoder of
Qualcomm's SigLIP2 export, each ``run`` once per request, each answer
dequantized and L2-normalised."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from .features import dequantize, l2_normalize, preprocess_image, tokenize


@dataclass
class Embedding:
    vector: list[float]
    modality: str
    inference_ms: float


class SiglipQnnEngine:
    """``image_session`` / ``text_session`` need ``run(None, feeds)``;
    ``metadata`` is the export's metadata.json (input/output specs);
    ``tokenizer`` needs ``encode(text).ids`` and ``token_to_id``."""

    def __init__(self, image_session: Any, text_session: Any, metadata: dict[str, Any], tokenizer: Any) -> None:
        files = metadata.get("model_files") or {}
        img = files.get("image_encoder.onnx") or {}
        txt = files.get("text_encoder.onnx") or {}
        self._img_in_name, self._img_in_spec = next(iter((img.get("inputs") or {"image": {}}).items()))
        self._img_out_name, self._img_out_spec = next(iter((img.get("outputs") or {"image_embeds": {}}).items()))
        self._txt_in_name, _ = next(iter((txt.get("inputs") or {"input_ids": {}}).items()))
        self._txt_out_name, self._txt_out_spec = next(iter((txt.get("outputs") or {"text_embeds": {}}).items()))
        self._image = image_session
        self._text = text_session
        self._tokenizer = tokenizer
        self.dim: int | None = None

    def embed_image(self, image_bytes: bytes) -> Embedding:
        tensor = preprocess_image(image_bytes, self._img_in_spec)
        t0 = time.perf_counter()
        out = self._image.run(None, {self._img_in_name: tensor})[0]
        vec = l2_normalize(dequantize(out, self._img_out_spec))
        self.dim = int(vec.shape[0])
        return Embedding([float(x) for x in vec], "image", (time.perf_counter() - t0) * 1000)

    def embed_text(self, text: str) -> Embedding:
        ids = tokenize(text, self._tokenizer)
        t0 = time.perf_counter()
        out = self._text.run(None, {self._txt_in_name: ids})[0]
        vec = l2_normalize(dequantize(out, self._txt_out_spec))
        self.dim = int(vec.shape[0])
        return Embedding([float(x) for x in vec], "text", (time.perf_counter() - t0) * 1000)
