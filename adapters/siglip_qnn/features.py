"""The two front ends of Qualcomm's SigLIP2 export, and its fixed-point
boundary.

Image: 224×224 RGB, bilinear resize with no crop (``SiglipImageProcessor``
with ``do_resize`` to 224×224, ``resample`` 2), scaled to [0, 1] — the
export then takes it as uint16 with the scale/zero-point metadata.json
names (the normalisation to mean 0.5 / std 0.5 is folded into the graph).

Text: the Gemma tokenizer SigLIP2 was trained with (256k pieces, no BOS,
an EOS appended by the tokenizer's own post-processor), lower-cased as
the model's processor does, padded with 0 or truncated to the export's
fixed 64 positions, keeping the EOS last when it truncates.

Both encoders return a uint16 [1, 768] vector with its own quantization
parameters; ``dequantize`` + ``l2_normalize`` make it the unit vector
the search store expects.
"""
from __future__ import annotations

import io
from typing import Any

import numpy as np

IMAGE_SIZE = 224
TEXT_LEN = 64
EMBED_DIM = 768
PAD_ID = 0
DEFAULT_EOS_ID = 1


def _spec_params(spec: dict[str, Any]) -> tuple[float, int, str]:
    qp = spec.get("quantization_parameters") or {}
    return float(qp.get("scale", 1.0)), int(qp.get("zero_point", 0)), str(spec.get("dtype", "uint16"))


def quantize(x: np.ndarray, spec: dict[str, Any]) -> np.ndarray:
    """Float → the export's fixed-point dtype, rounded and clipped."""
    scale, zp, dtype = _spec_params(spec)
    info = np.iinfo(np.dtype(dtype))
    q = np.round(np.asarray(x, np.float32) / scale) + zp
    return np.clip(q, info.min, info.max).astype(dtype)


def dequantize(q: np.ndarray, spec: dict[str, Any]) -> np.ndarray:
    scale, zp, _ = _spec_params(spec)
    return (np.asarray(q).astype(np.float32) - zp) * scale


def l2_normalize(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, np.float32).reshape(-1)
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


def preprocess_image(image_bytes: bytes, spec: dict[str, Any]) -> np.ndarray:
    """A JPEG/PNG → the encoder's ``image`` tensor [1, 3, 224, 224] in the
    export's dtype. Raises ``ValueError`` for bytes that are not an image."""
    from PIL import Image

    try:
        im = Image.open(io.BytesIO(image_bytes))
        im.load()
    except Exception as exc:  # noqa: BLE001 — PIL raises a family of errors
        raise ValueError(f"not a decodable image: {exc}") from exc
    im = im.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
    arr = np.asarray(im, dtype=np.float32) / 255.0            # HWC in [0, 1]
    chw = np.transpose(arr, (2, 0, 1))[None, ...]              # [1, 3, H, W]
    return np.ascontiguousarray(quantize(chw, spec))


def tokenize(text: str, tokenizer: Any, *, length: int = TEXT_LEN) -> np.ndarray:
    """The text encoder's ``input_ids`` [1, 64] (int32): lower-cased, the
    tokenizer's own EOS kept last, padded with 0 or truncated."""
    ids = list(tokenizer.encode(text.strip().lower()).ids)
    eos = None
    try:
        eos = tokenizer.token_to_id("<eos>")
    except Exception:  # noqa: BLE001 — a tokenizer without the lookup
        eos = None
    if eos is None:
        eos = DEFAULT_EOS_ID
    if len(ids) > length:
        ids = ids[:length - 1] + [eos]
    if len(ids) < length:
        ids = ids + [PAD_ID] * (length - len(ids))
    return np.asarray([ids], dtype=np.int32)
