"""Whisper's log-mel front end in numpy, and the fixed-point helpers the
quantized export needs.

The encoder was compiled for exactly one input: 80 mel bins over 3000
frames (30 s at 16 kHz, hop 160), as HuggingFace's WhisperFeatureExtractor
produces them — periodic Hann window of 400, reflect-padded centred frames,
power spectrum with the last frame dropped, the model's own mel filter bank
(80 x 201, shipped in ``preprocessor_config.json``), log10 clamped to eight
decades below the loudest bin, then ``(x + 4) / 4``. This implementation was
checked against that extractor on a tone-plus-noise clip and on silence
(max abs difference 1e-6 and 0.0); it exists so the image carries numpy and
not transformers.
"""
from __future__ import annotations

import numpy as np

SAMPLE_RATE = 16000
N_FFT = 400
HOP = 160
CHUNK_SECONDS = 30
N_SAMPLES = CHUNK_SECONDS * SAMPLE_RATE      # 480000
N_FRAMES = N_SAMPLES // HOP                  # 3000
N_MELS = 80


def log_mel(wave: np.ndarray, mel_filters: np.ndarray) -> np.ndarray:
    """``(80, 3000)`` float32 log-mel features of up to 30 s of 16 kHz mono
    audio (shorter input is zero-padded, longer is cut). ``mel_filters`` is
    the ``(80, 201)`` bank from the model's preprocessor config."""
    filters = np.asarray(mel_filters, dtype=np.float32)
    if filters.shape != (N_MELS, N_FFT // 2 + 1):
        raise ValueError(f"mel filters must be ({N_MELS}, {N_FFT // 2 + 1}), got {filters.shape}")
    x = np.asarray(wave, dtype=np.float32).reshape(-1)
    x = x[:N_SAMPLES] if x.shape[0] >= N_SAMPLES else np.pad(x, (0, N_SAMPLES - x.shape[0]))
    window = np.hanning(N_FFT + 1)[:-1].astype(np.float32)           # periodic Hann
    padded = np.pad(x, (N_FFT // 2, N_FFT // 2), mode="reflect")
    n_frames = 1 + (padded.shape[0] - N_FFT) // HOP                   # 3001
    idx = np.arange(N_FFT)[None, :] + HOP * np.arange(n_frames)[:, None]
    spec = np.fft.rfft(padded[idx] * window, n=N_FFT, axis=1)
    power = (spec.real ** 2 + spec.imag ** 2)[:-1]                     # drop the last frame
    mel = power @ filters.T                                            # (3000, 80)
    logm = np.log10(np.maximum(mel, 1e-10))
    logm = np.maximum(logm, logm.max() - 8.0)
    return ((logm + 4.0) / 4.0).T.astype(np.float32)


_DTYPES = {"uint8": np.uint8, "uint16": np.uint16, "int32": np.int32, "int8": np.int8, "int16": np.int16}


def quantize(x: np.ndarray, spec: dict) -> np.ndarray:
    """Float → the export's fixed-point input: ``round(x / scale) + zero_point``,
    clipped to the dtype's range. ``spec`` is one entry of metadata.json."""
    dtype = _DTYPES[spec["dtype"]]
    q = spec["quantization_parameters"]
    info = np.iinfo(dtype)
    v = np.rint(np.asarray(x, dtype=np.float32) / float(q["scale"])) + int(q["zero_point"])
    return np.clip(v, info.min, info.max).astype(dtype)


def dequantize(q: np.ndarray, spec: dict) -> np.ndarray:
    """The export's fixed-point output → float32."""
    p = spec["quantization_parameters"]
    return (np.asarray(q).astype(np.float32) - float(p["zero_point"])) * float(p["scale"])


def zero_tensor(spec: dict) -> np.ndarray:
    """A tensor of the spec's shape holding the fixed-point ZERO (its zero
    point, not 0): what an empty KV cache must look like to the decoder."""
    dtype = _DTYPES[spec["dtype"]]
    zp = spec.get("quantization_parameters", {}).get("zero_point", 0)
    return np.full(tuple(spec["shape"]), int(zp), dtype=dtype)
