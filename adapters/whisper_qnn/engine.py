"""Encoder → cached greedy decoder, as Qualcomm's precompiled Whisper-Small
(w8a16, QCS8275) expects to be driven.

The two ONNX files are EPContext wrappers around QNN context binaries:
the encoder takes the quantized log-mel ``input_features`` [1, 80, 3000]
and returns the 12 layers' cross-attention K/V caches; the decoder takes
ONE token, its position, an attention mask over a 200-slot window and the
12 self-attention caches of 199 slots, and returns the next-token logits
with the caches shifted by one (newest at the end — the mask therefore
opens from the END: slot ``199 - n`` becomes valid at step ``n``). All
tensors cross the boundary in the fixed-point dtypes metadata.json names;
the scale/zero-point pairs there are what this module applies.

Decoding is greedy with the English prefix forced
(``<|startoftranscript|><|en|><|transcribe|><|notimestamps|>``): a camera
agent wants words, not timestamps, and forcing the language stops the
Portuguese hallucinations a free-running decoder produced on noise. The
no-speech probability is the softmax mass on ``<|nospeech|>`` at the first
step, the gate the agent uses to drop a clip that is not speech.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .features import N_SAMPLES, dequantize, log_mel, quantize, zero_tensor

SOT = 50258              # <|startoftranscript|>
EOT = 50257              # <|endoftext|>
LANG_EN = 50259          # <|en|>
TRANSCRIBE = 50359       # <|transcribe|>
NO_TIMESTAMPS = 50363    # <|notimestamps|>
NO_SPEECH = 50362        # <|nospeech|> (spelled <|nocaptions|> in this vocabulary)
BLANK = 220              # " " — suppressed at the first generated step, as Whisper does
MEAN_DECODE_LEN = 200    # the decoder's attention window
CACHE_LEN = MEAN_DECODE_LEN - 1
MASK_NEG = -100.0        # what the export was traced with for a masked slot
N_LAYERS = 12
DEFAULT_MAX_TOKENS = 160


@dataclass
class Transcription:
    text: str
    tokens: list[int]
    no_speech_prob: float
    steps: int
    encoder_ms: float
    decoder_ms: float
    chunks: int = 1
    extra: dict[str, Any] = field(default_factory=dict)


class WhisperQnnEngine:
    """Drives the two sessions. ``encoder``/``decoder`` need ``run(None, feeds)``
    and ``get_outputs()``; ``tokenizer`` needs ``decode(ids, skip_special_tokens=True)``
    and ``token_to_id(str)``."""

    def __init__(self, encoder, decoder, metadata: dict, tokenizer, mel_filters,
                 *, max_tokens: int = DEFAULT_MAX_TOKENS) -> None:
        files = metadata["model_files"]
        self.enc_in = files["encoder.onnx"]["inputs"]
        self.dec_in = files["decoder.onnx"]["inputs"]
        self.dec_out = files["decoder.onnx"]["outputs"]
        self.encoder, self.decoder, self.tokenizer = encoder, decoder, tokenizer
        self.mel_filters = np.asarray(mel_filters, dtype=np.float32)
        self.max_tokens = int(max_tokens)
        self._enc_names = [o.name for o in encoder.get_outputs()]
        self._dec_names = [o.name for o in decoder.get_outputs()]
        for name in ("input_ids", "position_ids", "attention_mask"):
            if name not in self.dec_in:
                raise ValueError(f"decoder metadata lacks input {name!r}")
        self._mask_spec = self.dec_in["attention_mask"]
        self._logits_spec = self.dec_out["logits"]

    # ── public ──────────────────────────────────────────────────────

    def language_token(self, language: str | None) -> int:
        if not language or language.lower() in ("en", "english", "unknown"):
            return LANG_EN
        tid = self.tokenizer.token_to_id(f"<|{language.lower()}|>")
        return int(tid) if tid is not None else LANG_EN

    def transcribe(self, wave16k: np.ndarray, *, language: str | None = "en") -> Transcription:
        """Up to the first 30 s in one pass; longer audio in 30 s chunks whose
        texts are joined (an agent utterance never gets there)."""
        wave = np.asarray(wave16k, dtype=np.float32).reshape(-1)
        chunks = [wave[i:i + N_SAMPLES] for i in range(0, max(len(wave), 1), N_SAMPLES)] or [wave]
        parts = [self._transcribe_chunk(c, self.language_token(language)) for c in chunks]
        first = parts[0]
        if len(parts) == 1:
            return first
        return Transcription(
            text=" ".join(p.text for p in parts if p.text).strip(),
            tokens=[t for p in parts for t in p.tokens],
            no_speech_prob=min(p.no_speech_prob for p in parts),
            steps=sum(p.steps for p in parts),
            encoder_ms=sum(p.encoder_ms for p in parts),
            decoder_ms=sum(p.decoder_ms for p in parts),
            chunks=len(parts),
        )

    # ── one 30 s window ────────────────────────────────────────────

    def _transcribe_chunk(self, wave: np.ndarray, lang_token: int) -> Transcription:
        t0 = time.perf_counter()
        feats = log_mel(wave, self.mel_filters)[None]                       # (1, 80, 3000)
        enc_feeds = {"input_features": quantize(feats, self.enc_in["input_features"])}
        cross = dict(zip(self._enc_names, self.encoder.run(None, enc_feeds)))
        t1 = time.perf_counter()

        self_cache = {name: zero_tensor(spec) for name, spec in self.dec_in.items()
                      if "_cache_self_" in name}
        mask = np.full((1, 1, 1, MEAN_DECODE_LEN), MASK_NEG, dtype=np.float32)
        prefix = [SOT, lang_token, TRANSCRIBE, NO_TIMESTAMPS]
        tokens = list(prefix)
        generated: list[int] = []
        no_speech_prob = 0.0
        steps = 0
        for n in range(CACHE_LEN):
            mask[..., MEAN_DECODE_LEN - n - 1] = 0.0                       # this step's slot opens
            feeds = {
                "input_ids": np.array([[tokens[n]]], dtype=np.int32),
                "position_ids": np.array([n], dtype=np.int32),
                "attention_mask": quantize(mask, self._mask_spec),
            }
            feeds.update(self_cache)
            feeds.update(cross)
            outs = dict(zip(self._dec_names, self.decoder.run(None, feeds)))
            steps += 1
            for name in list(self_cache):
                self_cache[name] = outs[name.replace("_in", "_out")]
            logits = dequantize(outs["logits"], self._logits_spec).reshape(-1)
            if n == 0:
                no_speech_prob = float(_softmax(logits)[NO_SPEECH])
            if n < len(prefix) - 1:
                continue                                                   # the prefix is forced
            logits[EOT + 1:] = -np.inf                                     # no language/task/timestamp tokens
            if not generated:
                logits[BLANK] = -np.inf
            nxt = int(np.argmax(logits))
            if nxt == EOT or len(generated) >= self.max_tokens:
                break
            generated.append(nxt)
            tokens.append(nxt)
        t2 = time.perf_counter()
        text = self.tokenizer.decode(generated, skip_special_tokens=True).strip() if generated else ""
        return Transcription(text=text, tokens=generated, no_speech_prob=no_speech_prob, steps=steps,
                             encoder_ms=(t1 - t0) * 1000.0, decoder_ms=(t2 - t1) * 1000.0)


def _softmax(x: np.ndarray) -> np.ndarray:
    z = x - np.max(x)
    e = np.exp(z)
    return e / e.sum()
