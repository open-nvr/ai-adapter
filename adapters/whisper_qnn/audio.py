"""Clip bytes → 16 kHz mono float32. A 16 kHz PCM WAV is read with the
standard library; anything else (another rate, a browser's webm, an mp3)
goes through ffmpeg, which the image carries. A copy of the parakeet
adapter's loader with the resample added: this model takes exactly 16 kHz."""
from __future__ import annotations

import io
import os
import shutil
import subprocess
import tempfile
import wave

import numpy as np

SAMPLE_RATE = 16000


class AudioError(ValueError):
    pass


def load_audio_16k(audio: bytes) -> tuple[np.ndarray, float]:
    """``(waveform float32 mono at 16 kHz, duration in seconds)``."""
    if audio[:4] == b"RIFF" and audio[8:12] == b"WAVE":
        try:
            with wave.open(io.BytesIO(audio), "rb") as w:
                ch, width, rate, n = w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()
                pcm = w.readframes(n) if w.getcomptype() == "NONE" else None
        except wave.Error:
            pcm = None
        if pcm is not None and width in (1, 2, 4) and rate == SAMPLE_RATE:
            if width == 1:
                x = (np.frombuffer(pcm, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
            elif width == 2:
                x = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
            else:
                x = np.frombuffer(pcm, dtype="<i4").astype(np.float32) / 2147483648.0
            if ch > 1:
                x = x.reshape(-1, ch).mean(axis=1)
            return np.ascontiguousarray(x), len(x) / float(SAMPLE_RATE)
    if shutil.which("ffmpeg") is None:
        raise AudioError("Audio must be a 16 kHz PCM WAV (ffmpeg is not installed to convert anything else).")
    tmp_dir = tempfile.mkdtemp(prefix="whisper-qnn-")
    try:
        src = os.path.join(tmp_dir, "in.bin")
        out = os.path.join(tmp_dir, "in16k.wav")
        with open(src, "wb") as fh:
            fh.write(audio)
        proc = subprocess.run(
            ["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", src,
             "-ac", "1", "-ar", str(SAMPLE_RATE), "-sample_fmt", "s16", "-f", "wav", out],
            capture_output=True, text=True,
        )
        if proc.returncode != 0 or not os.path.exists(out):
            raise AudioError(f"Audio could not be decoded: {proc.stderr.strip()[:200]}")
        with open(out, "rb") as fh:
            converted = fh.read()
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    return load_audio_16k(converted)
