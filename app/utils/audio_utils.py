# Copyright (c) 2026 OpenNVR
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Audio loading utilities for the AI adapter.

Mirrors the ``opennvr://frames/...`` scheme with ``opennvr://audio/...`` so
audio adapters (Whisper STT, audio-event detectors, etc.) receive input via
the same URI contract as vision adapters.
"""
import pathlib
import uuid

from fastapi import HTTPException

from app.config.config import BASE_AUDIO_DIR

_AUDIO_URI_PREFIX = "opennvr://audio/"


def resolve_audio_uri(uri: str) -> pathlib.Path:
    """
    Resolve an ``opennvr://audio/<relative_path>`` URI to an absolute filesystem
    path under ``BASE_AUDIO_DIR``.

    Adapters receive a ``pathlib.Path`` (not raw PCM) so they can hand the path
    to their own decoder — ``faster-whisper`` uses PyAV, ``torchaudio`` uses
    soundfile, etc. Pre-decoding here would force a single format choice on
    every adapter.

    Raises:
        HTTPException 400: malformed URI or path-traversal attempt.
        HTTPException 404: file does not exist.
    """
    if not isinstance(uri, str) or not uri.startswith(_AUDIO_URI_PREFIX):
        raise HTTPException(status_code=400, detail="Invalid audio URI")

    relative_path = uri[len(_AUDIO_URI_PREFIX):]
    if not relative_path:
        raise HTTPException(status_code=400, detail="Invalid audio URI: empty path")

    base = pathlib.Path(BASE_AUDIO_DIR).resolve()
    audio_path = (base / relative_path).resolve()

    try:
        audio_path.relative_to(base)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid audio URI: path traversal detected") from exc

    if not audio_path.exists():
        raise HTTPException(status_code=404, detail="Audio file not found")
    if not audio_path.is_file():
        raise HTTPException(status_code=400, detail="Audio URI must reference a file")

    return audio_path


def mint_audio_path(subdir: str, extension: str = "wav") -> tuple[pathlib.Path, str]:
    """
    Allocate a fresh path for a newly-generated audio file (TTS output, clipped
    segment, etc.) and return its absolute filesystem path plus the matching
    ``opennvr://audio/...`` URI.

    The caller is responsible for writing the bytes — this helper only creates
    the parent directory and picks a collision-safe filename.
    """
    extension = extension.lstrip(".")
    base = pathlib.Path(BASE_AUDIO_DIR).resolve()
    target_dir = (base / subdir).resolve()

    try:
        target_dir.relative_to(base)
    except ValueError as exc:
        raise ValueError(f"Invalid audio subdir '{subdir}': escapes BASE_AUDIO_DIR") from exc

    target_dir.mkdir(parents=True, exist_ok=True)

    filename = f"{uuid.uuid4().hex}.{extension}"
    audio_path = target_dir / filename
    relative = audio_path.relative_to(base).as_posix()
    uri = f"{_AUDIO_URI_PREFIX}{relative}"
    return audio_path, uri
