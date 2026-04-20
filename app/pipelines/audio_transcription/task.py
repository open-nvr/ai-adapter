import time
from typing import Any, ClassVar, Literal

from pydantic import BaseModel

from app.interfaces.adapter import BaseAdapter
from app.interfaces.task import BaseTask
from app.schemas.responses import AudioTranscriptionResponse, TranscriptSegment


class _AudioTaskBase(BaseTask):
    """
    Shared logic for Whisper-backed audio tasks.

    Subclasses set ``name`` and ``_default_task`` to one of
    ``audio_transcription`` / ``audio_translation``.
    """

    _default_task: ClassVar[Literal["audio_transcription", "audio_translation"]] = "audio_transcription"

    def process(self, image: Any, adapter: BaseAdapter) -> AudioTranscriptionResponse:
        # ``image`` is the BaseTask contract parameter name; for audio tasks it
        # carries the audio payload dict ({"audio": {"uri": ...}, ...}).
        payload = image
        if isinstance(image, dict):
            payload = dict(image)
            payload.setdefault("task", self._default_task)

        raw_result = adapter.predict(payload)
        if isinstance(raw_result, BaseModel):
            raw_payload = raw_result.model_dump()
        elif isinstance(raw_result, dict):
            raw_payload = raw_result
        else:
            raise TypeError(
                f"{self.__class__.__name__} expected adapter payload dict/BaseModel, "
                f"got {type(raw_result).__name__}"
            )

        raw_segments = raw_payload.get("segments") or []
        if not isinstance(raw_segments, list):
            raise ValueError("Adapter response must provide segments as a list")

        segments = [
            TranscriptSegment(**seg) if isinstance(seg, dict) else seg
            for seg in raw_segments
        ]

        return AudioTranscriptionResponse(
            task=raw_payload.get("task", self._default_task),
            text=str(raw_payload.get("text", "")).strip(),
            segments=segments,
            language=str(raw_payload.get("language", "")).strip() or "unknown",
            language_confidence=raw_payload.get("language_confidence"),
            duration_seconds=float(raw_payload.get("duration_seconds", 0.0)),
            model=str(raw_payload.get("model", "")).strip() or "unknown",
            translated_to_english=bool(raw_payload.get("translated_to_english", False)),
            executed_at=int(raw_payload.get("executed_at", int(time.time() * 1000))),
            latency_ms=int(raw_payload.get("latency_ms", 0)),
        )


class AudioTranscriptionTask(_AudioTaskBase):
    name = "audio_transcription"
    _default_task = "audio_transcription"


class AudioTranslationTask(_AudioTaskBase):
    name = "audio_translation"
    _default_task = "audio_translation"
