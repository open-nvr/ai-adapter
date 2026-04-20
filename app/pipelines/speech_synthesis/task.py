import time
from typing import Any

from pydantic import BaseModel

from app.interfaces.adapter import BaseAdapter
from app.interfaces.task import BaseTask
from app.schemas.responses import SpeechSynthesisResponse


class SpeechSynthesisTask(BaseTask):
    name = "speech_synthesis"

    def process(self, image: Any, adapter: BaseAdapter) -> SpeechSynthesisResponse:
        # ``image`` carries the TTS payload ({"text": "...", "voice": "..."?}).
        payload = image
        if isinstance(image, dict):
            payload = dict(image)
            payload.setdefault("task", "speech_synthesis")

        raw_result = adapter.predict(payload)
        if isinstance(raw_result, BaseModel):
            raw_payload = raw_result.model_dump()
        elif isinstance(raw_result, dict):
            raw_payload = raw_result
        else:
            raise TypeError(
                f"SpeechSynthesisTask expected adapter payload dict/BaseModel, got {type(raw_result).__name__}"
            )

        audio_uri = raw_payload.get("audio_uri")
        if not isinstance(audio_uri, str) or not audio_uri:
            raise ValueError("Adapter response missing 'audio_uri'")

        return SpeechSynthesisResponse(
            audio_uri=audio_uri,
            duration_seconds=float(raw_payload.get("duration_seconds", 0.0)),
            sample_rate=int(raw_payload.get("sample_rate", 0)),
            voice=str(raw_payload.get("voice", "")).strip() or "unknown",
            text_length=int(raw_payload.get("text_length", 0)),
            executed_at=int(raw_payload.get("executed_at", int(time.time() * 1000))),
            latency_ms=int(raw_payload.get("latency_ms", 0)),
        )
