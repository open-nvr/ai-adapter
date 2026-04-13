from typing import Any

from pydantic import BaseModel

from app.interfaces.adapter import BaseAdapter
from app.interfaces.task import BaseTask
from app.schemas.responses import FaceRecognitionResponse


class FaceRecognitionTask(BaseTask):
    name = "face_recognition"

    @staticmethod
    def _parse_face_bbox(value: Any) -> list[int] | None:
        if value is None:
            return None
        if not isinstance(value, (list, tuple)) or len(value) != 4:
            raise ValueError("face_bbox must be [x1, y1, x2, y2] when provided")
        return [int(value[0]), int(value[1]), int(value[2]), int(value[3])]

    def process(self, image: Any, adapter: BaseAdapter) -> FaceRecognitionResponse:
        payload = image
        if isinstance(image, dict):
            payload = dict(image)
            payload.setdefault("task", "face_recognition")

        raw_result = adapter.predict(payload)

        if isinstance(raw_result, BaseModel):
            raw_payload = raw_result.model_dump()
        elif isinstance(raw_result, dict):
            raw_payload = raw_result
        else:
            raise TypeError(
                f"FaceRecognitionTask expected adapter payload dict/BaseModel, got {type(raw_result).__name__}"
            )

        recognized = raw_payload.get("recognized")
        if not isinstance(recognized, bool):
            raise ValueError("Adapter response must provide recognized as a boolean")

        similarity = raw_payload.get("similarity")
        return FaceRecognitionResponse(
            recognized=recognized,
            person_id=raw_payload.get("person_id"),
            name=raw_payload.get("name"),
            category=raw_payload.get("category"),
            similarity=float(similarity) if similarity is not None else None,
            face_bbox=self._parse_face_bbox(raw_payload.get("face_bbox")),
            message=raw_payload.get("message"),
            latency_ms=int(raw_payload.get("latency_ms", 0)),
        )
