import time
from typing import Any

from pydantic import BaseModel

from app.interfaces.adapter import BaseAdapter
from app.interfaces.task import BaseTask
from app.schemas.responses import BBox, Detection, PersonCountResponse


class PersonCountingTask(BaseTask):
    name = "person_counting"

    @staticmethod
    def _to_detection(item: dict[str, Any], index: int) -> Detection:
        bbox = item.get("bbox")
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            raise ValueError(
                f"Malformed detection at index {index}: expected bbox [left, top, width, height]"
            )

        class_id = item.get("class_id")
        if not isinstance(class_id, int):
            raise ValueError(f"Malformed detection at index {index}: missing integer class_id")

        confidence = item.get("confidence")
        if confidence is None:
            raise ValueError(f"Malformed detection at index {index}: missing confidence")

        bbox_model = BBox(
            left=int(bbox[0]),
            top=int(bbox[1]),
            width=int(bbox[2]),
            height=int(bbox[3]),
        )

        return Detection(
            bbox=bbox_model,
            class_id=class_id,
            confidence=float(confidence),
            class_name=item.get("class_name"),
            track_id=item.get("track_id"),
        )

    def process(self, image: Any, adapter: BaseAdapter) -> PersonCountResponse:
        raw_result = adapter.predict(image)

        if isinstance(raw_result, BaseModel):
            raw_payload = raw_result.model_dump()
        elif isinstance(raw_result, dict):
            raw_payload = raw_result
        else:
            raise TypeError(
                f"PersonCountingTask expected adapter payload dict/BaseModel, got {type(raw_result).__name__}"
            )

        predictions = raw_payload.get("predictions", [])
        if not isinstance(predictions, list):
            raise ValueError("Adapter response must provide predictions as a list")

        person_detections = []
        for index, item in enumerate(predictions):
            if not isinstance(item, dict):
                raise ValueError(f"Prediction at index {index} must be an object")
            if item.get("class_id") != 0:
                continue
            person_detections.append(self._to_detection(item, index))

        raw_prediction_count = raw_payload.get("raw_prediction_count", len(predictions))
        latency_ms = raw_payload.get("latency_ms", 0)

        return PersonCountResponse(
            count=len(person_detections),
            detections=person_detections,
            raw_prediction_count=int(raw_prediction_count),
            executed_at=int(time.time() * 1000),
            latency_ms=int(latency_ms),
        )
