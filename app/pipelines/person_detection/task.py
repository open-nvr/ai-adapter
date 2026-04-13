import time
from typing import Any

from pydantic import BaseModel

from app.interfaces.adapter import BaseAdapter
from app.interfaces.task import BaseTask
from app.schemas.responses import PersonDetectionResponse


class PersonDetectionTask(BaseTask):
    name = "person_detection"

    @staticmethod
    def _to_bbox(item: dict[str, Any], index: int) -> list[int]:
        bbox = item.get("bbox")
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            raise ValueError(
                f"Malformed detection at index {index}: expected bbox [left, top, width, height]"
            )

        return [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]

    def process(self, image: Any, adapter: BaseAdapter) -> PersonDetectionResponse:
        payload = image
        if isinstance(image, dict):
            payload = dict(image)
            payload.setdefault("task", "person_detection")

        raw_result = adapter.predict(payload)

        if isinstance(raw_result, BaseModel):
            raw_payload = raw_result.model_dump()
        elif isinstance(raw_result, dict):
            raw_payload = raw_result
        else:
            raise TypeError(
                f"PersonDetectionTask expected adapter payload dict/BaseModel, got {type(raw_result).__name__}"
            )

        predictions = raw_payload.get("predictions", [])
        if not isinstance(predictions, list):
            raise ValueError("Adapter response must provide predictions as a list")

        best_item: dict[str, Any] | None = None
        best_index = -1
        best_confidence = -1.0
        for index, item in enumerate(predictions):
            if not isinstance(item, dict):
                raise ValueError(f"Prediction at index {index} must be an object")
            if item.get("class_id") != 0:
                continue
            confidence = item.get("confidence")
            if confidence is None:
                raise ValueError(f"Malformed detection at index {index}: missing confidence")
            confidence_value = float(confidence)
            if confidence_value > best_confidence:
                best_confidence = confidence_value
                best_item = item
                best_index = index

        executed_at = int(raw_payload.get("executed_at", int(time.time() * 1000)))
        latency_ms = int(raw_payload.get("latency_ms", 0))

        if best_item is None:
            return PersonDetectionResponse(
                label="person",
                confidence=0.0,
                bbox=[0, 0, 0, 0],
                annotated_image_uri=raw_payload.get("annotated_image_uri"),
                executed_at=executed_at,
                latency_ms=latency_ms,
            )

        return PersonDetectionResponse(
            label="person",
            confidence=round(best_confidence, 2),
            bbox=self._to_bbox(best_item, best_index),
            annotated_image_uri=raw_payload.get("annotated_image_uri"),
            executed_at=executed_at,
            latency_ms=latency_ms,
        )
