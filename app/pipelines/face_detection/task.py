import time
from typing import Any

from pydantic import BaseModel

from app.interfaces.adapter import BaseAdapter
from app.interfaces.task import BaseTask
from app.schemas.responses import FaceDetection, FaceDetectionResponse


class FaceDetectionTask(BaseTask):
    name = "face_detection"

    @staticmethod
    def _parse_bbox(bbox: Any, index: int) -> list[int]:
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            raise ValueError(
                f"Malformed face detection at index {index}: expected bbox [x1, y1, x2, y2]"
            )
        return [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]

    @staticmethod
    def _parse_landmarks(value: Any, index: int) -> list[list[int]] | None:
        if value is None:
            return None
        if not isinstance(value, list):
            raise ValueError(f"Malformed face detection at index {index}: landmarks must be a list")

        landmarks: list[list[int]] = []
        for landmark_index, pair in enumerate(value):
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                raise ValueError(
                    f"Malformed landmark at face index {index}, landmark index {landmark_index}"
                )
            landmarks.append([int(pair[0]), int(pair[1])])
        return landmarks

    def process(self, image: Any, adapter: BaseAdapter) -> FaceDetectionResponse:
        payload = image
        if isinstance(image, dict):
            payload = dict(image)
            payload.setdefault("task", "face_detection")

        raw_result = adapter.predict(payload)

        if isinstance(raw_result, BaseModel):
            raw_payload = raw_result.model_dump()
        elif isinstance(raw_result, dict):
            raw_payload = raw_result
        else:
            raise TypeError(
                f"FaceDetectionTask expected adapter payload dict/BaseModel, got {type(raw_result).__name__}"
            )

        raw_faces = raw_payload.get("faces", [])
        if not isinstance(raw_faces, list):
            raise ValueError("Adapter response must provide faces as a list")

        faces: list[FaceDetection] = []
        for index, item in enumerate(raw_faces):
            if not isinstance(item, dict):
                raise ValueError(f"Face detection at index {index} must be an object")

            faces.append(
                FaceDetection(
                    bbox=self._parse_bbox(item.get("bbox"), index),
                    confidence=float(item.get("confidence")),
                    landmarks=self._parse_landmarks(item.get("landmarks"), index),
                    age=int(item["age"]) if item.get("age") is not None else None,
                    gender=item.get("gender"),
                )
            )

        return FaceDetectionResponse(
            faces=faces,
            face_count=int(raw_payload.get("face_count", len(faces))),
            annotated_image_uri=raw_payload.get("annotated_image_uri"),
            executed_at=int(raw_payload.get("executed_at", int(time.time() * 1000))),
            latency_ms=int(raw_payload.get("latency_ms", 0)),
        )
