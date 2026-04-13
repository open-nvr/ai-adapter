import time
from typing import Any

from pydantic import BaseModel

from app.interfaces.adapter import BaseAdapter
from app.interfaces.task import BaseTask
from app.schemas.responses import SceneDescriptionResponse


class SceneDescriptionTask(BaseTask):
    name = "scene_description"

    def process(self, image: Any, adapter: BaseAdapter) -> SceneDescriptionResponse:
        payload = image
        if isinstance(image, dict):
            payload = dict(image)
            payload.setdefault("task", "scene_description")

        raw_result = adapter.predict(payload)
        if isinstance(raw_result, BaseModel):
            raw_payload = raw_result.model_dump()
        elif isinstance(raw_result, dict):
            raw_payload = raw_result
        else:
            raise TypeError(
                f"SceneDescriptionTask expected adapter payload dict/BaseModel, got {type(raw_result).__name__}"
            )

        return SceneDescriptionResponse(
            caption=str(raw_payload.get("caption", "")).strip(),
            model_id=str(raw_payload.get("model_id", "")).strip(),
            executed_at=int(raw_payload.get("executed_at", int(time.time() * 1000))),
            latency_ms=int(raw_payload.get("latency_ms", 0)),
        )
