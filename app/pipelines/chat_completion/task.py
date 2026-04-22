import time
from typing import Any

from pydantic import BaseModel

from app.interfaces.adapter import BaseAdapter
from app.interfaces.task import BaseTask
from app.schemas.responses import ChatCompletionResponse, ChatMessage


class ChatCompletionTask(BaseTask):
    name = "chat_completion"

    def process(self, image: Any, adapter: BaseAdapter) -> ChatCompletionResponse:
        # ``image`` is the BaseTask contract parameter name; for chat tasks it
        # carries the LLM payload dict ({"messages": [...], ...}).
        payload = image
        if isinstance(image, dict):
            payload = dict(image)
            payload.setdefault("task", "chat_completion")

        raw_result = adapter.predict(payload)
        if isinstance(raw_result, BaseModel):
            raw_payload = raw_result.model_dump()
        elif isinstance(raw_result, dict):
            raw_payload = raw_result
        else:
            raise TypeError(
                f"ChatCompletionTask expected adapter payload dict/BaseModel, got {type(raw_result).__name__}"
            )

        message_data = raw_payload.get("message") or {}
        if not isinstance(message_data, dict):
            raise ValueError("Adapter response 'message' must be an object")

        message = ChatMessage(
            role=message_data.get("role", "assistant"),
            content=str(message_data.get("content", "")),
        )

        return ChatCompletionResponse(
            message=message,
            model=str(raw_payload.get("model", "")).strip() or "unknown",
            finish_reason=raw_payload.get("finish_reason", "stop"),
            prompt_tokens=raw_payload.get("prompt_tokens"),
            completion_tokens=raw_payload.get("completion_tokens"),
            total_tokens=raw_payload.get("total_tokens"),
            executed_at=int(raw_payload.get("executed_at", int(time.time() * 1000))),
            latency_ms=int(raw_payload.get("latency_ms", 0)),
        )
