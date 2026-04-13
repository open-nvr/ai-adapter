from abc import ABC, abstractmethod
from typing import Any

from app.interfaces.adapter import BaseAdapter, ResponsePayload


class BaseTask(ABC):
    """Contract for task-level business logic pipelines."""

    @abstractmethod
    def process(self, image: Any, adapter: BaseAdapter) -> ResponsePayload:
        """Process an image by delegating inference work to an adapter."""
