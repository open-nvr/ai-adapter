from abc import ABC, abstractmethod
from typing import Any, TypeAlias

from pydantic import BaseModel


ResponsePayload: TypeAlias = BaseModel | dict[str, Any]


class BaseAdapter(ABC):
    """Contract for lazy-loadable model adapters."""

    @abstractmethod
    def load_model(self) -> None:
        """Load model artifacts into memory."""

    @abstractmethod
    def predict(self, input_data: Any) -> ResponsePayload:
        """Run raw inference against the loaded model."""
