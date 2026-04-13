import logging
from abc import ABC, abstractmethod
from typing import Dict, Any

from app.interfaces.adapter import BaseAdapter as AdapterInterface, ResponsePayload

try:
    from app.utils.hardware import validate_hardware_for_model
except ImportError:
    validate_hardware_for_model = lambda n, t: None

logger = logging.getLogger(__name__)

class BaseAdapter(AdapterInterface, ABC):
    """
    Core Abstraction: All adapters (Vision, LLM, etc.) must inherit from this class.
    Implements mandatory lazy model loading.
    """
    
    # Unique identifier for the adapter
    name: str = ""
    # Type of adapter: e.g. "vision", "llm"
    type: str = ""
    
    def __init__(self, config: Dict[str, Any] = None):
        if not self.name or not self.type:
            raise ValueError(f"{self.__class__.__name__} must define 'name' and 'type'")
            
        self.config = config or {}
        self.model = None
        logger.info(f"Initializing adapter metadata: {self.name} | Enabled: {self.config.get('enabled', False)}")
        
    def ensure_model_loaded(self):
        """Lazy load the model if not already in memory."""
        if self.model is None:
            logger.info(f"Loading model for adapter: {self.name}...")
            validate_hardware_for_model(self.name, self.type)
            try:
                self.load_model()
                logger.info(f"Model successfully loaded and cached in memory for {self.name}.")
            except Exception as e:
                logger.error(f"Failed to load model for {self.name}: {str(e)}")
                raise RuntimeError(f"Error loading {self.name} model: {str(e)}")
        else:
            pass # Keep logs clean during high speed inference, can add logger.debug if needed
            
    @abstractmethod
    def load_model(self):
        """
        Implement heavy model loading logic here. Sets self.model.
        Do NOT call this in constructor.
        """
        pass
        
    @abstractmethod
    def infer_local(self, input_data: Any) -> ResponsePayload:
        """
        Actual inference logic on the loaded model.
        Implemented by child classes.
        """
        pass
        
    def infer(self, input_data: Any) -> ResponsePayload:
        """
        Execute inference after ensuring model availability via lazy loading.
        """
        logger.info(f"Inference requested on adapter: {self.name}")
        self.ensure_model_loaded()
        return self.infer_local(input_data)

    def predict(self, input_data: Any) -> ResponsePayload:
        """Interface-compatible raw inference entry point."""
        return self.infer(input_data)
    
    @property
    def schema(self) -> Dict[str, Any]:
        """
        Return OpenNVR UI schema for this adapter's tasks.
        Override in subclasses to provide task-specific schemas.
        """
        return {}
        
    def get_model_info(self) -> Dict[str, Any]:
        """
        Return metadata about the underlying model.
        Override in subclasses to provide framework, device, and task details.
        Used by monitoring dashboards and the /health endpoint.
        """
        return {
            "name": self.name,
            "type": self.type,
            "model_loaded": self.model is not None,
        }

    def health_check(self) -> Dict[str, Any]:
        return {
            "status": "healthy",
            "type": self.type,
            "model_loaded": self.model is not None,
            "model_info": self.get_model_info(),
        }
