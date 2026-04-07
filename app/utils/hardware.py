import psutil
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

def get_system_specs() -> Dict[str, Any]:
    """Retrieve current system hardware specifications."""
    try:
        vm = psutil.virtual_memory()
        free_ram_gb = vm.available / (1024 ** 3)
        total_ram_gb = vm.total / (1024 ** 3)
        cpu_cores = psutil.cpu_count(logical=True)
        cpu_usage = psutil.cpu_percent(interval=0.1)
        
        gpu_info = "None"
        vram_available = 0.0
        try:
            import torch
            if torch.cuda.is_available():
                gpu_info = torch.cuda.get_device_name(0)
                # rough approximation if torch doesn't report explicitly free early on
                vram_available = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        except ImportError:
            pass

        return {
            "ram_total_gb": round(total_ram_gb, 2),
            "ram_free_gb": round(free_ram_gb, 2),
            "cpu_cores": cpu_cores,
            "cpu_usage_percent": cpu_usage,
            "gpu_device": gpu_info,
            "vram_total_gb": round(vram_available, 2)
        }
    except Exception as e:
        logger.error(f"Error fetching system specs: {e}")
        return {"error": str(e)}

def validate_hardware_for_model(adapter_name: str, adapter_type: str):
    """
    Issues warnings if the system is underpowered.
    """
    specs = get_system_specs()
    # Simple heuristic checks
    if specs.get("ram_free_gb", 0) < 2.0:
        logger.warning(f"⚠️ [HARDWARE WARNING]: Very low RAM (Free: {specs.get('ram_free_gb')} GB). "
                       f"Loading {adapter_name} may fail or crash the system.")
                       
    if adapter_type == "llm":
        if specs.get("ram_free_gb", 0) < 6.0 and "None" in specs.get("gpu_device", "None"):
            logger.warning(f"⚠️ [HARDWARE WARNING]: Loading heavy LLM '{adapter_name}' on a CPU-only "
                           f"system with < 6GB RAM is highly discouraged. Expect freezing.")
