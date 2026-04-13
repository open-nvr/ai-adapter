from typing import List, Dict, Any

from pydantic import BaseModel


class PipelineEngine:
    """
    Executes an array of tasks sequentially.
    """
    def __init__(self, model_router):
        self.router = model_router
        
    @staticmethod
    def _normalize_result(result: Any) -> Any:
        if isinstance(result, BaseModel):
            return result.model_dump()
        return result

    async def run_pipeline(self, steps: List[str], initial_data: Any) -> Dict[str, Any]:
        """
        Run a sequence of tasks sequentially.
        The output of step N is passed as the input to step N+1.
        """
        current_data = initial_data
        results = {}
        
        for step in steps:
            try:
                step_result = await self.router.route_task(step, current_data)
                normalized_result = self._normalize_result(step_result)
                results[step] = normalized_result
                
                # Chain output to next module's input
                current_data = normalized_result
            except Exception as e:
                return {
                    "status": "error",
                    "failed_at": step,
                    "error": str(e),
                    "partial_results": results
                }
                
        return {"status": "success", "results": results}
