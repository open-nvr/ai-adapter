from typing import List, Dict, Any

class PipelineEngine:
    """
    Executes an array of tasks sequentially.
    """
    def __init__(self, model_router):
        self.router = model_router
        
    def run_pipeline(self, steps: List[str], initial_data: Any) -> Dict[str, Any]:
        """
        Run a sequence of tasks sequentially.
        The output of step N is passed as the input to step N+1.
        """
        current_data = initial_data
        results = {}
        
        for step in steps:
            try:
                step_result = self.router.route_task(step, current_data)
                results[step] = step_result
                
                # Chain output to next module's input
                current_data = step_result
            except Exception as e:
                return {
                    "status": "error",
                    "failed_at": step,
                    "error": str(e),
                    "partial_results": results
                }
                
        return {"status": "success", "results": results}
