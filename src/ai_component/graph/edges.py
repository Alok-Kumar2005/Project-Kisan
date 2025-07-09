import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

from src.ai_component.graph.state import AICompanionState

def select_workflow(state: AICompanionState) -> str:
    """
    Selects the workflow based on the current state.
    This function is used to determine which workflow to execute next.
    """
    workflow = state.get("workflow")
    if workflow == "GeneralNode":
        return "GeneralNode"
    elif workflow == "DiseaseNode":
        return "DiseaseNode"
    else:
        return "DefaultWorkflow" 