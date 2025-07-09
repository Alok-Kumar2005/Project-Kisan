import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

from src.ai_component.graph.state import AICompanionState
from src.ai_component.logger import logging
from src.ai_component.exception import CustomException

def select_workflow(state: AICompanionState) -> str:
    """
    Selects the workflow based on the current state.
    This function is used to determine which workflow to execute next.
    """
    try:
        logging.info(f"Selecting workflow based on state")
        workflow = state.get("workflow")
        if workflow == "GeneralNode":
            return "GeneralNode"
        elif workflow == "DiseaseNode":
            return "DiseaseNode"
        else:
            return "DefaultWorkflow" 
        logging.info(f"Selected workflow: {workflow}")
    except CustomException as e:
        logging.error(f"Error in Engineering Node : {str(e)}")
        raise CustomException(e, sys) from e
    