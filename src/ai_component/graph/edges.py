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
        elif workflow == "WeatherNode":
            return "WeatherNode"
        elif workflow == "MandiNode":
            return "MandiNode"
        elif workflow == "GovSchemeNode":
            return "GovSchemeNode"
        elif workflow == "CarbonFootprintNode":
            return "CarbonFootprintNode"
        else:   
            return "DefaultWorkflow" 
        logging.info(f"Selected workflow: {workflow}")
    except CustomException as e:
        logging.error(f"Error in select workflow Node : {str(e)}")
        raise CustomException(e, sys) from e
    

def select_output_workflow(state: AICompanionState) -> str:
    """
    these will return the format of output in whcih user want
    """
    try:
        logging.info(f"Selecting output workflow")
        output_workflow = state.get("output")
        if output_workflow == "ImageNode":
            return "ImageNode"
        elif output_workflow == "VoiceNode":
            return "VoiceNode"
        else:   
            return "TextNode" 
        logging.info(f"Selected workflow: {output_workflow}")
    except CustomException as e:
        logging.error(f"Error in selecing output workflow : {str(e)}")
        raise CustomException(e, sys) from e


def should_continue(state: AICompanionState) -> str:
    """
    Determine if we should continue to tools or end the conversation.
    """
    messages = state["messages"]
    last_message = messages[-1]
    
    # Check if the last message has tool calls
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "tools"
    else:
        return "__end__"