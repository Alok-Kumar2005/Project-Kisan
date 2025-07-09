import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

from src.ai_component.graph.utils.chains import router_chain
from src.ai_component.llm import LLMChainFactory
from src.ai_component.modules.schedule.context_generation import ScheduleContextGenerator
from src.ai_component.graph.state import AICompanionState
from src.ai_component.core.prompts import general_template
from langgraph.graph import Graph, Node
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate



async def route_node(state: AICompanionState) -> str:
    """
    Node to route the conversation based on the user's query.
    This node uses the router_chain to determine the type of response needed.
    """
    query = state["messages"][-1].content if state["messages"] else ""
    
    if not query:
        return "GeneralNode"  # Default to GeneralNode if no query is present
    
    chain = router_chain()
    response = await chain.invoke({"query": query})
    
    return {
        "workflow": response.route_node
    }




async def context_injestion_node(state: AICompanionState) -> AIMessage:
    """
    Node to inject context about Ramesh Kumar's current activity into the conversation.
    This node uses the ScheduleContextGenerator to get the current activity and returns it as an AI message.
    """
    from src.ai_component.modules.schedule.context_generation import ScheduleContextGenerator

    current_activity = ScheduleContextGenerator.get_current_activity()
    
    if current_activity:
        return {
            "current_activity": current_activity,
        }
    else:
        return {
            "current_activity": "Ramesh Kumar is currently not scheduled for any activity."
        }
    

async def GeneralNode(state: AICompanionState) -> AIMessage:
    """
    General node to handle queries that do not fit into specific categories.
    This node can be extended to provide general information or assistance.
    """
    query = state["messages"][-1].content if state["messages"] else ""
    prompt = PromptTemplate(
        input_variables=["current_activity", "query"],
        template=general_template
    )
    factory = LLMChainFactory(model_type="groq")
    chain = factory.get_llm_chain(prompt)
    response = await chain.invoke({"current_activity": state.current_activity, "query": query})
    return {
        "messages": response.content,
    }