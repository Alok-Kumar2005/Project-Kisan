import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

from src.ai_component.graph.utils.chains import async_router_chain
from src.ai_component.llm import LLMChainFactory
from src.ai_component.modules.schedule.context_generation import ScheduleContextGenerator
from src.ai_component.graph.state import AICompanionState
from src.ai_component.core.prompts import general_template
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate



async def route_node(state: AICompanionState) -> str:
    """
    Async version of route_node.
    Node to route the conversation based on the user's query.
    This node uses the async_router_chain to determine the type of response needed.
    """
    query = state["messages"][-1].content if state["messages"] else ""
    
    if not query:
        return "GeneralNode"  # Default to GeneralNode if no query is present
    
    chain = await async_router_chain()
    response = await chain.ainvoke({"query": query})
    
    return {
        "workflow": response.route_node
    }


async def context_injestion_node(state: AICompanionState) -> AIMessage:
    """
    Async version of context_injestion_node.
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
    Async version of GeneralNode.
    General node to handle queries that do not fit into specific categories.
    This node can be extended to provide general information or assistance.
    """
    query = state["messages"][-1].content if state["messages"] else ""
    prompt = PromptTemplate(
        input_variables=["current_activity", "query"],
        template=general_template
    )
    factory = LLMChainFactory(model_type="gemini")
    chain = await factory.get_llm_chain_async(prompt)
    response = await chain.ainvoke({
        "current_activity": state["current_activity"],
        "query": query
    })
    return {
        "messages": response.content,
    }