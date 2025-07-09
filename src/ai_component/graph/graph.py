import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

from functools import lru_cache
from langgraph.graph import END, START, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from src.ai_component.graph.state import AICompanionState
from src.ai_component.tools.web_seach_tool import web_tool
from src.ai_component.tools.rag_tool import rag_tool
from langgraph.prebuilt import ToolNode, tools_condition
from src.ai_component.graph.nodes import (
    route_node,
    context_injestion_node,
    GeneralNode,
    DiseaseNode
)
from src.ai_component.graph.edges import select_workflow
import asyncio
from typing import Optional

# Global memory saver instance
memory_saver = MemorySaver()
disease_tools = ToolNode(tools=[web_tool, rag_tool])

@lru_cache(maxsize=1)
def create_async_workflow_graph():
    """
    Creates an async version of the workflow graph with memory saver.
    This allows for better concurrency when processing multiple requests
    while maintaining conversation history.
    """
    graph_builder = StateGraph(AICompanionState)
    
    # Add nodes
    graph_builder.add_node("route_node", route_node)
    graph_builder.add_node("context_injestion_node", context_injestion_node)
    graph_builder.add_node("GeneralNode", GeneralNode)
    graph_builder.add_node("DiseaseNode", DiseaseNode)
    graph_builder.add_node("disease_tools", disease_tools)

    # Adding edges
    graph_builder.add_edge(START, "route_node")
    graph_builder.add_edge("route_node", "context_injestion_node")
    
    # Conditional edge from context_injestion_node to select workflow
    graph_builder.add_conditional_edges(
        "context_injestion_node", 
        select_workflow,
        {
            "GeneralNode": "GeneralNode",
            "DiseaseNode": "DiseaseNode", 
            "DefaultWorkflow": "GeneralNode"  # Default case
        }
    )
    
    graph_builder.add_conditional_edges(
        "DiseaseNode", 
        tools_condition,
        {
            "tools": "disease_tools",
            "__end__": END
        }
    )
    
    # After using tools, return to DiseaseNode
    graph_builder.add_edge("disease_tools", "DiseaseNode")
    
    # GeneralNode goes to END
    graph_builder.add_edge("GeneralNode", END)

    # Compile with memory saver for persistent conversation history
    return graph_builder.compile(checkpointer=memory_saver)


async_graph = create_async_workflow_graph()
try:
    img_data = async_graph.get_graph().draw_mermaid_png()
    with open("workflow.png", "wb") as f:
        f.write(img_data)
    print("Graph saved as workflow.png")
except Exception as e:
    print(f"Error: {e}")



async def process_query_async(
    query: str, 
    workflow: str = "GeneralNode",
    thread_id: str = "default_thread",
    config: Optional[dict] = None
):
    """
    Async function to process a query using the async workflow graph with memory.
    
    Args:
        query: The user's query
        workflow: The workflow to use (default: "GeneralNode")
        thread_id: Unique identifier for the conversation thread
        config: Optional configuration dict for the graph execution
    
    Returns:
        The result from the async graph execution
    """
    initial_state = {
        "messages": query,
        "current_activity": "",
        "workflow": workflow
    }
    
    # Configuration for memory management
    if config is None:
        config = {
            "configurable": {
                "thread_id": thread_id
            }
        }
    
    result = await async_graph.ainvoke(initial_state, config=config)
    return result


if __name__ == "__main__":
    async def test_async_execution():
        # Simple test
        query = "In my cauliflower there are some fungus are found can you suggest some medicine for that?"
        result = await process_query_async(query)
        print("Simple test result:")
        print(result["messages"][-2].content)
        print(result["messages"][-1].content)
        
        
    
    asyncio.run(test_async_execution())