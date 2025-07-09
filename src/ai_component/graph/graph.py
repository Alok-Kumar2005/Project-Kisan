import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

from functools import lru_cache
from langgraph.graph import END, START, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from src.ai_component.graph.state import AICompanionState
from src.ai_component.graph.nodes import (
    route_node,
    context_injestion_node,
    GeneralNode
)
from src.ai_component.graph.edges import select_workflow
import asyncio
from typing import Optional

# Global memory saver instance
memory_saver = MemorySaver()

@lru_cache(maxsize=1)
def create_async_workflow_graph():
    """
    Creates an async version of the workflow graph with memory saver.
    This allows for better concurrency when processing multiple requests
    while maintaining conversation history.
    """
    graph_builder = StateGraph(AICompanionState)
    graph_builder.add_node("route_node", route_node)
    graph_builder.add_node("context_injestion_node", context_injestion_node)
    graph_builder.add_node("GeneralNode", GeneralNode)

    ## adding edges
    graph_builder.add_edge(START, "route_node")
    graph_builder.add_edge("route_node", "context_injestion_node")
    graph_builder.add_conditional_edges(
        "context_injestion_node", select_workflow)
    graph_builder.add_edge("GeneralNode", END)

    # Compile with memory saver for persistent conversation history
    return graph_builder.compile(checkpointer=memory_saver)


async_graph = create_async_workflow_graph()


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


async def get_conversation_history(thread_id: str = "default_thread"):
    """
    Retrieve the conversation history for a specific thread.
    
    Args:
        thread_id: The thread identifier to get history for
    
    Returns:
        List of conversation states
    """
    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }
    
    # Get the conversation history
    history = []
    async for state in async_graph.aget_state_history(config):
        history.append(state)
    
    return history


async def clear_conversation_memory(thread_id: str = "default_thread"):
    """
    Clear the conversation memory for a specific thread.
    
    Args:
        thread_id: The thread identifier to clear memory for
    """
    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }
    
    # Clear the thread's memory
    await memory_saver.adelete(config)


async def get_current_state(thread_id: str = "default_thread"):
    """
    Get the current state of a conversation thread.
    
    Args:
        thread_id: The thread identifier
    
    Returns:
        Current state of the thread
    """
    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }
    
    return await async_graph.aget_state(config)


if __name__ == "__main__":
    async def test_async_execution():
        # Simple test
        query = "Hi, how are you? What is your name and what are you doing?"
        result = await process_query_async(query)
        print("Simple test result:")
        print(result["messages"][-1].content)
        
        print("\n=== Testing Memory Continuation ===")
        result1 = await process_query_async("My favorite color is blue")
        print(f"First: {result1['messages'][-1].content}")
        
        result2 = await process_query_async("What's my favorite color?")
        print(f"Second: {result2['messages'][-1].content}")
        
    
    asyncio.run(test_async_execution())