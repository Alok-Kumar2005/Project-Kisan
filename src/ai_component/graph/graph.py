import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

import asyncio
import aiosqlite
from functools import lru_cache
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from src.ai_component.graph.state import AICompanionState
from src.ai_component.tools.all_tools import Tools
from src.ai_component.graph.nodes import Nodes
from src.ai_component.graph.edges import select_workflow, should_continue, select_output_workflow
from typing import Optional, List

# Global database connection and saver
DB_PATH = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data2', 'chat_history.db')
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

async_saver = None

async def initialize_database():
    """Initialize the async SQLite database and saver"""
    global async_saver
    
    # Create async connection
    conn = await aiosqlite.connect(DB_PATH, check_same_thread=False)
    
    # Initialize AsyncSqliteSaver
    async_saver = AsyncSqliteSaver(conn)
    
    # Setup database tables
    await async_saver.setup()
    
    return async_saver

# Tool nodes
general_tool = ToolNode(tools=[Tools.rag_tool, Tools.call_tool])
disease_tools = ToolNode(tools=[Tools.web_tool])
weather_tools = ToolNode(tools=[Tools.weather_forecast_tool, Tools.weather_report_tool])
mandi_tools = ToolNode(tools=[Tools.mandi_report_tool])
gov_scheme_tools = ToolNode(tools=[Tools.gov_scheme_tool, Tools.web_tool])


async def create_async_workflow_graph():
    """Create the async workflow graph with SQLite persistence"""
    global async_saver
    
    if async_saver is None:
        async_saver = await initialize_database()
    
    graph_builder = StateGraph(AICompanionState)
    
    # Add nodes
    graph_builder.add_node("route_node", Nodes.route_node)
    graph_builder.add_node("UserNode", Nodes.UserNode)
    graph_builder.add_node("context_injestion_node", Nodes.context_injestion_node)
    graph_builder.add_node("GeneralNode", Nodes.GeneralNode)
    graph_builder.add_node("DiseaseNode", Nodes.DiseaseNode)
    graph_builder.add_node("WeatherNode", Nodes.WeatherNode)  
    graph_builder.add_node("MandiNode", Nodes.MandiNode)
    graph_builder.add_node("CarbonFootprintNode", Nodes.CarbonFootprintNode)
    graph_builder.add_node("GovSchemeNode", Nodes.GovSchemeNode)
    graph_builder.add_node("MemoryIngestionNode", Nodes.MemoryIngestionNode)
    graph_builder.add_node("ImageNode", Nodes.ImageNode)
    graph_builder.add_node("VoiceNode", Nodes.VoiceNode)
    graph_builder.add_node("TextNode", Nodes.TextNode)

    # Adding tool nodes
    graph_builder.add_node("disease_tools", disease_tools)
    graph_builder.add_node("weather_tools", weather_tools) 
    graph_builder.add_node("mandi_tools", mandi_tools) 
    graph_builder.add_node("gov_scheme_tools", gov_scheme_tools) 
    graph_builder.add_node("general_tool", general_tool) 

    # Adding edges
    graph_builder.add_edge(START, "route_node")
    graph_builder.add_edge("route_node", "UserNode")
    graph_builder.add_edge("UserNode", "context_injestion_node")
    
    graph_builder.add_conditional_edges(
        "context_injestion_node", 
        select_workflow,
        {
            "GeneralNode": "GeneralNode",
            "DiseaseNode": "DiseaseNode",
            "WeatherNode": "WeatherNode",  
            "MandiNode": "MandiNode",
            "CarbonFootprintNode": "CarbonFootprintNode",
            "GovSchemeNode": "GovSchemeNode",
            "DefaultWorkflow": "GeneralNode"
        }
    )
    
    # Use our custom should_continue function 
    graph_builder.add_conditional_edges(
        "GeneralNode", 
        should_continue,
        {
            "tools": "general_tool",
            "memory": "MemoryIngestionNode"
        }
    )
    graph_builder.add_conditional_edges(
        "DiseaseNode", 
        should_continue,
        {
            "tools": "disease_tools",
            "memory": "MemoryIngestionNode"
        }
    )
    graph_builder.add_conditional_edges(
        "WeatherNode", 
        should_continue,
        {
            "tools": "weather_tools",
            "memory": "MemoryIngestionNode"
        }
    )
    graph_builder.add_conditional_edges(
        "MandiNode", 
        should_continue,
        {
            "tools": "mandi_tools",
            "memory": "MemoryIngestionNode"
        }
    )
    graph_builder.add_conditional_edges(
        "GovSchemeNode", 
        should_continue,
        {
            "tools": "gov_scheme_tools",
            "memory": "MemoryIngestionNode"
        }
    )
    
    # After using tools, return to respective nodes
    graph_builder.add_edge("disease_tools", "DiseaseNode")
    graph_builder.add_edge("weather_tools", "WeatherNode")
    graph_builder.add_edge("mandi_tools", "MandiNode")
    graph_builder.add_edge("gov_scheme_tools", "GovSchemeNode")
    graph_builder.add_edge("general_tool", "GeneralNode")
    
    # Direct edges from nodes that don't use tools
    graph_builder.add_edge("CarbonFootprintNode", "MemoryIngestionNode")

    graph_builder.add_conditional_edges(
        "MemoryIngestionNode",
        select_output_workflow,
        {
            "ImageNode": "ImageNode",
            "VoiceNode": "VoiceNode",
            "TextNode": "TextNode"
        }
    )
    graph_builder.add_edge("ImageNode", END)
    graph_builder.add_edge("VoiceNode", END)
    graph_builder.add_edge("TextNode", END)

    return graph_builder.compile(checkpointer=async_saver)


# Initialize the graph
async_graph = None

async def get_async_graph():
    """Get or create the async graph instance"""
    global async_graph
    if async_graph is None:
        async_graph = await create_async_workflow_graph()
    return async_graph


async def retrieve_all_threads_for_user(user_id: str) -> List[str]:
    """Retrieve all thread IDs for a specific user"""
    global async_saver
    
    if async_saver is None:
        async_saver = await initialize_database()
    
    all_threads = set()
    
    async for checkpoint in async_saver.alist(None):
        thread_id = checkpoint.config.get('configurable', {}).get('thread_id')
        if thread_id and thread_id.startswith(f"user_{user_id}_"):
            all_threads.add(thread_id)
    
    return list(all_threads)


async def get_thread_messages(thread_id: str):
    """Get all messages for a specific thread"""
    global async_saver
    
    if async_saver is None:
        async_saver = await initialize_database()
    
    try:
        config = {"configurable": {"thread_id": thread_id}}
        state = await async_saver.aget(config)
        
        if state and 'messages' in state.values:
            return state.values['messages']
        return []
    except Exception as e:
        print(f"Error retrieving thread messages: {e}")
        return []


async def delete_thread(thread_id: str) -> bool:
    """Delete a specific thread from the database"""
    global async_saver
    
    if async_saver is None:
        async_saver = await initialize_database()
    
    try:
        # Get the database connection from the saver
        conn = async_saver.conn
        
        # Delete all checkpoints for this thread
        await conn.execute(
            "DELETE FROM checkpoints WHERE thread_id = ?", 
            (thread_id,)
        )
        await conn.commit()
        return True
    except Exception as e:
        print(f"Error deleting thread: {e}")
        return False


async def get_thread_summary(thread_id: str) -> dict:
    """Get summary information about a thread (first message, last activity, etc.)"""
    messages = await get_thread_messages(thread_id)
    
    if not messages:
        return {
            "thread_id": thread_id,
            "first_message": "",
            "message_count": 0,
            "last_activity": None
        }
    
    # Get first human message for preview
    first_message = ""
    for msg in messages:
        if hasattr(msg, 'content') and msg.content:
            first_message = msg.content[:100] + "..." if len(msg.content) > 100 else msg.content
            break
    
    return {
        "thread_id": thread_id,
        "first_message": first_message,
        "message_count": len(messages),
        "last_activity": messages[-1].additional_kwargs.get('timestamp') if messages else None
    }


async def process_query_async(
    query: str, 
    workflow: str = "GeneralNode",
    thread_id: str = "default_thread",
    collection_name: str = "default_collection",  
    config: Optional[dict] = None
):
    """Process query asynchronously with SQLite persistence"""
    graph = await get_async_graph()
    
    initial_state = {
        "messages": [{"role": "user", "content": query}],
        "collection_name": collection_name, 
        "current_activity": "",
        "workflow": workflow
    }
    
    if config is None:
        config = {
            "configurable": {
                "thread_id": thread_id
            }
        }
    
    result = await graph.ainvoke(initial_state, config=config)
    return result


async def process_query_stream(
    query: str, 
    workflow: str = "GeneralNode",
    thread_id: str = "default_thread",
    collection_name: str = "default_collection",
    config: Optional[dict] = None
):
    """Process query with streaming support and SQLite persistence"""
    graph = await get_async_graph()
    
    initial_state = {
        "messages": [{"role": "user", "content": query}],
        "collection_name": collection_name,
        "current_activity": "",
        "workflow": workflow
    }
    
    if config is None:
        config = {
            "configurable": {
                "thread_id": thread_id
            }
        }
    
    # Stream the graph execution
    async for event in graph.astream(initial_state, config=config):
        yield event


# Cleanup function
async def cleanup_database():
    """Cleanup database connections"""
    global async_saver
    if async_saver and hasattr(async_saver, 'conn'):
        await async_saver.conn.close()


if __name__ == "__main__":
    async def test_async_execution():
        print("TEST 1 ===========================================================")
        query = "I am facing problem related to pesticides in my area and my area is ( Kankar bhag, Patna , bihar) found any one if they are facing same problem and tell him that i wanted to meet him"
        result = await process_query_async(query, thread_id="test_thread_1")
        for msg in reversed(result["messages"]):
            if hasattr(msg, 'content') and msg.content:
                print(msg.content)
                break
        
        print("TEST 2 ===========================================================")
        query = "yes please, and share my message with him"
        result = await process_query_async(query, thread_id="test_thread_1")
        for msg in reversed(result["messages"]):
            if hasattr(msg, 'content') and msg.content:
                print(msg.content)
                break

    asyncio.run(test_async_execution())