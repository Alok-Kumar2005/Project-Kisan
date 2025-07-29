import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

from functools import lru_cache
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from src.ai_component.graph.state import AICompanionState
from src.ai_component.tools.web_seach_tool import web_tool
from src.ai_component.tools.rag_tool import rag_tool
from src.ai_component.tools.mandi_report_tool import mandi_report_tool
from src.ai_component.tools.weather_tool import weather_forecast_tool, weather_report_tool
from src.ai_component.tools.gov_scheme_tool import gov_scheme_tool
from src.ai_component.graph.nodes import (
    route_node, UserNode,
    context_injestion_node,MemoryIngestionNode,
    GeneralNode,DiseaseNode,WeatherNode,MandiNode,GovSchemeNode,CarbonFootprintNode,
    ImageNode, VoiceNode , TextNode
)
from src.ai_component.graph.edges import select_workflow, should_continue, select_output_workflow
import asyncio
from typing import Optional
from opik.integrations.langchain import OpikTracer
from dotenv import load_dotenv
load_dotenv()

os.environ["OPIK_API_KEY"] = os.getenv("OPIK_API_KEY")
os.environ["OPIK_WORKSPACE"] = os.getenv("OPIK_WORKSPACE")
os.environ["OPIK_PROJECT_NAME"] = os.getenv("OPIK_PROJECT_NAME")

# Global memory saver instance
memory_saver = MemorySaver()
disease_tools = ToolNode(tools=[web_tool, rag_tool])
weather_tools = ToolNode(tools=[weather_forecast_tool, weather_report_tool])
mandi_tools = ToolNode(tools = [mandi_report_tool])
gov_scheme_tools = ToolNode(tools = [gov_scheme_tool, web_tool])


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
    graph_builder.add_node("UserNode", UserNode)
    graph_builder.add_node("context_injestion_node", context_injestion_node)
    graph_builder.add_node("GeneralNode", GeneralNode)
    graph_builder.add_node("DiseaseNode", DiseaseNode)
    graph_builder.add_node("WeatherNode", WeatherNode)  
    graph_builder.add_node("MandiNode", MandiNode)
    graph_builder.add_node("CarbonFootprintNode", CarbonFootprintNode)
    graph_builder.add_node("GovSchemeNode", GovSchemeNode)
    graph_builder.add_node("MemoryIngestionNode", MemoryIngestionNode)
    graph_builder.add_node("ImageNode", ImageNode)
    graph_builder.add_node("VoiceNode", VoiceNode)
    graph_builder.add_node("TextNode", TextNode)

    ## Adding tools
    graph_builder.add_node("disease_tools", disease_tools)
    graph_builder.add_node("weather_tools", weather_tools) 
    graph_builder.add_node("mandi_tools", mandi_tools) 
    graph_builder.add_node("gov_scheme_tools", gov_scheme_tools) 

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
    
    # Use our custom should_continue function for DiseaseNode
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
    
    # Direct edges from nodes that don't use tools
    graph_builder.add_edge("CarbonFootprintNode", "MemoryIngestionNode")
    graph_builder.add_edge("GeneralNode", "MemoryIngestionNode")

    # Output workflow selection
    graph_builder.add_conditional_edges(
        "MemoryIngestionNode",
        select_output_workflow,
        {
            "ImageNode": "ImageNode",
            "VoiceNode": "VoiceNode",
            "TextNode": "TextNode"
        }
    )

    # End the graph after output nodes
    graph_builder.add_edge("ImageNode", END)
    graph_builder.add_edge("VoiceNode", END)
    graph_builder.add_edge("TextNode", END)

    return graph_builder.compile(checkpointer=memory_saver)


async_graph = create_async_workflow_graph()
tracer = OpikTracer(graph=async_graph.get_graph(xray=True))

# try:
#     img_data = async_graph.get_graph().draw_mermaid_png()
#     with open("workflow.png", "wb") as f:
#         f.write(img_data)
#     print("Graph saved as workflow.png")
# except Exception as e:
#     print(f"Error: {e}")


async def process_query_async(query: str, workflow: str = "GeneralNode",thread_id: str = "default_thread1",config: Optional[dict] = None):
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
        "messages": [{"role": "user", "content": query}],
        "collection_name": "alice123",
        "current_activity": "",
        "workflow": workflow
    }
    
    if config is None:
        config = {
            "configurable": {
                "thread_id": thread_id
            },
            "callbacks": [tracer]
        }
    result = await async_graph.ainvoke(initial_state, config=config)
    return result


if __name__ == "__main__":
    async def test_async_execution():
        print("TEST 1 ===========================================================")
        query = "can you tell me comodity prices?"
        result = await process_query_async(query)
        for msg in reversed(result["messages"]):
            if hasattr(msg, 'content') and msg.content:
                print(msg.content)
                break
        print("TEST 2 ===========================================================")
        query = "Onion in Uttar Pradesh mandi?"
        result = await process_query_async(query)
        for msg in reversed(result["messages"]):
            if hasattr(msg, 'content') and msg.content:
                print(msg.content)
                break

    asyncio.run(test_async_execution())