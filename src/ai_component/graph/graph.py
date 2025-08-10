import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

from functools import lru_cache
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from src.ai_component.graph.state import AICompanionState
from src.ai_component.tools.all_tools import Tools
from src.ai_component.graph.nodes import Nodes
from src.ai_component.graph.edges import select_workflow, should_continue, select_output_workflow
import asyncio
from typing import Optional

# Global memory saver instance
memory_saver = MemorySaver()
general_tool = ToolNode(tools= [Tools.rag_tool, Tools.call_tool])
disease_tools = ToolNode(tools=[Tools.web_tool, Tools.rag_tool])
weather_tools = ToolNode(tools=[Tools.weather_forecast_tool, Tools.weather_report_tool])
mandi_tools = ToolNode(tools = [Tools.mandi_report_tool])
gov_scheme_tools = ToolNode(tools = [Tools.gov_scheme_tool, Tools.web_tool])


@lru_cache(maxsize=1)
def create_async_workflow_graph():
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

    ## Adding tools
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
    # graph_builder.add_edge("GeneralNode", "MemoryIngestionNode")

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

    return graph_builder.compile(checkpointer=memory_saver)


async_graph = create_async_workflow_graph()

try:
    img_data = async_graph.get_graph().draw_mermaid_png()
    with open("workflow.png", "wb") as f:
        f.write(img_data)
    print("Graph saved as workflow.png")
except Exception as e:
    print(f"Error: {e}")


async def process_query_async(query: str, workflow: str = "GeneralNode",thread_id: str = "default_thread1",config: Optional[dict] = None):
    initial_state = {
        "messages": [{"role": "user", "content": query}],
        "collection_name": "ashok123",
        "current_activity": "",
        "workflow": workflow
    }
    
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
        print("TEST 1 ===========================================================")
        query = "I am suffering of fertilizer related problem in my area. can you find tell me is there anyone other who are suffering from same problem (my location is Varanasi, Uttar Pradesh, India) and tell me message to him that i wanted to met them and find the solution "
        result = await process_query_async(query)
        for msg in reversed(result["messages"]):
            if hasattr(msg, 'content') and msg.content:
                print(msg.content)
                break
        print("TEST 2 ===========================================================")
        query = "yes please, and share my message with him"
        result = await process_query_async(query)
        for msg in reversed(result["messages"]):
            if hasattr(msg, 'content') and msg.content:
                print(msg.content)
                break

    asyncio.run(test_async_execution())