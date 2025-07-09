import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

from functools import lru_cache
from langgraph.graph import END, START, StateGraph
from src.ai_component.graph.state import AICompanionState
from src.ai_component.graph.nodes import (
    route_node,
    context_injestion_node,
    GeneralNode
)
from src.ai_component.graph.edges import select_workflow

@lru_cache(maxsize=1)
def create_workflow_graph():
    graph_builder = StateGraph(AICompanionState)
    graph_builder.add_node("route_node", route_node)
    graph_builder.add_node("context_injestion_node", context_injestion_node)
    graph_builder.add_node("GeneralNode", GeneralNode)

    ## adding edges
    graph_builder.add_edge(START, "route_node")
    graph_builder.add_edge("route_node", "context_injestion_node")
    graph_builder.add_conditional_edges(
        "context_injestion_node",select_workflow)
    graph_builder.add_edge("GeneralNode", END)

    return graph_builder


graph = create_workflow_graph().compile()

if __name__ == "__main__":
    query = "Hii how are you, what is your name and what are you doing?" 
    initial_state = {
        "messages": query,
        "current_activity": "",
        "workflow": "GeneralNode"
    }
    result = graph.invoke(initial_state)

    # print(result)
    print(result["messages"][-1])