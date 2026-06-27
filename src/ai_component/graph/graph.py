import os
import asyncio
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.store.postgres.aio import AsyncPostgresStore
from src.ai_component.graph.state import AICompanionState
from src.ai_component.tools.all_tools import Tools
from src.ai_component.graph.nodes import Nodes
from src.ai_component.graph.edges import select_workflow, should_continue, select_output_workflow

# ---------------------------------------------------------------------------
# Singletons
# ---------------------------------------------------------------------------

_saver: AsyncPostgresSaver | None = None
_store: AsyncPostgresStore | None = None
_graph = None

# Hold the async context manager exits so we can close them on shutdown
_saver_ctx = None
_store_ctx = None


def _get_psycopg_conn_string() -> str:
    """
    Return a psycopg3-compatible connection string for AsyncPostgresSaver /
    AsyncPostgresStore.

    Neon's default URL looks like:
      postgresql://user:pass@host/db?sslmode=require&channel_binding=require

    psycopg3 understands 'sslmode' but NOT 'channel_binding' — remove it.
    Also strip the '+asyncpg' scheme variant if present.
    """
    from urllib.parse import urlparse, urlencode, parse_qs, urlunparse

    raw = os.getenv("NEON_API")
    if not raw:
        raise RuntimeError(
            "NEON_API environment variable is not set. "
            "Set it to a postgresql:// connection string before starting."
        )

    # Normalise scheme to plain postgresql:// for psycopg3
    url = raw.replace("postgresql+asyncpg://", "postgresql://", 1)

    # Remove unsupported params (channel_binding)
    parsed = urlparse(url)
    params = parse_qs(parsed.query, keep_blank_values=True)
    params.pop("channel_binding", None)
    clean_query = urlencode({k: v[0] for k, v in params.items()})
    return urlunparse(parsed._replace(query=clean_query))


# ---------------------------------------------------------------------------
# Public accessors used by main.py lifespan and chat.py router
# ---------------------------------------------------------------------------

async def get_saver() -> AsyncPostgresSaver:
    """Lazily initialise and return the singleton AsyncPostgresSaver (Neon)."""
    global _saver, _saver_ctx
    if _saver is None:
        conn_string = _get_psycopg_conn_string()
        # from_conn_string is an @asynccontextmanager — enter it and keep it
        # open for the lifetime of the process (pool_config creates a pool).
        _saver_ctx = AsyncPostgresSaver.from_conn_string(
            conn_string,
            pipeline=False,
        )
        _saver = await _saver_ctx.__aenter__()
        await _saver.setup()   # creates checkpoint tables if absent
    return _saver


async def get_store() -> AsyncPostgresStore:
    """Lazily initialise and return the singleton AsyncPostgresStore (Neon)."""
    global _store, _store_ctx
    if _store is None:
        conn_string = _get_psycopg_conn_string()
        _store_ctx = AsyncPostgresStore.from_conn_string(conn_string)
        _store = await _store_ctx.__aenter__()
        await _store.setup()   # creates store table if absent
    return _store


async def get_async_graph():
    """Return (and lazily initialise) the singleton compiled graph."""
    global _graph
    if _graph is None:
        _graph = await _build_graph()
    return _graph


async def get_graph():
    """Alias for get_async_graph — used by main.py lifespan."""
    return await get_async_graph()


# ---------------------------------------------------------------------------
# Thread helpers (used by chat.py router)
# ---------------------------------------------------------------------------

async def get_thread_messages(thread_id: str) -> list:
    """Return all messages stored in a thread's latest checkpoint."""
    saver = await get_saver()
    try:
        config = {"configurable": {"thread_id": thread_id}}
        state = await saver.aget(config)
        if state and state.channel_values and "messages" in state.channel_values:
            return state.channel_values["messages"]
    except Exception as e:
        print(f"Error retrieving thread messages for {thread_id}: {e}")
    return []


async def delete_thread(thread_id: str) -> bool:
    """Delete all checkpoints for a thread from Neon via the LangGraph API."""
    saver = await get_saver()
    try:
        await saver.adelete_thread(thread_id)
        return True
    except Exception as e:
        print(f"Error deleting thread {thread_id}: {e}")
        return False


# ---------------------------------------------------------------------------
# Non-streaming query helper (used by POST /message endpoint)
# ---------------------------------------------------------------------------

async def process_query_async(
    query: str,
    workflow: str = "GeneralNode",
    thread_id: str = "default_thread",
    collection_name: str = "default_collection",
    config: dict | None = None,
) -> dict:
    """Invoke the graph and return the final state dict (non-streaming)."""
    graph = await get_async_graph()
    state = {
        "messages": [{"role": "user", "content": query}],
        "collection_name": collection_name,
        "current_activity": "",
        "workflow": workflow,
    }
    if config is None:
        config = {"configurable": {"thread_id": thread_id}}
    return await graph.ainvoke(state, config=config)


# ---------------------------------------------------------------------------
# Shutdown
# ---------------------------------------------------------------------------

async def cleanup_database() -> None:
    """Close the checkpointer and store connections on shutdown."""
    global _saver, _saver_ctx, _store, _store_ctx
    if _saver_ctx is not None:
        try:
            await _saver_ctx.__aexit__(None, None, None)
        except Exception:
            pass
        _saver = None
        _saver_ctx = None
    if _store_ctx is not None:
        try:
            await _store_ctx.__aexit__(None, None, None)
        except Exception:
            pass
        _store = None
        _store_ctx = None


# ---------------------------------------------------------------------------
# Tool nodes
# ---------------------------------------------------------------------------

general_tool   = ToolNode(tools=[Tools.rag_tool, Tools.call_tool])
disease_tools  = ToolNode(tools=[Tools.web_tool])
weather_tools  = ToolNode(tools=[Tools.weather_forecast_tool, Tools.weather_report_tool])
mandi_tools    = ToolNode(tools=[Tools.mandi_report_tool])
gov_scheme_tools = ToolNode(tools=[Tools.gov_scheme_tool, Tools.web_tool])


# ---------------------------------------------------------------------------
# Graph construction (internal)
# ---------------------------------------------------------------------------

async def _build_graph():
    """Build and compile the async workflow graph with Postgres persistence."""
    saver = await get_saver()
    store = await get_store()

    graph_builder = StateGraph(AICompanionState)

    # Nodes
    graph_builder.add_node("route_node",              Nodes.route_node)
    graph_builder.add_node("UserNode",                Nodes.UserNode)
    graph_builder.add_node("context_injestion_node",  Nodes.context_injestion_node)
    graph_builder.add_node("GeneralNode",             Nodes.GeneralNode)
    graph_builder.add_node("DiseaseNode",             Nodes.DiseaseNode)
    graph_builder.add_node("WeatherNode",             Nodes.WeatherNode)
    graph_builder.add_node("MandiNode",               Nodes.MandiNode)
    graph_builder.add_node("CarbonFootprintNode",     Nodes.CarbonFootprintNode)
    graph_builder.add_node("GovSchemeNode",           Nodes.GovSchemeNode)
    graph_builder.add_node("MemoryIngestionNode",     Nodes.MemoryIngestionNode)
    graph_builder.add_node("ImageNode",               Nodes.ImageNode)
    graph_builder.add_node("VoiceNode",               Nodes.VoiceNode)
    graph_builder.add_node("TextNode",                Nodes.TextNode)

    # Tool nodes
    graph_builder.add_node("disease_tools",   disease_tools)
    graph_builder.add_node("weather_tools",   weather_tools)
    graph_builder.add_node("mandi_tools",     mandi_tools)
    graph_builder.add_node("gov_scheme_tools", gov_scheme_tools)
    graph_builder.add_node("general_tool",    general_tool)

    # Edges
    graph_builder.add_edge(START, "route_node")
    graph_builder.add_edge("route_node", "UserNode")
    graph_builder.add_edge("UserNode",   "context_injestion_node")

    graph_builder.add_conditional_edges(
        "context_injestion_node",
        select_workflow,
        {
            "GeneralNode":       "GeneralNode",
            "DiseaseNode":       "DiseaseNode",
            "WeatherNode":       "WeatherNode",
            "MandiNode":         "MandiNode",
            "CarbonFootprintNode": "CarbonFootprintNode",
            "GovSchemeNode":     "GovSchemeNode",
            "DefaultWorkflow":   "GeneralNode",
        },
    )

    graph_builder.add_conditional_edges(
        "GeneralNode",  should_continue, {"tools": "general_tool",   "memory": "MemoryIngestionNode"},
    )
    graph_builder.add_conditional_edges(
        "DiseaseNode",  should_continue, {"tools": "disease_tools",  "memory": "MemoryIngestionNode"},
    )
    graph_builder.add_conditional_edges(
        "WeatherNode",  should_continue, {"tools": "weather_tools",  "memory": "MemoryIngestionNode"},
    )
    graph_builder.add_conditional_edges(
        "MandiNode",    should_continue, {"tools": "mandi_tools",    "memory": "MemoryIngestionNode"},
    )
    graph_builder.add_conditional_edges(
        "GovSchemeNode", should_continue, {"tools": "gov_scheme_tools", "memory": "MemoryIngestionNode"},
    )

    # Return-to-node edges after tool calls
    graph_builder.add_edge("disease_tools",   "DiseaseNode")
    graph_builder.add_edge("weather_tools",   "WeatherNode")
    graph_builder.add_edge("mandi_tools",     "MandiNode")
    graph_builder.add_edge("gov_scheme_tools", "GovSchemeNode")
    graph_builder.add_edge("general_tool",    "GeneralNode")

    graph_builder.add_edge("CarbonFootprintNode", "MemoryIngestionNode")

    graph_builder.add_conditional_edges(
        "MemoryIngestionNode",
        select_output_workflow,
        {"ImageNode": "ImageNode", "VoiceNode": "VoiceNode", "TextNode": "TextNode"},
    )
    graph_builder.add_edge("ImageNode", END)
    graph_builder.add_edge("VoiceNode", END)
    graph_builder.add_edge("TextNode",  END)

    return graph_builder.compile(checkpointer=saver, store=store)
