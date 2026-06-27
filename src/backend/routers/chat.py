import asyncio
import json
import base64
from datetime import datetime
from typing import Dict, Any, AsyncGenerator, List, Optional
from fastapi import APIRouter, HTTPException, Query, status, Depends
from fastapi.responses import StreamingResponse
from sse_starlette.sse import EventSourceResponse

from src.backend.schemas.schemas import (
    ChatMessage, ChatResponse, ChatStreamChunk,
    MediaResponse, ThreadCreate, ThreadResponse
)
from src.backend.core.auth import verify_token
from src.ai_component.graph.graph import (
    process_query_async,
    get_thread_messages, delete_thread, get_async_graph
)
from src.database.database import chat_db

router = APIRouter()


# ---------------------------------------------------------------------------
# Ownership helper
# ---------------------------------------------------------------------------

async def _get_owned_chat(thread_id: str, user_id: int):
    """
    Fetch a chat by thread_id and verify it belongs to user_id.
    Raises HTTP 403 for both missing and wrong-owner cases (no existence leak).
    """
    chat = await chat_db.get_chat(thread_id)
    if chat is None or chat.user_id != user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied to this thread"
        )
    return chat


# ---------------------------------------------------------------------------
# POST /thread/create
# ---------------------------------------------------------------------------

@router.post("/thread/create", response_model=ThreadResponse)
async def create_thread(
    thread_data: ThreadCreate,
    current_user: Dict[str, Any] = Depends(verify_token)
):
    """Create a new chat thread — inserts a row into the chats table."""
    chat = await chat_db.create_chat(current_user["id"])
    if chat is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error creating thread"
        )

    return ThreadResponse(
        thread_id=chat.thread_id,
        thread_name=chat.name,
        created_at=chat.created_at,
        message_count=chat.message_count,
    )


# ---------------------------------------------------------------------------
# GET /threads
# ---------------------------------------------------------------------------

@router.get("/threads")
async def get_user_threads(
    limit: int = Query(default=50, ge=1, le=200, description="Max threads to return"),
    offset: int = Query(default=0, ge=0, description="Number of threads to skip"),
    current_user: Dict[str, Any] = Depends(verify_token)
):
    """Return paginated chat sessions for the current user, ordered by most-recently active."""
    user_id = current_user["id"]
    chats = await chat_db.list_chats(user_id, limit=limit, offset=offset)
    total = await chat_db.count_chats(user_id)

    return {
        "threads": [
            ThreadResponse(
                thread_id=c.thread_id,
                thread_name=c.name,
                created_at=c.created_at,
                message_count=c.message_count,
            )
            for c in chats
        ],
        "total": total,
        "limit": limit,
        "offset": offset,
        "has_more": (offset + limit) < total,
    }


# ---------------------------------------------------------------------------
# GET /thread/{thread_id}/messages
# ---------------------------------------------------------------------------

@router.get("/thread/{thread_id}/messages")
async def get_thread_messages_endpoint(
    thread_id: str,
    current_user: Dict[str, Any] = Depends(verify_token)
):
    """Return the ordered message history for a thread owned by the current user."""
    await _get_owned_chat(thread_id, current_user["id"])

    try:
        messages = await get_thread_messages(thread_id)

        formatted_messages = []
        for msg in messages:
            if hasattr(msg, "content") and msg.content:
                # Skip system messages
                if hasattr(msg, "type") and msg.type == "system":
                    continue
                role = "user" if hasattr(msg, "type") and msg.type == "human" else "assistant"
                formatted_messages.append({
                    "role": role,
                    "content": msg.content,
                    "timestamp": getattr(msg, "additional_kwargs", {}).get(
                        "timestamp", datetime.now().isoformat()
                    ),
                })

        return {"thread_id": thread_id, "messages": formatted_messages}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving thread messages: {str(e)}"
        )


# ---------------------------------------------------------------------------
# DELETE /thread/{thread_id}
# ---------------------------------------------------------------------------

@router.delete("/thread/{thread_id}")
async def delete_thread_endpoint(
    thread_id: str,
    current_user: Dict[str, Any] = Depends(verify_token)
):
    """Delete chat metadata from the chats table and its checkpoints from LangGraph."""
    await _get_owned_chat(thread_id, current_user["id"])

    # Delete from chats table
    deleted = await chat_db.delete_chat(thread_id, current_user["id"])
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Thread not found or could not be deleted"
        )

    # Delete LangGraph checkpoints (best-effort — don't fail if already gone)
    await delete_thread(thread_id)

    return {"message": "Thread deleted successfully"}


# ---------------------------------------------------------------------------
# Streaming helpers
# ---------------------------------------------------------------------------

STALL_TIMEOUT = 30.0  # seconds — if no event arrives within this window it's a stall


async def _token_stream(
    query: str,
    workflow: str,
    thread_id: str,
    collection_name: str,
) -> AsyncGenerator[str, None]:
    """
    Yield SSE frames: individual token chunks while graph runs, then event: done.
    Uses astream_events(version="v2") to get per-token chunks from the LLM.

    Implements a 30-second stall timeout (Requirement 3.5): if no event is
    delivered from the graph for a continuous 30-second period the generator
    emits an error event and stops.
    """
    try:
        graph = await get_async_graph()
        config = {"configurable": {"thread_id": thread_id}}
        state = {
            "messages": [{"role": "user", "content": query}],
            "collection_name": collection_name,
            "current_activity": "",
            "workflow": workflow,
        }

        # Wrap the async iterator so we can apply a per-event timeout.
        # asyncio.wait_for(anext(...), timeout) raises asyncio.TimeoutError
        # when no event arrives within STALL_TIMEOUT seconds, which satisfies
        # Requirement 3.5 ("terminate the stream and emit an error event").
        event_iter = graph.astream_events(state, config=config, version="v2").__aiter__()

        while True:
            try:
                event = await asyncio.wait_for(
                    event_iter.__anext__(),
                    timeout=STALL_TIMEOUT,
                )
            except StopAsyncIteration:
                # Graph finished without emitting on_chain_end/LangGraph —
                # emit the fallback completion signal.
                break
            except asyncio.TimeoutError:
                yield (
                    f"event: error\n"
                    f"data: {json.dumps({'detail': 'Streaming stall: no response within 30 seconds'})}\n\n"
                )
                return

            kind = event["event"]
            if kind == "on_chat_model_stream":
                chunk = event["data"]["chunk"]
                token = chunk.content if hasattr(chunk, "content") else ""
                if token:
                    yield f"data: {json.dumps({'content': token, 'type': 'token'})}\n\n"
            elif kind == "on_chain_end" and event.get("name") == "LangGraph":
                yield f"event: done\ndata: {{}}\n\n"
                return

        # Fallback completion signal if on_chain_end/LangGraph wasn't emitted
        yield f"event: done\ndata: {{}}\n\n"

    except Exception as e:
        yield (
            f"event: error\n"
            f"data: {json.dumps({'detail': str(e)})}\n\n"
        )


# ---------------------------------------------------------------------------
# POST /message/stream  (SSE streaming)
# ---------------------------------------------------------------------------

@router.post("/message/stream")
async def stream_chat_message(
    message: ChatMessage,
    current_user: Dict[str, Any] = Depends(verify_token)
):
    """Stream token-by-token assistant response over SSE."""
    user_id: int = current_user["id"]

    # Auto-create a thread if none supplied
    if not message.thread_id:
        chat = await chat_db.create_chat(user_id)
        if chat is None:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Could not create a new thread"
            )
        message.thread_id = chat.thread_id
    else:
        await _get_owned_chat(message.thread_id, user_id)

    thread_id = message.thread_id
    collection_name = current_user["unique_name"]

    # Check if this is the first message before streaming so we can name it after
    msg_count = await chat_db.get_message_count(thread_id)

    async def named_stream() -> AsyncGenerator[str, None]:
        async for frame in _token_stream(
            message.query, message.workflow, thread_id, collection_name
        ):
            yield frame
        # After stream completes, set name on first message and increment count
        if msg_count == 0:
            await chat_db.set_chat_name(thread_id, message.query)
        await chat_db.increment_message_count(thread_id)

    return EventSourceResponse(
        named_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ---------------------------------------------------------------------------
# GET /message/stream  (SSE streaming — query params, supports EventSource)
# ---------------------------------------------------------------------------

@router.get("/message/stream")
async def stream_chat_message_get(
    query: str = Query(..., description="The user's message text"),
    thread_id: str = Query(..., description="Thread identifier"),
    workflow: str = Query(default="GeneralNode", description="Workflow routing hint"),
    current_user: Dict[str, Any] = Depends(verify_token),
):
    """
    Stream token-by-token assistant response over SSE (GET variant).

    This endpoint accepts query parameters instead of a request body so that
    browser-native EventSource clients — which only support GET — can connect
    directly.  The POST variant (/message/stream) remains available for clients
    that send a JSON body.
    """
    user_id: int = current_user["id"]
    await _get_owned_chat(thread_id, user_id)

    collection_name = current_user["unique_name"]

    msg_count = await chat_db.get_message_count(thread_id)

    async def named_stream() -> AsyncGenerator[str, None]:
        async for frame in _token_stream(query, workflow, thread_id, collection_name):
            yield frame
        if msg_count == 0:
            await chat_db.set_chat_name(thread_id, query)
        await chat_db.increment_message_count(thread_id)

    return EventSourceResponse(
        named_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ---------------------------------------------------------------------------
# POST /message  (non-streaming)
# ---------------------------------------------------------------------------

@router.post("/message", response_model=MediaResponse)
async def send_chat_message(
    message: ChatMessage,
    current_user: Dict[str, Any] = Depends(verify_token)
):
    """Send a message and receive the complete assistant response (non-streaming)."""
    user_id: int = current_user["id"]

    # Auto-create a thread if none supplied
    if not message.thread_id:
        chat = await chat_db.create_chat(user_id)
        if chat is None:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Could not create a new thread"
            )
        message.thread_id = chat.thread_id
    else:
        await _get_owned_chat(message.thread_id, user_id)

    thread_id = message.thread_id

    try:
        config = {"configurable": {"thread_id": thread_id}}
        result = await process_query_async(
            query=message.query,
            workflow=message.workflow,
            thread_id=thread_id,
            collection_name=current_user["unique_name"],
            config=config,
        )

        # Determine response media type
        media_type = "text"
        content = "No response generated"

        if result.get("voice"):
            media_type = "voice"
            content = base64.b64encode(result["voice"]).decode("utf-8")
        elif result.get("image"):
            media_type = "image"
            content = base64.b64encode(result["image"]).decode("utf-8")
        else:
            media_type = "text"
            for msg in reversed(result.get("messages", [])):
                if hasattr(msg, "content") and msg.content and msg.content.strip():
                    content = msg.content
                    break

        # Auto-name on first message, then increment counter
        msg_count = await chat_db.get_message_count(thread_id)
        if msg_count == 0:
            await chat_db.set_chat_name(thread_id, message.query)
        await chat_db.increment_message_count(thread_id)

        return MediaResponse(
            content=content,
            media_type=media_type,
            thread_id=thread_id,
            timestamp=datetime.now(),
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing message: {str(e)}"
        )
