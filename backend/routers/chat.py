import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import json
import uuid
import asyncio
import base64
from datetime import datetime
from typing import Dict, Any, AsyncGenerator, List
from fastapi import APIRouter, HTTPException, status, Depends
from fastapi.responses import StreamingResponse, Response
from sse_starlette.sse import EventSourceResponse

from backend.schemas.schemas import (
    ChatMessage, ChatResponse, ChatStreamChunk, 
    MediaResponse, ThreadCreate, ThreadResponse
)
from backend.core.auth import verify_token
from src.ai_component.graph.graph import (
    process_query_async, process_query_stream,
    retrieve_all_threads_for_user, get_thread_messages,
    delete_thread, get_thread_summary, get_async_graph
)

router = APIRouter()


def generate_thread_id(user_id: int) -> str:
    """Generate unique thread ID for user"""
    return f"user_{user_id}_{uuid.uuid4().hex[:8]}"


@router.post("/thread/create", response_model=ThreadResponse)
async def create_thread(
    thread_data: ThreadCreate,
    current_user: Dict[str, Any] = Depends(verify_token)
):
    """Create a new chat thread"""
    thread_id = generate_thread_id(current_user["id"])
    
    # Initialize the thread by sending a system message
    try:
        config = {
            "configurable": {
                "thread_id": thread_id
            }
        }
        
        graph = await get_async_graph()
        initial_state = {
            "messages": [{"role": "system", "content": f"Thread created: {thread_data.thread_name or 'New Conversation'}"}],
            "collection_name": current_user["unique_name"],
            "current_activity": "",
            "workflow": "GeneralNode"
        }
        
        # Initialize the thread in the database
        await graph.ainvoke(initial_state, config=config)
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error creating thread: {str(e)}"
        )
    
    return ThreadResponse(
        thread_id=thread_id,
        thread_name=thread_data.thread_name,
        created_at=datetime.now(),
        message_count=0
    )


async def stream_chat_response(
    query: str, 
    workflow: str, 
    thread_id: str, 
    collection_name: str
) -> AsyncGenerator[str, None]:
    """Stream chat response from LangGraph with SQLite persistence"""
    try:
        config = {
            "configurable": {
                "thread_id": thread_id
            }
        }
        
        async for event in process_query_stream(
            query=query,
            workflow=workflow,
            thread_id=thread_id,
            collection_name=collection_name,
            config=config
        ):
            # Process different types of events from LangGraph
            for node_name, node_output in event.items():
                if isinstance(node_output, dict) and "messages" in node_output:
                    messages = node_output["messages"]
                    if messages:
                        last_message = messages[-1]
                        if hasattr(last_message, 'content') and last_message.content:
                            chunk_data = ChatStreamChunk(
                                content=last_message.content,
                                chunk_type="text",
                                thread_id=thread_id,
                                timestamp=datetime.now()
                            )
                            yield f"data: {chunk_data.model_dump_json()}\n\n"
                
                # Handle tool calls
                if node_name.endswith("_tools"):
                    chunk_data = ChatStreamChunk(
                        content=f"Using {node_name.replace('_tools', '')} tools...",
                        chunk_type="tool_call",
                        thread_id=thread_id,
                        timestamp=datetime.now()
                    )
                    yield f"data: {chunk_data.model_dump_json()}\n\n"
        
        # Send final completion message
        chunk_data = ChatStreamChunk(
            content="",
            chunk_type="final",
            thread_id=thread_id,
            timestamp=datetime.now()
        )
        yield f"data: {chunk_data.model_dump_json()}\n\n"
        
    except Exception as e:
        error_chunk = ChatStreamChunk(
            content=f"Error: {str(e)}",
            chunk_type="text",
            thread_id=thread_id,
            timestamp=datetime.now()
        )
        yield f"data: {error_chunk.model_dump_json()}\n\n"


@router.post("/message/stream")
async def stream_chat_message(
    message: ChatMessage,
    current_user: Dict[str, Any] = Depends(verify_token)
):
    """Stream chat message response with SQLite persistence"""
    # Generate thread ID if not provided
    if not message.thread_id:
        message.thread_id = generate_thread_id(current_user["id"])
    
    # Validate thread ownership by checking if it belongs to the user
    if not message.thread_id.startswith(f"user_{current_user['id']}_"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied to this thread"
        )
    
    # Use user's unique name as collection name
    collection_name = current_user["unique_name"]
    
    return EventSourceResponse(
        stream_chat_response(
            message.query,
            message.workflow,
            message.thread_id,
            collection_name
        ),
        media_type="text/event-stream"
    )


@router.post("/message", response_model=MediaResponse)
async def send_chat_message(
    message: ChatMessage,
    current_user: Dict[str, Any] = Depends(verify_token)
):
    """Send chat message and get complete response (non-streaming) with SQLite persistence"""
    try:
        # Generate thread ID if not provided
        if not message.thread_id:
            message.thread_id = generate_thread_id(current_user["id"])
        
        # Validate thread ownership
        if not message.thread_id.startswith(f"user_{current_user['id']}_"):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied to this thread"
            )
        
        config = {
            "configurable": {
                "thread_id": message.thread_id
            }
        }
        
        result = await process_query_async(
            query=message.query,
            workflow=message.workflow,
            thread_id=message.thread_id,
            collection_name=current_user["unique_name"],
            config=config
        )
        
        # Determine response type based on final state
        media_type = "text"
        content = "No response generated"
        
        # Check for voice output
        if "voice" in result and result["voice"]:
            media_type = "voice"
            content = base64.b64encode(result["voice"]).decode('utf-8')
        
        # Check for image output
        elif "image" in result and result["image"]:
            media_type = "image"
            content = base64.b64encode(result["image"]).decode('utf-8')
        
        # Default to text output
        else:
            media_type = "text"
            # Get the last meaningful message
            messages = result.get("messages", [])
            for msg in reversed(messages):
                if hasattr(msg, 'content') and msg.content and msg.content.strip():
                    content = msg.content
                    break
        
        return MediaResponse(
            content=content,
            media_type=media_type,
            thread_id=message.thread_id,
            timestamp=datetime.now()
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing message: {str(e)}"
        )


@router.get("/threads")
async def get_user_threads(
    current_user: Dict[str, Any] = Depends(verify_token)
):
    """Get all threads for the current user from SQLite database"""
    try:
        user_id = str(current_user["id"])
        thread_ids = await retrieve_all_threads_for_user(user_id)
        
        user_threads = []
        for thread_id in thread_ids:
            try:
                summary = await get_thread_summary(thread_id)
                user_threads.append(
                    ThreadResponse(
                        thread_id=thread_id,
                        thread_name=summary.get("first_message", "New Conversation")[:50] + "...",
                        created_at=datetime.now(),  # You might want to store this in the database
                        message_count=summary.get("message_count", 0)
                    )
                )
            except Exception as e:
                print(f"Error getting thread summary for {thread_id}: {e}")
                continue
        
        # Sort by thread_id (which includes timestamp info) descending
        user_threads.sort(key=lambda x: x.thread_id, reverse=True)
        
        return {"threads": user_threads}
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving threads: {str(e)}"
        )


@router.get("/thread/{thread_id}/messages")
async def get_thread_messages_endpoint(
    thread_id: str,
    current_user: Dict[str, Any] = Depends(verify_token)
):
    """Get all messages from a specific thread"""
    # Validate thread ownership
    if not thread_id.startswith(f"user_{current_user['id']}_"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied to this thread"
        )
    
    try:
        messages = await get_thread_messages(thread_id)
        
        # Convert messages to a format suitable for frontend
        formatted_messages = []
        for msg in messages:
            if hasattr(msg, 'content') and msg.content:
                # Skip system messages
                if hasattr(msg, 'type') and msg.type == 'system':
                    continue
                
                role = 'user' if hasattr(msg, 'type') and msg.type == 'human' else 'assistant'
                formatted_messages.append({
                    'role': role,
                    'content': msg.content,
                    'timestamp': getattr(msg, 'additional_kwargs', {}).get('timestamp', datetime.now().isoformat())
                })
        
        return {
            "thread_id": thread_id,
            "messages": formatted_messages
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving thread messages: {str(e)}"
        )


@router.delete("/thread/{thread_id}")
async def delete_thread_endpoint(
    thread_id: str,
    current_user: Dict[str, Any] = Depends(verify_token)
):
    """Delete a specific thread from SQLite database"""
    # Validate thread ownership
    if not thread_id.startswith(f"user_{current_user['id']}_"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied to this thread"
        )
    
    try:
        success = await delete_thread(thread_id)
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Thread not found or could not be deleted"
            )
        
        return {"message": "Thread deleted successfully"}
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error deleting thread: {str(e)}"
        )