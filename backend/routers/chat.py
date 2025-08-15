import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import json
import uuid
import asyncio
import base64
from datetime import datetime
from typing import Dict, Any, AsyncGenerator
from fastapi import APIRouter, HTTPException, status, Depends
from fastapi.responses import StreamingResponse, Response
from sse_starlette.sse import EventSourceResponse

from backend.schemas.schemas import (
    ChatMessage, ChatResponse, ChatStreamChunk, 
    MediaResponse, ThreadCreate, ThreadResponse
)
from backend.core.auth import verify_token
from src.ai_component.graph.graph import async_graph

router = APIRouter()

# In-memory storage for active threads (in production, use Redis or database)
active_threads: Dict[str, Dict[str, Any]] = {}


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
    
    active_threads[thread_id] = {
        "user_id": current_user["id"],
        "thread_name": thread_data.thread_name,
        "created_at": datetime.now(),
        "message_count": 0,
        "last_activity": datetime.now()
    }
    
    return ThreadResponse(
        thread_id=thread_id,
        thread_name=thread_data.thread_name,
        created_at=active_threads[thread_id]["created_at"],
        message_count=0
    )


async def stream_chat_response(
    query: str, 
    workflow: str, 
    thread_id: str, 
    collection_name: str
) -> AsyncGenerator[str, None]:
    """Stream chat response from LangGraph"""
    try:
        initial_state = {
            "messages": [{"role": "user", "content": query}],
            "collection_name": collection_name,
            "current_activity": "",
            "workflow": workflow
        }
        
        config = {
            "configurable": {
                "thread_id": thread_id
            }
        }
        
        async for event in async_graph.astream(initial_state, config=config):
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
    """Stream chat message response"""
    # Generate thread ID if not provided
    if not message.thread_id:
        message.thread_id = generate_thread_id(current_user["id"])
    
    # Validate thread ownership
    if message.thread_id in active_threads:
        if active_threads[message.thread_id]["user_id"] != current_user["id"]:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied to this thread"
            )
    else:
        # Create new thread entry
        active_threads[message.thread_id] = {
            "user_id": current_user["id"],
            "thread_name": None,
            "created_at": datetime.now(),
            "message_count": 0,
            "last_activity": datetime.now()
        }
    
    # Update thread activity
    active_threads[message.thread_id]["last_activity"] = datetime.now()
    active_threads[message.thread_id]["message_count"] += 1
    
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
    """Send chat message and get complete response (non-streaming)"""
    try:
        # Generate thread ID if not provided
        if not message.thread_id:
            message.thread_id = generate_thread_id(current_user["id"])
        
        # Validate thread ownership
        if message.thread_id in active_threads:
            if active_threads[message.thread_id]["user_id"] != current_user["id"]:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Access denied to this thread"
                )
        else:
            # Create new thread entry
            active_threads[message.thread_id] = {
                "user_id": current_user["id"],
                "thread_name": None,
                "created_at": datetime.now(),
                "message_count": 0,
                "last_activity": datetime.now()
            }
        
        # Update thread activity
        active_threads[message.thread_id]["last_activity"] = datetime.now()
        active_threads[message.thread_id]["message_count"] += 1
        
        initial_state = {
            "messages": [{"role": "user", "content": message.query}],
            "collection_name": current_user["unique_name"],
            "current_activity": "",
            "workflow": message.workflow
        }
        
        config = {
            "configurable": {
                "thread_id": message.thread_id
            }
        }
        
        result = await async_graph.ainvoke(initial_state, config=config)
        
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
    """Get all threads for the current user"""
    user_threads = []
    for thread_id, thread_data in active_threads.items():
        if thread_data["user_id"] == current_user["id"]:
            user_threads.append(
                ThreadResponse(
                    thread_id=thread_id,
                    thread_name=thread_data["thread_name"],
                    created_at=thread_data["created_at"],
                    message_count=thread_data["message_count"]
                )
            )
    
    return {"threads": user_threads}


@router.delete("/thread/{thread_id}")
async def delete_thread(
    thread_id: str,
    current_user: Dict[str, Any] = Depends(verify_token)
):
    """Delete a specific thread"""
    if thread_id not in active_threads:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Thread not found"
        )
    
    if active_threads[thread_id]["user_id"] != current_user["id"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied to this thread"
        )
    
    del active_threads[thread_id]
    return {"message": "Thread deleted successfully"}