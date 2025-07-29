from langgraph.graph import MessagesState
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class AICompanionState(TypedDict):
    """State class for the AI Companion workflow.
    Extends MessagesState to track conversation history and maintains the last message received.
    """
    messages: Annotated[list[BaseMessage], add_messages]
    collection_name:str
    workflow: str
    output: str
    current_activity: str
    image: bytes
    voice: bytes