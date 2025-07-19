from langgraph.graph import MessagesState


class AICompanionState(MessagesState):
    """State class for the AI Companion workflow.
    Extends MessagesState to track conversation history and maintains the last message received.
    """
    collection_name:str
    workflow: str
    output: str
    current_activity: str
    image: bytes
    voice: bytes