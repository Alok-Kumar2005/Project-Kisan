from langgraph.graph import MessagesState


class AICompanionState(MessagesState):
    """State class for the AI Companion workflow.
    Extends MessagesState to track conversation history and maintains the last message received.
    """
    workflow: str
    current_activity: str