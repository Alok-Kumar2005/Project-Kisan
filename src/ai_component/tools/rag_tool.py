from langchain.tools import BaseTool
from typing import Type
from pydantic import BaseModel, Field

class RAGToolInput(BaseModel):
    query: str = Field(..., description="The query to search for relevant information in the RAG system.")

class RAGTool(BaseTool):
    name: str = "rag_tool"
    description: str = "A tool to search for relevant information in the RAG system based on the user's query."
    args_schema: Type[RAGToolInput] = RAGToolInput

    def _run(self, query: str) -> str:
        """
        Run the RAG tool with the provided query.
        This method should implement the logic to search for relevant information in the RAG system.
        """
        return f"Searching for information related to: {query}"

rag_tool = RAGTool()