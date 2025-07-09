from langchain.tools import BaseTool
from typing import Type
import asyncio
from pydantic import BaseModel, Field

class RAGToolInput(BaseModel):
    query: str = Field(..., description="The query to search for relevant information in the RAG system.")

class RAGTool(BaseTool):
    name: str = "rag_tool"
    description: str = "A tool to search for relevant information about plant diseases and treatments in the RAG system based on the user's query. Use this to find specific disease information, symptoms, and treatment recommendations."
    args_schema: Type[RAGToolInput] = RAGToolInput

    async def _arun(self, query: str) -> str:
        """
        Async version of the RAG tool.
        """
        # Placeholder response - replace with actual async RAG implementation
        if "fungus" in query.lower() or "cauliflower" in query.lower():
            return """
            Based on RAG database search for cauliflower fungal diseases:
            
            Common fungal diseases in cauliflower:
            1. Alternaria leaf spot (Alternaria brassicae)
            2. Black rot (Xanthomonas campestris)
            3. Downy mildew (Peronospora parasitica)
            4. White rust (Albugo candida)
            
            Recommended treatments:
            - Copper-based fungicides (Copper oxychloride 50% WP @ 2-3 g/L)
            - Mancozeb 75% WP @ 2-2.5 g/L
            - Metalaxyl + Mancozeb @ 2-2.5 g/L for downy mildew
            - Ensure proper drainage and avoid overhead watering
            - Remove infected plant debris
            """
        
        return f"RAG database search completed for: {query}. Found relevant information about plant diseases and treatments."

    def _run(self, query: str) -> str:
        """
        Sync version calls async version.
        """
        try:
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(self._arun(query))
        except RuntimeError:
            # If no event loop is running, create a new one
            return asyncio.run(self._arun(query))

rag_tool = RAGTool()