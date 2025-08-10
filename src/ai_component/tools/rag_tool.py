import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

from langchain.tools import BaseTool
from typing import Type
import asyncio
from pydantic import BaseModel, Field
from src.ai_component.logger import logging
from src.ai_component.exception import CustomException
from src.ai_component.modules.memory.vector_store import memory

class RAGToolInput(BaseModel):
    query: str = Field(..., description="The query to search for people with similar problems or expertise in specific locations")

def search_people_from_vector_store(query: str, k: int = 10) -> str:
    """Search for people with similar problems using vector store"""
    try:
        logging.info(f"Searching vector store for query: {query}")
        search_results = memory.search_across_collections(query, k)
        if not search_results:
            return "No people found with similar problems in the database."
        
        all_matches = []
        for collection_name, docs_with_scores in search_results.items():
            if docs_with_scores: 
                for doc, score in docs_with_scores:
                    metadata = doc.metadata
                    content = doc.page_content
                    person = {
                        "collection_name": collection_name,
                        "name": metadata.get("user_name", "Unknown"),
                        "phone": metadata.get("user_phone", "Not available"),
                        "address": metadata.get("user_address", "Location not specified"),
                        "user_id": metadata.get("user_id", "N/A"),
                        "problem_summary": content,
                        "similarity_score": float(score)
                    }
                    all_matches.append(person)
        
        if not all_matches:
            return "No people found with similar problems."
        
        # Sort by similarity score
        all_matches.sort(key=lambda x: x["similarity_score"])
        
        # Removing duplicates
        seen_users = set()
        unique_matches = []
        for match in all_matches:
            user_key = f"{match['name']}_{match['phone']}"
            if user_key not in seen_users:
                seen_users.add(user_key)
                unique_matches.append(match)
        
        # Format results
        result = f"Found {len(unique_matches)} people with similar problems:\n\n"
        
        # Show top 3 most relevant matches
        for i, person in enumerate(unique_matches[:3], 1):
            result += f"{i}. **{person['name']}**\n"
            result += f" Phone: {person['phone']}\n"
            result += f" Location: {person['address']}\n"
            result += f" Problem: {person['problem_summary'][:150]}{'...' if len(person['problem_summary']) > 150 else ''}\n"
            result += f" Similarity: {person['similarity_score']:.3f}\n\n"
        
        if len(unique_matches) > 3:
            result += f"... and {len(unique_matches) - 3} more matches available.\n\n"
        
        result += "Would you like me to help you connect with any of these people?"
        return result
        
    except Exception as e:
        logging.error(f"Error searching vector store: {str(e)}")
        return f"Error searching for people: {str(e)}"

class RAGTool(BaseTool):
    name: str = "rag_tool"
    description: str = """Search for people in specific locations who have similar agricultural problems. 
    Use this tool when users want to find others facing similar issues or connect with people in their area."""
    args_schema: Type[RAGToolInput] = RAGToolInput

    async def _arun(self, query: str) -> str:
        """Async version of the RAG tool."""
        try:
            logging.info(f"Running RAG tool with query: {query}")
            result = search_people_from_vector_store(query)
            logging.info(f"RAG tool completed search")
            return result
        except Exception as e:
            logging.error(f"Error in RAG Tool: {str(e)}")
            return f"Sorry, I encountered an error while searching for people: {str(e)}"

    def _run(self, query: str) -> str:
        """Sync version calls async version."""
        try:
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(self._arun(query))
        except Exception as e:
            logging.error(f"Error in RAG Tool sync method: {str(e)}")
            return f"Sorry, I encountered an error: {str(e)}"

rag_tool = RAGTool()