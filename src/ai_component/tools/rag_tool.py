import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

from langchain.tools import BaseTool
from typing import Type, List, Dict
import asyncio
import json
from pydantic import BaseModel, Field
from src.ai_component.logger import logging
from src.ai_component.exception import CustomException

class RAGToolInput(BaseModel):
    query: str = Field(..., description="The query to search for people with similar problems or expertise in specific locations")

# Mock database - in production, this would be a real database
PEOPLE_DATABASE = [
    {
        "name": "Rajesh Sharma",
        "phone": "+918090175358", 
        "location": "Varanasi, Uttar Pradesh, India",
        "problem": "fertilizer issues with wheat crop",
        "description": "Facing nitrogen deficiency in wheat, looking for organic fertilizer solutions"
    }
]

def search_people(query: str) -> str:
    """Simple search function for people with similar problems"""
    try:
        query_lower = query.lower()
        matching_people = []
        
        # Simple keyword matching
        for person in PEOPLE_DATABASE:
            score = 0
            
            # Check location match
            if any(loc in person["location"].lower() for loc in ["varanasi", "uttar pradesh"] 
                   if loc in query_lower):
                score += 3
            
            # Check problem match
            if any(keyword in person["problem"].lower() or keyword in person["description"].lower()
                   for keyword in ["fertilizer", "pest", "soil", "organic"] 
                   if keyword in query_lower):
                score += 2
            
            if score > 0:
                person_copy = person.copy()
                person_copy["score"] = score
                matching_people.append(person_copy)
        
        # Sort by score
        matching_people.sort(key=lambda x: x["score"], reverse=True)
        
        if not matching_people:
            return "No people found with similar problems in your area."
        
        # Format results
        result = f"Found {len(matching_people)} people with similar problems:\n\n"
        
        for i, person in enumerate(matching_people[:3], 1):  # Show top 3
            result += f"{i}. **{person['name']}**\n"
            result += f"   📞 Phone: {person['phone']}\n"
            result += f"   📍 Location: {person['location']}\n"
            result += f"   🌾 Problem: {person['problem']}\n"
            result += f"   📝 Details: {person['description']}\n\n"
        
        result += "Would you like me to call any of these people?"
        return result
        
    except Exception as e:
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
            
            # Simulate some processing time
            await asyncio.sleep(0.1)
            
            result = search_people(query)
            
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

# Create the tool instance
rag_tool = RAGTool()