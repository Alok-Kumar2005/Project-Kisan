import sys
import os
from datetime import datetime
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

from src.ai_component.modules.memory.vector_store import memory
from src.ai_component.llm import LLMChainFactory
from src.ai_component.logger import logging
from src.ai_component.exception import CustomException
from src.ai_component.core.prompts import memory_template1, memory_template2
from langchain.prompts import PromptTemplate
from pydantic import BaseModel, Field
from typing import Literal


class MemoryAnalysis1(BaseModel):
    is_important: Literal['Yes', 'No'] = Field(..., description="Is these conversation is important to store or not")

class MemoryAnalysis2(BaseModel):
    summary: str = Field(..., description="The short summary of the conversation between user and LLM")

class MemoryManager: 
    def __init__(self):
        self.vector_store = memory
        self.llm = LLMChainFactory(model_type="gemini")
        self.output_schema1 = MemoryAnalysis1
        self.output_schema2 = MemoryAnalysis2

    async def _should_store(self, conversation: str) -> Literal["Yes", "No"]:
        """Tell that weather to store the conversation in the memory or not"""
        try:
            prompt = PromptTemplate(
                input_variables=["conversation"],
                template=memory_template1
            )
            chain = await self.llm.get_structured_llm_chain_async(prompt=prompt, output_schema=self.output_schema1)
            response = await chain.ainvoke({"conversation": conversation})
            return response.is_important
        except CustomException as e:
            logging.error(f"Error in finding should store in memory or not: {str(e)}")
            raise CustomException(e, sys) from e
        
    async def _summary(self, conversation: str):
        """
        Generate the summary of the conversation 
        """
        try:
            should_store = await self._should_store(conversation)
            if should_store == "No":
                logging.info("Not important to store")
                return None
            
            logging.info("Getting summary of conversation")
            prompt = PromptTemplate(
                input_variables=["conversation"],
                template=memory_template2
            )
            chain = await self.llm.get_structured_llm_chain_async(prompt=prompt, output_schema=self.output_schema2)
            response = await chain.ainvoke({"conversation": conversation})
            return response.summary
        except CustomException as e:
            logging.error(f"Error in generating summary: {str(e)}")
            raise CustomException(e, sys) from e
        
    async def store_in_memory(self, collection_name: str, conversation: str):
        """Store the conversation in memory"""
        try:
            logging.info("Checking if conversation should be stored")
            summary = await self._summary(conversation)
            
            if summary is None:
                logging.info("Conversation not important enough to store")
                return False
            
            logging.info("Storing the message")
            self.vector_store.ingest_data(
                collection_name=collection_name,
                data=summary,
                additional_metadata={"type": "conversation_summary"}
            )
            logging.info("Message stored successfully")
            return True
        except CustomException as e:
            logging.error(f"Error in storing the conversation {str(e)}")
            raise CustomException(e, sys) from e
        

memory_manager = MemoryManager()  