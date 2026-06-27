import sys
from uuid import uuid4
from src.ai_component.modules.memory.vector_store import memory
from src.ai_component.llm import LLMChainFactory
from src.ai_component.logger import logging
from src.ai_component.exception import CustomException
from src.ai_component.core.prompts import Template
from src.ai_component.config import default_model
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
        self.llm = LLMChainFactory(model_type=default_model)
        self.output_schema1 = MemoryAnalysis1
        self.output_schema2 = MemoryAnalysis2

    async def _should_store(self, conversation: str) -> Literal["Yes", "No"]:
        """Tell that weather to store the conversation in the memory or not"""
        try:
            prompt = PromptTemplate(
                input_variables=["conversation"],
                template=Template.memory_template1
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
                template=Template.memory_template2
            )
            chain = await self.llm.get_structured_llm_chain_async(prompt=prompt, output_schema=self.output_schema2)
            response = await chain.ainvoke({"conversation": conversation})
            return response.summary
        except CustomException as e:
            logging.error(f"Error in generating summary: {str(e)}")
            raise CustomException(e, sys) from e

    async def store_in_memory(self, collection_name: str, conversation: str):
        """Store the conversation summary in the long-term store (AsyncPostgresStore) and
        the embedding in Qdrant Cloud for vector search.  PII is never written to Qdrant.
        """
        try:
            logging.info("Checking if conversation should be stored")
            summary = await self._summary(conversation)

            if summary is None:
                logging.info("Conversation not important enough to store")
                return False

            # ------------------------------------------------------------------
            # 1. Write durable summary to AsyncPostgresStore (Neon Postgres)
            #    namespace = ("long_term", collection_name) ensures per-user isolation.
            # ------------------------------------------------------------------
            try:
                from src.ai_component.graph.graph import get_store
                store = await get_store()
                if store is not None:
                    namespace = ("long_term", collection_name)
                    await store.aput(namespace, str(uuid4()), {"summary": summary})
                    logging.info(f"Summary stored in AsyncPostgresStore for namespace {namespace}")
                else:
                    logging.warning("AsyncPostgresStore not available; skipping long-term store write")
            except Exception as e:
                logging.error(f"Failed to write to AsyncPostgresStore: {str(e)}")
                # Do not raise — fall through to Qdrant write so embeddings still work

            # ------------------------------------------------------------------
            # 2. Write embedding to Qdrant Cloud for vector similarity search.
            #    Only allowed metadata fields are written; NO PII.
            # ------------------------------------------------------------------
            try:
                # PII-free metadata — phone, name, user_id, age, address are intentionally excluded
                safe_metadata = {"type": "conversation_summary"}
                result = self.vector_store.ingest_data(
                    collection_name=collection_name,
                    data=summary,
                    additional_metadata=safe_metadata
                )
                logging.info("Embedding stored in Qdrant Cloud successfully")
            except Exception as e:
                logging.error(f"Failed to write to Qdrant: {str(e)}")
                # Qdrant is best-effort; long-term store write above is the source of truth

            return True
        except CustomException as e:
            logging.error(f"Error in storing the conversation {str(e)}")
            raise CustomException(e, sys) from e


memory_manager = MemoryManager()
