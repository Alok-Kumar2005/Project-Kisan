"""
Government Scheme RAG Tool
──────────────────────────
Searches the 'Government_scheme' Qdrant Cloud collection for relevant content.

On first use (or if the collection is empty) the tool automatically ingests the
PDFs from the `data/` folder into Qdrant Cloud.  Because Qdrant Cloud is
persistent, the data only needs to be uploaded once — subsequent restarts skip
ingestion automatically.
"""

import os
import asyncio
from typing import Type, ClassVar

from langchain.tools import BaseTool
from pydantic import BaseModel, Field, ConfigDict

from src.ai_component.logger import logging
from src.ai_component.modules.memory.vector_store import memory


COLLECTION_NAME = "Government_scheme"
# Relative to the project root (same dir as Dockerfile / docker-compose.yml)
DATA_PATH = "data"


class SchemeToolInput(BaseModel):
    query: str = Field(
        ...,
        description="Query to search in the government scheme knowledge base.",
    )


class SchemeTool(BaseTool):
    name: str = "gov_scheme_tool"
    description: str = (
        "Search for Indian government agricultural schemes, subsidies, and policies. "
        "Use this to answer questions about PM-Kisan, crop insurance, soil health cards, "
        "irrigation schemes, and other government support for farmers."
    )
    args_schema: Type[SchemeToolInput] = SchemeToolInput
    model_config = ConfigDict(extra="allow")
    memory: ClassVar = memory

    # ------------------------------------------------------------------ #
    #  Ingestion check — based purely on whether the cloud collection     #
    #  already has vectors.  No local metadata collection needed.         #
    # ------------------------------------------------------------------ #

    def _collection_has_data(self) -> bool:
        """Return True if the Qdrant Cloud collection exists and is non-empty."""
        try:
            if not self.memory._collection_exists(COLLECTION_NAME):
                return False
            info = self.memory.client.get_collection(COLLECTION_NAME)
            return info.vectors_count is not None and info.vectors_count > 0
        except Exception as e:
            logging.warning(f"Could not check collection state: {e}")
            return False

    async def _ingest_pdfs(self) -> bool:
        """Upload PDFs from DATA_PATH into the Qdrant Cloud collection."""
        if not os.path.isdir(DATA_PATH):
            logging.error(f"Data directory '{DATA_PATH}' not found — cannot ingest PDFs.")
            return False

        pdf_files = [f for f in os.listdir(DATA_PATH) if f.lower().endswith(".pdf")]
        if not pdf_files:
            logging.error(f"No PDF files found in '{DATA_PATH}'.")
            return False

        logging.info(f"Ingesting {len(pdf_files)} PDF(s) into '{COLLECTION_NAME}' on Qdrant Cloud…")
        try:
            result = await self.memory.StoreInMemory2(
                collection_name=COLLECTION_NAME,
                data_path=DATA_PATH,
            )
            if result:
                logging.info("PDF ingestion into Qdrant Cloud completed successfully.")
            return result
        except Exception as e:
            logging.error(f"PDF ingestion failed: {e}")
            return False

    async def _ensure_data_ready(self) -> bool:
        """Ingest PDFs if the cloud collection is empty."""
        if self._collection_has_data():
            logging.info(f"'{COLLECTION_NAME}' collection already has data — skipping ingestion.")
            return True

        logging.info(f"'{COLLECTION_NAME}' is empty or missing — running ingestion pipeline.")
        return await self._ingest_pdfs()

    # ------------------------------------------------------------------ #
    #  Tool run methods                                                    #
    # ------------------------------------------------------------------ #

    async def _arun(self, query: str) -> str:
        """Async entry point called by LangGraph."""
        try:
            logging.info(f"gov_scheme_tool query: {query}")

            if not await self._ensure_data_ready():
                return (
                    "⚠️ Government scheme data is not available. "
                    "Please ensure the PDF files are present in the data/ directory."
                )

            results = self.memory.search_in_collection(
                query=query,
                collection_name=COLLECTION_NAME,
                k=4,
            )

            if not results:
                return f"No relevant government scheme information found for: {query}"

            lines = [f"**Government Scheme Search Results** for: _{query}_\n"]
            for i, (doc, score) in enumerate(results, 1):
                lines.append(f"**Result {i}** (score: {score:.3f})")
                lines.append(doc.page_content.strip())
                lines.append("─" * 50)

            return "\n".join(lines)

        except Exception as e:
            logging.error(f"gov_scheme_tool error: {e}")
            return f"❌ Error searching government schemes: {e}"

    def _run(self, query: str) -> str:
        """Sync wrapper — runs the async version in a new event loop."""
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(self._arun(query))
            finally:
                loop.close()
        except Exception as e:
            logging.error(f"gov_scheme_tool sync error: {e}")
            return f"❌ Error: {e}"


gov_scheme_tool = SchemeTool()
