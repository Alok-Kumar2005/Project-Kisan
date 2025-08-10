import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

from langchain.tools import BaseTool
from typing import Type, ClassVar
import asyncio
import hashlib
from datetime import datetime
from pydantic import BaseModel, Field, ConfigDict
from src.ai_component.logger import logging
from src.ai_component.exception import CustomException
from src.ai_component.modules.memory.vector_store import memory

class SchemeToolInput(BaseModel):
    query: str = Field(..., description="The query to search for relevant information in the RAG system.")

class SchemeTool(BaseTool):
    name: str = "rag_tool"
    description: str = "A tool to search for relevant information about plant diseases and treatments in the RAG system based on the user's query. Use this to find specific disease information, symptoms, and treatment recommendations."
    args_schema: Type[SchemeToolInput] = SchemeToolInput
    
    model_config = ConfigDict(extra='allow')
    memory: ClassVar = memory
    
    def __init__(self, data_path: str = "data", collection_name: str = "Government_scheme", **kwargs):
        super().__init__(**kwargs)
        self.collection_name = collection_name
        self.data_path = data_path
        self.metadata_collection = f"{collection_name}_metadata"

    def _get_pdf_files(self) -> list:
        """Get list of PDF files in the data directory"""
        if not os.path.exists(self.data_path):
            logging.warning(f"Data directory '{self.data_path}' does not exist")
            return []
        
        pdf_files = []
        for file in os.listdir(self.data_path):
            if file.endswith('.pdf'):
                file_path = os.path.join(self.data_path, file)
                pdf_files.append({
                    'name': file,
                    'path': file_path,
                    'modified_time': os.path.getmtime(file_path),
                    'size': os.path.getsize(file_path)
                })
        return pdf_files

    def _get_file_hash(self, file_path: str) -> str:
        """Generate hash for file to track changes"""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    def _is_data_ingested(self) -> dict:
        """Check if data is already ingested and get status"""
        try:
            # Check if main collection exists
            if not self.memory._collection_exists(self.collection_name):
                return {"ingested": False, "reason": "Collection doesn't exist"}
            
            # Check if metadata collection exists
            if not self.memory._collection_exists(self.metadata_collection):
                return {"ingested": False, "reason": "Metadata collection doesn't exist"}
            
            # Get current PDF files
            current_files = self._get_pdf_files()
            if not current_files:
                return {"ingested": False, "reason": "No PDF files found in data directory"}
            
            # Search for file metadata in metadata collection
            stored_files_info = self.memory.search_in_collection(
                query="file_metadata",
                collection_name=self.metadata_collection,
                k=100  # Get all stored files
            )
            
            if not stored_files_info:
                return {"ingested": False, "reason": "No file metadata found"}
            
            # Extract stored file information
            stored_files = {}
            for doc, score in stored_files_info:
                if doc.metadata and 'filename' in doc.metadata:
                    filename = doc.metadata['filename']
                    stored_files[filename] = {
                        'hash': doc.metadata.get('file_hash', ''),
                        'modified_time': doc.metadata.get('modified_time', 0)
                    }
            
            # Check if all current files are stored and up-to-date
            missing_files = []
            changed_files = []
            
            for file_info in current_files:
                filename = file_info['name']
                current_hash = self._get_file_hash(file_info['path'])
                
                if filename not in stored_files:
                    missing_files.append(filename)
                elif stored_files[filename]['hash'] != current_hash:
                    changed_files.append(filename)
            
            if missing_files or changed_files:
                reason = f"Missing files: {missing_files}, Changed files: {changed_files}"
                return {"ingested": False, "reason": reason, "missing": missing_files, "changed": changed_files}
            
            return {"ingested": True, "reason": "All files are up-to-date"}
            
        except Exception as e:
            logging.error(f"Error checking ingestion status: {str(e)}")
            return {"ingested": False, "reason": f"Error checking status: {str(e)}"}

    async def _ingest_data_pipeline(self) -> bool:
        """Smart pipeline to ingest data only if needed"""
        try:
            logging.info("Running data ingestion pipeline...")
            status = self._is_data_ingested()
            
            if status["ingested"]:
                logging.info("Data is already up-to-date, skipping ingestion")
                return True
            
            logging.info(f"Data ingestion needed: {status['reason']}")
            
            # Get PDF files
            pdf_files = self._get_pdf_files()
            if not pdf_files:
                logging.error("No PDF files found for ingestion")
                return False
            
            # Create collections if they don't exist
            self.memory.create_collection(self.collection_name)
            self.memory.create_collection(self.metadata_collection)
            logging.info(f"Ingesting {len(pdf_files)} PDF files...")
            
            # Ingest PDF data using existing method
            result = await self.memory.StoreInMemory2(
                collection_name=self.collection_name,
                data_path=self.data_path
            )
            
            if not result:
                logging.error("Failed to ingest PDF data")
                return False
            
            # Store file metadata for future checks
            for file_info in pdf_files:
                file_hash = self._get_file_hash(file_info['path'])
                metadata = {
                    'filename': file_info['name'],
                    'file_hash': file_hash,
                    'modified_time': file_info['modified_time'],
                    'file_size': file_info['size'],
                    'ingestion_date': datetime.now().isoformat(),
                    'type': 'file_metadata'
                }
                
                # Store metadata
                self.memory.ingest_data(
                    collection_name=self.metadata_collection,
                    data=f"file_metadata for {file_info['name']}",
                    additional_metadata=metadata
                )
            
            logging.info("Data ingestion pipeline completed successfully")
            return True
            
        except Exception as e:
            logging.error(f"Error in data ingestion pipeline: {str(e)}")
            return False

    async def _ensure_data_ready(self) -> bool:
        """Ensure data is ready for search"""
        try:
            # Check if we need to run ingestion
            if not self._is_data_ingested()["ingested"]:
                logging.info("Data not ready, running ingestion pipeline...")
                return await self._ingest_data_pipeline()
            
            return True
        except Exception as e:
            logging.error(f"Error ensuring data ready: {str(e)}")
            return False

    async def _arun(self, query: str) -> str:
        """
        Async version of the RAG tool with automatic data pipeline.
        """
        try:
            logging.info(f"Running RAG tool with query: {query}")
            
            # Ensure data is ready (run pipeline if needed)
            data_ready = await self._ensure_data_ready()
            if not data_ready:
                return "❌ Failed to prepare data for search. Please check the logs for more details."
            
            # Search for relevant information
            search_results = self.memory.search_in_collection(
                query=query, 
                collection_name=self.collection_name, 
                k=3
            )
            
            if not search_results:
                return f"No relevant information found in the knowledge base for query: {query}"
            
            # Format the results
            formatted_response = f"🔍 **RAG Search Results for**: {query}\n\n"
            
            for i, (doc, score) in enumerate(search_results, 1):
                formatted_response += f"📄 **Result {i}** (Relevance: {score:.3f})\n"
                formatted_response += f"{doc.page_content}\n"
                
                if doc.metadata:
                    formatted_response += f"📅 Created: {doc.metadata.get('created_at', 'Unknown')}\n"
                
                formatted_response += "\n" + "─" * 50 + "\n"
            
            return formatted_response
            
        except Exception as e:
            logging.error(f"Error in RAG Tool: {str(e)}")
            return f"❌ Error occurred while searching: {str(e)}"

    def _run(self, query: str) -> str:
        """
        Sync version with automatic data pipeline.
        """
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(self._arun(query))
                return result
            finally:
                loop.close()
                
        except Exception as e:
            logging.error(f"Error in RAG Tool (sync): {str(e)}")
            return f"❌ Error occurred while searching: {str(e)}"

    def force_reingest(self) -> bool:
        """Force re-ingestion of all data"""
        try:
            if self.memory._collection_exists(self.collection_name):
                self.memory.delete_collection(self.collection_name)
            if self.memory._collection_exists(self.metadata_collection):
                self.memory.delete_collection(self.metadata_collection)
            
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(self._ingest_data_pipeline())
                return result
            finally:
                loop.close()
                
        except Exception as e:
            logging.error(f"Error in force re-ingestion: {str(e)}")
            return False

gov_scheme_tool = SchemeTool(data_path="data", collection_name="Government_scheme")

if __name__ == "__main__":
    # Example usage - Everything is now automatic!
    
    print("🚀 Testing Automatic RAG Pipeline...")
    tool = SchemeTool(data_path="data")
    
    print("\n1️⃣ First search - will automatically ingest data if needed:")
    result1 = tool._run("What are the schemes for farmers from government to grow business")
    print(result1)
    
    print("\n2️⃣ Second search - will skip ingestion since data is already there:")
    result2 = tool._run("tWhat are the schemes for farmers from government to grow business")
    print(result2)
    
    print("\n3️⃣ Check ingestion status:")
    status = tool._is_data_ingested()
    print(f"Data status: {status}")
