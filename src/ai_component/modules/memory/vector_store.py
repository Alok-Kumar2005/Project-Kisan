import sys
import os
from datetime import datetime
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

from typing import List , Dict , Optional
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from langchain_qdrant import Qdrant
from langchain.schema import Document
from src.ai_component.config import top_collection_search, top_database_search
from src.ai_component.logger import logging
from src.ai_component.exception import CustomException
from dotenv import load_dotenv

load_dotenv()

class LongTermMemeory:
    def __init__(self, qdrant_url: str = os.getenv("QDRANT_URL"), google_api_key: str = os.getenv("GOOGLE_API_KEY")):
        self.qdrant_url = qdrant_url
        self.google_api_key = google_api_key

        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=self.google_api_key
        )

        self.client = QdrantClient(
            url = self.qdrant_url,
            prefer_grpc = False
        )

    def _list_collection(self)->List[str]:
        """Give all the collection in Vector Database"""
        try:
            logging.info("checking list of collections")
            collections = self.client.get_collections()
            return [collection.name for collection in collections.collections]
        except CustomException as e:
            logging.error(f"Error in finding list of collections {str(e)}")
            raise CustomException(e, sys) from e
        
    def _collection_exists(self, collection_name: str)-> bool:
        """Check the collection exist or not"""
        try:
            logging.info("Checking for collection in Vector Database")
            collections = self._list_collection()
            return collection_name in collections
        except CustomException as e:
            logging.error(f"Error in checking collection exist or not {str(e)}")
            raise CustomException(e, sys) from e
    
    def create_collection(self, collection_name: str , vector_size: int = 768) -> bool:
        """Create new collection"""
        try:
            logging.info("Checking for collection exist or not")
            if self._collection_exists(collection_name):
                logging.info("Collection already exist")
                return True
            
            logging.info("Creating new collection")
            self.client.create_collection(
                collection_name = collection_name,
                vectors_config = VectorParams(size = vector_size , distance = Distance.COSINE)
            )
            logging.info("New collection created")
            return True
        except CustomException as e:
            logging.error(f"Error in creating collection; {str(e)}")
            raise CustomException(e,sys) from e
        
    def delete_collection(self, collecton_name: str) -> bool:
        """Delete the collection from the vector database"""
        try:
            logging.info("Checking for collection exist or not")
            if not self._collection_exists(collection_name = collecton_name):
                logging.info("collection not exist")
                return False
            
            logging.info("Deleing Collection")
            self.client.delete_collection(collecton_name)
            logging.info("Collection removed")
            return True
        except CustomException as e:
            logging.error(f"Error in deleing collecton {str(e)}")
            raise CustomException(e, sys) from e
        
    def ingest_data(self, collection_name: str , data: str, additional_metadata: Dict = None)-> bool:
        """Ingest the data in the collection of the Vector Database with datetime metadata"""
        try:
            logging.info("Checking for collection exist or not")
            if not self._collection_exists(collection_name=collection_name):
                logging.info("Collection not exist")
                self.create_collection(collection_name=collection_name , vector_size= 768)
            
            logging.info("Preparing data with metadata")
            
            # Create metadata with datetime
            metadata = {
                "created_at": datetime.now().isoformat(),
                "timestamp": datetime.now().timestamp(),
                "collection": collection_name
            }
            
            # Add any additional metadata provided
            if additional_metadata:
                metadata.update(additional_metadata)
            
            # Convert data to Document objects with metadata
            if isinstance(data, str):
                documents = [Document(page_content=data, metadata=metadata)]
            elif isinstance(data, list):
                documents = []
                for item in data:
                    if isinstance(item, str):
                        documents.append(Document(page_content=item, metadata=metadata))
                    elif isinstance(item, Document):
                        # If it's already a Document, update its metadata
                        item.metadata.update(metadata)
                        documents.append(item)
                    else:
                        # Convert other types to string
                        documents.append(Document(page_content=str(item), metadata=metadata))
            else:
                # Handle other data types
                documents = [Document(page_content=str(data), metadata=metadata)]
            
            logging.info("Ingesting data")
            qdrant = Qdrant.from_documents(
                documents,
                self.embeddings,
                url = self.qdrant_url,
                collection_name = collection_name,
                prefer_grpc = False
            )
            logging.info("Data ingested successfully")
            return True
        except CustomException as e:
            logging.error(f"Error in inserting data : {str(e)}")
            raise CustomException(e, sys) from e
        
    def search_in_collection(self, query: str, collection_name: str, k: int = top_collection_search)-> List:
        """Search in the collection"""
        try:
            if not self._collection_exists(collection_name=collection_name):
                return []
            logging.info("Search in collecton ")
            db = Qdrant(
                client= self.client,
                collection_name= collection_name,
                embeddings=self.embeddings
            )
            docs = db.similarity_search_with_score(query=query , k= k)
            logging.info("relavent docs find with score")
            return docs
        except CustomException as e:
            logging.info(f"Error in similariy search {str(e)}")
            raise CustomException(e, sys) from e
    
    def search_across_collections(self, query: str, k: int = top_database_search) -> Dict:
        """Search across multiple collections"""
        try:
            logging.info("Search in database")
            results = {}
            
            for collection_name in self._list_collection:
                if self.collection_exists(collection_name):
                    docs = self.search_in_collection(collection_name, query, k)
                    results[collection_name] = docs
                else:
                    results[collection_name] = []
                    print(f"Collection '{collection_name}' not found")
            
            return results
        except CustomException as e:
            logging.info(f"Error in collections search {str(e)}")
            raise CustomException(e,sys) from e
        

if __name__ == "__main__":
    memory = LongTermMemeory()
    # memory.create_collection("ay7472")
    # memory.ingest_data("ay7472", "hii, My name is Alok and i am from varanasi Uttar pradesh")

    result = memory.search_in_collection("where alok live", "ay7472", 1)
    print(result)