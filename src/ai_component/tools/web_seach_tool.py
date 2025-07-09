import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

from src.ai_component.config import max_result
from langchain_tavily import TavilySearch
from dotenv import load_dotenv

load_dotenv()
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")


tool = TavilySearch(
    max_results=max_result,
    topic="general",
)