import os
from src.ai_component.config import max_result
from langchain_tavily import TavilySearch
from dotenv import load_dotenv

load_dotenv()
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")


web_tool = TavilySearch(
    max_results=max_result,
    topic="general",
)