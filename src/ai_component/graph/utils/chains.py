import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

from pydantic import BaseModel , Field
from typing import Optional, Literal , Union
from src.ai_component.llm import LLMChainFactory
from src.ai_component.core.prompts import router_template
from langchain.prompts import PromptTemplate, ChatPromptTemplate

class Router(BaseModel):
        route_node: Literal['DiseaseNode', 'WeatherNode', 'CropNode', 'MarketNode', 'GeneralNode'] = Field(..., description="Just the type of node to route to, e.g., 'DiseaseNode")


async def async_router_chain() -> Literal["DiseaseNode", "WeatherNode", "CropNode", "MarketNode", 'GeneralNode']:
    """
    Async version of router_chain.
    Return the node according to user query and the prompt
    """
    prompt = PromptTemplate(
        input_variables=["query"],
        template= router_template
    )

    factory = LLMChainFactory(model_type="groq")
    chain = await factory.get_structured_llm_chain_async(prompt, Router)
    
    return chain


if __name__ == "__main__":
    import asyncio
    
    async def test_async():
        query = "What are the symptoms of leaf blight in rice?"
        chain = await async_router_chain()
        response = await chain.ainvoke({"query": query})
        print(f"Async result: {response.route_node}")

    asyncio.run(test_async())