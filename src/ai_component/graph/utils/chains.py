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


def router_chain() -> Literal["DiseaseNode", "WeatherNode", "CropNode", "MarketNode", 'GeneralNode']:
    """
    Return the node according to user query and the prompt
    """
    prompt = PromptTemplate(
        input_variables=["query"],
        template= router_template
    )

    factory = LLMChainFactory(model_type="groq")
    chain = factory.get_structured_llm_chain(prompt, Router)
    
    return chain


if __name__ == "__main__":
    # Example usage
    query = "What are the symptoms of leaf blight in rice?"
    chain = router_chain()
    response = chain.invoke({"query": query})
    
    print(response.route_node) 
    # This should print the type of node to route to, e.g., "DiseaseNode"