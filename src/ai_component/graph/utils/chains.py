import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

import asyncio
from pydantic import BaseModel , Field
from typing import Optional, Literal , Union
from src.ai_component.llm import LLMChainFactory
from src.ai_component.core.prompts import router_template
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from src.ai_component.logger import logging
from src.ai_component.exception import CustomException

class Router(BaseModel):
        route_node: Literal['DiseaseNode', 'WeatherNode', 'MandiNode', 'GovSchemeNode', 'CarbonFootprintNode', 'GeneralNode'] = Field(..., description="Just the type of node to route to, e.g., 'DiseaseNode")
        output: Literal["TextNode", "ImageNode", "VoiceNode"] = Field(..., description="Give in which format user want answer")



async def async_router_chain():
    """
    Async version of router_chain.
    Return the node according to user query and the prompt
    """
    try:
        logging.info("Calling Router Chain")
        prompt = PromptTemplate(
            input_variables=["query"],
            template= router_template.prompt
        )

        factory = LLMChainFactory(model_type="groq")
        chain = await factory.get_structured_llm_chain_async(prompt, Router)
        
        return chain
    except CustomException as e:
        logging.error(f"Error in Engineering Node : {str(e)}")
        raise CustomException(e, sys) from e


if __name__ == "__main__":
    
    async def test_async():
        query = "What are the symptoms of leaf blight in rice?"
        chain = await async_router_chain()
        response = await chain.ainvoke({"query": query})
        print(f"result: {response.route_node}")

    asyncio.run(test_async())