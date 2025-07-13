import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

import requests
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.prompts import PromptTemplate
from together import Together
from pydantic import BaseModel
from dotenv import load_dotenv
import os

from typing import Annotated
from langchain_core.messages import BaseMessage
from typing_extensions import TypedDict

from src.ai_component.config import (
    gemini_model_kwargs,gemini_model_name,
    groq_model_kwargs,groq_model_name,
    image_model, image_height, image_width , steps, image_url
)
from src.ai_component.logger import logging
from src.ai_component.exception import CustomException

load_dotenv()

os.environ['LANGSMITH_API_KEY'] = os.getenv("LANGSMITH_API_KEY")
os.environ['LANGSMITH_TRACING'] = os.getenv("LANGSMITH_TRACING")
os.environ['LANGSMITH_PROJECT'] = os.getenv("LANGSMITH_PROJECT")
os.environ["TOGETHER_API_KEY"]  = os.getenv("TOGETHER_API_KEY")

class LLMChainFactory:
    def __init__(self, model_type: str = "gemini"):
        """
        Initializes the factory with the model type.
        """
        self.model_type = model_type
        self.gemini_model_name = gemini_model_name
        self.groq_model_name = groq_model_name
        self.gemini_model_kwargs = gemini_model_kwargs
        self.groq_model_kwargs = groq_model_kwargs
        self.google_api_key = os.getenv("GOOGLE_API_KEY")
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        self.imgClient = Together()

    def _get_llm(self):
        """
        Returns the appropriate LLM instance based on model type.
        """
        if self.model_type == "gemini":
            return ChatGoogleGenerativeAI(
                model=self.gemini_model_name,
                google_api_key=self.google_api_key,
                **self.gemini_model_kwargs 
            )
        elif self.model_type == "groq":
            return ChatGroq(
                model=self.groq_model_name,
                api_key=self.groq_api_key,
                **self.groq_model_kwargs  
            )
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
    @staticmethod
    def _convert_url_to_bytes(img_url: str) -> bytes:
        """
        Get image as raw bytes and convert it into bytes
        """
        try:
            logging.info("Converting image url to bytes")
            response = requests.get(url=img_url)
            response.raise_for_status()
            return response.content
        except CustomException as e:
            logging.error(f"Error in converting url to bytes {str(e)}")
            print(f"Error in converting url to bytes {e}")
            return b""

    async def get_llm_chain_async(self, prompt: PromptTemplate | ChatPromptTemplate):
        """
        Returns an async LangChain chain object based on the selected model type.
        
        Args:
            prompt: PromptTemplate object
        
        Returns:
            An async LangChain chain that can be invoked with input variables.
        """
        try:
            logging.info("Calling llm chain ")
            llm = self._get_llm()
            chain = prompt | llm
            return chain
        except CustomException as e:
            logging.error(f"Error in llm chain : {str(e)}")
            raise CustomException(e, sys) from e

    async def get_structured_llm_chain_async(self, prompt: PromptTemplate | ChatPromptTemplate, output_schema: BaseModel):
        """
        Returns an async LangChain chain that returns structured output based on a Pydantic model.
        
        Args:
            prompt: PromptTemplate object
            output_schema: Pydantic BaseModel class defining the output structure
        
        Returns:
            An async LangChain chain that returns structured output according to the schema.
        """
        try:
            logging.info("Calling structured LLM model")
            llm = self._get_llm()
            structured_llm = llm.with_structured_output(output_schema)
            chain = prompt | structured_llm
        except CustomException as e:
            logging.error(f"Error in structured llm model : {str(e)}")
            raise CustomException(e, sys) from e
        
        return chain

    async def get_llm_tool_chain(self, prompt: PromptTemplate | ChatPromptTemplate, tools: list):
        """
        Returns a LangChain chain that integrates tools with the LLM.
        
        Args:
            prompt: PromptTemplate object
            tools: List of tools to integrate with the LLM
        
        Returns:
            A LangChain chain that can be invoked with input variables and uses the specified tools.
        """
        try:
            logging.info("Callin tool llm model")
            llm = self._get_llm()
            llm_with_tools = llm.bind_tools(tools)
            chain = prompt | llm_with_tools 
            return chain
        except CustomException as e:
            logging.error(f"Error in calling tool llm model : {str(e)}")
            raise CustomException(e, sys) from e
    
    def get_image_model(self, prompt: str, model: str = None, 
                    img_width: int = None, img_height: int = None,
                    img_steps: int = None, img_url: str = None):
        """
        Generate an image using the Together API.
        """
        try:
            logging.info("Calling image model")
            model = model or image_model
            img_width = img_width or image_width
            img_height = img_height or image_height
            img_steps = img_steps or steps
            img_url = img_url or image_url
            
            result = self.imgClient.images.generate(
                model=model,
                width=img_width,
                height=img_height,
                steps=img_steps,
                prompt=prompt,
                image_url=img_url
            )
            result_url = result.data[0].url
            logging.info(f"Generated image url : {result_url}")
            print(result_url)
            result_bytes = self._convert_url_to_bytes(result_url)
            return result_bytes
        except CustomException as e:
            logging.error(f"Error in generating image : {str(e)}")
            raise CustomException(e, sys) from e
    
    async def get_image_model_async(self, prompt: str, model: str = None, 
                                img_width: int = None, img_height: int = None,
                                img_steps: int = None, img_url: str = None):
        """
        Async wrapper for image generation.
        """
        try:
            logging.info("Calling image model")
            model = model or image_model
            img_width = img_width or image_width
            img_height = img_height or image_height
            img_steps = img_steps or steps
            img_url = img_url or image_url
            
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: self.imgClient.images.generate(
                    model=model,
                    width=img_width,
                    height=img_height,
                    steps=img_steps,
                    prompt=prompt,
                    image_url=img_url
                )
            )
            
            result_url = result.data[0].url
            logging.info(f"Generated image Url: {result_url}")
            result_bytes = self._convert_url_to_bytes(result_url)
            return result_bytes, result_url
        except CustomException as e:
            logging.error(f"Error in generating image : {str(e)}")
            raise CustomException(e, sys) from e


if __name__ == "__main__":
    import asyncio
    
    async def test_async():
        factory = LLMChainFactory(model_type="groq")
        # prompt = ChatPromptTemplate.from_messages([
        #     ("system", "You are a helpful assistant."),
        #     ("user", "{input}")
        # ])
        
        # chain = await factory.get_llm_chain_async(prompt)
        # response = await chain.ainvoke({"input": "What is the capital of France?"})
        # print(response.content)

        response, url = await factory.get_image_model_async(prompt= "generate image of drinking man")
        print(response)
        print(url)

    
    # Test sync version (original)
    # factory = LLMChainFactory(model_type="groq")
    # prompt = ChatPromptTemplate.from_messages([
    #     ("system", "You are a helpful assistant."),
    #     ("user", "{input}")
    # ])
    
    # chain = factory.get_llm_chain(prompt)
    # response = chain.invoke({"input": "What is the capital of France?"})
    # print(response.content)
    
    # Test async version
    asyncio.run(test_async())



# ## for sync model of image generation
# if __name__ == "__main__":
#     import asyncio
    
#     async def test_async():
#         factory = LLMChainFactory(model_type="groq")
        
#         # Remove 'await' since get_image_model is now synchronous
#         response = factory.get_image_model(prompt="generate image of drunk man")
#         print(response)

#     asyncio.run(test_async())