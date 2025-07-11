import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.prompts import PromptTemplate
from pydantic import BaseModel
from dotenv import load_dotenv
import os

from typing import Annotated
from langchain_core.messages import BaseMessage
from typing_extensions import TypedDict

from src.ai_component.config import (
    gemini_model_kwargs,
    gemini_model_name,
    groq_model_kwargs,
    groq_model_name,
)

load_dotenv()

os.environ['LANGSMITH_API_KEY'] = os.getenv("LANGSMITH_API_KEY")
os.environ['LANGSMITH_TRACING'] = os.getenv("LANGSMITH_TRACING")
os.environ['LANGSMITH_PROJECT'] = os.getenv("LANGSMITH_PROJECT")

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

    async def get_llm_chain_async(self, prompt: PromptTemplate | ChatPromptTemplate):
        """
        Returns an async LangChain chain object based on the selected model type.
        
        Args:
            prompt: PromptTemplate object
        
        Returns:
            An async LangChain chain that can be invoked with input variables.
        """
        llm = self._get_llm()
        chain = prompt | llm
        return chain

    async def get_structured_llm_chain_async(self, prompt: PromptTemplate | ChatPromptTemplate, output_schema: BaseModel):
        """
        Returns an async LangChain chain that returns structured output based on a Pydantic model.
        
        Args:
            prompt: PromptTemplate object
            output_schema: Pydantic BaseModel class defining the output structure
        
        Returns:
            An async LangChain chain that returns structured output according to the schema.
        """
        llm = self._get_llm()
        structured_llm = llm.with_structured_output(output_schema)
        chain = prompt | structured_llm
        
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
        llm = self._get_llm()
        llm_with_tools = llm.bind_tools(tools)
        chain = prompt | llm_with_tools 
        return chain


if __name__ == "__main__":
    import asyncio
    
    async def test_async():
        factory = LLMChainFactory(model_type="groq")
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant."),
            ("user", "{input}")
        ])
        
        chain = await factory.get_llm_chain_async(prompt)
        response = await chain.ainvoke({"input": "What is the capital of France?"})
        print(response.content)
    
    # Test sync version (original)
    factory = LLMChainFactory(model_type="groq")
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant."),
        ("user", "{input}")
    ])
    
    chain = factory.get_llm_chain(prompt)
    response = chain.invoke({"input": "What is the capital of France?"})
    print(response.content)
    
    # Test async version
    asyncio.run(test_async())