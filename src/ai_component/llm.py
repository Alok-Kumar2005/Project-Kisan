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

from src.ai_component.config import (
    gemini_model_kwargs,
    gemini_model_name,
    groq_model_kwargs,
    groq_model_name,
)

load_dotenv()


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

    def get_llm_chain(self, prompt: PromptTemplate | ChatPromptTemplate):
        """
        Returns a LangChain chain object based on the selected model type.
        
        Args:
            prompt: PromptTemplate object
        
        Returns:
            A LangChain chain that can be invoked with input variables.
        """
        if self.model_type == "gemini":
            llm = ChatGoogleGenerativeAI(
                model=self.gemini_model_name,
                google_api_key=self.google_api_key,
                **self.gemini_model_kwargs 
            )
        elif self.model_type == "groq":
            llm = ChatGroq(
                model=self.groq_model_name,
                api_key=self.groq_api_key,
                **self.groq_model_kwargs  
            )
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

        chain = prompt | llm
        return chain

    def get_structured_llm_chain(self, prompt: PromptTemplate | ChatPromptTemplate, output_schema: BaseModel):
        """
        Returns a LangChain chain that returns structured output based on a Pydantic model.
        
        Args:
            prompt: PromptTemplate object
            output_schema: Pydantic BaseModel class defining the output structure
        
        Returns:
            A LangChain chain that returns structured output according to the schema.
        """

        if self.model_type == "gemini":
            llm = ChatGoogleGenerativeAI(
                model=self.gemini_model_name,
                google_api_key=self.google_api_key,
                **self.gemini_model_kwargs  
            )
        elif self.model_type == "groq":
            llm = ChatGroq(
                model=self.groq_model_name,
                api_key=self.groq_api_key,
                **self.groq_model_kwargs 
            )
        
        structured_llm = llm.with_structured_output(output_schema)
        chain = prompt | structured_llm
        
        return chain

    # NEW: Async versions of the chain methods
    async def get_llm_chain_async(self, prompt: PromptTemplate | ChatPromptTemplate):
        """
        Returns an async LangChain chain object based on the selected model type.
        
        Args:
            prompt: PromptTemplate object
        
        Returns:
            An async LangChain chain that can be invoked with input variables.
        """
        if self.model_type == "gemini":
            llm = ChatGoogleGenerativeAI(
                model=self.gemini_model_name,
                google_api_key=self.google_api_key,
                **self.gemini_model_kwargs 
            )
        elif self.model_type == "groq":
            llm = ChatGroq(
                model=self.groq_model_name,
                api_key=self.groq_api_key,
                **self.groq_model_kwargs  
            )
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

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

        if self.model_type == "gemini":
            llm = ChatGoogleGenerativeAI(
                model=self.gemini_model_name,
                google_api_key=self.google_api_key,
                **self.gemini_model_kwargs  
            )
        elif self.model_type == "groq":
            llm = ChatGroq(
                model=self.groq_model_name,
                api_key=self.groq_api_key,
                **self.groq_model_kwargs 
            )
        
        structured_llm = llm.with_structured_output(output_schema)
        chain = prompt | structured_llm
        
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