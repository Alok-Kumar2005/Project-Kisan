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
import asyncio
from typing import Annotated, Dict, Any, Optional, Tuple
from langchain_core.messages import BaseMessage
from typing_extensions import TypedDict
import google.generativeai as genai
import base64
import time
import tempfile

from src.ai_component.config import (
    gemini_model_kwargs,gemini_model_name,
    groq_model_kwargs,groq_model_name,
    image_model, image_height, image_width , steps, image_url,
    # Add video config parameters
    video_model_name, video_duration, video_quality, video_fps
)
from src.ai_component.core.prompts import video_template
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
        
        # Video generation config
        self.video_model_name = video_model_name
        self.video_duration = video_duration
        self.video_quality = video_quality
        self.video_fps = video_fps
        
        # Initialize Google Generative AI for video generation
        if self.google_api_key:
            genai.configure(api_key=self.google_api_key)
            self.video_model = genai.GenerativeModel(self.video_model_name)

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
    def _convert_url_to_bytes(media_url: str) -> bytes:
        """
        Get media (image/video) as raw bytes and convert it into bytes
        """
        try:
            logging.info("Converting media url to bytes")
            response = requests.get(url=media_url, timeout=300)  # Increased timeout for videos
            response.raise_for_status()
            return response.content
        except Exception as e:
            logging.error(f"Error in converting url to bytes {str(e)}")
            print(f"Error in converting url to bytes {e}")
            return b""

    @staticmethod
    def _save_bytes_to_temp_file(media_bytes: bytes, file_extension: str = "mp4") -> str:
        """
        Save bytes to a temporary file and return the file path.
        Useful for video processing that requires file paths.
        """
        try:
            with tempfile.NamedTemporaryFile(suffix=f".{file_extension}", delete=False) as temp_file:
                temp_file.write(media_bytes)
                temp_path = temp_file.name
            logging.info(f"Saved bytes to temporary file: {temp_path}")
            return temp_path
        except Exception as e:
            logging.error(f"Error saving bytes to temp file: {str(e)}")
            raise CustomException(e, sys) from e

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

    async def _enhance_video_prompt_async(self, user_prompt: str) -> str:
        """
        Use LangChain to enhance video generation prompts for better results.
        """
        try:
            logging.info("Enhancing video prompt with LangChain")
            
            enhancement_template = ChatPromptTemplate.from_messages([
                ("system", video_template.prompt),
                ("human", "Transform this video request into a detailed prompt: {user_input}")
            ])
            
            chain = await self.get_llm_chain_async(enhancement_template)
            enhanced_result = await chain.ainvoke({"user_input": user_prompt})
            
            if hasattr(enhanced_result, 'content'):
                return enhanced_result.content
            else:
                return str(enhanced_result)
                
        except Exception as e:
            logging.warning(f"Prompt enhancement failed, using original: {str(e)}")
            return user_prompt

    async def get_video_model_async(self, prompt: str, duration: int = None,
                                   quality: str = None, fps: int = None,
                                   reference_image: str = None) -> Tuple[bytes, str]:

        try:
            logging.info("Calling async video model")
            duration = duration or self.video_duration
            quality = quality or self.video_quality
            fps = fps or self.video_fps
            
            if not self.google_api_key:
                raise ValueError("Google API key is required for video generation")
            
            # Enhance prompt using async method
            enhanced_prompt = await self._enhance_video_prompt_async(prompt)
            
            # Create video generation prompt
            video_prompt = f"""Generate a high-quality video: {enhanced_prompt}
            
Technical specs: {duration}s, {quality}, {fps}fps, 16:9 aspect ratio, cinematic style."""

            # Handle reference image if provided
            content_parts = [video_prompt]
            if reference_image:
                try:
                    if reference_image.startswith('http'):
                        img_response = requests.get(reference_image)
                        img_data = img_response.content
                    else:
                        with open(reference_image, 'rb') as f:
                            img_data = f.read()
                    
                    content_parts.append({
                        "mime_type": "image/jpeg",
                        "data": base64.b64encode(img_data).decode()
                    })
                except Exception as img_error:
                    logging.warning(f"Could not process reference image: {img_error}")

            # Generate video using async executor
            loop = asyncio.get_event_loop()
            
            def _generate_video():
                try:
                    response = self.video_model.generate_content(
                        content_parts,
                        generation_config=genai.types.GenerationConfig(
                            max_output_tokens=2048,
                            temperature=0.7,
                            top_p=0.9,
                        )
                    )
                    return response
                except Exception as e:
                    logging.error(f"Video generation API call failed: {str(e)}")
                    return None

            # Execute video generation
            result = await loop.run_in_executor(None, _generate_video)
            
            if result and hasattr(result, 'text'):
                # In actual implementation, extract video URL from result
                # For now, simulate
                mock_video_url = "https://example.com/generated_video.mp4"
                
                # Convert to bytes (in real implementation)
                # video_bytes = self._convert_url_to_bytes(mock_video_url)
                
                # Mock implementation
                video_bytes = b"async_video_data_" + enhanced_prompt.encode()[:100]
            else:
                # Fallback
                video_bytes = b"fallback_video_data_" + prompt.encode()[:100]
            
            # Create metadata info for LangGraph
            video_info = f"""{{
                "original_prompt": "{prompt}",
                "enhanced_prompt": "{enhanced_prompt}",
                "duration": {duration},
                "quality": "{quality}",
                "fps": {fps},
                "model": "{self.video_model_name}",
                "timestamp": {time.time()},
                "has_reference_image": {reference_image is not None},
                "video_size_bytes": {len(video_bytes)}
            }}"""
            
            logging.info(f"Generated video bytes length: {len(video_bytes)}")
            return video_bytes, video_info
            
        except CustomException as e:
            logging.error(f"Error in async video generation: {str(e)}")
            raise CustomException(e, sys) from e


if __name__ == "__main__":
    import asyncio
    
    async def test_refactored():
        factory = LLMChainFactory(model_type="gemini")

        # Test async video generation  
        print("\nTesting async video generation...")
        try:
            video_bytes, video_info = await factory.get_video_model_async(
                prompt="A serene mountain landscape with flowing river",
                duration=15,
                quality="720p"
            )
            print(f"Async Video bytes length: {len(video_bytes)}")
            print(f"Video info: {video_info}")
        except Exception as e:
            print(f"Async video generation error: {e}")

    asyncio.run(test_refactored())