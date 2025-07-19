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
from google import genai
from google.genai import types
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
        self.video_model_name = "veo-3.0-generate-preview"  # Use Veo 3
        self.video_duration = 15
        self.video_quality = "720p"
        self.video_fps = 24
        
        # Initialize Google Generative AI for video generation
        if self.google_api_key:
            self.genai_client = genai.Client(api_key=self.google_api_key)
        else:
            raise ValueError("Google API key is required for video generation")

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
                ("system", video_template),
                ("human", "{user_input}")
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
        """
        Generate video using Google's Veo 3.0 API with proper async handling.
        """
        try:
            logging.info("Starting async video generation with Veo 3.0")
            duration = duration or self.video_duration
            quality = quality or self.video_quality
            fps = fps or self.video_fps
            
            if not self.google_api_key:
                raise ValueError("Google API key is required for video generation")
            
            # Enhance prompt using async method
            enhanced_prompt = await self._enhance_video_prompt_async(prompt)
            logging.info(f"Enhanced prompt: {enhanced_prompt}")
            
            # Generate video using async executor to avoid blocking
            loop = asyncio.get_event_loop()
            
            def _generate_video_sync():
                """Synchronous video generation to run in executor"""
                try:
                    logging.info("Starting Veo 3.0 video generation operation")
                    
                    # Create the video generation operation
                    operation = self.genai_client.models.generate_videos(
                        model=self.video_model_name,
                        prompt=enhanced_prompt,
                        config=types.GenerateVideosConfig(
                            person_generation="allow_all",
                            aspect_ratio="16:9",
                        ),
                    )
                    
                    logging.info(f"Video generation operation started: {operation.name}")
                    
                    # Poll for completion with timeout
                    max_wait_time = 300  # 5 minutes timeout
                    start_time = time.time()
                    
                    while not operation.done:
                        if time.time() - start_time > max_wait_time:
                            raise TimeoutError("Video generation timed out after 5 minutes")
                        
                        logging.info("Waiting for video generation to complete...")
                        time.sleep(10)  # Check every 10 seconds
                        
                        # Refresh operation status
                        operation = self.genai_client.operations.get(operation)
                        
                        if operation.error:
                            raise Exception(f"Video generation failed: {operation.error}")
                    
                    logging.info("Video generation completed successfully")
                    
                    # Extract the first generated video
                    if not operation.response.generated_videos:
                        raise Exception("No videos were generated")
                    
                    generated_video = operation.response.generated_videos[0]
                    
                    # Download the video as bytes
                    video_file = generated_video.video
                    video_bytes = self.genai_client.files.download(file=video_file)
                    
                    # Create video info
                    video_info = {
                        "original_prompt": prompt,
                        "enhanced_prompt": enhanced_prompt,
                        "duration": duration,
                        "quality": quality,
                        "fps": fps,
                        "model": self.video_model_name,
                        "timestamp": time.time(),
                        "operation_name": operation.name,
                        "video_size_bytes": len(video_bytes),
                        "aspect_ratio": "16:9"
                    }
                    
                    logging.info(f"Video generation successful. Size: {len(video_bytes)} bytes")
                    return video_bytes, video_info
                    
                except Exception as e:
                    logging.error(f"Synchronous video generation failed: {str(e)}")
                    raise e
            
            # Run the synchronous video generation in a thread executor
            video_bytes, video_info = await loop.run_in_executor(None, _generate_video_sync)
            
            return video_bytes, str(video_info)
            
        except CustomException as e:
            logging.error(f"Error in async video generation: {str(e)}")
            raise CustomException(e, sys) from e
        except Exception as e:
            logging.error(f"Unexpected error in video generation: {str(e)}")
            raise CustomException(e, sys) from e
        
    def save_video_bytes_to_file(self, video_bytes: bytes, filename: str = None) -> str:
        """
        Save video bytes to a file for testing/debugging purposes.
        
        Args:
            video_bytes: The video data as bytes
            filename: Optional filename, if not provided, creates a temp file
            
        Returns:
            The path to the saved file
        """
        try:
            if filename:
                filepath = filename
            else:
                # Create a temporary file
                with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_file:
                    filepath = temp_file.name
            
            with open(filepath, 'wb') as f:
                f.write(video_bytes)
            
            logging.info(f"Video saved to: {filepath}")
            return filepath
            
        except Exception as e:
            logging.error(f"Error saving video bytes: {str(e)}")
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
            print(f"Vide bytes : {video_bytes}")
            print(f"Async Video bytes length: {len(video_bytes)}")
            print(f"Video info: {video_info}")
        except Exception as e:
            print(f"Async video generation error: {e}")

    asyncio.run(test_refactored())

