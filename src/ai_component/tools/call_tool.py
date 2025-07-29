import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

import asyncio
import aiohttp
import requests
from pydantic import BaseModel, Field
from typing import Any, Optional, List, Type
from langchain.tools import BaseTool
from src.ai_component.logger import logging
from src.ai_component.exception import CustomException

from dotenv import load_dotenv
load_dotenv()

api_key = os.getenv("BLAND_API_KEY")

class InputSchema(BaseModel):
    phone_number: str = Field(..., description="Valid phone number of user with country code", examples=["+914567345893"])
    instructions: str = Field(..., description="message to send the other receiver on the other side of the call")

class CallTool(BaseTool):
    """
    Call the other farmer to collaborate with other farmers 
    """
    name: str = "call_tool"
    description: str = "Phone call to other farmers as per request by farmer"
    args_schema: Type[InputSchema] = InputSchema

    async def _arun(self, phone_number: str, instructions: str):
        """
        makes a confirmation call using the Bland.ai API (async version)

        Parameters:
            phone_number (str): the phone number to call
            instructions (str): the instructions to send with the call

        Returns:
            dict: the response from the API
        """
        try:
            logging.info(f"Running Call tool for phone number: {phone_number}")
            
            if not api_key:
                raise ValueError("BLAND_API_KEY is not set in environment variables.")
            
            url = "https://api.bland.ai/v1/calls"
            payload = {
                "phone_number": phone_number,
                "voice": "Alena",
                "wait_for_greeting": False,
                "record": True,
                "answered_by_enabled": True,
                "noise_cancellation": False,
                "interruption_threshold": 100,
                "block_interruptions": False,
                "max_duration": 3,
                "model": "base",
                "language": "hi",
                "background_track": "none",
                "endpoint": "https://api.bland.ai",
                "voicemail_action": "hangup",
                "first_sentence": "Namaste ",
                "task": instructions
            }
            headers = {
                "authorization": api_key,
                "Content-Type": "application/json"
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data
                    else:
                        error_msg = f"API request failed with status {response.status}"
                        logging.error(error_msg)
                        return f"Error: {error_msg}"
        except Exception as e:
            logging.error(f"Error in Call tool: {str(e)}")
            return f"Error making call: {str(e)}"

    def _run(self, phone_number: str, instructions: str):
        """
        makes a confirmation call using the Bland.ai API (sync version)

        Parameters:
            phone_number (str): the phone number to call
            instructions (str): the instructions to send with the call

        Returns:
            dict: the response from the API
        """
        try:
            logging.info(f"Running Call tool for phone number: {phone_number}")
            
            if not api_key:
                raise ValueError("BLAND_API_KEY is not set in environment variables.")
            
            url = "https://api.bland.ai/v1/calls"
            payload = {
                "phone_number": phone_number,
                "voice": "Alena",
                "wait_for_greeting": False,
                "record": True,
                "answered_by_enabled": True,
                "noise_cancellation": False,
                "interruption_threshold": 100,
                "block_interruptions": False,
                "max_duration": 3,
                "model": "base",
                "language": "hi",
                "background_track": "none",
                "endpoint": "https://api.bland.ai",
                "voicemail_action": "hangup",
                "first_sentence": "Namaste ",
                "task": instructions
            }
            headers = {
                "authorization": api_key,
                "Content-Type": "application/json"
            }
            
            response = requests.post(url, json=payload, headers=headers)
            if response.status_code == 200:
                data = response.json()
                return data
            else:
                error_msg = f"API request failed with status {response.status_code}"
                logging.error(error_msg)
                return f"Error: {error_msg}"
        except Exception as e:
            logging.error(f"Error in Call tool: {str(e)}")
            return f"Error making call: {str(e)}"

call_tool = CallTool()


async def test_async():
    phone = "+918090175358"
    instructions = "say he is too good guy"
    result = await call_tool._arun(phone_number=phone, instructions=instructions)
    print("Result:", result)


if __name__ == "__main__":
    asyncio.run(test_async())
    