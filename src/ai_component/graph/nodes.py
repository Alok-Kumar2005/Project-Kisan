import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

import io
import wave
import struct
import base64
import asyncio
import tempfile
from cartesia import Cartesia
from datetime import datetime
from src.ai_component.graph.utils.chains import async_router_chain
from src.ai_component.llm import LLMChainFactory
from src.ai_component.modules.schedule.context_generation import ScheduleContextGenerator
from src.ai_component.graph.state import AICompanionState
from src.ai_component.core.prompts import general_template, disease_template, weather_template, mandi_template, image_template
from src.ai_component.tools.web_seach_tool import web_tool
from src.ai_component.tools.rag_tool import rag_tool
from src.ai_component.tools.weather_tool import weather_forecast_tool, weather_report_tool
from src.ai_component.tools.mandi_report_tool import mandi_report_tool
from src.ai_component.modules.memory.memory_manager import memory_manager
from src.ai_component.logger import logging
from src.ai_component.exception import CustomException
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

cartesia_client = None
try:
    cartesia_api_key = os.getenv('CARTESIA_API_KEY')
    if cartesia_api_key:
        cartesia_client = Cartesia(api_key=cartesia_api_key)
except Exception as e:
    logging.error(f"Failed to initialize Cartesia client: {str(e)}")

async def route_node(state: AICompanionState) -> dict:
    """
    Route the conversation based on the user's query, preserving full history.
    """
    try:
        logging.info("Calling Route Node")
        query = state["messages"][-1].content if state["messages"] else ""
        # default route
        workflow = "GeneralNode"
        output = None
        if query:
            chain = await async_router_chain()
            response = await chain.ainvoke({"query": query})
            workflow = response.route_node
            output = response.output
            logging.info(f"Route Node selected: {workflow}")
        return {
            "workflow": workflow,
            "output": output,
            "messages": state["messages"]
        }
    except CustomException as e:
        logging.error(f"Error in route_node: {e}")
        raise CustomException(e, sys) from e

async def context_injestion_node(state: AICompanionState) -> dict:
    """
    Inject current activity context while preserving history.
    """
    try:
        logging.info("Calling Context Ingestion Node")
        activity = ScheduleContextGenerator.get_current_activity() or "No scheduled activity."
        logging.info(f"Current activity: {activity}")
        return {
            "current_activity": activity,
            "messages": state["messages"]
        }
    except CustomException as e:
        logging.error(f"Error in context_ingestion_node: {e}")
        raise CustomException(e, sys) from e


async def GeneralNode(state: AICompanionState) -> dict:
    """
    General fallback node: appends AI response to conversation history.
    """
    try:
        logging.info("Calling General Node")
        query = state["messages"][-1].content
        history_text = "\n".join(m.content for m in state["messages"])
        prompt = PromptTemplate(
            input_variables=["history", "current_activity", "query"],
            template=general_template.prompt
        )
        factory = LLMChainFactory(model_type="gemini")
        chain = await factory.get_llm_chain_async(prompt)
        response = await chain.ainvoke({
            "history": history_text,
            "current_activity": state.get("current_activity", ""),
            "query": query
        })
        logging.info("GeneralNode got response")
        return {"messages": state["messages"] + [AIMessage(content=response.content)]}
    except CustomException as e:
        logging.error(f"Error in GeneralNode: {e}")
        raise CustomException(e, sys) from e


async def DiseaseNode(state: AICompanionState) -> dict:
    """
    Handle plant disease queries, including tool calls, appending results.
    """
    try:
        logging.info("Calling Disease Node")
        messages = state["messages"]
        last = messages[-1]
        history_text = "\n".join(m.content for m in messages)
        # if last message is a tool result, generate final report
        if isinstance(last, ToolMessage):
            # find original query
            query = next((m.content for m in reversed(messages) if isinstance(m, HumanMessage)), "")
            tool_results = "\n".join(f"Tool: {m.name}\nResult: {m.content}" for m in messages if isinstance(m, ToolMessage))
            enhanced = f"{disease_template}\nOriginal Query: {{query}}\nTool Results:\n{{tool_results}}"
            prompt = PromptTemplate(input_variables=["query", "tool_results"], template=enhanced)
            factory = LLMChainFactory(model_type="gemini")
            chain = await factory.get_llm_chain_async(prompt)
            resp = await chain.ainvoke({"query": query, "tool_results": tool_results})
            return {"messages": messages + [AIMessage(content=resp.content)]}
        # otherwise, call tools as needed
        query = last.content
        prompt = PromptTemplate(input_variables=["history", "query"], template=disease_template.prompt)
        factory = LLMChainFactory(model_type="gemini")
        chain = await factory.get_llm_tool_chain(prompt, [web_tool, rag_tool])
        resp = await chain.ainvoke({"history": history_text, "query": query})
        if hasattr(resp, 'tool_calls') and resp.tool_calls:
            return {"messages": messages + [resp]}
        return {"messages": messages + [AIMessage(content=resp.content)]}
    except CustomException as e:
        logging.error(f"Error in DiseaseNode: {e}")
        raise CustomException(e, sys) from e


async def WeatherNode(state: AICompanionState) -> dict:
    """
    Handle weather requests, using web or forecast tools, preserving history.
    """
    try:
        logging.info("Calling Weather Node")
        messages = state["messages"]
        last = messages[-1]
        history_text = "\n".join(m.content for m in messages)
        # tool result branch
        if isinstance(last, ToolMessage):
            query = next((m.content for m in reversed(messages) if isinstance(m, HumanMessage)), "")
            tool_results = "\n".join(m.content for m in messages if isinstance(m, ToolMessage))
            enhanced = f"{weather_template}\nOriginal Query: {{query}}\nResults:\n{{tool_results}}"
            prompt = PromptTemplate(input_variables=["date","query","tool_results"], template=enhanced)
            chain = await LLMChainFactory(model_type="gemini").get_llm_chain_async(prompt)
            resp = await chain.ainvoke({
                "date": datetime.now().strftime("%Y-%m-%d"),
                "query": query,
                "tool_results": tool_results
            })
            return {"messages": messages + [AIMessage(content=resp.content)]}
        # initial branch
        query = last.content
        prompt = PromptTemplate(input_variables=["date","history","query"], template=weather_template.prompt)
        chain = await LLMChainFactory(model_type="gemini").get_llm_tool_chain(prompt, [weather_forecast_tool, weather_report_tool])
        resp = await chain.ainvoke({
            "date": datetime.now().strftime("%Y-%m-%d"),
            "history": history_text,
            "query": query
        })
        if hasattr(resp, 'tool_calls') and resp.tool_calls:
            return {"messages": messages + [resp]}
        return {"messages": messages + [AIMessage(content=resp.content)]}
    except CustomException as e:
        logging.error(f"Error in WeatherNode: {e}")
        raise CustomException(e, sys) from e


async def MandiNode(state: AICompanionState) -> dict:
    """
    Provide mandi reports via tool, preserving history.
    """
    try:
        logging.info("Calling Mandi Node")
        messages = state["messages"]
        last = messages[-1]
        history_text = "\n".join(m.content for m in messages)
        if isinstance(last, ToolMessage):
            query = next((m.content for m in reversed(messages) if isinstance(m, HumanMessage)), "")
            tool_results = "\n".join(m.content for m in messages if isinstance(m, ToolMessage))
            enhanced = f"{mandi_template}\nOriginal Query: {{query}}\nResults:\n{{tool_results}}"
            prompt = PromptTemplate(input_variables=["date","query","tool_results"], template=enhanced)
            chain = await LLMChainFactory(model_type="gemini").get_llm_chain_async(prompt)
            resp = await chain.ainvoke({
                "date": datetime.now().strftime("%Y-%m-%d"),
                "query": query,
                "tool_results": tool_results
            })
            return {"messages": messages + [AIMessage(content=resp.content)]}
        query = last.content
        prompt = PromptTemplate(input_variables=["date","history","query"], template=mandi_template.prompt)
        chain = await LLMChainFactory(model_type="gemini").get_llm_tool_chain(prompt, [mandi_report_tool])
        resp = await chain.ainvoke({
            "date": datetime.now().strftime("%Y-%m-%d"),
            "history": history_text,
            "query": query
        })
        if hasattr(resp, 'tool_calls') and resp.tool_calls:
            return {"messages": messages + [resp]}
        return {"messages": messages + [AIMessage(content=resp.content)]}
    except CustomException as e:
        logging.error(f"Error in MandiNode: {e}")
        raise CustomException(e, sys) from e


async def GovSchemeNode(state: AICompanionState) -> dict:
    """
    Return government schemes, appending to history.
    """
    try:
        logging.info("Calling GovSchemeNode")
        messages = state["messages"]
        # For now static response; can be expanded with tools
        response = "Here are the top government schemes for farmers: ..."
        return {"messages": messages + [AIMessage(content=response)]}
    except CustomException as e:
        logging.error(f"Error in GovSchemeNode: {e}")
        raise CustomException(e, sys) from e


async def CarbonFootprintNode(state: AICompanionState) -> dict:
    """
    Calculate/return carbon footprint info, appending to history.
    """
    try:
        logging.info("Calling CarbonFootprintNode")
        messages = state["messages"]
        response = "Total carbon footprint generated: ..."
        return {"messages": messages + [AIMessage(content=response)]}
    except CustomException as e:
        logging.error(f"Error in CarbonFootprintNode: {e}")
        raise CustomException(e, sys) from e


async def MemoryIngestionNode(state: AICompanionState) -> dict:
    """
    Persist only the last user query and final LLM response to memory.
    """
    try:
        logging.info("Memory Ingestion Node -----------")
        messages = state["messages"]
        ### finding last user query
        last_user_message = None
        for message in reversed(messages):
            if isinstance(message, HumanMessage):
                last_user_message = message.content
                break
        ### finding last AI Message
        last_ai_message = None
        for message in reversed(messages):
            if isinstance(message, AIMessage):
                last_ai_message = message.content
                break
        
        ### only store if we get both
        if last_user_message and last_ai_message:
            conversation = f"User: {last_user_message}\nAI: {last_ai_message}"
            
            # Store in memory with the conversation content
            await memory_manager.store_in_memory(state["collection_name"], conversation)
        else:
            logging.info("No valid query-response pair found to store")
        
        return {}
    except CustomException as e:
        logging.error(f"Error in Memory Ingestion Node: {str(e)}")
        raise CustomException(e, sys) from e


async def ImageNode(state: AICompanionState) -> dict:
    """
    Generate image from last user prompt, preserving history.
    """
    try:
        logging.info("Calling ImageNode")
        messages = state["messages"]
        query = messages[-1].content
        prompt = PromptTemplate(input_variables=["text"], template=image_template.prompt)
        factory = LLMChainFactory(model_type="gemini")
        chain = await factory.get_llm_chain_async(prompt)
        img_prompt = (await chain.ainvoke({"text": query})).content
        loop = asyncio.get_event_loop()
        img_bytes = await loop.run_in_executor(None, lambda: factory.get_image_model(img_prompt))
        return {"messages": messages, "image": img_bytes}
    except CustomException as e:
        logging.error(f"Error in ImageNode: {e}")
        raise CustomException(e, sys) from e


async def VoiceNode(state: AICompanionState) -> dict:
    """
    Synthesize voice from last AI response, preserving history.
    """
    try:
        logging.info("Calling VoiceNode")
        messages = state["messages"]
        response_text = messages[-1].content
        if not response_text:
            return {"audio": ""}
        
        if not cartesia_client:
            logging.error("Cartesia client not initialized")
            return {"audio": ""}
        voice_id = "ef8390dc-0fc0-473b-bbc0-7277503793f7"

        audio_generator = cartesia_client.tts.bytes(
            model_id="sonic",
            transcript=response_text,
            voice={
                "mode": "id",
                "id": voice_id
            },
            language="en",
            output_format={
                "container": "raw",
                "sample_rate": 16000,
                "encoding": "pcm_f32le"
            }
        )
        # Collect all audio chunks
        audio_chunks = []
        for chunk in audio_generator:
            audio_chunks.append(chunk)
        
        # Combine all chunks into a single bytes object
        audio_data = b''.join(audio_chunks)

        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, 'wb') as wav_file:
            wav_file.setnchannels(1) 
            wav_file.setsampwidth(4)  
            wav_file.setframerate(16000)
            wav_file.writeframes(audio_data)
        
        wav_buffer.seek(0)
        wav_bytes = wav_buffer.read()
        audio_base64 = base64.b64encode(wav_bytes).decode('utf-8')

        return {"messages": response_text, "voice": audio_base64}
    except CustomException as e:
        logging.error(f"Error in VoiceNode: {e}")
        raise CustomException(e, sys) from e


async def TextNode(state: AICompanionState) -> dict:
    """
    Pass-through final text output.
    """
    return {"messages": state["messages"]}

async def VideoNode(state: AICompanionState) -> dict:
    """
    Generate the video for farmer using the fixed implementation.
    """
    try:
        logging.info("Calling VideoNode with fixed implementation")
        messages = state["messages"]
        message = messages[-1].content

        factory = LLMChainFactory(model_type="gemini")
        
        # Generate video with proper error handling
        video_bytes, video_info = await factory.get_video_model_async(
            prompt=message,
            duration=15,
            quality="720p"
        )
        
        logging.info(f"VideoNode generated video: {len(video_bytes)} bytes")
        
        return {
            "messages": message,
            "video": video_bytes,
            "video_info": video_info
        }
        
    except CustomException as e:
        logging.error(f"Error in VideoNode: {str(e)}")
        # Return fallback response instead of failing completely
        return {
            "messages": f"Sorry, video generation failed: {str(e)}",
            "video": b"",  # Empty bytes as fallback
            "video_info": "Video generation failed"
        }
    except Exception as e:
        logging.error(f"Unexpected error in VideoNode: {str(e)}")
        raise CustomException(e, sys) from e