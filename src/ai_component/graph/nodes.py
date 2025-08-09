import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

import io
import wave
import base64
import asyncio
from cartesia import Cartesia
from datetime import datetime
from src.ai_component.graph.utils.chains import async_router_chain
from src.ai_component.llm import LLMChainFactory
from src.ai_component.modules.schedule.context_generation import ScheduleContextGenerator
from src.ai_component.graph.state import AICompanionState
from src.ai_component.core.prompts import general_template, disease_template, weather_template, mandi_template, image_template, gov_scheme_template
from src.ai_component.tools.web_seach_tool import web_tool
from src.ai_component.tools.rag_tool import rag_tool
from src.ai_component.tools.gov_scheme_tool import gov_scheme_tool
from src.ai_component.tools.weather_tool import weather_forecast_tool, weather_report_tool
from src.ai_component.tools.mandi_report_tool import mandi_report_tool
from src.ai_component.tools.call_tool import call_tool
from src.ai_component.modules.memory.memory_manager import memory_manager
from src.ai_component.modules.memory.vector_store import memory
from src.ai_component.logger import logging
from src.ai_component.exception import CustomException
from Database.database import user_db
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

async def UserNode(state: AICompanionState) -> dict:
    """
    Just check user and store their data if collection is new.
    Only stores user profile once to avoid duplicates.
    """
    try:
        logging.info("Calling Simple User Node")
        
        user_unique_name = state['collection_name']
        if not user_unique_name:
            return {"messages": state["messages"], "error": "No user provided"}
        
        if not user_db.user_exists(user_unique_name):  ## check if user exists in database
            return {"messages": state["messages"], "error": "User not found"}
        
        user_data = user_db.get_user_by_unique_name(user_unique_name)  ## get user data
        if not user_data:
            return {"messages": state["messages"], "error": "Could not retrieve user data"}
        
        collection_created = memory.create_collection(collection_name=user_unique_name) ## create collection name
        
        # Check if user profile already exists in vector database
        existing_profile = memory.search_in_collection(
            query="user profile information name age location", 
            collection_name=user_unique_name, 
            k=1
        )
        
        # Only store user profile if it doesn't exist
        if not existing_profile or len(existing_profile) == 0:
            user_info = f"""
            User: {user_data.name} ({user_data.unique_name})
            Age: {user_data.age}
            Location: {user_data.city or ''}, {user_data.district}, {user_data.state}, {user_data.country}
            Address: {user_data.resident or 'Not provided'}
            """.strip()
            
            memory.ingest_data(
                collection_name=user_unique_name,
                data=user_info,
                additional_metadata={"type": "user_profile", "user_id": user_data.id}
            )
            logging.info(f"User data stored for {user_unique_name}")
        else:
            logging.info(f"User profile already exists for {user_unique_name}, skipping storage")
        
        return {"messages": state["messages"]}  
        
    except CustomException as e:
        logging.error(f"Error in user node : {str(e)}")
        raise CustomException(e, sys) from e
    except Exception as e:
        logging.error(f"Error in SimpleUserNode: {str(e)}")
        return {"messages": state["messages"], "error": str(e)}


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
            "messages": state["messages"]  # Keep original messages list
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
            template=general_template
        )
        
        factory = LLMChainFactory(model_type="groq")
        chain = await factory.get_llm_tool_chain(prompt, [rag_tool, call_tool])
        
        response = await chain.ainvoke({
            "history": history_text,
            "current_activity": state.get("current_activity", ""),
            "query": query
        })
        if hasattr(response, 'tool_calls') and response.tool_calls:
            return {"messages": [response]}
        else:
            return {"messages": [AIMessage(content=response.content)]}
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
            query = next((m.content for m in reversed(messages) if isinstance(m, HumanMessage)), "")
            tool_results = "\n".join(f"Tool: {m.name}\nResult: {m.content}" for m in messages if isinstance(m, ToolMessage))
            enhanced = f"{disease_template}\nOriginal Query: {{query}}\nTool Results:\n{{tool_results}}"
            prompt = PromptTemplate(input_variables=["query", "tool_results"], template=enhanced)
            factory = LLMChainFactory(model_type="gemini")
            chain = await factory.get_llm_chain_async(prompt)
            resp = await chain.ainvoke({"query": query, "tool_results": tool_results})
            return {"messages": [AIMessage(content=resp.content)]}
        
        # otherwise, call tools as needed
        query = last.content
        prompt = PromptTemplate(input_variables=["history", "query"], template=disease_template)
        factory = LLMChainFactory(model_type="groq")
        chain = await factory.get_llm_tool_chain(prompt, [web_tool, rag_tool])
        resp = await chain.ainvoke({"history": history_text, "query": query})
        
        if hasattr(resp, 'tool_calls') and resp.tool_calls:
            return {"messages": [resp]}  # Tool call message
        return {"messages": [AIMessage(content=resp.content)]}
        
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
        
        # Handle tool result branch
        if isinstance(last, ToolMessage):
            query = next((m.content for m in reversed(messages) if isinstance(m, HumanMessage)), "")
            tool_results = "\n".join(m.content for m in messages if isinstance(m, ToolMessage))
            
            # Enhanced template for processing tool results
            enhanced_template = f"""{weather_template}

Original Query: {{query}}
Weather Tool Results:
{{tool_results}}

Based on the weather data above, provide a comprehensive weather report. Include current conditions, temperature, humidity, wind speed, and any other relevant information. Format the response in a clear, detailed text format that can be easily understood."""
            
            prompt = PromptTemplate(
                input_variables=["date", "query", "tool_results"], 
                template=enhanced_template
            )
            chain = await LLMChainFactory(model_type="gemini").get_llm_chain_async(prompt)
            resp = await chain.ainvoke({
                "date": datetime.now().strftime("%Y-%m-%d"),
                "query": query,
                "tool_results": tool_results
            })
            return {"messages": [AIMessage(content=resp.content)]}
        
        # Handle initial query branch
        query = last.content
        prompt = PromptTemplate(
            input_variables=["date", "query"], 
            template=weather_template
        )
        
        # Use tool chain with weather tools
        chain = await LLMChainFactory(model_type="gemini").get_llm_tool_chain(
            prompt, 
            [weather_forecast_tool, weather_report_tool]
        )
        
        resp = await chain.ainvoke({
            "date": datetime.now().strftime("%Y-%m-%d"),
            "query": query
        })
        
        # Check if response has tool calls
        if hasattr(resp, 'tool_calls') and resp.tool_calls:
            return {"messages": [resp]}  # Tool call message
        
        return {"messages": [AIMessage(content=resp.content)]}
        
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
        
        if isinstance(last, ToolMessage):
            query = next((m.content for m in reversed(messages) if isinstance(m, HumanMessage)), "")
            tool_results = "\n".join(m.content for m in messages if isinstance(m, ToolMessage))
            
            enhanced_template = f"""{mandi_template}

Original Query: {{query}}
Tool Results:
{{tool_results}}

Based on the mandi data above, provide a comprehensive market analysis with proper formatting and emojis for better readability. Include price statistics, trends, and forecasts as applicable."""
            
            prompt = PromptTemplate(
                input_variables=["date", "query", "tool_results"], 
                template=enhanced_template
            )
            chain = await LLMChainFactory(model_type="gemini").get_llm_chain_async(prompt)
            resp = await chain.ainvoke({
                "date": datetime.now().strftime("%Y-%m-%d"),
                "query": query,
                "tool_results": tool_results
            })
            return {"messages": [AIMessage(content=resp.content)]}
        
        # Handle initial query case
        query = last.content
        prompt = PromptTemplate(
            input_variables=["date", "query"], 
            template=mandi_template
        )
        
        chain = await LLMChainFactory(model_type="groq").get_llm_tool_chain(
            prompt, 
            [mandi_report_tool]
        )
        
        resp = await chain.ainvoke({
            "date": datetime.now().strftime("%Y-%m-%d"),
            "query": query
        })
        if hasattr(resp, 'tool_calls') and resp.tool_calls:
            return {"messages": [resp]}  # Tool call message
        
        return {"messages": [AIMessage(content=resp.content)]}
        
    except CustomException as e:
        logging.error(f"Error in MandiNode: {e}")
        raise CustomException(e, sys) from e


async def GovSchemeNode(state: AICompanionState) -> dict:
    """
    Handle government scheme requests, using gov_scheme_tool and web_tool, preserving history.
    """
    try:
        logging.info("Calling Gov Scheme Node")
        messages = state["messages"]
        last = messages[-1]
        
        # Handle tool result branch
        if isinstance(last, ToolMessage):
            query = next((m.content for m in reversed(messages) if isinstance(m, HumanMessage)), "")
            tool_results = "\n".join(m.content for m in messages if isinstance(m, ToolMessage))
            
            # Enhanced template for processing tool results
            enhanced_template = f"""{gov_scheme_template}

Original Query: {{query}}
Government Scheme Tool Results:
{{tool_results}}

Based on the government scheme data above, provide a comprehensive response about the relevant government schemes for farmers. Include scheme details, eligibility criteria, benefits, and application process where available. Format the response in a clear, detailed text format that can be easily understood."""
            
            prompt = PromptTemplate(
                input_variables=["date", "query", "tool_results"], 
                template=enhanced_template
            )
            chain = await LLMChainFactory(model_type="gemini").get_llm_chain_async(prompt)
            resp = await chain.ainvoke({
                "date": datetime.now().strftime("%Y-%m-%d"),
                "query": query,
                "tool_results": tool_results
            })
            return {"messages": [AIMessage(content=resp.content)]}
        
        # Handle initial query branch
        query = last.content
        prompt = PromptTemplate(
            input_variables=["date", "query"], 
            template=gov_scheme_template
        )
        chain = await LLMChainFactory(model_type="gemini").get_llm_tool_chain(
            prompt, 
            [gov_scheme_tool, web_tool]
        )
        
        resp = await chain.ainvoke({
            "date": datetime.now().strftime("%Y-%m-%d"),
            "query": query
        })
        if hasattr(resp, 'tool_calls') and resp.tool_calls:
            return {"messages": [resp]}  # Tool call message
        
        return {"messages": [AIMessage(content=resp.content)]}
        
    except CustomException as e:
        logging.error(f"Error in GovSchemeNode: {e}")
        raise CustomException(e, sys) from e


async def CarbonFootprintNode(state: AICompanionState) -> dict:
    """
    Calculate/return carbon footprint info, appending to history.
    """
    try:
        logging.info("Calling CarbonFootprintNode")
        response = "Total carbon footprint generated: ..."
        return {"messages": [AIMessage(content=response)]}
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
        
        return {}  # Don't return messages - let state pass through
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
        prompt = PromptTemplate(input_variables=["text"], template=image_template)
        factory = LLMChainFactory(model_type="gemini")
        chain = await factory.get_llm_chain_async(prompt)
        img_prompt = (await chain.ainvoke({"text": query})).content
        loop = asyncio.get_event_loop()
        img_bytes = await loop.run_in_executor(None, lambda: factory.get_image_model(img_prompt))
        return {"image": img_bytes}  
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

        return {"voice": audio_base64}  # Don't modify messages
    except CustomException as e:
        logging.error(f"Error in VoiceNode: {e}")
        raise CustomException(e, sys) from e


async def TextNode(state: AICompanionState) -> dict:
    """
    Pass-through final text output.
    """
    return {}