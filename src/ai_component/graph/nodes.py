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
from src.ai_component.core.prompts import Template
from src.ai_component.tools.all_tools import Tools
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


class Nodes:
    @staticmethod
    async def route_node(state: AICompanionState) -> dict:
        try:
            logging.info("Calling Route Node")
            query = state["messages"][-1].content if state["messages"] else ""
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

    @staticmethod
    async def UserNode(state: AICompanionState) -> dict:
        try:
            logging.info("Calling Simple User Node")
            user_unique_name = state['collection_name']
            if not user_unique_name:
                return {"messages": state["messages"], "error": "No user provided"}
            if not user_db.user_exists(user_unique_name):
                return {"messages": state["messages"], "error": "User not found"}
            user_data = user_db.get_user_by_unique_name(user_unique_name)
            if not user_data:
                return {"messages": state["messages"], "error": "Could not retrieve user data"}
            memory.create_collection(collection_name=user_unique_name)
            existing_profile = memory.search_in_collection(
                query="user profile information name age location",
                collection_name=user_unique_name,
                k=1
            )
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

    @staticmethod
    async def context_injestion_node(state: AICompanionState) -> dict:
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

    @staticmethod
    async def GeneralNode(state: AICompanionState) -> dict:
        try:
            logging.info("Calling General Node")
            messages = state["messages"]
            last_message = messages[-1]
            if isinstance(last_message, ToolMessage):
                query = next((m.content for m in reversed(messages) if isinstance(m, HumanMessage)), "")
                tool_results = "\n".join(
                    f"Tool: {m.name}\nResult: {m.content}" 
                    for m in messages if isinstance(m, ToolMessage)
                )
                enhanced_template = f"""{Template.general_template}

    Original Query: {{query}}

    Tool Results:
    {{tool_results}}

    Based on the tool results above, provide a comprehensive and helpful response to the user's query.
    Do NOT make any more tool calls. Use only the information from the tool results."""
                
                prompt = PromptTemplate(
                    input_variables=["history", "current_activity", "query", "tool_results"],
                    template=enhanced_template
                )
                
                factory = LLMChainFactory(model_type="groq")
                chain = await factory.get_llm_chain_async(prompt)
                
                history_text = "\n".join(m.content for m in messages[:-len([m for m in messages if isinstance(m, ToolMessage)])])
                
                response = await chain.ainvoke({
                    "history": history_text,
                    "current_activity": state.get("current_activity", ""),
                    "query": query,
                    "tool_results": tool_results
                })
                
                return {"messages": [AIMessage(content=response.content)]}
            query = last_message.content
            history_text = "\n".join(m.content for m in messages)
            
            prompt = PromptTemplate(
                input_variables=["history", "current_activity", "query"],
                template=Template.general_template
            )
            
            factory = LLMChainFactory(model_type="groq")
            chain = await factory.get_llm_tool_chain(prompt, [Tools.rag_tool, Tools.call_tool])
            
            response = await chain.ainvoke({
                "history": history_text,
                "current_activity": state.get("current_activity", ""),
                "query": query
            })
            
            # Return the response (either with tool_calls or final answer)
            if hasattr(response, 'tool_calls') and response.tool_calls:
                return {"messages": [response]}
            else:
                return {"messages": [AIMessage(content=response.content)]}
                
        except CustomException as e:
            logging.error(f"Error in GeneralNode: {e}")
            raise CustomException(e, sys) from e

    @staticmethod
    async def DiseaseNode(state: AICompanionState) -> dict:
        try:
            logging.info("Calling Disease Node")
            messages = state["messages"]
            last = messages[-1]
            history_text = "\n".join(m.content for m in messages)
            if isinstance(last, ToolMessage):
                query = next((m.content for m in reversed(messages) if isinstance(m, HumanMessage)), "")
                tool_results = "\n".join(f"Tool: {m.name}\nResult: {m.content}" for m in messages if isinstance(m, ToolMessage))
                enhanced = f"{Template.disease_template}\nOriginal Query: {{query}}\nTool Results:\n{{tool_results}}"
                prompt = PromptTemplate(input_variables=["query", "tool_results"], template=enhanced)
                factory = LLMChainFactory(model_type="gemini")
                chain = await factory.get_llm_chain_async(prompt)
                resp = await chain.ainvoke({"query": query, "tool_results": tool_results})
                return {"messages": [AIMessage(content=resp.content)]}
            query = last.content
            prompt = PromptTemplate(input_variables=["history", "query"], template=Template.disease_template)
            factory = LLMChainFactory(model_type="groq")
            chain = await factory.get_llm_tool_chain(prompt, [Tools.web_tool, Tools.rag_tool])
            resp = await chain.ainvoke({"history": history_text, "query": query})
            if hasattr(resp, 'tool_calls') and resp.tool_calls:
                return {"messages": [resp]}
            return {"messages": [AIMessage(content=resp.content)]}
        except CustomException as e:
            logging.error(f"Error in DiseaseNode: {e}")
            raise CustomException(e, sys) from e

    @staticmethod
    async def WeatherNode(state: AICompanionState) -> dict:
        try:
            logging.info("Calling Weather Node")
            messages = state["messages"]
            last = messages[-1]
            if isinstance(last, ToolMessage):
                query = next((m.content for m in reversed(messages) if isinstance(m, HumanMessage)), "")
                tool_results = "\n".join(m.content for m in messages if isinstance(m, ToolMessage))
                enhanced_template = f"""{Template.weather_template}
Original Query: {{query}}
Weather Tool Results:
{{tool_results}}
Based on the weather data above, provide a comprehensive weather report."""
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
            query = last.content
            prompt = PromptTemplate(input_variables=["date", "query"], template=Template.weather_template)
            chain = await LLMChainFactory(model_type="gemini").get_llm_tool_chain(
                prompt, [Tools.weather_forecast_tool, Tools.weather_report_tool]
            )
            resp = await chain.ainvoke({
                "date": datetime.now().strftime("%Y-%m-%d"),
                "query": query
            })
            if hasattr(resp, 'tool_calls') and resp.tool_calls:
                return {"messages": [resp]}
            return {"messages": [AIMessage(content=resp.content)]}
        except CustomException as e:
            logging.error(f"Error in WeatherNode: {e}")
            raise CustomException(e, sys) from e

    @staticmethod
    async def MandiNode(state: AICompanionState) -> dict:
        try:
            logging.info("Calling Mandi Node")
            messages = state["messages"]
            last = messages[-1]
            if isinstance(last, ToolMessage):
                query = next((m.content for m in reversed(messages) if isinstance(m, HumanMessage)), "")
                tool_results = "\n".join(m.content for m in messages if isinstance(m, ToolMessage))
                enhanced_template = f"""{Template.mandi_template}
Original Query: {{query}}
Tool Results:
{{tool_results}}
Based on the mandi data above, provide a comprehensive market analysis."""
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
            query = last.content
            prompt = PromptTemplate(input_variables=["date", "query"], template=Template.mandi_template)
            chain = await LLMChainFactory(model_type="groq").get_llm_tool_chain(
                prompt, [Tools.mandi_report_tool]
            )
            resp = await chain.ainvoke({
                "date": datetime.now().strftime("%Y-%m-%d"),
                "query": query
            })
            if hasattr(resp, 'tool_calls') and resp.tool_calls:
                return {"messages": [resp]}
            return {"messages": [AIMessage(content=resp.content)]}
        except CustomException as e:
            logging.error(f"Error in MandiNode: {e}")
            raise CustomException(e, sys) from e

    @staticmethod
    async def GovSchemeNode(state: AICompanionState) -> dict:
        try:
            logging.info("Calling Gov Scheme Node")
            messages = state["messages"]
            last = messages[-1]
            if isinstance(last, ToolMessage):
                query = next((m.content for m in reversed(messages) if isinstance(m, HumanMessage)), "")
                tool_results = "\n".join(m.content for m in messages if isinstance(m, ToolMessage))
                enhanced_template = f"""{Template.gov_scheme_template}
Original Query: {{query}}
Government Scheme Tool Results:
{{tool_results}}
Based on the government scheme data above, provide details."""
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
            query = last.content
            prompt = PromptTemplate(input_variables=["date", "query"], template=Template.gov_scheme_template)
            chain = await LLMChainFactory(model_type="gemini").get_llm_tool_chain(
                prompt, [Tools.gov_scheme_tool, Tools.web_tool]
            )
            resp = await chain.ainvoke({
                "date": datetime.now().strftime("%Y-%m-%d"),
                "query": query
            })
            if hasattr(resp, 'tool_calls') and resp.tool_calls:
                return {"messages": [resp]}
            return {"messages": [AIMessage(content=resp.content)]}
        except CustomException as e:
            logging.error(f"Error in GovSchemeNode: {e}")
            raise CustomException(e, sys) from e

    @staticmethod
    async def CarbonFootprintNode(state: AICompanionState) -> dict:
        try:
            logging.info("Calling CarbonFootprintNode")
            response = "Total carbon footprint generated: ..."
            return {"messages": [AIMessage(content=response)]}
        except CustomException as e:
            logging.error(f"Error in CarbonFootprintNode: {e}")
            raise CustomException(e, sys) from e

    @staticmethod
    async def MemoryIngestionNode(state: AICompanionState) -> dict:
        try:
            logging.info("Memory Ingestion Node -----------")
            messages = state["messages"]
            last_user_message = next((m.content for m in reversed(messages) if isinstance(m, HumanMessage)), None)
            last_ai_message = next((m.content for m in reversed(messages) if isinstance(m, AIMessage)), None)
            if last_user_message and last_ai_message:
                conversation = f"User: {last_user_message}\nAI: {last_ai_message}"
                await memory_manager.store_in_memory(state["collection_name"], conversation)
            else:
                logging.info("No valid query-response pair found to store")
            return {}
        except CustomException as e:
            logging.error(f"Error in Memory Ingestion Node: {str(e)}")
            raise CustomException(e, sys) from e

    @staticmethod
    async def ImageNode(state: AICompanionState) -> dict:
        try:
            logging.info("Calling ImageNode")
            query = state["messages"][-1].content
            prompt = PromptTemplate(input_variables=["text"], template=Template.image_template)
            factory = LLMChainFactory(model_type="gemini")
            chain = await factory.get_llm_chain_async(prompt)
            img_prompt = (await chain.ainvoke({"text": query})).content
            loop = asyncio.get_event_loop()
            img_bytes = await loop.run_in_executor(None, lambda: factory.get_image_model(img_prompt))
            return {"image": img_bytes}
        except CustomException as e:
            logging.error(f"Error in ImageNode: {e}")
            raise CustomException(e, sys) from e

    @staticmethod
    async def VoiceNode(state: AICompanionState) -> dict:
        try:
            logging.info("Calling VoiceNode")
            response_text = state["messages"][-1].content
            if not response_text:
                return {"audio": ""}
            if not cartesia_client:
                logging.error("Cartesia client not initialized")
                return {"audio": ""}
            voice_id = "ef8390dc-0fc0-473b-bbc0-7277503793f7"
            audio_generator = cartesia_client.tts.bytes(
                model_id="sonic",
                transcript=response_text,
                voice={"mode": "id", "id": voice_id},
                language="en",
                output_format={
                    "container": "raw",
                    "sample_rate": 16000,
                    "encoding": "pcm_f32le"
                }
            )
            audio_chunks = [chunk for chunk in audio_generator]
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
            return {"voice": audio_base64}
        except CustomException as e:
            logging.error(f"Error in VoiceNode: {e}")
            raise CustomException(e, sys) from e

    @staticmethod
    async def TextNode(state: AICompanionState) -> dict:
        return {}
