import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

from datetime import datetime
from src.ai_component.graph.utils.chains import async_router_chain
from src.ai_component.llm import LLMChainFactory
from src.ai_component.modules.schedule.context_generation import ScheduleContextGenerator
from src.ai_component.graph.state import AICompanionState
from src.ai_component.core.prompts import general_template, disease_template, weather_template
from src.ai_component.tools.web_seach_tool import web_tool
from src.ai_component.tools.rag_tool import rag_tool
from src.ai_component.logger import logging
from src.ai_component.exception import CustomException
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage




async def route_node(state: AICompanionState) -> str:
    """
    Async version of route_node.
    Node to route the conversation based on the user's query.
    This node uses the async_router_chain to determine the type of response needed.
    """
    try:
        logging.info("Calling Route Node")
        query = state["messages"][-1].content if state["messages"] else ""
        
        if not query:
            return "GeneralNode"  
        
        chain = await async_router_chain()
        response = await chain.ainvoke({"query": query})

        logging.info(f"Route Node Response: {response.route_node}")
        
        return {
            "workflow": response.route_node
        }
    except CustomException as e:
        logging.error(f"Error in Engineering Node : {str(e)}")
        raise CustomException(e, sys) from e


async def context_injestion_node(state: AICompanionState) -> AIMessage:
    """
    Async version of context_injestion_node.
    Node to inject context about Ramesh Kumar's current activity into the conversation.
    This node uses the ScheduleContextGenerator to get the current activity and returns it as an AI message.
    """
    try:
        logging.info("Calling Context Ingestion Node")
        current_activity = ScheduleContextGenerator.get_current_activity()
        
        if current_activity:
            return {
                "current_activity": current_activity,
            }
        else:
            return {
                "current_activity": "Ramesh Kumar is currently not scheduled for any activity."
            }
    except CustomException as e:
        logging.error(f"Error in Engineering Node : {str(e)}")
        raise CustomException(e, sys) from e
    

async def GeneralNode(state: AICompanionState) -> AIMessage:
    """
    Async version of GeneralNode.
    General node to handle queries that do not fit into specific categories.
    This node can be extended to provide general information or assistance.
    """
    try:
        logging.info("Calling General Node")
        query = state["messages"][-1].content if state["messages"] else ""
        prompt = PromptTemplate(
            input_variables=["current_activity", "query"],
            template=general_template
        )
        factory = LLMChainFactory(model_type="gemini")
        chain = await factory.get_llm_chain_async(prompt)
        response = await chain.ainvoke({
            "current_activity": state["current_activity"],
            "query": query
        })
        logging.info(f"General Node Response: {response.content}")
        return {
            "messages":[AIMessage(content=response.content)],
        }
    except CustomException as e:
        logging.error(f"Error in Engineering Node : {str(e)}")
        raise CustomException(e, sys) from e
    

async def DiseaseNode(state: AICompanionState):
    """
    Disease node to handle queries related to plant diseases, symptoms, or treatments.
    It processes plant problems from text and can use tools to find solutions.
    """
    try:
        logging.info("Calling Disease Node")
        messages = state["messages"]
        
        # Check if the last message is a tool message (result from tools)
        logging.info("Checking if last message is a tool message")
        if messages and isinstance(messages[-1], ToolMessage):
            query = ""
            tool_results = []
            
            for msg in reversed(messages):
                if isinstance(msg, HumanMessage):
                    query = msg.content
                    break
                elif isinstance(msg, ToolMessage):
                    tool_results.append(f"Tool: {msg.name}\nResult: {msg.content}")

            enhanced_template = f"""
    {disease_template}

    Original Query: {{query}}

    Tool Results:
    {{tool_results}}

    Based on the tool results above, provide a comprehensive answer about the plant disease and treatment recommendations.
    """
            
            prompt = PromptTemplate(
                input_variables=["query", "tool_results"],
                template=enhanced_template
            )
            
            factory = LLMChainFactory(model_type="gemini")
            chain = await factory.get_llm_chain_async(prompt)
            response = await chain.ainvoke({
                "query": query,
                "tool_results": "\n\n".join(tool_results)
            })
            
            return {
                "messages": [AIMessage(content=response.content)]
            }
        
        else:
            logging.info("No tool message found, processing query directly")
            # Initial processing - decide whether to use tools or respond directly
            query = messages[-1].content if messages else ""
            
            # Use LLM with tools to process the query
            prompt = PromptTemplate(
                input_variables=["query"],
                template=disease_template
            )
            
            tools = [web_tool, rag_tool]
            factory = LLMChainFactory(model_type="gemini")
            chain = await factory.get_llm_tool_chain(prompt, tools)
            response = await chain.ainvoke({"query": query})
            
            # Check if the response contains tool calls
            if hasattr(response, 'tool_calls') and response.tool_calls:
                # Return the AI message with tool calls
                return {
                    "messages": [response]
                }
            else:
                # Direct response without tools
                return {
                    "messages": [AIMessage(content=response.content)]
                }
    except CustomException as e:
        logging.error(f"Error in Engineering Node : {str(e)}")
        raise CustomException(e, sys) from e
    

async def WeatherNode(state: AICompanionState):
    """
    Weather node to handle queries related to weather conditions, forecasts, or climate-related information.
    It uses the web search tool to find the latest weather data and provides a response.
    """
    try:
        logging.info("Calling Weather Node")
        
        messages = state["messages"]
        
        # Check if the last message is a tool message (result from web search)
        if messages and isinstance(messages[-1], ToolMessage):
            # Process the tool results and generate final response
            query = ""
            tool_results = []
            
            # Find the original human message and collect tool results
            for msg in reversed(messages):
                if isinstance(msg, HumanMessage):
                    query = msg.content
                    break
                elif isinstance(msg, ToolMessage):
                    tool_results.append(f"Search Result: {msg.content}")
            
            # Create enhanced prompt with tool results
            enhanced_template = f"""
{weather_template}

Original Query: {{query}}

Weather Search Results:
{{tool_results}}

Based on the search results above, provide a comprehensive weather report with current conditions and/or forecast as requested.
"""
            
            prompt = PromptTemplate(
                input_variables=["query", "tool_results", "date"],
                template=enhanced_template
            )
            
            factory = LLMChainFactory(model_type="gemini")
            chain = await factory.get_llm_chain_async(prompt)
            response = await chain.ainvoke({
                "query": query,
                "tool_results": "\n\n".join(tool_results),
                "date": datetime.now().strftime("%Y-%m-%d")
            })
            
            return {
                "messages": [AIMessage(content=response.content)]
            }
        
        else:
            # Initial processing - use web search tool to get weather data
            query = messages[-1].content if messages else ""
            
            # Use LLM with web search tool to process the weather query
            prompt = PromptTemplate(
                input_variables=["query", "date"],
                template=weather_template
            )
            
            tools = [web_tool]  # Only web search tool for weather
            factory = LLMChainFactory(model_type="gemini")
            chain = await factory.get_llm_tool_chain(prompt, tools)
            response = await chain.ainvoke({
                "query": query,
                "date": datetime.now().strftime("%Y-%m-%d")
            })
            
            # Check if the response contains tool calls
            if hasattr(response, 'tool_calls') and response.tool_calls:
                # Return the AI message with tool calls
                return {
                    "messages": [response]
                }
            else:
                # Direct response without tools
                return {
                    "messages": [AIMessage(content=response.content)]
                }
                
    except CustomException as e:
        logging.error(f"Error in Weather Node: {str(e)}")
        raise CustomException(e, sys) from e
    except Exception as e:
        logging.error(f"Unexpected error in Weather Node: {str(e)}")
        raise CustomException(e, sys) from e