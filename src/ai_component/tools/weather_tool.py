import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

import asyncio
import aiohttp
import requests
from pydantic import BaseModel, Field
from typing import Type
from langchain.tools import BaseTool
from src.ai_component.config import DEFAULT_FORECAST_COUNT, DEFAULT_DAYS
from src.ai_component.logger import logging
from src.ai_component.exception import CustomException

from dotenv import load_dotenv
load_dotenv()


class WeatherForecastToolInput(BaseModel):
    place: str = Field(..., description="The place for which the weather forecast information is requested.")
    days: int = Field(default=DEFAULT_DAYS, description="Number of days for weather forecast (1-5 days, default is 5).")


class WeatherForecastTool(BaseTool):  
    name: str = "weather_forecast_tool"
    description: str = "Provides weather forecast information for a specified place using OpenWeatherMap API for a given number of days (1-5 days)."
    args_schema: Type[WeatherForecastToolInput] = WeatherForecastToolInput
    
    def _validate_days(self, days: int) -> int:
        """
        Validate and adjust the number of days
        OpenWeatherMap 5-day forecast API provides data for up to 5 days
        """
        if days < 1:
            logging.warning(f"Days parameter {days} is less than 1, setting to 1")
            return 1
        elif days > 7:
            logging.warning(f"Days parameter {days} exceeds maximum of 5, setting to 5")
            return 5
        return days
    
    def _calculate_forecast_count(self, days: int) -> int:
        """
        Calculate the number of forecast entries based on days
        OpenWeatherMap provides 8 forecasts per day (every 3 hours)
        """
        validated_days = self._validate_days(days)
        return validated_days * 8
    
    async def _arun(self, place: str, days: int = DEFAULT_DAYS) -> str:
        """
        Async version: Provide the forecast of the place for specified number of days
        """
        try:
            logging.info(f"Running Weather Forecast tool for place: {place}, days: {days}")
            api_key = os.getenv("OPENWEATHER_API_KEY")
            if not api_key:
                raise ValueError("OPENWEATHER_API_KEY is not set in environment variables.")
            
            forecast_count = self._calculate_forecast_count(days)
            
            base_url = "https://api.openweathermap.org/data/2.5/forecast"
            params = {
                "q": place,
                "appid": api_key,
                "cnt": forecast_count, 
                "units": "metric" 
            }
            
            # Use aiohttp for async requests
            async with aiohttp.ClientSession() as session:
                async with session.get(base_url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        return self._format_weather_data(data, days)
                    else:
                        error_msg = f"API request failed with status {response.status}"
                        logging.error(error_msg)
                        return f"Error: {error_msg}"
                        
        except CustomException as e:
            logging.error(f"Error in Weather Forecast tool: {str(e)}")
            return f"Error fetching weather data: {str(e)}"
    
    def _run(self, place: str, days: int = DEFAULT_DAYS) -> str:
        """
        Sync version: Provide the forecast of the place for specified number of days
        """
        try:
            logging.info(f"Running Weather Forecast tool for place: {place}, days: {days}")
            api_key = os.getenv("OPENWEATHER_API_KEY")
            if not api_key:
                raise ValueError("OPENWEATHER_API_KEY is not set in environment variables.")
            
            forecast_count = self._calculate_forecast_count(days)
            
            base_url = "https://api.openweathermap.org/data/2.5/forecast"
            params = {
                "q": place,
                "appid": api_key,
                "cnt": forecast_count, 
                "units": "metric"
            }
            
            response = requests.get(base_url, params=params)
            if response.status_code == 200:
                data = response.json()
                return self._format_weather_data(data, days)
            else:
                error_msg = f"API request failed with status {response.status_code}"
                logging.error(error_msg)
                return f"Error: {error_msg}"
                
        except CustomException as e:
            logging.error(f"Error in Weather Forecast tool: {str(e)}")
            return f"Error fetching weather data: {str(e)}"
    
    def _format_weather_data(self, data: dict, days: int) -> str:
        """
        Format the weather data into a readable string
        """
        try:
            if not data or 'list' not in data:
                return "No weather data available"
            
            city = data.get('city', {}).get('name', 'Unknown')
            country = data.get('city', {}).get('country', '')
            validated_days = self._validate_days(days)
            
            forecast_text = f"Weather Forecast for {city}, {country} - {validated_days} day{'s' if validated_days > 1 else ''}:\n\n"
            
            forecast_count = self._calculate_forecast_count(validated_days)
            for forecast in data['list'][:forecast_count]: 
                dt_txt = forecast.get('dt_txt', '')
                temp = forecast.get('main', {}).get('temp', 'N/A')
                feels_like = forecast.get('main', {}).get('feels_like', 'N/A')
                humidity = forecast.get('main', {}).get('humidity', 'N/A')
                description = forecast.get('weather', [{}])[0].get('description', 'N/A')
                
                forecast_text += f"Date/Time: {dt_txt}\n"
                forecast_text += f"Temperature: {temp}°C (feels like {feels_like}°C)\n"
                forecast_text += f"Humidity: {humidity}%\n"
                forecast_text += f"Conditions: {description.title()}\n"
                forecast_text += "-" * 40 + "\n"
            
            return forecast_text
            
        except CustomException as e:
            logging.error(f"Error formatting weather data: {str(e)}")
            return f"Error formatting weather data: {str(e)}"
        

class WeatherReportToolInput(BaseModel):
    place: str = Field(..., description="The place for which the current weather report is requested.")

class WeatherReportTool(BaseTool):
    name: str = "weather_report_tool"
    description: str = "Provides the current weather report for a specified place."
    args_schema: Type[WeatherReportToolInput] = WeatherReportToolInput

    async def _arun(self, place: str) -> str:
        """
        this tool provides the current weather report for a specified place
        """
        try:
            logging.info(f"Running Weather Report tool for place: {place}")
            api_key = os.getenv("WEATHERSTACK_API_KEY")
            if not api_key:
                raise ValueError("WEATHERSTACK_API_KEY is not set in environment variables.")
            
            base_url = f'https://api.weatherstack.com/current?access_key={api_key}&query={place}'
            async with aiohttp.ClientSession() as session:
                async with session.get(base_url) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data
                    else:
                        error_msg = f"API request failed with status {response.status}"
                        logging.error(error_msg)
                        return f"Error: {error_msg}"
        except CustomException as e:
            logging.error(f"Error in Weather Report tool: {str(e)}")
            return f"Error fetching weather data: {str(e)}"
    
    def _run(self, place: str) -> str:
        """
        this tool provides the current weather report for a specified place
        """
        try:
            logging.info(f"Running Weather Report tool for place: {place}")
            api_key = os.getenv("WEATHERSTACK_API_KEY")
            if not api_key:
                raise ValueError("WEATHERSTACK_API_KEY is not set in environment variables.")
            
            base_url = f'https://api.weatherstack.com/current?access_key={api_key}&query={place}'
            response = requests.get(base_url)
            if response.status_code == 200:
                data = response.json()
                return data
            else:
                error_msg = f"API request failed with status {response.status_code}"
                logging.error(error_msg)
                return f"Error: {error_msg}"
        except CustomException as e:
            logging.error(f"Error in Weather Report tool: {str(e)}")
            return f"Error fetching weather data: {str(e)}"



weather_forecast_tool = WeatherForecastTool()
weather_report_tool = WeatherReportTool()

if __name__ == "__main__":
    async def test_tool():
        # result = await weather_forecast_tool._arun("London", days=3)
        # print("\nAsync result (3 days):")
        # print(result)
        

        result = await weather_report_tool._arun("Varanasi, Uttar Pradesh, India")
        print("\nAsync result (current weather):")
        print(result)
    
    asyncio.run(test_tool())