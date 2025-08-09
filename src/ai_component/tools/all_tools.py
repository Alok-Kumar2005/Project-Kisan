import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

from src.ai_component.tools.web_seach_tool import web_tool
from src.ai_component.tools.rag_tool import rag_tool
from src.ai_component.tools.gov_scheme_tool import gov_scheme_tool
from src.ai_component.tools.weather_tool import weather_forecast_tool, weather_report_tool
from src.ai_component.tools.mandi_report_tool import mandi_report_tool
from src.ai_component.tools.call_tool import call_tool

class Tools:
    web_tool = web_tool
    rag_tool = rag_tool
    gov_scheme_tool = gov_scheme_tool
    weather_forecast_tool = weather_forecast_tool 
    weather_report_tool = weather_report_tool
    mandi_report_tool = mandi_report_tool
    call_tool = call_tool