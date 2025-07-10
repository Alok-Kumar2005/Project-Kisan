import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

from pydantic import BaseModel, Field
from typing import Type
from langchain.tools import BaseTool
import asyncio
import requests
from bs4 import BeautifulSoup
import pandas as pd
from src.ai_component.logger import logging
from src.ai_component.exception import CustomException


class MandiReportToolInput(BaseModel):
    state: str = Field(..., description="The state for which the mandi report is requested.", examples=["Uttar Pradesh", "Maharashtra", "Punjab", "Haryana", "Delhi", "Bihar", "West Bengal", "Tamil Nadu", "Karnataka", "Andhra Pradesh"])
    district: str = Field(..., description="The district for which the mandi report is requested.", examples=["Varanasi", "Mumbai", "Amritsar", "Delhi", "Benares", "Lucknow", "Patna", "Kolkata", "Chennai", "Bangalore"])
    market: str = Field(..., description="The market for which the mandi report is requested.", examples=["Varanasi Mandi", "Mumbai APMC", "Amritsar Mandi", "Delhi Azadpur Mandi", "Benares Mandi"])
    commodity: str = Field(..., description="The commodity for which the mandi report is requested.", examples=["Wheat", "Rice", "Sugarcane", "Cotton", "Soybean", "Maize", "Barley", "Pulses", "Groundnut", "Mustard"])
    from_date: str = Field(..., description="The start date for the mandi report in YYYY-MM-DD format.")
    to_date: str = Field(..., description="The end date for the mandi report in YYYY-MM-DD format.")

class MandiReportTool(BaseTool):
    """
    This tool is used to fetch mandi report for a given state, district, market, commodity and date range.
    """
    name: str = "mandi_report_tool"
    description: str = "Fetches mandi report for a given state, district, market, commodity and date range."
    args_schema: Type[MandiReportToolInput] = MandiReportToolInput

    def _fetch_report(self, state: str, district: str, market: str, commodity: str, from_date: str, to_date: str) -> pd.DataFrame:
        """
        Fetches mandi report data from the government website.
        """
        session = requests.Session()
        url = "https://agmarknet.gov.in/PriceAndArrivals/DatewiseCommodityReport.aspx"
        
        try:
            # GET the page to pick up VIEWSTATE and other tokens
            r0 = session.get(url)
            r0.raise_for_status()
            soup = BeautifulSoup(r0.text, 'html.parser')

            # extracting hidden inputs
            def get_token(name):
                tag = soup.find('input', {'id': name})
                return tag['value'] if tag else ""
            
            viewstate = get_token("__VIEWSTATE")
            eventvalidation = get_token("__EVENTVALIDATION")
            
            payload = {
                "__VIEWSTATE": viewstate,
                "__EVENTVALIDATION": eventvalidation,
                "ddlStateName": state,
                "ddlDistrict": district,
                "ddlMarket": market,
                "ddlCommodity": commodity,
                "txtFromDate": from_date,
                "txtToDate": to_date,
                "btnSubmit": "Submit"
            }
            
            # POST back to same url with the payload
            r1 = session.post(url, data=payload)
            r1.raise_for_status()

            # parse the resulting HTML table in data frame
            df_list = pd.read_html(r1.text)
            if df_list:
                df = df_list[0]
                return df
            else:
                return pd.DataFrame()
                
        except Exception as e:
            logging.error(f"Error fetching mandi report: {str(e)}")
            raise
        finally:
            session.close()

    def _run(self, state: str, district: str, market: str, commodity: str, from_date: str, to_date: str) -> str:
        """
        Synchronous version of the mandi report tool.
        """
        try:
            logging.info(f"Running Mandi Report tool with parameters: state={state}, district={district}, market={market}, commodity={commodity}, from_date={from_date}, to_date={to_date}")
            
            report_df = self._fetch_report(state, district, market, commodity, from_date, to_date)
            
            if report_df.empty:
                logging.warning("No data found for the given parameters.")
                return "No data found for the given parameters."
            
            logging.info("Mandi report fetched successfully.")
            return report_df.to_json(orient='records', date_format='iso')
            
        except Exception as e:
            logging.error(f"Error in Mandi Report Tool: {str(e)}")
            raise CustomException(e, sys) from e
        
    async def _arun(self, state: str, district: str, market: str, commodity: str, from_date: str, to_date: str) -> str:
        """
        Asynchronous version of the mandi report tool.
        """
        try:
            logging.info(f"Running Mandi Report tool with parameters: state={state}, district={district}, market={market}, commodity={commodity}, from_date={from_date}, to_date={to_date}")
            
            report_df = await asyncio.to_thread(self._fetch_report, state, district, market, commodity, from_date, to_date)
            
            if report_df.empty:
                logging.warning("No data found for the given parameters.")
                return "No data found for the given parameters."
            
            logging.info("Mandi report fetched successfully.")
            return report_df.to_json(orient='records', date_format='iso')
            
        except Exception as e:
            logging.error(f"Error in Mandi Report Tool: {str(e)}")
            raise CustomException(e, sys) from e
        
mandi_report_tool = MandiReportTool()
        
if __name__ == "__main__":
    async def test_tool():
        # Example parameters
        state = "Uttar Pradesh"
        district = "Varanasi"
        market = "Varanasi Mandi"
        commodity = "Wheat"
        from_date = "2023-10-01"
        to_date = "2023-10-31"
        
        mandi_tool = MandiReportTool()
        result = await mandi_tool._arun(state, district, market, commodity, from_date, to_date)
        print("\nAsync result:")
        print(result)
    
    asyncio.run(test_tool())