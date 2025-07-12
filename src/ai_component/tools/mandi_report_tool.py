import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

from pydantic import BaseModel, Field
from typing import Type, Optional, List, Dict, Any
from langchain.tools import BaseTool
import asyncio
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.ai_component.logger import logging
from src.ai_component.exception import CustomException

from dotenv import load_dotenv
load_dotenv()

api_key = os.getenv("GOV_DATA_API_KEY")


class MandiPriceForecastInput(BaseModel):
    commodity: str = Field(..., description="The commodity for which price forecast is requested.", examples=["Wheat", "Rice", "Cotton", "Soybean", "Maize", "Barley", "Pulses", "Groundnut", "Mustard", "Onion"])
    state: Optional[str] = Field(None, description="The state to filter prices (optional).", examples=["Uttar Pradesh", "Maharashtra", "Punjab", "Haryana", "Delhi", "Bihar", "West Bengal", "Tamil Nadu", "Karnataka", "Andhra Pradesh"])
    district: Optional[str] = Field(None, description="The district to filter prices (optional).", examples=["Varanasi", "Mumbai", "Amritsar", "Delhi", "Lucknow", "Patna", "Kolkata", "Chennai", "Bangalore"])
    market: Optional[str] = Field(None, description="The market to filter prices (optional).", examples=["Varanasi Mandi", "Mumbai APMC", "Amritsar Mandi", "Delhi Azadpur Mandi"])
    days: Optional[int] = Field(10, description="Number of days to analyze for price trends (default: 10, max: 30)")
    forecast_days: Optional[int] = Field(7, description="Number of days to forecast prices (default: 7, max: 15)")


class MandiPriceForecastTool(BaseTool):
    """
    This tool fetches mandi prices for specific commodities and provides price forecasting analysis.
    It analyzes historical price trends and provides simple forecasting for the next few days.
    """
    name: str = "mandi_price_forecast_tool"
    description: str = "Fetches mandi prices for specific commodities and provides price trend analysis and forecasting. Analyzes price patterns and predicts future prices based on historical data."
    args_schema: Type[MandiPriceForecastInput] = MandiPriceForecastInput
    
    # Define class attributes
    api_key: str = api_key
    base_url: str = "https://api.data.gov.in/resource/9ef84268-d588-465a-a308-a864a43d0070"

    def _build_filters(self, **kwargs) -> Dict[str, str]:
        """Build filters dictionary for API request."""
        filters = {}
        
        # Map the parameters to filter format
        filter_mappings = {
            'state': 'filters[state.keyword]',
            'district': 'filters[district]',
            'market': 'filters[market]',
            'commodity': 'filters[commodity]'
        }
        
        for param, filter_name in filter_mappings.items():
            value = kwargs.get(param)
            if value:
                filters[filter_name] = value
        
        return filters

    def _fetch_commodity_prices(self, **kwargs) -> List[Dict[str, Any]]:
        """
        Fetches commodity price data from data.gov.in API.
        """
        try:
            # Build query parameters - fetch more records to get diverse data
            params = {
                'api-key': self.api_key,
                'format': 'json',
                'limit': 1000,  # Fetch more records to get historical data
                'offset': 0
            }
            
            # Add filters
            filters = self._build_filters(**kwargs)
            params.update(filters)
            
            # Make API request
            response = requests.get(self.base_url, params=params, timeout=30)
            response.raise_for_status()
            
            # Parse JSON response
            data = response.json()
            
            # Extract records from the response
            if 'records' in data:
                return data['records']
            else:
                logging.warning("No 'records' key found in API response")
                return []
                
        except requests.exceptions.RequestException as e:
            logging.error(f"HTTP error fetching commodity prices: {str(e)}")
            raise
        except Exception as e:
            logging.error(f"Error fetching commodity prices: {str(e)}")
            raise

    def _process_price_data(self, records: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Process and clean the price data for analysis.
        """
        if not records:
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame(records)
        
        # Common column name variations in the API
        price_columns = ['modal_price', 'Modal_Price', 'modal price', 'price', 'Price']
        date_columns = ['price_date', 'Price_Date', 'date', 'Date', 'arrival_date', 'Arrival_Date']
        
        # Find the correct price column
        price_col = None
        for col in price_columns:
            if col in df.columns:
                price_col = col
                break
        
        # Find the correct date column
        date_col = None
        for col in date_columns:
            if col in df.columns:
                date_col = col
                break
        
        if not price_col:
            logging.warning("No price column found in data")
            return pd.DataFrame()
        
        # Clean and process data
        df_clean = df.copy()
        
        # Convert price to numeric
        df_clean[price_col] = pd.to_numeric(df_clean[price_col], errors='coerce')
        
        # Remove rows with invalid prices
        df_clean = df_clean.dropna(subset=[price_col])
        df_clean = df_clean[df_clean[price_col] > 0]
        
        # If date column exists, try to parse it
        if date_col:
            try:
                df_clean[date_col] = pd.to_datetime(df_clean[date_col], errors='coerce')
                df_clean = df_clean.dropna(subset=[date_col])
                df_clean = df_clean.sort_values(date_col)
            except:
                logging.warning("Could not parse date column")
        
        # Standardize column names
        df_clean.rename(columns={price_col: 'price'}, inplace=True)
        if date_col:
            df_clean.rename(columns={date_col: 'date'}, inplace=True)
        
        return df_clean

    def _calculate_price_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate comprehensive price statistics.
        """
        if df.empty or 'price' not in df.columns:
            return {}
        
        prices = df['price'].values
        
        stats = {
            'total_records': len(df),
            'current_price': float(prices[-1]) if len(prices) > 0 else None,
            'average_price': float(np.mean(prices)),
            'min_price': float(np.min(prices)),
            'max_price': float(np.max(prices)),
            'price_std': float(np.std(prices)),
            'price_variance': float(np.var(prices))
        }
        
        # Calculate price trend
        if len(prices) >= 2:
            recent_prices = prices[-min(5, len(prices)):]  # Last 5 records
            older_prices = prices[:min(5, len(prices))]    # First 5 records
            
            if len(recent_prices) > 0 and len(older_prices) > 0:
                recent_avg = np.mean(recent_prices)
                older_avg = np.mean(older_prices)
                
                price_change = recent_avg - older_avg
                price_change_percent = (price_change / older_avg) * 100
                
                stats['price_trend'] = 'increasing' if price_change > 0 else 'decreasing' if price_change < 0 else 'stable'
                stats['price_change'] = float(price_change)
                stats['price_change_percent'] = float(price_change_percent)
        
        return stats

    def _simple_price_forecast(self, df: pd.DataFrame, forecast_days: int = 7) -> Dict[str, Any]:
        """
        Simple price forecasting using moving averages and trend analysis.
        """
        if df.empty or 'price' not in df.columns:
            return {}
        
        prices = df['price'].values
        
        if len(prices) < 3:
            return {'error': 'Not enough data for forecasting'}
        
        # Calculate moving averages
        if len(prices) >= 5:
            short_ma = np.mean(prices[-5:])  # 5-period moving average
        else:
            short_ma = np.mean(prices)
        
        if len(prices) >= 10:
            long_ma = np.mean(prices[-10:])  # 10-period moving average
        else:
            long_ma = np.mean(prices)
        
        # Calculate trend
        if len(prices) >= 5:
            recent_trend = np.polyfit(range(len(prices[-5:])), prices[-5:], 1)[0]
        else:
            recent_trend = np.polyfit(range(len(prices)), prices, 1)[0]
        
        # Simple forecast using trend and moving averages
        forecast = []
        last_price = prices[-1]
        
        for i in range(1, forecast_days + 1):
            # Simple linear projection with some smoothing
            predicted_price = last_price + (recent_trend * i)
            
            # Add some influence from moving averages
            if abs(predicted_price - short_ma) > (short_ma * 0.1):  # If prediction deviates too much
                predicted_price = (predicted_price + short_ma) / 2  # Average with moving average
            
            forecast.append({
                'day': i,
                'predicted_price': round(float(predicted_price), 2),
                'confidence': 'medium' if i <= 3 else 'low'
            })
        
        return {
            'forecast': forecast,
            'trend_direction': 'upward' if recent_trend > 0 else 'downward' if recent_trend < 0 else 'stable',
            'trend_strength': abs(float(recent_trend)),
            'short_ma': float(short_ma),
            'long_ma': float(long_ma)
        }

    def _generate_price_report(self, df: pd.DataFrame, stats: Dict[str, Any], forecast: Dict[str, Any], **kwargs) -> str:
        """
        Generate a comprehensive price report with forecasting.
        """
        if df.empty:
            return f"❌ No price data found for {kwargs.get('commodity', 'the specified commodity')}."
        
        commodity = kwargs.get('commodity', 'Unknown Commodity')
        state = kwargs.get('state', 'All States')
        
        report = []
        report.append(f"📊 **{commodity} Price Analysis Report**")
        report.append(f"📍 **Location:** {state}")
        report.append(f"📈 **Data Points:** {stats.get('total_records', 0)} records")
        report.append("")
        
        # Current price information
        if stats.get('current_price'):
            report.append(f"💰 **Current Price:** ₹{stats['current_price']:.2f} per quintal")
        
        # Price statistics
        report.append(f"📊 **Price Statistics:**")
        report.append(f"   • **Average Price:** ₹{stats.get('average_price', 0):.2f}")
        report.append(f"   • **Minimum Price:** ₹{stats.get('min_price', 0):.2f}")
        report.append(f"   • **Maximum Price:** ₹{stats.get('max_price', 0):.2f}")
        
        # Price trend
        if 'price_trend' in stats:
            trend_emoji = "📈" if stats['price_trend'] == 'increasing' else "📉" if stats['price_trend'] == 'decreasing' else "➡️"
            report.append(f"   • **Price Trend:** {trend_emoji} {stats['price_trend'].title()}")
            
            if 'price_change_percent' in stats:
                report.append(f"   • **Price Change:** {stats['price_change_percent']:.1f}%")
        
        report.append("")
        
        # Forecasting section
        if 'forecast' in forecast and forecast['forecast']:
            report.append(f"🔮 **Price Forecast (Next {len(forecast['forecast'])} days):**")
            
            for day_forecast in forecast['forecast'][:7]:  # Show only first 7 days
                confidence_emoji = "🟢" if day_forecast['confidence'] == 'high' else "🟡" if day_forecast['confidence'] == 'medium' else "🔴"
                report.append(f"   • **Day {day_forecast['day']}:** ₹{day_forecast['predicted_price']:.2f} {confidence_emoji}")
            
            trend_direction = forecast.get('trend_direction', 'stable')
            trend_emoji = "📈" if trend_direction == 'upward' else "📉" if trend_direction == 'downward' else "➡️"
            report.append(f"   • **Forecast Trend:** {trend_emoji} {trend_direction.title()}")
            
            report.append("")
            report.append(f"ℹ️ **Note:** This is a simple forecast based on recent price trends. Market prices can be influenced by many factors including weather, demand, supply, and government policies.")
        
        return "\n".join(report)

    def _run(self, commodity: str, state: Optional[str] = None, district: Optional[str] = None, 
            market: Optional[str] = None, days: Optional[int] = 10, forecast_days: Optional[int] = 7) -> str:
        """
        Synchronous version of the price forecast tool.
        """
        try:
            logging.info(f"Running Price Forecast tool for commodity: {commodity}")
            
            # Fetch data from API
            records = self._fetch_commodity_prices(
                commodity=commodity, 
                state=state, 
                district=district, 
                market=market
            )
            
            if not records:
                logging.warning(f"No data found for commodity: {commodity}")
                return f"❌ No price data found for {commodity}. Please check the commodity name and try again."
            
            # Process data
            df = self._process_price_data(records)
            
            if df.empty:
                return f"❌ No valid price data found for {commodity}."
            
            # Calculate statistics
            stats = self._calculate_price_statistics(df)
            
            # Generate forecast
            forecast = self._simple_price_forecast(df, forecast_days or 7)
            
            # Generate report
            report = self._generate_price_report(df, stats, forecast, 
                                               commodity=commodity, state=state, 
                                               district=district, market=market)
            
            logging.info(f"Price forecast generated successfully for {commodity}")
            return report
            
        except Exception as e:
            logging.error(f"Error in Price Forecast Tool: {str(e)}")
            raise CustomException(e, sys) from e
        
    async def _arun(self, commodity: str, state: Optional[str] = None, district: Optional[str] = None, 
                   market: Optional[str] = None, days: Optional[int] = 10, forecast_days: Optional[int] = 7) -> str:
        """
        Asynchronous version of the price forecast tool.
        """
        try:
            logging.info(f"Running async Price Forecast tool for commodity: {commodity}")
            
            # Use asyncio.to_thread for async execution
            records = await asyncio.to_thread(
                self._fetch_commodity_prices,
                commodity=commodity, 
                state=state, 
                district=district, 
                market=market
            )
            
            if not records:
                logging.warning(f"No data found for commodity: {commodity}")
                return f"❌ No price data found for {commodity}. Please check the commodity name and try again."
            
            # Process data
            df = await asyncio.to_thread(self._process_price_data, records)
            
            if df.empty:
                return f"❌ No valid price data found for {commodity}."
            
            # Calculate statistics
            stats = await asyncio.to_thread(self._calculate_price_statistics, df)
            
            # Generate forecast
            forecast = await asyncio.to_thread(self._simple_price_forecast, df, forecast_days or 7)
            
            # Generate report
            report = await asyncio.to_thread(
                self._generate_price_report, df, stats, forecast, 
                commodity=commodity, state=state, district=district, market=market
            )
            
            logging.info(f"Price forecast generated successfully for {commodity}")
            return report
            
        except Exception as e:
            logging.error(f"Error in async Price Forecast Tool: {str(e)}")
            raise CustomException(e, sys) from e

    def get_available_commodities(self) -> List[str]:
        """
        Get list of available commodities from the API.
        """
        try:
            params = {
                'api-key': self.api_key,
                'format': 'json',
                'limit': 1000
            }
            
            response = requests.get(self.base_url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            if 'records' in data:
                df = pd.DataFrame(data['records'])
                commodity_columns = ['commodity', 'Commodity', 'COMMODITY']
                
                for col in commodity_columns:
                    if col in df.columns:
                        return sorted(df[col].unique().tolist())
            
            return []
            
        except Exception as e:
            logging.error(f"Error fetching available commodities: {str(e)}")
            return []


# Create tool instance
mandi_report_tool = MandiPriceForecastTool()

        
if __name__ == "__main__":
    async def test_price_forecast():
        """Test the price forecast tool with various commodities."""
        
        print("🔮 Testing Mandi Price Forecast Tool")
        print("=" * 50)
        
        forecast_tool = MandiPriceForecastTool()
        
        # Test 1: Wheat price forecast
        print("\n🌾 Test 1: Wheat Price Forecast")
        result1 = await forecast_tool._arun(commodity="Rice", state="Uttar Pradesh",days = 10 )
        print(result1)
        
        # # Test 2: Rice price forecast
        # print("\n🍚 Test 2: Rice Price Forecast")
        # result2 = await forecast_tool._arun(commodity="Rice", forecast_days=5)
        # print(result2)
        
        # # Test 3: Onion price forecast (commonly volatile)
        # print("\n🧅 Test 3: Onion Price Forecast")
        # result3 = await forecast_tool._arun(commodity="Onion", forecast_days=10)
        # print(result3)
        
        # # Test 4: Get available commodities
        # print("\n📋 Test 4: Available Commodities")
        # commodities = forecast_tool.get_available_commodities()
        # print(f"Found {len(commodities)} commodities:")
        # for i, commodity in enumerate(commodities[:20]):  # Show first 20
        #     print(f"  {i+1}. {commodity}")
        # if len(commodities) > 20:
        #     print(f"  ... and {len(commodities)-20} more")
    
    # Run the test
    asyncio.run(test_price_forecast())