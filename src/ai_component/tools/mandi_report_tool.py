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
    historical_days: Optional[int] = Field(10, description="Number of historical days to analyze and show detailed analysis (default: 10, max: 30)")
    forecast_days: Optional[int] = Field(7, description="Number of days to forecast prices (default: 7, max: 15)")
    include_historical_analysis: Optional[bool] = Field(True, description="Include detailed analysis of historical prices (default: True)")
    include_future_forecast: Optional[bool] = Field(True, description="Include future price forecasting (default: True)")


class MandiPriceForecastTool(BaseTool):
    """
    This tool fetches mandi prices for specific commodities and provides comprehensive analysis including:
    - Historical price analysis for past days
    - Current price trends and patterns
    - Future price forecasting for next few days
    """
    name: str = "mandi_price_forecast_tool"
    description: str = "Fetches mandi prices for specific commodities and provides comprehensive analysis including historical price trends for past days and future price forecasting. Analyzes both historical patterns and predicts future prices."
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

    def _get_historical_analysis(self, df: pd.DataFrame, historical_days: int = 10) -> Dict[str, Any]:
        """
        Analyze historical price data for the specified number of days.
        """
        if df.empty or 'price' not in df.columns:
            return {}
        
        # If we have date column, filter by recent days
        if 'date' in df.columns:
            latest_date = df['date'].max()
            cutoff_date = latest_date - timedelta(days=historical_days)
            recent_df = df[df['date'] >= cutoff_date].copy()
        else:
            # If no date column, take last N records
            recent_df = df.tail(historical_days).copy()
        
        if recent_df.empty:
            return {}
        
        prices = recent_df['price'].values
        
        # Calculate daily changes if we have dates
        daily_analysis = []
        if 'date' in recent_df.columns and len(recent_df) > 1:
            recent_df = recent_df.sort_values('date')
            for i in range(len(recent_df)):
                row = recent_df.iloc[i]
                analysis_item = {
                    'date': row['date'].strftime('%Y-%m-%d') if pd.notna(row['date']) else f"Day {i+1}",
                    'price': float(row['price']),
                }
                
                # Calculate change from previous day
                if i > 0:
                    prev_price = recent_df.iloc[i-1]['price']
                    price_change = row['price'] - prev_price
                    price_change_percent = (price_change / prev_price) * 100
                    analysis_item.update({
                        'change_from_prev': float(price_change),
                        'change_percent': float(price_change_percent),
                        'trend': 'up' if price_change > 0 else 'down' if price_change < 0 else 'stable'
                    })
                
                daily_analysis.append(analysis_item)
        
        # Overall historical statistics
        historical_stats = {
            'period_days': len(recent_df),
            'average_price': float(np.mean(prices)),
            'min_price': float(np.min(prices)),
            'max_price': float(np.max(prices)),
            'price_volatility': float(np.std(prices)),
            'daily_analysis': daily_analysis
        }
        
        # Calculate overall trend for the period
        if len(prices) >= 2:
            overall_trend = np.polyfit(range(len(prices)), prices, 1)[0]
            historical_stats.update({
                'overall_trend': 'upward' if overall_trend > 0 else 'downward' if overall_trend < 0 else 'stable',
                'trend_strength': abs(float(overall_trend))
            })
        
        return historical_stats

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
        
        # Get current date for future predictions
        current_date = datetime.now()
        
        for i in range(1, forecast_days + 1):
            # Simple linear projection with some smoothing
            predicted_price = last_price + (recent_trend * i)
            
            # Add some influence from moving averages
            if abs(predicted_price - short_ma) > (short_ma * 0.1):  # If prediction deviates too much
                predicted_price = (predicted_price + short_ma) / 2  # Average with moving average
            
            forecast_date = current_date + timedelta(days=i)
            
            forecast.append({
                'day': i,
                'date': forecast_date.strftime('%Y-%m-%d'),
                'predicted_price': round(float(predicted_price), 2),
                'confidence': 'high' if i <= 2 else 'medium' if i <= 5 else 'low',
                'change_from_current': round(float(predicted_price - last_price), 2),
                'change_percent': round(float(((predicted_price - last_price) / last_price) * 100), 2)
            })
        
        return {
            'forecast': forecast,
            'trend_direction': 'upward' if recent_trend > 0 else 'downward' if recent_trend < 0 else 'stable',
            'trend_strength': abs(float(recent_trend)),
            'short_ma': float(short_ma),
            'long_ma': float(long_ma)
        }

    def _generate_comprehensive_report(self, df: pd.DataFrame, stats: Dict[str, Any], 
                                     historical_analysis: Dict[str, Any], forecast: Dict[str, Any], 
                                     include_historical: bool, include_future: bool, **kwargs) -> str:
        """
        Generate a comprehensive price report with both historical analysis and forecasting.
        """
        if df.empty:
            return f"❌ No price data found for {kwargs.get('commodity', 'the specified commodity')}."
        
        commodity = kwargs.get('commodity', 'Unknown Commodity')
        state = kwargs.get('state', 'All States')
        
        report = []
        report.append(f"📊 **{commodity} Comprehensive Price Analysis Report**")
        report.append(f"📍 **Location:** {state}")
        report.append(f"📈 **Data Points:** {stats.get('total_records', 0)} records")
        report.append(f"🕒 **Analysis Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Current price information
        if stats.get('current_price'):
            report.append(f"💰 **Current Price:** ₹{stats['current_price']:.2f} per quintal")
        
        # Overall price statistics
        report.append(f"📊 **Overall Statistics:**")
        report.append(f"   • **Average Price:** ₹{stats.get('average_price', 0):.2f}")
        report.append(f"   • **Price Range:** ₹{stats.get('min_price', 0):.2f} - ₹{stats.get('max_price', 0):.2f}")
        report.append(f"   • **Price Volatility:** ₹{stats.get('price_std', 0):.2f}")
        
        # Price trend
        if 'price_trend' in stats:
            trend_emoji = "📈" if stats['price_trend'] == 'increasing' else "📉" if stats['price_trend'] == 'decreasing' else "➡️"
            report.append(f"   • **Overall Trend:** {trend_emoji} {stats['price_trend'].title()}")
            
            if 'price_change_percent' in stats:
                report.append(f"   • **Trend Change:** {stats['price_change_percent']:.1f}%")
        
        report.append("")
        
        # Historical Analysis Section
        if include_historical and historical_analysis:
            report.append(f"📅 **Historical Analysis (Last {historical_analysis.get('period_days', 0)} days):**")
            
            if 'daily_analysis' in historical_analysis and historical_analysis['daily_analysis']:
                # Show last 7 days of detailed analysis
                recent_days = historical_analysis['daily_analysis'][-7:]
                for day_data in recent_days:
                    date_str = day_data['date']
                    price = day_data['price']
                    
                    day_line = f"   • **{date_str}:** ₹{price:.2f}"
                    
                    if 'change_from_prev' in day_data:
                        change = day_data['change_from_prev']
                        change_pct = day_data['change_percent']
                        trend = day_data['trend']
                        
                        if trend == 'up':
                            day_line += f" 📈 (+₹{change:.2f}, +{change_pct:.1f}%)"
                        elif trend == 'down':
                            day_line += f" 📉 (-₹{abs(change):.2f}, {change_pct:.1f}%)"
                        else:
                            day_line += f" ➡️ (No change)"
                    
                    report.append(day_line)
            
            # Historical period summary
            if 'overall_trend' in historical_analysis:
                hist_trend_emoji = "📈" if historical_analysis['overall_trend'] == 'upward' else "📉" if historical_analysis['overall_trend'] == 'downward' else "➡️"
                report.append(f"   • **Period Trend:** {hist_trend_emoji} {historical_analysis['overall_trend'].title()}")
                report.append(f"   • **Period Average:** ₹{historical_analysis.get('average_price', 0):.2f}")
                report.append(f"   • **Period Volatility:** ₹{historical_analysis.get('price_volatility', 0):.2f}")
            
            report.append("")
        
        # Future Forecasting Section
        if include_future and 'forecast' in forecast and forecast['forecast']:
            report.append(f"🔮 **Future Price Forecast (Next {len(forecast['forecast'])} days):**")
            
            for day_forecast in forecast['forecast']:
                confidence_emoji = "🟢" if day_forecast['confidence'] == 'high' else "🟡" if day_forecast['confidence'] == 'medium' else "🔴"
                change_emoji = "📈" if day_forecast['change_from_current'] > 0 else "📉" if day_forecast['change_from_current'] < 0 else "➡️"
                
                report.append(f"   • **{day_forecast['date']} (Day {day_forecast['day']}):** ₹{day_forecast['predicted_price']:.2f} {confidence_emoji}")
                report.append(f"     └─ Change: {change_emoji} {day_forecast['change_from_current']:+.2f} ({day_forecast['change_percent']:+.1f}%)")
            
            trend_direction = forecast.get('trend_direction', 'stable')
            trend_emoji = "📈" if trend_direction == 'upward' else "📉" if trend_direction == 'downward' else "➡️"
            report.append(f"   • **Forecast Trend:** {trend_emoji} {trend_direction.title()}")
            
            report.append("")
        
        # Recommendations and Notes
        report.append("📋 **Analysis Summary:**")
        
        if include_historical and historical_analysis:
            if 'overall_trend' in historical_analysis:
                report.append(f"   • Recent historical trend shows {historical_analysis['overall_trend']} movement")
        
        if include_future and 'trend_direction' in forecast:
            report.append(f"   • Future forecast suggests {forecast['trend_direction']} trend")
        
        report.append("")
        report.append("ℹ️ **Important Notes:**")
        report.append("   • Historical analysis is based on actual recorded prices")
        report.append("   • Future forecasts are predictions based on recent trends")
        report.append("   • Market prices can be influenced by weather, demand, supply, and policies")
        report.append("   • Use this analysis as a reference, not absolute prediction")
        
        return "\n".join(report)

    def _run(self, commodity: str, state: Optional[str] = None, district: Optional[str] = None, 
            market: Optional[str] = None, historical_days: Optional[int] = 10, 
            forecast_days: Optional[int] = 7, include_historical_analysis: Optional[bool] = True,
            include_future_forecast: Optional[bool] = True) -> str:
        """
        Synchronous version of the enhanced price analysis tool.
        """
        try:
            logging.info(f"Running Enhanced Price Analysis tool for commodity: {commodity}")
            
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
            
            # Calculate overall statistics
            stats = self._calculate_price_statistics(df)
            
            # Get historical analysis
            historical_analysis = {}
            if include_historical_analysis:
                historical_analysis = self._get_historical_analysis(df, historical_days or 10)
            
            # Generate forecast
            forecast = {}
            if include_future_forecast:
                forecast = self._simple_price_forecast(df, forecast_days or 7)
            
            # Generate comprehensive report
            report = self._generate_comprehensive_report(
                df, stats, historical_analysis, forecast, 
                include_historical_analysis, include_future_forecast,
                commodity=commodity, state=state, district=district, market=market
            )
            
            logging.info(f"Enhanced price analysis generated successfully for {commodity}")
            return report
            
        except Exception as e:
            logging.error(f"Error in Enhanced Price Analysis Tool: {str(e)}")
            raise CustomException(e, sys) from e
        
    async def _arun(self, commodity: str, state: Optional[str] = None, district: Optional[str] = None, 
                   market: Optional[str] = None, historical_days: Optional[int] = 10, 
                   forecast_days: Optional[int] = 7, include_historical_analysis: Optional[bool] = True,
                   include_future_forecast: Optional[bool] = True) -> str:
        """
        Asynchronous version of the enhanced price analysis tool.
        """
        try:
            logging.info(f"Running async Enhanced Price Analysis tool for commodity: {commodity}")
            
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
            
            # Get historical analysis
            historical_analysis = {}
            if include_historical_analysis:
                historical_analysis = await asyncio.to_thread(
                    self._get_historical_analysis, df, historical_days or 10
                )
            
            # Generate forecast
            forecast = {}
            if include_future_forecast:
                forecast = await asyncio.to_thread(
                    self._simple_price_forecast, df, forecast_days or 7
                )
            
            # Generate comprehensive report
            report = await asyncio.to_thread(
                self._generate_comprehensive_report,
                df, stats, historical_analysis, forecast, 
                include_historical_analysis, include_future_forecast,
                commodity=commodity, state=state, district=district, market=market
            )
            
            logging.info(f"Enhanced price analysis generated successfully for {commodity}")
            return report
            
        except Exception as e:
            logging.error(f"Error in async Enhanced Price Analysis Tool: {str(e)}")
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
    async def test_enhanced_analysis():
        """Test the enhanced analysis tool with both historical and future analysis."""
        
        print("🔮 Testing Enhanced Mandi Price Analysis Tool")
        print("=" * 60)
        
        analysis_tool = MandiPriceForecastTool()
        
        # Test 1: Full analysis (both historical and future)
        print("\n🌾 Test 1: Complete Analysis - Rice (Historical + Future)")
        result1 = await analysis_tool._arun(
            commodity="Rice", 
            state="Uttar Pradesh",
            historical_days=15,
            forecast_days=10,
            include_historical_analysis=True,
            include_future_forecast=True
        )
        print(result1)
        
        # Test 2: Only historical analysis
        print("\n📅 Test 2: Historical Analysis Only - Wheat")
        result2 = await analysis_tool._arun(
            commodity="Wheat",
            historical_days=20,
            include_historical_analysis=True,
            include_future_forecast=False
        )
        print(result2)
        
        # Test 3: Only future forecast
        print("\n🔮 Test 3: Future Forecast Only - Onion")
        result3 = await analysis_tool._arun(
            commodity="Onion",
            forecast_days=7,
            include_historical_analysis=False,
            include_future_forecast=True
        )
        print(result3)
    
    # Run the test
    asyncio.run(test_enhanced_analysis())