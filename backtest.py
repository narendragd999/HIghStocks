import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import pytz

# Set Streamlit page configuration
st.set_page_config(page_title="Historical Stock Screener (Extended)", layout="wide")
st.title("Historical Stock Screener: 9:15 AM to 10:00 AM Candles")

# Sidebar for user input
st.sidebar.header("Input Parameters")
ticker = st.sidebar.text_input("Enter Stock Ticker (e.g., RELIANCE.NS)", "RELIANCE.NS").strip().upper()
start_date = st.sidebar.date_input("Start Date", value=datetime.today() - timedelta(days=60))
end_date = st.sidebar.date_input("End Date", value=datetime.today())

# Function to screen historical data for a ticker
def screen_historical_data(ticker, start_date, end_date):
    results = []
    skipped_dates = []
    ist = pytz.timezone("Asia/Kolkata")
    
    # Convert date range to datetime
    start_datetime = datetime.combine(start_date, datetime.min.time()).astimezone(ist)
    end_datetime = datetime.combine(end_date, datetime.min.time()).astimezone(ist)
    
    # Fetch daily data for the entire period plus one extra day for previous day checks
    stock = yf.Ticker(ticker)
    daily_data = stock.history(start=start_date - timedelta(days=1), end=end_date + timedelta(days=1), interval="1d")
    
    if daily_data.empty:
        st.error(f"No daily data available for {ticker} in the specified date range.")
        return pd.DataFrame(), []
    
    # Iterate over each trading day in the date range
    current_date = start_datetime
    while current_date <= end_datetime:
        try:
            # Convert current_date to date for indexing
            date_key = current_date.date()
            
            # Check if data exists for the current date
            if date_key not in daily_data.index.date:
                skipped_dates.append((date_key, "No daily data available"))
                current_date += timedelta(days=1)
                continue
            
            # Get previous day's data (most recent trading day before current_date)
            prev_trading_days = daily_data.index[daily_data.index.date < date_key]
            if len(prev_trading_days) == 0:
                skipped_dates.append((date_key, "No previous trading day data"))
                current_date += timedelta(days=1)
                continue
            prev_day_index = prev_trading_days[-1]
            prev_day = daily_data.loc[prev_day_index]
            prev_high = prev_day["High"]
            prev_close = prev_day["Close"]
            
            # Condition 1: Previous high > close
            if prev_high <= prev_close:
                current_date += timedelta(days=1)
                continue
            
            # Get current day's data
            current_day = daily_data.loc[daily_data.index.date == date_key]
            if current_day.empty:
                skipped_dates.append((date_key, "No daily data for current day"))
                current_date += timedelta(days=1)
                continue
            current_day = current_day.iloc[0]
            today_open = current_day["Open"]
            
            # Condition 2: Opened above previous high
            if today_open <= prev_high:
                current_date += timedelta(days=1)
                continue
            
            # Fetch 5-minute data for 9:15 AM to 10:00 AM
            start_time = current_date.replace(hour=9, minute=15, second=0, microsecond=0)
            end_time = current_date.replace(hour=10, minute=0, second=0, microsecond=0)
            intraday_data = stock.history(start=start_time, end=end_time, interval="5m")
            
            if intraday_data.empty:
                skipped_dates.append((date_key, "No 5-minute data available (possibly outside 60-day window)"))
                current_date += timedelta(days=1)
                continue
            
            # Check each 5-minute candle for Condition 3
            for index, candle in intraday_data.iterrows():
                current_price = candle["Close"]
                candle_time = index.strftime("%H:%M")
                
                # Condition 3: Current price within 0.2% of previous close
                price_diff_percent = abs(current_price - prev_close) / prev_close * 100
                if price_diff_percent <= 0.2:
                    # If conditions are met, add to results
                    results.append({
                        "Date": date_key,
                        "Candle Time": candle_time,
                        "Ticker": ticker,
                        "Prev High": round(prev_high, 2),
                        "Prev Close": round(prev_close, 2),
                        "Today Open": round(today_open, 2),
                        "Candle Close": round(current_price, 2),
                        "Price Diff %": round(price_diff_percent, 4)
                    })
        
        except Exception as e:
            skipped_dates.append((date_key, f"Error: {str(e)}"))
        
        current_date += timedelta(days=1)
    
    return pd.DataFrame(results), skipped_dates

# Run screening when button is clicked
if st.button("Screen Historical Data"):
    if ticker and start_date <= end_date:
        # Warn about yfinance 60-day limitation for 5-minute data
        if (datetime.today().date() - start_date).days > 60:
            st.warning("Note: yfinance only provides 5-minute intraday data for the last 60 days. Dates earlier than that may be skipped.")
        
        with st.spinner(f"Screening {ticker} from {start_date} to {end_date}..."):
            results_df, skipped_dates = screen_historical_data(ticker, start_date, end_date)
        
        if not results_df.empty:
            st.subheader(f"Results for {ticker}")
            st.dataframe(results_df.style.format({
                "Date": lambda x: x.strftime("%Y-%m-%d"),
                "Candle Time": "{}",
                "Prev High": "{:.2f}",
                "Prev Close": "{:.2f}",
                "Today Open": "{:.2f}",
                "Candle Close": "{:.2f}",
                "Price Diff %": "{:.4f}"
            }))
        else:
            st.warning(f"No dates found for {ticker} where all criteria were met.")
        
        if skipped_dates:
            st.subheader("Skipped Dates")
            skipped_df = pd.DataFrame(skipped_dates, columns=["Date", "Reason"])
            st.dataframe(skipped_df.style.format({"Date": lambda x: x.strftime("%Y-%m-%d")}))
    else:
        st.error("Please enter a valid ticker and ensure start date is before end date.")

# Instructions
st.sidebar.markdown("""
### Screening Criteria
- **Previous High > Close**: Previous day's high must be greater than its close.
- **Opened Above High**: Current day's open price must be higher than the previous day's high.
- **Candle Close within 0.2% of Previous Close**: Any 5-minute candle between 9:15 AM and 10:00 AM IST must have a close price within ±0.2% of the previous day's close.
- **Note**: Use NSE tickers (e.g., RELIANCE.NS). 5-minute data is only available for the last 60 days via yfinance. Earlier dates may be skipped.
""")