import streamlit as st
import pandas as pd
import yfinance as yf
import requests
from datetime import datetime, timedelta
import pytz
import os
import time
from st_aggrid import AgGrid
import plotly.express as px
import plotly.graph_objects as go

# Set timezone to IST
ist = pytz.timezone('Asia/Kolkata')

# NSE headers
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nseindia.com/market-data/equity-derivatives-watch",
}

# Initialize NSE session
nse_session = None
def initialize_nse_session():
    global nse_session
    if nse_session is None:
        nse_session = requests.Session()
        try:
            response = nse_session.get("https://www.nseindia.com/", headers=headers)
            if response.status_code != 200:
                st.warning(f"Failed to load NSE homepage: {response.status_code}")
                return False
            time.sleep(2)
            response = nse_session.get("https://www.nseindia.com/market-data/equity-derivatives-watch", headers=headers)
            time.sleep(2)
            if response.status_code != 200:
                st.warning(f"Failed to load NSE derivatives page: {response.status_code}")
                return False
        except Exception as e:
            st.error(f"Error initializing NSE session: {e}")
            return False
    return True

# Function to fetch current price from NSE
def get_current_price_nse(ticker):
    global nse_session
    try:
        if nse_session is None and not initialize_nse_session():
            return None
        quote_url = f"https://www.nseindia.com/api/quote-equity?symbol={ticker}"
        response = nse_session.get(quote_url, headers=headers)
        if response.status_code == 200:
            quote_data = response.json()
            last_price = quote_data.get('priceInfo', {}).get('lastPrice', 0)
            if last_price > 0:
                return last_price
        return None
    except Exception as e:
        st.warning(f"Error fetching NSE current price for {ticker}: {e}")
        return None

# Function to fetch current price from yfinance (fallback)
@st.cache_data
def get_current_price_yfinance(symbol):
    try:
        stock = yf.Ticker(symbol)
        data = stock.history(period="1d")
        if not data.empty:
            return data['Close'].iloc[-1]
        return None
    except Exception as e:
        st.warning(f"Error fetching yfinance current price for {symbol}: {e}")
        return None

# Function to calculate resistance level (highest high over last 30 days)
@st.cache_data
def get_resistance_level(symbol, end_date):
    try:
        stock = yf.Ticker(symbol)
        start_date = end_date - timedelta(days=30)
        data = stock.history(start=start_date, end=end_date + timedelta(days=1), interval="1d")
        if not data.empty:
            return data['High'].max()
        return None
    except Exception as e:
        st.warning(f"Error calculating resistance for {symbol}: {e}")
        return None

# Function to get stock tickers from Nifty500.csv
def get_stock_tickers():
    try:
        df = pd.read_csv('Nifty500.csv')
        if not all(col in df.columns for col in ['Company Name', 'Industry', 'Symbol']):
            st.error("Nifty500.csv is missing required columns: 'Company Name', 'Industry', 'Symbol'")
            return pd.DataFrame()
        df = df[df['Series'] == 'EQ']
        df = df.rename(columns={'Company Name': 'Company_Name', 'Industry': 'Industry', 'Symbol': 'Symbol'})
        return df[['Company_Name', 'Industry', 'Symbol']]
    except FileNotFoundError:
        st.error("Nifty500.csv file not found in the current directory.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error reading Nifty500.csv: {e}")
        return pd.DataFrame()

# Function to fetch stock data for multiple tickers
@st.cache_data
def get_stock_data(symbols, start_date, end_date):
    try:
        tickers = " ".join([f"{s}.NS" for s in symbols])
        data = yf.download(tickers, start=start_date, end=end_date + timedelta(days=1), interval="1d", group_by='ticker')
        return data
    except Exception as e:
        st.warning(f"Error fetching historical data for {symbols}: {e}")
        return None

# Function to find the previous trading day
def get_previous_trading_day(data, symbol, selected_date):
    try:
        ticker_data = data[symbol] if symbol in data else None
        if ticker_data is None or ticker_data.empty:
            return None
        trading_days = ticker_data.index
        selected_date_str = selected_date.strftime('%Y-%m-%d')
        if selected_date_str not in trading_days.strftime('%Y-%m-%d'):
            return None
        selected_idx = trading_days.get_loc(selected_date_str)
        for i in range(selected_idx - 1, max(selected_idx - 10, -1), -1):
            prev_date = trading_days[i].date()
            if prev_date.weekday() < 5:  # Assume Monday-Friday are trading days
                return prev_date
        return None
    except Exception as e:
        st.warning(f"Error finding previous trading day for {symbol}: {e}")
        return None

# Function to load results from CSV if it exists
def load_results_from_csv(date):
    result_file = f"results_{date.strftime('%Y-%m-%d')}.csv"
    if os.path.exists(result_file):
        try:
            df = pd.read_csv(result_file)
            return df
        except Exception as e:
            st.warning(f"Error loading results from {result_file}: {e}")
            return pd.DataFrame()
    return pd.DataFrame()

# Function to save results to CSV
def save_results_to_csv(results, date):
    result_file = f"results_{date.strftime('%Y-%m-%d')}.csv"
    try:
        df = pd.DataFrame(results)
        df.to_csv(result_file, index=False)
        st.success(f"Results saved to {result_file}")
    except Exception as e:
        st.error(f"Error saving results to {result_file}: {e}")

# Streamlit app
st.title("NIFTY 500 Stocks: Open Price Above Previous Trading Day's High and Close")
st.write("Select a trading day and sectors to compare open price with the previous trading day's high and closing prices.")
st.write("Note: Today is Monday, June 30, 2025, 05:35 PM IST. June 28 and June 29, 2025, were non-trading days. Please select a valid trading day.")

# Date input
today = datetime.now(ist).date()
default_date = today - timedelta(days=0)  # Default to June 27, 2025 (Friday)
selected_date = st.date_input("Select a trading day", value=default_date, min_value=today - timedelta(days=365), max_value=today)

# Checkboxes
force_reprocess = st.checkbox("Force Reprocess (ignore existing results)", value=False)
filter_current_price = st.checkbox("Filter by Current Price Above Previous High and Close", value=False)
filter_resistance = st.checkbox("Exclude Stocks Near Resistance (within 2%)", value=False)

# Fetch all tickers and industries
df_tickers = get_stock_tickers()
if df_tickers.empty:
    st.stop()

# Get unique sectors
sectors = sorted(df_tickers['Industry'].unique())

# Initialize session state for checkboxes
if 'sector_states' not in st.session_state:
    st.session_state.sector_states = {sector: True for sector in sectors}

# Sector selection
st.subheader("Select Sectors")
col1, col2 = st.columns(2)
with col1:
    if st.button("Select All"):
        for sector in sectors:
            st.session_state.sector_states[sector] = True
with col2:
    if st.button("Select None"):
        for sector in sectors:
            st.session_state.sector_states[sector] = False

# Display checkboxes
selected_sectors = []
for sector in sectors:
    st.session_state.sector_states[sector] = st.checkbox(sector, value=st.session_state.sector_states[sector])
    if st.session_state.sector_states[sector]:
        selected_sectors.append(sector)

# Filter tickers based on selected sectors
selected_tickers = df_tickers[df_tickers['Industry'].isin(selected_sectors)][['Company_Name', 'Industry', 'Symbol']]

# Display total number of selected tickers
st.write(f"Total number of tickers to process: {len(selected_tickers)}")

# Stock selection for candlestick chart
st.subheader("Select Stock for Candlestick Chart")
selected_stock = st.selectbox("Choose a stock symbol", options=['None'] + selected_tickers['Symbol'].tolist(), index=0)

# Process button
process_button = st.button("Process Tickers")

# Define grid options for AgGrid
grid_options = {
    "rowSelection": "multiple",
    "enableSorting": True,
    "enableFilter": True,
    "enableColResize": True,
    "pagination": True,
    "paginationPageSize": 10,
    "domLayout": "autoWidth",
    "autoSizeStrategy": {
        "type": "fitGridWidth",
        "defaultMinWidth": 100
    }
}

# Container for on-the-fly results
results_container = st.container()

# Check for existing results
existing_results = load_results_from_csv(selected_date)

# Process tickers or load existing results
if process_button:
    if selected_tickers.empty:
        st.error("No tickers selected. Please select at least one sector.")
        st.stop()

    results = []
    near_resistance_stocks = []
    near_previous_high_stocks = []
    resistance_above_3pct_stocks = []
    opened_above_high_near_close_stocks = []
    open_near_prev_close_cross_high_stocks = []
    open_near_prev_close_high_within_half_pct_stocks = []

    if not force_reprocess and not existing_results.empty:
        # Filter existing results and populate additional tables
        filtered_results = existing_results
        near_previous_high_df = pd.DataFrame(columns=['Symbol', 'Company Name', 'Industry', 'Selected Trading Day', 'Open Price', 
                                                      'Previous Trading Day', 'Previous High', 'Percentage Above Previous High'])
        opened_above_high_near_close_df = pd.DataFrame(columns=['Symbol', 'Company Name', 'Industry', 'Previous Trading Day', 
                                                                'Previous High', 'Previous Close', 'Current Price', 'Percentage to Previous Close'])
        open_near_prev_close_cross_high_df = pd.DataFrame(columns=['Symbol', 'Company Name', 'Industry', 'Selected Trading Day', 'Open Price', 
                                                                  'Previous Trading Day', 'Previous Close', 'Current Price', 'Percentage to Previous Close'])
        open_near_prev_close_high_within_half_pct_df = pd.DataFrame(columns=['Symbol', 'Company Name', 'Industry', 'Selected Trading Day', 'Open Price', 
                                                                            'Previous Trading Day', 'Previous Close', 'Current Price', 'Percentage to Previous Close'])
        for idx, row in existing_results.iterrows():
            open_price = row['Open Price']
            prev_high = row['Previous High']
            prev_close = row['Previous Close']
            symbol = row['Symbol']
            yf_symbol = f"{symbol}.NS"
            # Check near-previous-high condition
            if prev_high <= open_price <= prev_high * 1.005:
                percentage_above = ((open_price - prev_high) / prev_high) * 100
                near_previous_high_stocks.append({
                    'Symbol': symbol,
                    'Company Name': row['Company Name'],
                    'Industry': row['Industry'],
                    'Selected Trading Day': row['Selected Trading Day'],
                    'Open Price': round(open_price, 2),
                    'Previous Trading Day': row['Previous Trading Day'],
                    'Previous High': round(prev_high, 2),
                    'Percentage Above Previous High': round(percentage_above, 2)
                })
            # Check opened-above-high-near-close condition
            if prev_high > prev_close:
                current_price = get_current_price_nse(symbol) or get_current_price_yfinance(yf_symbol)
                if current_price is not None and open_price > prev_high and prev_close * 0.998 <= current_price <= prev_close * 1.002:
                    percentage_to_close = ((current_price - prev_close) / prev_close) * 100
                    opened_above_high_near_close_stocks.append({
                        'Symbol': symbol,
                        'Company Name': row['Company Name'],
                        'Industry': row['Industry'],
                        'Previous Trading Day': row['Previous Trading Day'],
                        'Previous High': round(prev_high, 2),
                        'Previous Close': round(prev_close, 2),
                        'Current Price': round(current_price, 2),
                        'Percentage to Previous Close': round(percentage_to_close, 2)
                    })
            # Check open near previous close and current price above previous high
            current_price = get_current_price_nse(symbol) or get_current_price_yfinance(yf_symbol)
            if current_price is not None and prev_close * 0.998 <= open_price <= prev_close * 1.002 and current_price > prev_high:
                percentage_to_close = ((open_price - prev_close) / prev_close) * 100
                open_near_prev_close_cross_high_stocks.append({
                    'Symbol': symbol,
                    'Company Name': row['Company Name'],
                    'Industry': row['Industry'],
                    'Selected Trading Day': row['Selected Trading Day'],
                    'Open Price': round(open_price, 2),
                    'Previous Trading Day': row['Previous Trading Day'],
                    'Previous Close': round(prev_close, 2),
                    'Current Price': round(current_price, 2),
                    'Percentage to Previous Close': round(percentage_to_close, 2)
                })
            # Check new condition: open near previous close, current price above previous high, and prev high-close <= 0.5%
            if current_price is not None and prev_close * 0.998 <= open_price <= prev_close * 1.002 and current_price > prev_high and ((prev_high - prev_close) / prev_close) <= 0.005:
                percentage_to_close = ((open_price - prev_close) / prev_close) * 100
                open_near_prev_close_high_within_half_pct_stocks.append({
                    'Symbol': symbol,
                    'Company Name': row['Company Name'],
                    'Industry': row['Industry'],
                    'Selected Trading Day': row['Selected Trading Day'],
                    'Open Price': round(open_price, 2),
                    'Previous Trading Day': row['Previous Trading Day'],
                    'Previous Close': round(prev_close, 2),
                    'Current Price': round(current_price, 2),
                    'Percentage to Previous Close': round(percentage_to_close, 2)
                })

        if filter_current_price or filter_resistance:
            filtered_results = filtered_results.copy()
            filtered_results['Current Price Valid'] = True
            if filter_resistance:
                filtered_results['Resistance Valid'] = True
                near_resistance_df = pd.DataFrame(columns=['Symbol', 'Company Name', 'Industry', 'Current Price', 'Resistance Level', 'Percentage to Resistance'])
                resistance_above_3pct_df = pd.DataFrame(columns=['Symbol', 'Company Name', 'Industry', 'Selected Trading Day', 'Open Price', 
                                                                'Previous Trading Day', 'Previous High', 'Previous Close', 'Current Price', 
                                                                'Resistance Level', 'Percentage to Resistance'])
            for idx, row in filtered_results.iterrows():
                symbol = row['Symbol']
                yf_symbol = f"{symbol}.NS"
                current_price = get_current_price_nse(symbol) or get_current_price_yfinance(yf_symbol)
                if filter_current_price and (current_price is None or not (current_price > row['Previous High'] and current_price > row['Previous Close'])):
                    filtered_results.at[idx, 'Current Price Valid'] = False
                if filter_resistance and current_price is not None:
                    resistance = get_resistance_level(yf_symbol, selected_date)
                    if resistance is not None:
                        percentage_to_resistance = ((resistance - current_price) / resistance) * 100
                        if current_price >= resistance * 0.98:
                            filtered_results.at[idx, 'Resistance Valid'] = False
                            near_resistance_stocks.append({
                                'Symbol': symbol,
                                'Company Name': row['Company Name'],
                                'Industry': row['Industry'],
                                'Current Price': round(current_price, 2),
                                'Resistance Level': round(resistance, 2),
                                'Percentage to Resistance': round(percentage_to_resistance, 2)
                            })
                        elif current_price <= resistance * 0.97:
                            resistance_above_3pct_stocks.append({
                                'Symbol': row['Symbol'],
                                'Company Name': row['Company Name'],
                                'Industry': row['Industry'],
                                'Selected Trading Day': row['Selected Trading Day'],
                                'Open Price': row['Open Price'],
                                'Previous Trading Day': row['Previous Trading Day'],
                                'Previous High': row['Previous High'],
                                'Previous Close': row['Previous Close'],
                                'Current Price': round(current_price, 2),
                                'Resistance Level': round(resistance, 2),
                                'Percentage to Resistance': round(percentage_to_resistance, 2)
                            })
            if filter_resistance:
                near_resistance_df = pd.DataFrame(near_resistance_stocks)
                filtered_results = filtered_results[filtered_results['Resistance Valid']]
                resistance_above_3pct_df = pd.DataFrame(resistance_above_3pct_stocks)
            filtered_results = filtered_results[filtered_results['Current Price Valid']].drop(columns=['Current Price Valid'] + (['Resistance Valid'] if filter_resistance else []))
        near_previous_high_df = pd.DataFrame(near_previous_high_stocks)
        opened_above_high_near_close_df = pd.DataFrame(opened_above_high_near_close_stocks)
        open_near_prev_close_cross_high_df = pd.DataFrame(open_near_prev_close_cross_high_stocks)
        open_near_prev_close_high_within_half_pct_df = pd.DataFrame(open_near_prev_close_high_within_half_pct_stocks)

        # Display existing results with AgGrid
        with results_container:
            st.write(f"Loaded existing results for {selected_date.strftime('%Y-%m-%d')} from CSV:")
            st.write("Stocks meeting the condition (open > previous high and close):")
            AgGrid(filtered_results, grid_options=grid_options, key="main_results")
            st.write(f"Found {len(filtered_results)} stocks meeting the condition.")
            if filter_resistance and not near_resistance_df.empty:
                st.write("Stocks excluded due to being near resistance (within 2%):")
                AgGrid(near_resistance_df, grid_options=grid_options, key="near_resistance")
                st.write(f"Found {len(near_resistance_df)} stocks near resistance.")
            if filter_resistance and not resistance_above_3pct_df.empty:
                st.write("Stocks meeting the condition with resistance at least 3% above current price:")
                AgGrid(resistance_above_3pct_df, grid_options=grid_options, key="resistance_above_3pct")
                st.write(f"Found {len (resistance_above_3pct_df)} stocks with resistance above 3%.")
            if not near_previous_high_df.empty:
                st.write("Stocks with open price within 0.5% above previous trading day's high:")
                AgGrid(near_previous_high_df, grid_options=grid_options, key="near_previous_high")
                st.write(f"Found {len(near_previous_high_df)} stocks near previous high.")
            if not opened_above_high_near_close_df.empty:
                st.write("Stocks where previous high > close, opened above high, and current price within 0.2% of previous close:")
                AgGrid(opened_above_high_near_close_df, grid_options=grid_options, key="opened_above_high_near_close")
                st.write(f"Found {len(opened_above_high_near_close_df)} stocks meeting this condition.")
            if not open_near_prev_close_cross_high_df.empty:
                st.write("Stocks where open price is within 0.2% of previous close and current price is above previous high:")
                AgGrid(open_near_prev_close_cross_high_df, grid_options=grid_options, key="open_near_prev_close_cross_high")
                st.write(f"Found {len(open_near_prev_close_cross_high_df)} stocks meeting this condition.")
            if not open_near_prev_close_high_within_half_pct_df.empty:
                st.write("Stocks where open price is within 0.2% of previous close, current price is above previous high, and previous high-close difference is not more than 0.5%:")
                AgGrid(open_near_prev_close_high_within_half_pct_df, grid_options=grid_options, key="open_near_prev_close_high_within_half_pct")
                st.write(f"Found {len(open_near_prev_close_high_within_half_pct_df)} stocks meeting this condition.")
                # Histogram for Percentage to Previous Close
                fig_hist = px.histogram(open_near_prev_close_high_within_half_pct_df, x='Percentage to Previous Close', 
                                        title='Distribution of Percentage to Previous Close')
                st.plotly_chart(fig_hist)
            # Sector distribution pie chart
            if not filtered_results.empty:
                sector_counts = filtered_results['Industry'].value_counts()
                fig_pie = px.pie(values=sector_counts.values, names=sector_counts.index, title="Sector Distribution of Qualifying Stocks")
                st.plotly_chart(fig_pie)
    else:
        # Initialize NSE session
        initialize_nse_session()

        # Calculate start date for fetching data
        start_date = selected_date - timedelta(days=10)
        end_date = selected_date

        # Fetch data for all selected tickers
        data = get_stock_data(selected_tickers['Symbol'].tolist(), start_date, end_date)

        # Progress bar and counter
        progress_bar = st.progress(0)
        progress_text = st.empty()
        total_tickers = len(selected_tickers)

        # Create placeholders for on-the-fly updates
        with results_container:
            st.write("Stocks meeting the condition (open > previous high and close):")
            results_placeholder = st.empty()
            results_df = pd.DataFrame(columns=['Symbol', 'Company Name', 'Industry', 'Selected Trading Day', 'Open Price', 
                                             'Previous Trading Day', 'Previous High', 'Previous Close'])
            if filter_resistance:
                st.write("Stocks near resistance (within 2%):")
                near_resistance_placeholder = st.empty()
                near_resistance_df = pd.DataFrame(columns=['Symbol', 'Company Name', 'Industry', 'Current Price', 'Resistance Level', 'Percentage to Resistance'])
                st.write("Stocks meeting the condition with resistance at least 3% above current price:")
                resistance_above_3pct_placeholder = st.empty()
                resistance_above_3pct_df = pd.DataFrame(columns=['Symbol', 'Company Name', 'Industry', 'Selected Trading Day', 'Open Price', 
                                                                'Previous Trading Day', 'Previous High', 'Previous Close', 'Current Price', 
                                                                'Resistance Level', 'Percentage to Resistance'])
            st.write("Stocks with open price within 0.5% above previous trading day's high:")
            near_previous_high_placeholder = st.empty()
            near_previous_high_df = pd.DataFrame(columns=['Symbol', 'Company Name', 'Industry', 'Selected Trading Day', 'Open Price', 
                                                         'Previous Trading Day', 'Previous High', 'Percentage Above Previous High'])
            st.write("Stocks where previous high > close, opened above high, and current price within 0.2% of previous close:")
            opened_above_high_near_close_placeholder = st.empty()
            opened_above_high_near_close_df = pd.DataFrame(columns=['Symbol', 'Company Name', 'Industry', 'Previous Trading Day', 
                                                                   'Previous High', 'Previous Close', 'Current Price', 'Percentage to Previous Close'])
            st.write("Stocks where open price is within 0.2% of previous close and current price is above previous high:")
            open_near_prev_close_cross_high_placeholder = st.empty()
            open_near_prev_close_cross_high_df = pd.DataFrame(columns=['Symbol', 'Company Name', 'Industry', 'Selected Trading Day', 'Open Price', 
                                                                      'Previous Trading Day', 'Previous Close', 'Current Price', 'Percentage to Previous Close'])
            st.write("Stocks where open price is within 0.2% of previous close, current price is above previous high, and previous high-close difference is not more than 0.5%:")
            open_near_prev_close_high_within_half_pct_placeholder = st.empty()
            open_near_prev_close_high_within_half_pct_df = pd.DataFrame(columns=['Symbol', 'Company Name', 'Industry', 'Selected Trading Day', 'Open Price', 
                                                                                'Previous Trading Day', 'Previous Close', 'Current Price', 'Percentage to Previous Close'])

        for i, row in enumerate(selected_tickers.itertuples()):
            symbol = row.Symbol
            company_name = row.Company_Name
            industry = row.Industry
            yf_symbol = f"{symbol}.NS"

            # Update progress bar and counter
            progress_bar.progress((i + 1) / total_tickers)
            progress_text.text(f"Processed {i + 1}/{total_tickers} tickers: {symbol}")

            if data is not None and yf_symbol in data and not data[yf_symbol].empty:
                # Find previous trading day
                prev_trading_date = get_previous_trading_day(data, yf_symbol, selected_date)
                if prev_trading_date is None:
                    st.warning(f"No trading data available for {symbol} on {selected_date} or no previous trading day found.")
                    continue

                # Filter data for the selected date
                selected_date_str = selected_date.strftime('%Y-%m-%d')
                if selected_date_str in data[yf_symbol].index.strftime('%Y-%m-%d'):
                    selected_data = data[yf_symbol].loc[selected_date_str]

                    # Get previous trading day data
                    prev_data = data[yf_symbol].loc[prev_trading_date.strftime('%Y-%m-%d')]

                    # Extract dates
                    selected_trading_date = selected_data.name.date()

                    # Get price data
                    open_price = selected_data['Open']
                    prev_high = prev_data['High']
                    prev_close = prev_data['Close']

                    # Fetch current price for additional conditions
                    current_price = get_current_price_nse(symbol) or get_current_price_yfinance(yf_symbol)

                    # Check open near previous close and current price above previous high
                    if current_price is not None and prev_close * 0.998 <= open_price <= prev_close * 1.002 and current_price > prev_high:
                        percentage_to_close = ((open_price - prev_close) / prev_close) * 100
                        open_near_prev_close_cross_high_stocks.append({
                            'Symbol': symbol,
                            'Company Name': company_name,
                            'Industry': industry,
                            'Selected Trading Day': selected_trading_date,
                            'Open Price': round(open_price, 2),
                            'Previous Trading Day': prev_trading_date,
                            'Previous Close': round(prev_close, 2),
                            'Current Price': round(current_price, 2),
                            'Percentage to Previous Close': round(percentage_to_close, 2)
                        })

                    # Check new condition: open near previous close, current price above previous high, and prev high-close <= 0.5%
                    if current_price is not None and prev_close * 0.998 <= open_price <= prev_close * 1.002 and current_price > prev_high and ((prev_high - prev_close) / prev_close) <= 0.005:
                        percentage_to_close = ((open_price - prev_close) / prev_close) * 100
                        open_near_prev_close_high_within_half_pct_stocks.append({
                            'Symbol': symbol,
                            'Company Name': company_name,
                            'Industry': industry,
                            'Selected Trading Day': selected_trading_date,
                            'Open Price': round(open_price, 2),
                            'Previous Trading Day': prev_trading_date,
                            'Previous Close': round(prev_close, 2),
                            'Current Price': round(current_price, 2),
                            'Percentage to Previous Close': round(percentage_to_close, 2)
                        })

                    # Check opened-above-high-near-close condition
                    if prev_high > prev_close and current_price is not None and open_price > prev_high and prev_close * 0.998 <= current_price <= prev_close * 1.002:
                        percentage_to_close = ((current_price - prev_close) / prev_close) * 100
                        opened_above_high_near_close_stocks.append({
                            'Symbol': symbol,
                            'Company Name': company_name,
                            'Industry': industry,
                            'Previous Trading Day': prev_trading_date,
                            'Previous High': round(prev_high, 2),
                            'Previous Close': round(prev_close, 2),
                            'Current Price': round(current_price, 2),
                            'Percentage to Previous Close': round(percentage_to_close, 2)
                        })

                    # Check if open price is within 0.5% above previous high
                    if prev_high <= open_price <= prev_high * 1.005:
                        percentage_above = ((open_price - prev_high) / prev_high) * 100
                        near_previous_high_stocks.append({
                            'Symbol': symbol,
                            'Company Name': company_name,
                            'Industry': industry,
                            'Selected Trading Day': selected_trading_date,
                            'Open Price': round(open_price, 2),
                            'Previous Trading Day': prev_trading_date,
                            'Previous High': round(prev_high, 2),
                            'Percentage Above Previous High': round(percentage_above, 2)
                        })

                    # Check if selected day's open is above previous day's high and close
                    if open_price > prev_high and open_price > prev_close:
                        # Check current price and resistance conditions
                        include_stock = True
                        if filter_current_price and (current_price is None or not (current_price > prev_high and current_price > prev_close)):
                            include_stock = False
                        if filter_resistance and current_price is not None:
                            resistance = get_resistance_level(yf_symbol, selected_date)
                            if resistance is not None:
                                percentage_to_resistance = ((resistance - current_price) / resistance) * 100
                                if current_price >= resistance * 0.98:
                                    include_stock = False
                                    near_resistance_stocks.append({
                                        'Symbol': symbol,
                                        'Company Name': company_name,
                                        'Industry': industry,
                                        'Current Price': round(current_price, 2),
                                        'Resistance Level': round(resistance, 2),
                                        'Percentage to Resistance': round(percentage_to_resistance, 2)
                                    })
                                elif current_price <= resistance * 0.97:
                                    resistance_above_3pct_stocks.append({
                                        'Symbol': symbol,
                                        'Company Name': company_name,
                                        'Industry': industry,
                                        'Selected Trading Day': selected_trading_date,
                                        'Open Price': round(open_price, 2),
                                        'Previous Trading Day': prev_trading_date,
                                        'Previous High': round(prev_high, 2),
                                        'Previous Close': round(prev_close, 2),
                                        'Current Price': round(current_price, 2),
                                        'Resistance Level': round(resistance, 2),
                                        'Percentage to Resistance': round(percentage_to_resistance, 2)
                                    })

                        if include_stock:
                            result = {
                                'Symbol': symbol,
                                'Company Name': company_name,
                                'Industry': industry,
                                'Selected Trading Day': selected_trading_date,
                                'Open Price': round(open_price, 2),
                                'Previous Trading Day': prev_trading_date,
                                'Previous High': round(prev_high, 2),
                                'Previous Close': round(prev_close, 2)
                            }
                            results.append(result)

        # Update DataFrames
        results_df = pd.DataFrame(results)
        near_resistance_df = pd.DataFrame(near_resistance_stocks)
        resistance_above_3pct_df = pd.DataFrame(resistance_above_3pct_stocks)
        near_previous_high_df = pd.DataFrame(near_previous_high_stocks)
        opened_above_high_near_close_df = pd.DataFrame(opened_above_high_near_close_stocks)
        open_near_prev_close_cross_high_df = pd.DataFrame(open_near_prev_close_cross_high_stocks)
        open_near_prev_close_high_within_half_pct_df = pd.DataFrame(open_near_prev_close_high_within_half_pct_stocks)

        # Update all tables at once
        with results_container:
            st.write("Stocks meeting the condition (open > previous high and close):")
            results_placeholder.write(AgGrid(results_df, grid_options=grid_options, key="main_results_dynamic"))
            if filter_resistance:
                st.write("Stocks near resistance (within 2%):")
                near_resistance_placeholder.write(AgGrid(near_resistance_df, grid_options=grid_options, key="near_resistance_dynamic"))
                st.write("Stocks meeting the condition with resistance at least 3% above current price:")
                resistance_above_3pct_placeholder.write(AgGrid(resistance_above_3pct_df, grid_options=grid_options, key="resistance_above_3pct_dynamic"))
            st.write("Stocks with open price within 0.5% above previous trading day's high:")
            near_previous_high_placeholder.write(AgGrid(near_previous_high_df, grid_options=grid_options, key="near_previous_high_dynamic"))
            st.write("Stocks where previous high > close, opened above high, and current price within 0.2% of previous close:")
            opened_above_high_near_close_placeholder.write(AgGrid(opened_above_high_near_close_df, grid_options=grid_options, key="opened_above_high_near_close_dynamic"))
            st.write("Stocks where open price is within 0.2% of previous close and current price is above previous high:")
            open_near_prev_close_cross_high_placeholder.write(AgGrid(open_near_prev_close_cross_high_df, grid_options=grid_options, key="open_near_prev_close_cross_high_dynamic"))
            st.write("Stocks where open price is within 0.2% of previous close, current price is above previous high, and previous high-close difference is not more than 0.5%:")
            open_near_prev_close_high_within_half_pct_placeholder.write(AgGrid(open_near_prev_close_high_within_half_pct_df, grid_options=grid_options, key="open_near_prev_close_high_within_half_pct_dynamic"))

        # Save results to CSV
        if results:
            save_results_to_csv(results, selected_date)

        # Clear progress text after completion
        progress_text.empty()

        # Final message and visualizations
        with results_container:
            if not results:
                st.info(f"No stocks found where the open price on {selected_date} was above the previous trading day's high and closing prices" + 
                        (f" and current price is above previous high and close" if filter_current_price else "") + 
                        (f" and current price is not near resistance" if filter_resistance else "") + ".")
            else:
                st.write(f"Found {len(results)} stocks meeting the condition.")
                # Sector distribution pie chart
                sector_counts = results_df['Industry'].value_counts()
                fig_pie = px.pie(values=sector_counts.values, names=sector_counts.index, title="Sector Distribution of Qualifying Stocks")
                st.plotly_chart(fig_pie)
            if filter_resistance and near_resistance_stocks:
                st.write(f"Found {len(near_resistance_stocks)} stocks near resistance.")
            if filter_resistance and resistance_above_3pct_stocks:
                st.write(f"Found {len(resistance_above_3pct_stocks)} stocks with resistance at least 3% above current price.")
            if near_previous_high_stocks:
                st.write(f"Found {len(near_previous_high_stocks)} stocks with open price within 0.5% above previous trading day's high.")
            if opened_above_high_near_close_stocks:
                st.write(f"Found {len(opened_above_high_near_close_stocks)} stocks where previous high > close, opened above high, and current price within 0.2% of previous close.")
            if open_near_prev_close_cross_high_stocks:
                st.write(f"Found {len(open_near_prev_close_cross_high_stocks)} stocks where open price is within 0.2% of previous close and current price is above previous high.")
            if open_near_prev_close_high_within_half_pct_stocks:
                st.write(f"Found {len(open_near_prev_close_high_within_half_pct_stocks)} stocks where open price is within 0.2% of previous close, current price is above previous high, and previous high-close difference is not more than 0.5%.")
                # Histogram for Percentage to Previous Close
                fig_hist = px.histogram(open_near_prev_close_high_within_half_pct_df, x='Percentage to Previous Close', 
                                        title='Distribution of Percentage to Previous Close')
                st.plotly_chart(fig_hist)

    # Candlestick chart for selected stock
    if selected_stock != 'None':
        yf_symbol = f"{selected_stock}.NS"
        start_date = selected_date - timedelta(days=30)
        stock_data = get_stock_data([selected_stock], start_date, selected_date)
        if stock_data is not None and yf_symbol in stock_data and not stock_data[yf_symbol].empty:
            fig_candle = go.Figure(data=[
                go.Candlestick(
                    x=stock_data[yf_symbol].index,
                    open=stock_data[yf_symbol]['Open'],
                    high=stock_data[yf_symbol]['High'],
                    low=stock_data[yf_symbol]['Low'],
                    close=stock_data[yf_symbol]['Close']
                )
            ])
            fig_candle.update_layout(title=f"Candlestick Chart for {selected_stock}", xaxis_title="Date", yaxis_title="Price (₹)")
            st.plotly_chart(fig_candle)
        else:
            st.warning(f"No data available for {selected_stock} to display candlestick chart.")

# Footer
st.write("Data sourced from NSE and yfinance. Note: Market data is subject to delays and availability.")