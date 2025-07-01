import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import pytz
import os

# Set timezone to IST
ist = pytz.timezone('Asia/Kolkata')

# Function to fetch current price from yfinance
def get_current_price_yfinance(symbols):
    try:
        data = yf.download(symbols, period="1d", threads=True)['Close']
        if isinstance(data, pd.Series):
            return data.to_dict()
        elif isinstance(data, pd.DataFrame):
            return data.iloc[-1].to_dict()
        return {}
    except Exception as e:
        st.warning(f"Error fetching yfinance current prices: {e}")
        return {}

# Function to fetch stock data for multiple tickers
def get_stock_data(symbols, start_date, end_date):
    try:
        data = yf.download(symbols, start=start_date, end=end_date + timedelta(days=1), interval="1d", threads=True, group_by='ticker')
        return data
    except Exception as e:
        st.warning(f"Error fetching historical data: {e}")
        return None

# Function to find the previous trading day
def get_previous_trading_day(data, symbol, selected_date):
    if symbol in data:
        trading_days = data[symbol].index
        selected_date_str = selected_date.strftime('%Y-%m-%d')
        if selected_date_str not in trading_days.strftime('%Y-%m-%d'):
            return None
        selected_idx = trading_days.get_loc(selected_date_str)
        if selected_idx == 0:
            return None
        return trading_days[selected_idx - 1].date()
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
st.write("Note: Today is Tuesday, July 01, 2025, 12:12 PM IST. June 28 and June 29, 2025, were non-trading days. Please select a valid trading day.")

# Date input
today = datetime.now(ist).date()
default_date = today - timedelta(days=1)  # Default to June 30, 2025 (Monday)
selected_date = st.date_input("Select a trading day", value=default_date, min_value=today - timedelta(days=365), max_value=today)

# Textbox for low-high difference percentage
user_low_high_diff_pct = st.number_input("Enter maximum previous low-high difference percentage (%)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)

# Checkboxes
force_reprocess = st.checkbox("Force Reprocess (ignore existing results)", value=False)
filter_current_price = st.checkbox("Filter by Current Price Above Previous High and Close", value=False)

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

# Process button
process_button = st.button("Process Tickers")

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
    near_previous_high_stocks = []
    opened_above_high_near_close_stocks = []
    open_near_prev_close_cross_high_stocks = []
    open_near_prev_close_high_within_half_pct_stocks = []
    open_near_prev_close_high_within_half_pct_low_high_below_1pct_stocks = []

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
        open_near_prev_close_high_within_half_pct_low_high_below_1pct_df = pd.DataFrame(columns=['Symbol', 'Company Name', 'Industry', 'Selected Trading Day', 'Open Price', 
                                                                                               'Previous Trading Day', 'Previous Close', 'Current Price', 'Percentage to Previous Close', 
                                                                                               'Previous Low', 'Previous High', 'Previous Low-High %'])
        
        # Fetch current prices for all symbols
        yf_symbols = [f"{symbol}.NS" for symbol in existing_results['Symbol']]
        current_prices = get_current_price_yfinance(yf_symbols)

        # Fetch historical data for validation
        start_date = selected_date - timedelta(days=10)
        end_date = selected_date
        data = get_stock_data(yf_symbols, start_date, end_date)

        for idx, row in existing_results.iterrows():
            symbol = row['Symbol']
            yf_symbol = f"{symbol}.NS"
            open_price = row['Open Price']
            prev_high = row['Previous High']
            prev_close = row['Previous Close']
            
            # Validate previous trading day's data
            if data is not None and yf_symbol in data:
                prev_trading_date = get_previous_trading_day(data, yf_symbol, selected_date)
                if prev_trading_date is not None:
                    prev_data = data[yf_symbol].loc[prev_trading_date.strftime('%Y-%m-%d')]
                    if not prev_data.isna().any():  # Ensure no NaN values
                        prev_low = prev_data['Low']
                        prev_high = prev_data['High']
                        prev_close = prev_data['Close']
                        if prev_low > 0:  # Ensure valid low price
                            low_high_diff_pct = ((prev_high - prev_low) / prev_low) * 100
                        else:
                            low_high_diff_pct = float('inf')
                            st.warning(f"Skipping {symbol}: Previous trading day low price is zero or invalid.")
                            continue
                    else:
                        low_high_diff_pct = float('inf')
                        st.warning(f"Skipping {symbol}: Previous trading day data contains NaN values.")
                        continue
                else:
                    low_high_diff_pct = float('inf')
                    st.warning(f"Skipping {symbol}: No previous trading day found for {selected_date}.")
                    continue
            else:
                low_high_diff_pct = float('inf')
                st.warning(f"Skipping {symbol}: No historical data available.")
                continue

            # Check near-previous-high condition
            if prev_high <= open_price <= prev_high * 1.005:
                percentage_above = ((open_price - prev_high) / prev_high) * 100
                near_previous_high_stocks.append({
                    'Symbol': symbol,
                    'Company Name': row['Company Name'],
                    'Industry': row['Industry'],
                    'Selected Trading Day': selected_date,
                    'Open Price': round(open_price, 2),
                    'Previous Trading Day': row['Previous Trading Day'],
                    'Previous High': round(prev_high, 2),
                    'Percentage Above Previous High': round(percentage_above, 2)
                })

            # Check opened-above-high-near-close condition
            current_price = current_prices.get(yf_symbol)
            if prev_high > prev_close and current_price is not None and open_price > prev_high and prev_close * 0.998 <= current_price <= prev_close * 1.002:
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
            if current_price is not None and prev_close * 0.998 <= open_price <= prev_close * 1.002 and current_price > prev_high:
                percentage_to_close = ((open_price - prev_close) / prev_close) * 100
                open_near_prev_close_cross_high_stocks.append({
                    'Symbol': symbol,
                    'Company Name': row['Company Name'],
                    'Industry': row['Industry'],
                    'Selected Trading Day': selected_date,
                    'Open Price': round(open_price, 2),
                    'Previous Trading Day': row['Previous Trading Day'],
                    'Previous Close': round(prev_close, 2),
                    'Current Price': round(current_price, 2),
                    'Percentage to Previous Close': round(percentage_to_close, 2)
                })

            # Check condition: open near previous close, current price above previous high, and prev high-close <= 0.5%
            if current_price is not None and prev_close * 0.998 <= open_price <= prev_close * 1.002 and current_price > prev_high and ((prev_high - prev_close) / prev_close) <= 0.005:
                percentage_to_close = ((open_price - prev_close) / prev_close) * 100
                open_near_prev_close_high_within_half_pct_stocks.append({
                    'Symbol': symbol,
                    'Company Name': row['Company Name'],
                    'Industry': row['Industry'],
                    'Selected Trading Day': selected_date,
                    'Open Price': round(open_price, 2),
                    'Previous Trading Day': row['Previous Trading Day'],
                    'Previous Close': round(prev_close, 2),
                    'Current Price': round(current_price, 2),
                    'Percentage to Previous Close': round(percentage_to_close, 2)
                })

            # Check condition: open near previous close, current price above previous high, prev high-close <= 0.5%, and prev low-high < user-defined threshold
            if (current_price is not None and 
                prev_close * 0.998 <= open_price <= prev_close * 1.002 and 
                current_price > prev_high and 
                ((prev_high - prev_close) / prev_close) <= 0.005 and 
                low_high_diff_pct < user_low_high_diff_pct):
                percentage_to_close = ((open_price - prev_close) / prev_close) * 100
                open_near_prev_close_high_within_half_pct_low_high_below_1pct_stocks.append({
                    'Symbol': symbol,
                    'Company Name': row['Company Name'],
                    'Industry': row['Industry'],
                    'Selected Trading Day': selected_date,
                    'Open Price': round(open_price, 2),
                    'Previous Trading Day': prev_trading_date,
                    'Previous Close': round(prev_close, 2),
                    'Current Price': round(current_price, 2),
                    'Percentage to Previous Close': round(percentage_to_close, 2),
                    'Previous Low': round(prev_low, 2),
                    'Previous High': round(prev_high, 2),
                    'Previous Low-High %': round(low_high_diff_pct, 2)
                })
            elif low_high_diff_pct >= user_low_high_diff_pct:
                st.warning(f"{symbol} excluded from low-high table: Previous low-high difference ({low_high_diff_pct:.2f}%) >= {user_low_high_diff_pct}%.")

        if filter_current_price:
            filtered_results = filtered_results.copy()
            filtered_results['Current Price Valid'] = True
            for idx, row in filtered_results.iterrows():
                symbol = row['Symbol']
                yf_symbol = f"{symbol}.NS"
                current_price = current_prices.get(yf_symbol)
                if current_price is None or not (current_price > row['Previous High'] and current_price > row['Previous Close']):
                    filtered_results.at[idx, 'Current Price Valid'] = False
            filtered_results = filtered_results[filtered_results['Current Price Valid']].drop(columns=['Current Price Valid'])

        near_previous_high_df = pd.DataFrame(near_previous_high_stocks)
        opened_above_high_near_close_df = pd.DataFrame(opened_above_high_near_close_stocks)
        open_near_prev_close_cross_high_df = pd.DataFrame(open_near_prev_close_cross_high_stocks)
        open_near_prev_close_high_within_half_pct_df = pd.DataFrame(open_near_prev_close_high_within_half_pct_stocks)
        open_near_prev_close_high_within_half_pct_low_high_below_1pct_df = pd.DataFrame(open_near_prev_close_high_within_half_pct_low_high_below_1pct_stocks)

        # Display existing results
        with results_container:
            st.write(f"Loaded existing results for {selected_date.strftime('%Y-%m-%d')} from CSV:")
            st.write("Stocks meeting the condition (open > previous high and close):")
            st.dataframe(filtered_results)
            st.write(f"Found {len(filtered_results)} stocks meeting the condition.")
            if not near_previous_high_df.empty:
                st.write("Stocks with open price within 0.5% above previous trading day's high:")
                st.dataframe(near_previous_high_df)
                st.write(f"Found {len(near_previous_high_df)} stocks near previous high.")
            if not opened_above_high_near_close_df.empty:
                st.write("Stocks where previous high > close, opened above high, and current price within 0.2% of previous close:")
                st.dataframe(opened_above_high_near_close_df)
                st.write(f"Found {len(opened_above_high_near_close_df)} stocks meeting this condition.")
            if not open_near_prev_close_cross_high_df.empty:
                st.write("Stocks where open price is within 0.2% of previous close and current price is above previous high:")
                st.dataframe(open_near_prev_close_cross_high_df)
                st.write(f"Found {len(open_near_prev_close_cross_high_df)} stocks meeting this condition.")
            if not open_near_prev_close_high_within_half_pct_df.empty:
                st.write("Stocks where open price is within 0.2% of previous close, current price is above previous high, and previous high-close difference is not more than 0.5%:")
                st.dataframe(open_near_prev_close_high_within_half_pct_df)
                st.write(f"Found {len(open_near_prev_close_high_within_half_pct_df)} stocks meeting this condition.")
            if not open_near_prev_close_high_within_half_pct_low_high_below_1pct_df.empty:
                st.write("Stocks where open price is within 0.2% of previous close, current price is above previous high, previous high-close difference is not more than 0.5%, and previous low-high difference is below {user_low_high_diff_pct}%:")
                st.dataframe(open_near_prev_close_high_within_half_pct_low_high_below_1pct_df)
                st.write(f"Found {len(open_near_prev_close_high_within_half_pct_low_high_below_1pct_df)} stocks meeting this condition.")
    else:
        # Calculate start date for fetching data
        start_date = selected_date - timedelta(days=10)
        end_date = selected_date

        # Prepare symbols for batch download
        yf_symbols = [f"{symbol}.NS" for symbol in selected_tickers['Symbol']]
        
        # Fetch historical data for all tickers
        data = get_stock_data(yf_symbols, start_date, end_date)
        
        # Fetch current prices
        current_prices = get_current_price_yfinance(yf_symbols)

        # Progress bar and counter
        progress_bar = st.progress(0)
        progress_text = st.empty()
        total_tickers = len(selected_tickers)

        # Display initial empty DataFrames
        with results_container:
            st.write("Stocks meeting the condition (open > previous high and close, updated on the fly):")
            results_df = pd.DataFrame(columns=['Symbol', 'Company Name', 'Industry', 'Selected Trading Day', 'Open Price', 
                                             'Previous Trading Day', 'Previous High', 'Previous Close'])
            results_placeholder = st.dataframe(results_df)
            st.write("Stocks with open price within 0.5% above previous trading day's high (updated on the fly):")
            near_previous_high_df = pd.DataFrame(columns=['Symbol', 'Company Name', 'Industry', 'Selected Trading Day', 'Open Price', 
                                                         'Previous Trading Day', 'Previous High', 'Percentage Above Previous High'])
            near_previous_high_placeholder = st.dataframe(near_previous_high_df)
            st.write("Stocks where previous high > close, opened above high, and current price within 0.2% of previous close (updated on the fly):")
            opened_above_high_near_close_df = pd.DataFrame(columns=['Symbol', 'Company Name', 'Industry', 'Previous Trading Day', 
                                                                   'Previous High', 'Previous Close', 'Current Price', 'Percentage to Previous Close'])
            opened_above_high_near_close_placeholder = st.dataframe(opened_above_high_near_close_df)
            st.write("Stocks where open price is within 0.2% of previous close and current price is above previous high (updated on the fly):")
            open_near_prev_close_cross_high_df = pd.DataFrame(columns=['Symbol', 'Company Name', 'Industry', 'Selected Trading Day', 'Open Price', 
                                                                      'Previous Trading Day', 'Previous Close', 'Current Price', 'Percentage to Previous Close'])
            open_near_prev_close_cross_high_placeholder = st.dataframe(open_near_prev_close_cross_high_df)
            st.write("Stocks where open price is within 0.2% of previous close, current price is above previous high, and previous high-close difference is not more than 0.5% (updated on the fly):")
            open_near_prev_close_high_within_half_pct_df = pd.DataFrame(columns=['Symbol', 'Company Name', 'Industry', 'Selected Trading Day', 'Open Price', 
                                                                                'Previous Trading Day', 'Previous Close', 'Current Price', 'Percentage to Previous Close'])
            open_near_prev_close_high_within_half_pct_placeholder = st.dataframe(open_near_prev_close_high_within_half_pct_df)
            st.write("Stocks where open price is within 0.2% of previous close, current price is above previous high, previous high-close difference is not more than 0.5%, and previous low-high difference is below {user_low_high_diff_pct}% (updated on the fly):")
            open_near_prev_close_high_within_half_pct_low_high_below_1pct_df = pd.DataFrame(columns=['Symbol', 'Company Name', 'Industry', 'Selected Trading Day', 'Open Price', 
                                                                                                   'Previous Trading Day', 'Previous Close', 'Current Price', 'Percentage to Previous Close', 
                                                                                                   'Previous Low', 'Previous High', 'Previous Low-High %'])
            open_near_prev_close_high_within_half_pct_low_high_below_1pct_placeholder = st.dataframe(open_near_prev_close_high_within_half_pct_low_high_below_1pct_df)

        for i, row in enumerate(selected_tickers.itertuples()):
            symbol = row.Symbol
            company_name = row.Company_Name
            industry = row.Industry
            yf_symbol = f"{symbol}.NS"

            # Update progress bar and counter
            progress_bar.progress((i + 1) / total_tickers)
            progress_text.text(f"Processed {i + 1}/{total_tickers} tickers: {symbol}")

            if data is not None and yf_symbol in data:
                # Find previous trading day
                prev_trading_date = get_previous_trading_day(data, yf_symbol, selected_date)
                if prev_trading_date is None:
                    st.warning(f"No trading data available for {symbol} on {selected_date} or no previous trading day found.")
                    continue

                # Filter data for the selected date
                selected_date_str = selected_date.strftime('%Y-%m-%d')
                if selected_date_str in data[yf_symbol].index.strftime('%Y-%m-%d'):
                    selected_data = data[yf_symbol].loc[selected_date_str]
                    prev_data = data[yf_symbol].loc[prev_trading_date.strftime('%Y-%m-%d')]

                    # Check for valid data
                    if prev_data.isna().any() or selected_data.isna().any():
                        st.warning(f"Skipping {symbol}: Missing or invalid data for selected or previous trading day.")
                        continue

                    # Get price data
                    open_price = selected_data['Open']
                    prev_high = prev_data['High']
                    prev_close = prev_data['Close']
                    prev_low = prev_data['Low']
                    if prev_low > 0:
                        low_high_diff_pct = ((prev_high - prev_low) / prev_low) * 100
                    else:
                        low_high_diff_pct = float('inf')
                        st.warning(f"Skipping {symbol}: Previous trading day low price is zero or invalid.")
                        continue

                    # Fetch current price
                    current_price = current_prices.get(yf_symbol)

                    # Check open near previous close and current price above previous high
                    if current_price is not None and prev_close * 0.998 <= open_price <= prev_close * 1.002 and current_price > prev_high:
                        percentage_to_close = ((open_price - prev_close) / prev_close) * 100
                        open_near_prev_close_cross_high_stocks.append({
                            'Symbol': symbol,
                            'Company Name': company_name,
                            'Industry': industry,
                            'Selected Trading Day': selected_date,
                            'Open Price': round(open_price, 2),
                            'Previous Trading Day': prev_trading_date,
                            'Previous Close': round(prev_close, 2),
                            'Current Price': round(current_price, 2),
                            'Percentage to Previous Close': round(percentage_to_close, 2)
                        })
                        with results_container:
                            open_near_prev_close_cross_high_df = pd.DataFrame(open_near_prev_close_cross_high_stocks)
                            open_near_prev_close_cross_high_placeholder.dataframe(open_near_prev_close_cross_high_df)

                    # Check condition: open near previous close, current price above previous high, and prev high-close <= 0.5%
                    if current_price is not None and prev_close * 0.998 <= open_price <= prev_close * 1.002 and current_price > prev_high and ((prev_high - prev_close) / prev_close) <= 0.005:
                        percentage_to_close = ((open_price - prev_close) / prev_close) * 100
                        open_near_prev_close_high_within_half_pct_stocks.append({
                            'Symbol': symbol,
                            'Company Name': company_name,
                            'Industry': industry,
                            'Selected Trading Day': selected_date,
                            'Open Price': round(open_price, 2),
                            'Previous Trading Day': prev_trading_date,
                            'Previous Close': round(prev_close, 2),
                            'Current Price': round(current_price, 2),
                            'Percentage to Previous Close': round(percentage_to_close, 2)
                        })
                        with results_container:
                            open_near_prev_close_high_within_half_pct_df = pd.DataFrame(open_near_prev_close_high_within_half_pct_stocks)
                            open_near_prev_close_high_within_half_pct_placeholder.dataframe(open_near_prev_close_high_within_half_pct_df)

                    # Check condition: open near previous close, current price above previous high, prev high-close <= 0.5%, and prev low-high < user-defined threshold
                    if (current_price is not None and 
                        prev_close * 0.998 <= open_price <= prev_close * 1.002 and 
                        current_price > prev_high and 
                        ((prev_high - prev_close) / prev_close) <= 0.005 and 
                        low_high_diff_pct < user_low_high_diff_pct):
                        percentage_to_close = ((open_price - prev_close) / prev_close) * 100
                        open_near_prev_close_high_within_half_pct_low_high_below_1pct_stocks.append({
                            'Symbol': symbol,
                            'Company Name': company_name,
                            'Industry': industry,
                            'Selected Trading Day': selected_date,
                            'Open Price': round(open_price, 2),
                            'Previous Trading Day': prev_trading_date,
                            'Previous Close': round(prev_close, 2),
                            'Current Price': round(current_price, 2),
                            'Percentage to Previous Close': round(percentage_to_close, 2),
                            'Previous Low': round(prev_low, 2),
                            'Previous High': round(prev_high, 2),
                            'Previous Low-High %': round(low_high_diff_pct, 2)
                        })
                        with results_container:
                            open_near_prev_close_high_within_half_pct_low_high_below_1pct_df = pd.DataFrame(open_near_prev_close_high_within_half_pct_low_high_below_1pct_stocks)
                            open_near_prev_close_high_within_half_pct_low_high_below_1pct_placeholder.dataframe(open_near_prev_close_high_within_half_pct_low_high_below_1pct_df)
                    elif low_high_diff_pct >= user_low_high_diff_pct:
                        st.warning(f"{symbol} excluded from low-high table: Previous low-high difference ({low_high_diff_pct:.2f}%) >= {user_low_high_diff_pct}%.")

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
                        with results_container:
                            opened_above_high_near_close_df = pd.DataFrame(opened_above_high_near_close_stocks)
                            opened_above_high_near_close_placeholder.dataframe(opened_above_high_near_close_df)

                    # Check if open price is within 0.5% above previous high
                    if prev_high <= open_price <= prev_high * 1.005:
                        percentage_above = ((open_price - prev_high) / prev_high) * 100
                        near_previous_high_stocks.append({
                            'Symbol': symbol,
                            'Company Name': company_name,
                            'Industry': industry,
                            'Selected Trading Day': selected_date,
                            'Open Price': round(open_price, 2),
                            'Previous Trading Day': prev_trading_date,
                            'Previous High': round(prev_high, 2),
                            'Percentage Above Previous High': round(percentage_above, 2)
                        })
                        with results_container:
                            near_previous_high_df = pd.DataFrame(near_previous_high_stocks)
                            near_previous_high_placeholder.dataframe(near_previous_high_df)

                    # Check if selected day's open is above previous day's high and close
                    if open_price > prev_high and open_price > prev_close:
                        include_stock = True
                        if filter_current_price and (current_price is None or not (current_price > prev_high and current_price > prev_close)):
                            include_stock = False
                        if include_stock:
                            result = {
                                'Symbol': symbol,
                                'Company Name': company_name,
                                'Industry': industry,
                                'Selected Trading Day': selected_date,
                                'Open Price': round(open_price, 2),
                                'Previous Trading Day': prev_trading_date,
                                'Previous High': round(prev_high, 2),
                                'Previous Close': round(prev_close, 2)
                            }
                            results.append(result)
                            with results_container:
                                results_df = pd.DataFrame(results)
                                results_placeholder.dataframe(results_df)

        # Save results to CSV
        if results:
            save_results_to_csv(results, selected_date)

        # Clear progress text after completion
        progress_text.empty()

        # Final message
        with results_container:
            if not results:
                st.info(f"No stocks found where the open price on {selected_date} was above the previous trading day's high and closing prices" + 
                        (f" and current price is above previous high and close" if filter_current_price else "") + ".")
            else:
                st.write(f"Found {len(results)} stocks meeting the condition.")
            if near_previous_high_stocks:
                st.write(f"Found {len(near_previous_high_stocks)} stocks with open price within 0.5% above previous trading day's high.")
            if opened_above_high_near_close_stocks:
                st.write(f"Found {len(opened_above_high_near_close_stocks)} stocks where previous high > close, opened above high, and current price within 0.2% of previous close.")
            if open_near_prev_close_cross_high_stocks:
                st.write(f"Found {len(open_near_prev_close_cross_high_stocks)} stocks where open price is within 0.2% of previous close and current price is above previous high.")
            if open_near_prev_close_high_within_half_pct_stocks:
                st.write(f"Found {len(open_near_prev_close_high_within_half_pct_stocks)} stocks where open price is within 0.2% of previous close, current price is above previous high, and previous high-close difference is not more than 0.5%.")
            if open_near_prev_close_high_within_half_pct_low_high_below_1pct_stocks:
                st.write(f"Found {len(open_near_prev_close_high_within_half_pct_low_high_below_1pct_stocks)} stocks where open price is within 0.2% of previous close, current price is above previous high, previous high-close difference is not more than 0.5%, and previous low-high difference is below {user_low_high_diff_pct}%.")

# Footer
st.write("Data sourced from yfinance. Note: Market data is subject to delays and availability.")