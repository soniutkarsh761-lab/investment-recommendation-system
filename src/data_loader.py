import os
import pandas as pd
import yfinance as yf

# List of 10 NSE stock tickers in Yahoo Finance format plus NIFTY 50 index (^NSEI) as benchmark
TICKERS = [
    "RELIANCE.NS",
    "TCS.NS",
    "HDFCBANK.NS",
    "INFY.NS",
    "ICICIBANK.NS",
    "HINDUNILVR.NS",
    "ITC.NS",
    "BHARTIARTL.NS",
    "M&M.NS",
    "SUNPHARMA.NS",
    "^NSEI"  # NIFTY 50 Index benchmark
]

# Path to save the stock prices CSV file
DATA_FILE = os.path.join("data", "stock_prices.csv")

def fetch_data():
    """
    Downloads 5 years of daily historical data for the tickers, cleans it,
    prints a summary, and saves it to a CSV file.
    """
    print("Starting data download for tickers...")
    
    # Download 5 years of daily history for all tickers
    # yfinance returns a DataFrame with multi-index columns (Price, Ticker)
    data = yf.download(TICKERS, period="5y", interval="1d")
    
    # Keep only 'Close' prices
    # This selects the 'Close' column group, giving a DataFrame with dates as index and tickers as columns
    close_prices = data['Close']
    
    # Ensure close_prices is a DataFrame (in case only 1 ticker was downloaded)
    if isinstance(close_prices, pd.Series):
        close_prices = close_prices.to_frame()
        
    # Cleaning the data:
    # 1. Forward-fill missing values (propagates last valid observation forward to next valid)
    cleaned_data = close_prices.ffill()
    
    # 2. Drop rows where all stock prices are missing
    cleaned_data = cleaned_data.dropna(how='all')
    
    # Ensure the target directory exists
    os.makedirs("data", exist_ok=True)
    
    # Save the cleaned data to CSV
    cleaned_data.to_csv(DATA_FILE)
    print(f"Cleaned data saved to {DATA_FILE}")
    
    # Print summary: date range and row count per stock
    print("\n--- Data Summary ---")
    start_date = cleaned_data.index.min().strftime('%Y-%m-%d')
    end_date = cleaned_data.index.max().strftime('%Y-%m-%d')
    print(f"Date Range: {start_date} to {end_date}")
    print("Row count per stock (non-missing values):")
    
    # Print non-missing count for each stock in our ticker list
    for ticker in TICKERS:
        if ticker in cleaned_data.columns:
            count = cleaned_data[ticker].notna().sum()
            print(f"  {ticker}: {count} rows")
        else:
            print(f"  {ticker}: Ticker not found in downloaded columns")
            
    return cleaned_data

def load_data():
    """
    Reads the saved stock prices CSV file back into a pandas DataFrame.
    """
    if not os.path.exists(DATA_FILE):
        print(f"Error: {DATA_FILE} not found. Please run fetch_data() first.")
        return None
    
    # Read the CSV file, parsing the first column (Date) as the index and convert it to datetime
    df = pd.read_csv(DATA_FILE, index_col=0, parse_dates=True)
    return df

if __name__ == "__main__":
    fetch_data()
