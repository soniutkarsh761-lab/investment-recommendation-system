import pandas as pd
try:
    from src.data_loader import load_data
except ModuleNotFoundError:
    from data_loader import load_data

def daily_returns(prices):
    """
    Computes daily percentage returns for each stock.
    """
    # pct_change calculates the percentage change between current and prior elements
    return prices.pct_change()

def annualized_return(prices):
    """
    Computes the Compound Annual Growth Rate (CAGR) for all columns.
    Formula: (End Value / Start Value) ** (365.25 / Calendar Days) - 1
    """
    # Calculate the total duration in calendar days and convert to years
    days = (prices.index[-1] - prices.index[0]).days
    years = days / 365.25
    
    # Calculate CAGR for all columns (returns a Series)
    return (prices.iloc[-1] / prices.iloc[0]) ** (1 / years) - 1

def annualized_volatility(prices):
    """
    Computes the annualized volatility of daily returns.
    Formula: standard deviation of daily returns * sqrt(252)
    """
    returns = daily_returns(prices)
    # Annualized volatility is standard deviation of daily returns scaled by trading days (252)
    return returns.std() * (252 ** 0.5)

def sharpe_ratio(prices, risk_free_rate=0.065):
    """
    Computes the Sharpe Ratio for all columns.
    Formula: (Annualized Return - Risk-Free Rate) / Annualized Volatility
    The default risk_free_rate of 6.5% approximates the Indian 10-year G-Sec yield.
    """
    ann_ret = annualized_return(prices)
    ann_vol = annualized_volatility(prices)
    # Sharpe ratio represents excess return per unit of standard deviation
    return (ann_ret - risk_free_rate) / ann_vol

def max_drawdown(prices):
    """
    Computes the worst peak-to-trough decline (Maximum Drawdown) as a negative percentage.
    Formula: (Price - Running Max Price) / Running Max Price
    """
    # Running peak price for each stock
    running_max = prices.cummax()
    # Drawdown series from the peak
    drawdown = (prices - running_max) / running_max
    # Worst drawdown over the entire period
    return drawdown.min()

def moving_averages(prices):
    """
    Computes the 50-day and 200-day Simple Moving Averages (SMA).
    Returns a tuple of two DataFrames: (sma_50, sma_200)
    """
    sma_50 = prices.rolling(window=50).mean()
    sma_200 = prices.rolling(window=200).mean()
    return sma_50, sma_200

def beta(prices, benchmark_col="^NSEI"):
    """
    Computes the Beta of each stock relative to the benchmark.
    Formula: Covariance(Stock Returns, Benchmark Returns) / Variance(Benchmark Returns)
    """
    returns = daily_returns(prices)
    
    # Variance of the benchmark returns
    benchmark_var = returns[benchmark_col].var()
    
    # Covariance of all stocks with the benchmark
    covariance = returns.cov()[benchmark_col]
    
    # Beta = Covariance / Variance
    return covariance / benchmark_var

def metrics_summary(prices):
    """
    Generates a summary DataFrame of financial metrics for all stocks.
    The benchmark itself is excluded from the final rows.
    Columns: Annualized Return, Volatility, Sharpe Ratio, Max Drawdown, Beta
    """
    # Compute individual metrics
    ann_ret = annualized_return(prices)
    ann_vol = annualized_volatility(prices)
    sharpe = sharpe_ratio(prices, risk_free_rate=0.065)
    max_dd = max_drawdown(prices)
    betas = beta(prices, benchmark_col="^NSEI")
    
    # Combine into a single DataFrame
    summary = pd.DataFrame({
        "Annualized Return": ann_ret,
        "Volatility": ann_vol,
        "Sharpe Ratio": sharpe,
        "Max Drawdown": max_dd,
        "Beta": betas
    })
    
    # Exclude NIFTY 50 index (^NSEI) from the summary table rows
    if "^NSEI" in summary.index:
        summary = summary.drop(index="^NSEI")
        
    # Round all metrics to 4 decimal places
    return summary.round(4)

if __name__ == "__main__":
    # Load the stock price data from data/stock_prices.csv
    print("Loading data from data/stock_prices.csv...")
    prices = load_data()
    
    if prices is not None:
        # Calculate and show the metrics summary table
        summary_table = metrics_summary(prices)
        print("\n=== Stock Metrics Summary Table (Benchmark: ^NSEI) ===")
        print(summary_table.to_string())
