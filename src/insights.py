import pandas as pd

def growth_simulation(prices, ticker, amount=100000.0):
    """
    Simulates the value over time of an initial investment in a given ticker
    at the start of the price data period.
    Returns:
      - simulation_series: Series of investment value over time
      - final_value: Final value of the investment
      - multiple: Growth multiple string (e.g. "2.40x")
    """
    stock_prices = prices[ticker]
    initial_price = stock_prices.dropna().iloc[0]
    
    # Calculate value over time
    simulation_series = amount * (stock_prices / initial_price)
    final_value = simulation_series.iloc[-1]
    multiple_val = final_value / amount
    multiple = f"{multiple_val:.2f}x"
    
    return simulation_series, final_value, multiple

def correlation_matrix(prices):
    """
    Computes the Pearson correlation matrix of daily returns for the stocks
    (excluding the NIFTY 50 index benchmark).
    Returns:
      - corr_matrix: Correlation DataFrame
      - most_corr: Tuple of (Stock 1, Stock 2, Correlation Value)
      - least_corr: Tuple of (Stock 1, Stock 2, Correlation Value)
    """
    # Exclude benchmark ^NSEI
    stocks = sorted([col for col in prices.columns if col != "^NSEI"])
    
    # Calculate daily percentage returns
    daily_returns = prices[stocks].pct_change()
    
    # Calculate correlation matrix
    corr_matrix = daily_returns.corr()
    
    # Find the most and least correlated unique pairs
    pairs = []
    for i in range(len(stocks)):
        for j in range(i + 1, len(stocks)):
            s1 = stocks[i]
            s2 = stocks[j]
            corr_val = corr_matrix.loc[s1, s2]
            # Skip NaN values if any ticker has incomplete data
            if not pd.isna(corr_val):
                pairs.append((corr_val, s1, s2))
                
    if not pairs:
        return corr_matrix, ("N/A", "N/A", 0.0), ("N/A", "N/A", 0.0)
        
    # Sort pairs by correlation value
    pairs.sort()
    
    least_corr = (pairs[0][1], pairs[0][2], pairs[0][0])
    most_corr = (pairs[-1][1], pairs[-1][2], pairs[-1][0])
    
    return corr_matrix, most_corr, least_corr
