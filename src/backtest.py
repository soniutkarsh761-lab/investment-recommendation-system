import pandas as pd

# Handle relative or absolute imports depending on execution context
try:
    from src.data_loader import load_data
    from src.recommender import recommend
except ModuleNotFoundError:
    from data_loader import load_data
    from recommender import recommend

def run_backtest(prices, as_of_date, holding_days=252):
    """
    Validates the recommendation engine on historical data.
    Slices price history up to as_of_date, generates recommendations,
    and computes the actual forward return for each stock.
    """
    as_of_dt = pd.to_datetime(as_of_date)
    
    # Truncate price history to everything up to as_of_date (prevents lookahead bias)
    truncated_prices = prices.loc[:as_of_dt]
    
    # Run existing recommendation logic on the truncated data
    rec_df = recommend(truncated_prices, profile="moderate")
    
    # Find the row index of the closest trading date to as_of_date in the full dataset
    idx = prices.index.get_indexer([as_of_dt], method="nearest")[0]
    
    # Make sure we have enough forward data to compute return
    if idx + holding_days >= len(prices):
        raise ValueError(
            f"Not enough forward data. Date index {idx} + holding period {holding_days} "
            f"exceeds total dataset length ({len(prices)})."
        )
        
    start_date = prices.index[idx]
    end_date = prices.index[idx + holding_days]
    
    # Get prices at start and end of the holding period
    start_prices = prices.loc[start_date]
    end_prices = prices.loc[end_date]
    
    # Compute forward returns for all assets (including benchmark)
    forward_returns = (end_prices - start_prices) / start_prices
    
    # Assemble the results
    results = []
    for stock in rec_df.index:
        results.append({
            "Stock": stock,
            "Composite Score": rec_df.loc[stock, "Composite Score"],
            "Recommendation": rec_df.loc[stock, "Recommendation"],
            "Actual Forward Return": forward_returns[stock]
        })
        
    backtest_df = pd.DataFrame(results).set_index("Stock")
    return backtest_df

def evaluate(backtest_df, prices, as_of_date, holding_days=252):
    """
    Prints a summary evaluating backtest performance against benchmark returns.
    """
    as_of_dt = pd.to_datetime(as_of_date)
    idx = prices.index.get_indexer([as_of_dt], method="nearest")[0]
    
    # NIFTY 50 benchmark performance
    nifty_start = prices["^NSEI"].iloc[idx]
    nifty_end = prices["^NSEI"].iloc[idx + holding_days]
    nifty_fwd_return = (nifty_end - nifty_start) / nifty_start
    
    # Group results by recommendation class and calculate average forward returns
    summary = backtest_df.groupby("Recommendation")["Actual Forward Return"].mean()
    
    print("\n" + "=" * 60)
    print(f" EVALUATION SUMMARY (as of {as_of_date}, holding: {holding_days} days)")
    print("=" * 60)
    
    # Print average return for each class if any stock fell into that recommendation category
    for rec_type in ["Buy", "Hold", "Sell"]:
        if rec_type in summary.index:
            avg_ret = summary[rec_type] * 100.0
            print(f"Average Return of {rec_type}-rated stocks:  {avg_ret:+.2f}%")
        else:
            print(f"Average Return of {rec_type}-rated stocks:  N/A (No stocks rated {rec_type})")
            
    print(f"NIFTY 50 Benchmark Forward Return:        {nifty_fwd_return * 100:+.2f}%")
    print("=" * 60)

if __name__ == "__main__":
    # Load prices
    print("Loading prices from data/stock_prices.csv...")
    prices = load_data()
    
    if prices is not None:
        # Run backtest as of 2 years ago (2024-07-15) with 1-year holding period (252 trading days)
        as_of = "2024-07-15"
        print(f"Running backtest as of {as_of} with 252-day holding period...")
        backtest_results = run_backtest(prices, as_of, holding_days=252)
        
        # Display full stock recommendation table from backtest
        print("\n=== Backtest Stock Recommendation Table ===")
        # Format returns for display in terminal
        display_df = backtest_results.copy()
        display_df["Actual Forward Return"] = display_df["Actual Forward Return"].map(lambda x: f"{x * 100:+.2f}%")
        print(display_df.to_string())
        
        # Display performance evaluation summary
        evaluate(backtest_results, prices, as_of, holding_days=252)
