import pandas as pd

# Handle relative or absolute imports depending on execution context
try:
    from src.data_loader import load_data
    from src.analysis import moving_averages, metrics_summary
except ModuleNotFoundError:
    from data_loader import load_data
    from analysis import moving_averages, metrics_summary

def score_stock(prices, metrics_df, ticker):
    """
    Computes a composite score (0-100) for a given stock ticker based on four components:
    1. Trend (30 pts)
    2. Risk-adjusted return (30 pts)
    3. Momentum (20 pts)
    4. Stability (20 pts)
    """
    # --- 1. Trend (30 pts) ---
    latest_price = prices[ticker].iloc[-1]
    sma_50_df, sma_200_df = moving_averages(prices)
    latest_sma_50 = sma_50_df[ticker].iloc[-1]
    latest_sma_200 = sma_200_df[ticker].iloc[-1]
    
    trend_pts = 0.0
    if latest_price > latest_sma_200:
        trend_pts += 15.0
    if latest_sma_50 > latest_sma_200:
        trend_pts += 15.0
        
    # --- 2. Risk-adjusted Return (30 pts) ---
    sharpe = metrics_df.loc[ticker, "Sharpe Ratio"]
    if sharpe <= 0:
        sharpe_pts = 0.0
    elif sharpe >= 1.5:
        sharpe_pts = 30.0
    else:
        # Scale Sharpe ratio linearly from 0 to 1.5
        sharpe_pts = (sharpe / 1.5) * 30.0
        
    # --- 3. Momentum (20 pts) ---
    # Find price 6 months ago (approx. 180 calendar days)
    target_date = prices.index[-1] - pd.DateOffset(months=6)
    # Find the nearest available trading index for the target date
    idx_6m = prices.index.get_indexer([target_date], method='nearest')[0]
    price_6m = prices[ticker].iloc[idx_6m]
    
    # Calculate 6-month return
    momentum_return = (latest_price - price_6m) / price_6m
    if momentum_return <= -0.10:
        momentum_pts = 0.0
    elif momentum_return >= 0.20:
        momentum_pts = 20.0
    else:
        # Scale linearly between -10% (0 pts) and +20% (20 pts)
        momentum_pts = (momentum_return - (-0.10)) / (0.20 - (-0.10)) * 20.0
        
    # --- 4. Stability (20 pts) ---
    vol = metrics_df.loc[ticker, "Volatility"]
    if vol <= 0.18:
        stability_pts = 20.0
    elif vol >= 0.35:
        stability_pts = 0.0
    else:
        # Scale linearly between 18% (20 pts) and 35% (0 pts)
        stability_pts = (0.35 - vol) / (0.35 - 0.18) * 20.0
        
    raw_score = trend_pts + sharpe_pts + momentum_pts + stability_pts
    return raw_score, trend_pts, sharpe, momentum_return, vol

def adjust_for_risk_profile(score, volatility, beta, profile, short_term_trend_positive=False):
    """
    Adjusts the stock's composite score based on the risk profile.
    Profiles:
    - conservative: penalizes high volatility (>25%) or high beta (>1.1) up to -15 pts
    - moderate: no adjustment
    - aggressive: rewards high beta (>1.1) up to +10 pts when trend is positive
    """
    adjusted_score = score
    
    if profile == "conservative":
        # Deduct up to 7.5 pts for vol > 25% (max deduction at vol >= 35%)
        vol_deduct = 0.0
        if volatility > 0.25:
            vol_deduct = min(7.5, (volatility - 0.25) / (0.35 - 0.25) * 7.5)
            
        # Deduct up to 7.5 pts for beta > 1.1 (max deduction at beta >= 1.6)
        beta_deduct = 0.0
        if beta > 1.1:
            beta_deduct = min(7.5, (beta - 1.1) / (1.6 - 1.1) * 7.5)
            
        adjusted_score = score - (vol_deduct + beta_deduct)
        adjusted_score = max(0.0, adjusted_score)
        
    elif profile == "aggressive":
        # Boost up to 10 pts for beta > 1.1 (max boost at beta >= 1.6) when short-term trend is positive.
        # Rationale: aggressive investors act on shorter-horizon trend signals (50-day MA).
        if short_term_trend_positive and beta > 1.1:
            beta_boost = min(10.0, (beta - 1.1) / (1.6 - 1.1) * 10.0)
            adjusted_score = score + beta_boost
            adjusted_score = min(100.0, adjusted_score)
            
    return adjusted_score

def recommend(prices, profile="moderate"):
    """
    Computes stock recommendation results including composite score, Buy/Hold/Sell decision,
    and a plain-English reasoning string.
    """
    # Get standard metrics summary for all stocks
    metrics_df = metrics_summary(prices)
    
    # Calculate simple moving averages once to check trends
    sma_50_df, sma_200_df = moving_averages(prices)
    
    results = []
    
    # Iterate through all stocks in the metrics DataFrame (excludes benchmark ^NSEI)
    for ticker in metrics_df.index:
        latest_price = prices[ticker].iloc[-1]
        latest_sma_50 = sma_50_df[ticker].iloc[-1]
        # Short-term uptrend condition (latest price > 50-day MA) for aggressive boost
        short_term_trend_positive = latest_price > latest_sma_50
        
        # Calculate raw scores and intermediate components
        raw_score, trend_pts, sharpe, momentum, vol = score_stock(prices, metrics_df, ticker)
        
        # Adjust for risk profile
        beta_val = metrics_df.loc[ticker, "Beta"]
        adj_score = adjust_for_risk_profile(raw_score, vol, beta_val, profile, short_term_trend_positive)
        adj_score_rounded = int(round(adj_score))
        
        # Determine recommendation label
        if adj_score_rounded >= 65:
            rec_label = "Buy"
        elif adj_score_rounded >= 40:
            rec_label = "Hold"
        else:
            rec_label = "Sell"
            
        # Build plain-English reasoning
        drivers = []
        if trend_pts == 30:
            drivers.append("strong uptrend (price and 50-day MA above 200-day MA)")
        elif trend_pts == 15:
            drivers.append("moderate uptrend (price above 200-day MA)")
        else:
            drivers.append("downtrend (price below 200-day MA)")
            
        if sharpe >= 1.0:
            drivers.append(f"strong risk-adjusted returns (Sharpe {sharpe:.2f})")
        elif sharpe >= 0.5:
            drivers.append(f"moderate risk-adjusted returns (Sharpe {sharpe:.2f})")
        elif sharpe > 0:
            drivers.append(f"weak risk-adjusted returns (Sharpe {sharpe:.2f})")
        else:
            drivers.append(f"negative risk-adjusted returns (Sharpe {sharpe:.2f})")
            
        max_dd = metrics_df.loc[ticker, "Max Drawdown"]
        if max_dd >= -0.20:
            drivers.append(f"low drawdown history ({max_dd * 100:.1f}%)")
        elif max_dd <= -0.35:
            drivers.append(f"high drawdown history ({max_dd * 100:.1f}%)")
            
        if momentum >= 0.15:
            drivers.append(f"strong 6-month momentum (+{momentum * 100:.1f}%)")
        elif momentum <= -0.05:
            drivers.append(f"weak 6-month momentum ({momentum * 100:.1f}%)")
            
        # Mention risk-profile adjustments if any
        if profile == "conservative" and (vol > 0.25 or beta_val > 1.1):
            drivers.append("score penalized for high volatility/beta")
        elif profile == "aggressive" and beta_val > 1.1 and short_term_trend_positive:
            drivers.append("score boosted for high-beta stock in short-term uptrend (aggressive profile)")
            
        reasoning = f"{rec_label}: " + ", ".join(drivers) + "."
        
        results.append({
            "Stock": ticker,
            "Composite Score": adj_score_rounded,
            "Recommendation": rec_label,
            "Reasoning": reasoning
        })
        
    return pd.DataFrame(results).set_index("Stock")

if __name__ == "__main__":
    # Load prices
    print("Loading prices from data/stock_prices.csv...")
    prices = load_data()
    
    if prices is not None:
        profiles = ["conservative", "moderate", "aggressive"]
        
        for profile in profiles:
            print(f"\n=============================================================")
            print(f" RECOMMENDATION TABLE: {profile.upper()} PROFILE")
            print(f"=============================================================")
            rec_df = recommend(prices, profile=profile)
            print(rec_df.to_string())
