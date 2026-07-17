import pandas as pd

# Handle relative or absolute imports depending on execution context
try:
    from src.recommender import recommend
except ModuleNotFoundError:
    from recommender import recommend

def allocate(prices, profile, amount=100000.0):
    """
    Computes score-based asset allocations.
    Filters to Buy/Hold stocks, weights them proportionally to their composite scores,
    caps single stocks at 25%, and ensures risk-profile minimum cash floors.
    """
    # 1. Get recommendations for the selected profile
    rec_df = recommend(prices, profile=profile.lower())
    
    # 2. Eligible stocks: rated Buy or Hold only
    eligible_df = rec_df[rec_df["Recommendation"].isin(["Buy", "Hold"])]
    
    # 3. Risk-profile cash floors
    cash_floors = {
        "conservative": 0.30,
        "moderate": 0.15,
        "aggressive": 0.05
    }
    floor = cash_floors.get(profile.lower(), 0.15)
    equity_space = 1.0 - floor
    
    weights = {}
    
    if not eligible_df.empty:
        total_score = eligible_df["Composite Score"].sum()
        
        if total_score > 0:
            # Initial proportional weights
            weights = {s: (eligible_df.loc[s, "Composite Score"] / total_score) * equity_space for s in eligible_df.index}
            
            # Capping algorithm: Cap individual stock weight at 25% and redistribute remaining
            capped = set()
            while True:
                exceeding = [s for s, w in weights.items() if w > 0.25 and s not in capped]
                if not exceeding:
                    break
                
                for s in exceeding:
                    capped.add(s)
                    weights[s] = 0.25
                    
                remaining_equity = equity_space - 0.25 * len(capped)
                if remaining_equity <= 0:
                    for s in weights:
                        if s not in capped:
                            weights[s] = 0.0
                    break
                    
                uncapped_stocks = [s for s in weights if s not in capped]
                if not uncapped_stocks:
                    break
                    
                uncapped_score_sum = sum(eligible_df.loc[s, "Composite Score"] for s in uncapped_stocks)
                if uncapped_score_sum > 0:
                    for s in uncapped_stocks:
                        weights[s] = remaining_equity * (eligible_df.loc[s, "Composite Score"] / uncapped_score_sum)
                else:
                    for s in uncapped_stocks:
                        weights[s] = remaining_equity / len(uncapped_stocks)

    # 4. Assemble the results
    sorted_stocks = sorted(weights.items(), key=lambda x: x[1], reverse=True)
    rows = []
    
    for stock, w in sorted_stocks:
        if w > 0:
            rows.append({
                "Asset": stock,
                "Weight %": w * 100.0,
                "Rupee Amount": w * amount,
                "Recommendation": eligible_df.loc[stock, "Recommendation"],
                "Composite Score": int(eligible_df.loc[stock, "Composite Score"])
            })
            
    cash_weight = 1.0 - sum(weights.values())
    rows.append({
        "Asset": "Cash / Liquid Funds",
        "Weight %": cash_weight * 100.0,
        "Rupee Amount": cash_weight * amount,
        "Recommendation": "-",
        "Composite Score": "-"
    })
    
    alloc_df = pd.DataFrame(rows).set_index("Asset")
    return alloc_df

def allocation_summary(alloc_df, profile, prices=None):
    """
    Compiles a clean, plain-English summary line from the allocation.
    """
    cash_row = alloc_df.loc["Cash / Liquid Funds"]
    cash_pct = cash_row["Weight %"]
    equity_pct = 100.0 - cash_pct
    
    equity_df = alloc_df.drop("Cash / Liquid Funds")
    num_names = len(equity_df[equity_df["Weight %"] > 0])
    
    # Check market regime if prices is available
    if prices is not None:
        sma_200 = prices["^NSEI"].rolling(200).mean().iloc[-1]
        latest_nifty = prices["^NSEI"].iloc[-1]
        regime = "correction" if latest_nifty < sma_200 else "uptrend"
    else:
        regime = "correction"  # default standard
        
    label = profile.capitalize()
    
    return f"{label} allocation: {cash_pct:.0f}% cash, {equity_pct:.0f}% equities across {num_names} names — reflecting {regime} conditions."

if __name__ == "__main__":
    from data_loader import load_data
    print("Loading historical price data...")
    prices = load_data()
    
    if prices is not None:
        amt = 100000.0
        for p in ["Conservative", "Moderate", "Aggressive"]:
            print(f"\n==================================================")
            print(f" ALLOCATION FOR PROFILE: {p.upper()} (Rs. 1,00,000)")
            print(f"==================================================")
            df = allocate(prices, p, amt)
            print(df.to_string())
            print(f"\nSummary: {allocation_summary(df, p, prices)}")
