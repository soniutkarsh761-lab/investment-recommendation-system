import datetime
import os
import pandas as pd

# Handle relative or absolute imports depending on execution context
try:
    from src.data_loader import load_data
    from src.analysis import moving_averages, metrics_summary
    from src.recommender import recommend
except ModuleNotFoundError:
    from data_loader import load_data
    from analysis import moving_averages, metrics_summary
    from recommender import recommend

def build_report(prices, profile):
    """
    Assembles a plain-text stock recommendation and investment summary report as a string.
    """
    # Pre-calculate simple moving averages once to check trends
    sma_50_df, sma_200_df = moving_averages(prices)
    metrics_df = metrics_summary(prices)
    rec_df = recommend(prices, profile=profile)
    
    latest_nifty = prices["^NSEI"].iloc[-1]
    nifty_sma_200 = sma_200_df["^NSEI"].iloc[-1]
    
    # Determine Market Regime
    if latest_nifty >= nifty_sma_200:
        regime = "Uptrend"
        regime_desc = "NIFTY above its 200-day average."
    else:
        regime = "Correction"
        regime_desc = "NIFTY below its 200-day average. Recommendations reflect defensive conditions."
        
    # Calculate 6-Month NIFTY Return
    target_date = prices.index[-1] - pd.DateOffset(months=6)
    idx_6m = prices.index.get_indexer([target_date], method="nearest")[0]
    nifty_6m_price = prices["^NSEI"].iloc[idx_6m]
    nifty_6m_return = (latest_nifty - nifty_6m_price) / nifty_6m_price
    
    # Calculate % of tracked stocks above their 200-day MA
    all_stocks = sorted([col for col in prices.columns if col != "^NSEI"])
    above_200_count = sum(1 for stock in all_stocks if prices[stock].iloc[-1] > sma_200_df[stock].iloc[-1])
    pct_above_200 = (above_200_count / len(all_stocks)) * 100
    
    # Assemble the text report
    lines = []
    lines.append("=" * 80)
    lines.append("                         INVESTMENT RECOMMENDATION REPORT")
    lines.append("=" * 80)
    lines.append(f"Generated Date:  {datetime.date.today().strftime('%Y-%m-%d')}")
    lines.append(f"Risk Profile:    {profile.upper()}")
    lines.append("Prepared by:     Utkarsh Soni · BBA (International Business), MIT-WPU")
    lines.append("-" * 80)
    
    lines.append("\nMARKET OVERVIEW")
    lines.append("-" * 15)
    lines.append(f"NIFTY 50 Latest Close:    {latest_nifty:,.2f}")
    lines.append(f"NIFTY 50 6-Month Return:  {nifty_6m_return * 100:+.2f}%")
    lines.append(f"Market Regime:            {regime} — {regime_desc}")
    lines.append(f"Tracked Stocks > 200 MA:  {pct_above_200:.2f}%")
    lines.append("-" * 80)
    
    lines.append("\nPER-STOCK PERFORMANCE & RECOMMENDATIONS")
    lines.append("-" * 40)
    
    for ticker in all_stocks:
        m = metrics_df.loc[ticker]
        r = rec_df.loc[ticker]
        
        lines.append(f"Ticker: {ticker}")
        lines.append(f"  Annualized CAGR:     {m['Annualized Return'] * 100:.2f}%")
        lines.append(f"  Volatility:          {m['Volatility'] * 100:.2f}%")
        lines.append(f"  Sharpe Ratio:        {m['Sharpe Ratio']:.2f}")
        lines.append(f"  Max Drawdown:        {m['Max Drawdown'] * 100:.2f}%")
        lines.append(f"  Beta:                {m['Beta']:.2f}")
        lines.append(f"  Composite Score:     {r['Composite Score']}")
        lines.append(f"  Recommendation:      {r['Recommendation']}")
        lines.append(f"  Reasoning:           {r['Reasoning']}")
        lines.append("-" * 60)
        
    # Summary of Recommendations Counts
    buy_count = sum(1 for stock in all_stocks if rec_df.loc[stock, "Recommendation"] == "Buy")
    hold_count = sum(1 for stock in all_stocks if rec_df.loc[stock, "Recommendation"] == "Hold")
    sell_count = sum(1 for stock in all_stocks if rec_df.loc[stock, "Recommendation"] == "Sell")
    
    lines.append("\nRECOMMENDATION SUMMARY")
    lines.append("-" * 22)
    lines.append(f"  BUY:  {buy_count}")
    lines.append(f"  HOLD: {hold_count}")
    lines.append(f"  SELL: {sell_count}")
    lines.append("-" * 80)
    
    lines.append("\nDisclaimer: Educational project. Not investment advice.")
    lines.append("=" * 80)
    
    return "\n".join(lines)

if __name__ == "__main__":
    print("Loading historical price data...")
    prices_data = load_data()
    if prices_data is not None:
        print("Generating moderate-profile investment report...")
        report_txt = build_report(prices_data, "moderate")
        
        # Save to data/sample_report.txt
        os.makedirs("data", exist_ok=True)
        report_path = os.path.join("data", "sample_report.txt")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_txt)
            
        print(f"Success: Report generated and saved to {report_path}")
