import streamlit as st
import pandas as pd

# Import data loading, analysis, recommendation, and report modules
from src.data_loader import load_data
from src.analysis import moving_averages, metrics_summary
from src.recommender import recommend
from src.report import build_report
from src.backtest import run_backtest
from src.insights import growth_simulation, correlation_matrix
from src.charts import (
    nifty_chart, stock_chart, normalized_chart, sharpe_chart,
    growth_chart, correlation_heatmap
)

# Page Setup
st.set_page_config(page_title="Investment Recommendation System", layout="wide")
st.title("Investment Recommendation System")
st.caption("Built by Utkarsh Soni · BBA (International Business), MIT-WPU")

# Cache data loading using Streamlit's cache decorator
@st.cache_data
def get_cached_data():
    return load_data()

# Cache backtest execution to prevent lookahead validation lag
@st.cache_data
def get_backtest_results(_prices):
    return run_backtest(_prices, "2024-07-15", holding_days=252)

# Load historical stock prices
prices = get_cached_data()

if prices is None:
    st.error("Could not load price data. Make sure data/stock_prices.csv exists.")
    st.stop()

# Get available stock tickers (excludes the NIFTY 50 index benchmark)
all_stocks = sorted([col for col in prices.columns if col != "^NSEI"])

# Sidebar selections
st.sidebar.header("User Settings")
profile = st.sidebar.radio("Select Risk Profile", ["Conservative", "Moderate", "Aggressive"], index=1)
selected_stocks = st.sidebar.multiselect("Select Tracked Stocks", all_stocks, default=all_stocks)

if not selected_stocks:
    st.warning("Please select at least one stock to display dashboard metrics.")
    st.stop()

# Pre-calculate common metrics and SMAs
sma_50_df, sma_200_df = moving_averages(prices)
metrics_df = metrics_summary(prices)
latest_nifty = prices["^NSEI"].iloc[-1]
nifty_sma_200 = sma_200_df["^NSEI"].iloc[-1]

# -------------------------------------------------------------
# Section 1 — Market Overview
# -------------------------------------------------------------
st.header("Section 1 — Market Overview")

# Market Regime Banner
if latest_nifty < nifty_sma_200:
    st.warning("⚠️ **Market Regime: Correction** — NIFTY below its 200-day average. Recommendations reflect defensive conditions.")
else:
    st.success("📈 **Market Regime: Uptrend** — NIFTY 50 is trending positive.")

# Calculate 6-month NIFTY return
target_date_6m = prices.index[-1] - pd.DateOffset(months=6)
idx_6m = prices.index.get_indexer([target_date_6m], method="nearest")[0]
nifty_price_6m = prices["^NSEI"].iloc[idx_6m]
nifty_return_6m = (latest_nifty - nifty_price_6m) / nifty_price_6m

# Calculate % of tracked stocks above 200-day MA
above_200 = sum(1 for stock in all_stocks if prices[stock].iloc[-1] > sma_200_df[stock].iloc[-1])
pct_above_200 = (above_200 / len(all_stocks)) * 100

# Metric Cards
col1, col2, col3 = st.columns(3)
col1.metric("NIFTY 50 Latest Close", f"{latest_nifty:,.2f}")
col2.metric("NIFTY 50 6-Month Return", f"{nifty_return_6m * 100:+.2f}%")
col3.metric("Tracked Stocks > 200-day MA", f"{pct_above_200:.1f}%")

# NIFTY 50 Chart
st.plotly_chart(nifty_chart(prices, sma_50_df["^NSEI"], sma_200_df["^NSEI"]), use_container_width=True)

# -------------------------------------------------------------
# Section 2 — Recommendations
# -------------------------------------------------------------
st.header("Section 2 — Stock Recommendations")
st.subheader(f"Risk Profile: {profile}")

rec_df = recommend(prices, profile=profile.lower())
rec_df_filtered = rec_df.loc[selected_stocks]

def color_recommendation(val):
    if val == "Buy":
        return "background-color: #d4edda; color: #155724; font-weight: bold;"
    elif val == "Hold":
        return "background-color: #fff3cd; color: #856404; font-weight: bold;"
    elif val == "Sell":
        return "background-color: #f8d7da; color: #721c24; font-weight: bold;"
    return ""

if hasattr(rec_df_filtered.style, "map"):
    styled_df = rec_df_filtered.style.map(color_recommendation, subset=["Recommendation"])
else:
    styled_df = rec_df_filtered.style.applymap(color_recommendation, subset=["Recommendation"])

st.dataframe(styled_df, use_container_width=True)

# -------------------------------------------------------------
# Section 3 — Stock Deep-Dive
# -------------------------------------------------------------
st.header("Section 3 — Stock Deep-Dive")
deep_dive_stock = st.selectbox("Select Stock for Deep-Dive Analysis", selected_stocks)

if deep_dive_stock:
    stock_metrics = metrics_df.loc[deep_dive_stock]
    m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)
    m_col1.metric("Annualized CAGR", f"{stock_metrics['Annualized Return'] * 100:.2f}%")
    m_col2.metric("Annualized Volatility", f"{stock_metrics['Volatility'] * 100:.2f}%")
    m_col3.metric("Sharpe Ratio", f"{stock_metrics['Sharpe Ratio']:.2f}")
    m_col4.metric("Max Drawdown", f"{stock_metrics['Max Drawdown'] * 100:.2f}%")
    m_col5.metric("CAPM Beta", f"{stock_metrics['Beta']:.2f}")
    
    st.plotly_chart(stock_chart(prices, deep_dive_stock, sma_50_df[deep_dive_stock], sma_200_df[deep_dive_stock]), use_container_width=True)

# -------------------------------------------------------------
# Section 4 — Comparison
# -------------------------------------------------------------
st.header("Section 4 — Comparative Analysis")
comp_col1, comp_col2 = st.columns(2)

with comp_col1:
    normalized_prices = prices[selected_stocks].div(prices[selected_stocks].iloc[0]) * 100.0
    st.plotly_chart(normalized_chart(normalized_prices, selected_stocks), use_container_width=True)

with comp_col2:
    st.plotly_chart(sharpe_chart(metrics_df, selected_stocks), use_container_width=True)

# -------------------------------------------------------------
# Section 5 — Wealth Growth Simulator
# -------------------------------------------------------------
st.header("Section 5 — Wealth Growth Simulator")
sim_col1, sim_col2 = st.columns([1, 3])

with sim_col1:
    sim_stock = st.selectbox("Select Stock to Simulate", selected_stocks)
    initial_amount = st.number_input("Initial Investment Amount (₹)", min_value=1000.0, value=100000.0, step=5000.0)

if sim_stock:
    sim_series, final_val, multiple = growth_simulation(prices, sim_stock, initial_amount)
    with sim_col1:
        st.metric("Final Portfolio Value", f"₹{final_val:,.2f}")
        st.metric("Growth Multiple", multiple)
    with sim_col2:
        st.plotly_chart(growth_chart(sim_series, sim_stock), use_container_width=True)

# -------------------------------------------------------------
# Section 6 — Diversification Insights
# -------------------------------------------------------------
st.header("Section 6 — Diversification Insights")
corr_matrix_df, most_corr, least_corr = correlation_matrix(prices)
corr_matrix_subset = corr_matrix_df.loc[selected_stocks, selected_stocks]

st.plotly_chart(correlation_heatmap(corr_matrix_subset), use_container_width=True)
st.caption(
    f"**Correlations**: Most correlated pair is **{most_corr[0]}** and **{most_corr[1]}** ({most_corr[2]:.2f}). "
    f"Least correlated pair is **{least_corr[0]}** and **{least_corr[1]}** ({least_corr[2]:.2f})."
)

# -------------------------------------------------------------
# Section 7 — Validation (Backtest)
# -------------------------------------------------------------
st.header("Section 7 — Validation (Backtest)")
backtest_df = get_backtest_results(prices)
backtest_filtered = backtest_df.loc[selected_stocks]

# Group performance metrics
avg_buy_return = backtest_filtered[backtest_filtered["Recommendation"] == "Buy"]["Actual Forward Return"].mean()
avg_hold_return = backtest_filtered[backtest_filtered["Recommendation"] == "Hold"]["Actual Forward Return"].mean()

idx_bt = prices.index.get_indexer([pd.to_datetime("2024-07-15")], method="nearest")[0]
nifty_bt_start = prices["^NSEI"].iloc[idx_bt]
nifty_bt_end = prices["^NSEI"].iloc[idx_bt + 252]
nifty_bt_return = (nifty_bt_end - nifty_bt_start) / nifty_bt_start

b_col1, b_col2, b_col3 = st.columns(3)
b_col1.metric("Average Buy Return (Backtest)", f"{avg_buy_return * 100:+.2f}%" if not pd.isna(avg_buy_return) else "N/A")
b_col2.metric("Average Hold Return (Backtest)", f"{avg_hold_return * 100:+.2f}%" if not pd.isna(avg_hold_return) else "N/A")
b_col3.metric("NIFTY 50 Benchmark Return", f"{nifty_bt_return * 100:+.2f}%")

# Display backtest table
display_bt_df = backtest_filtered.copy()
display_bt_df["Actual Forward Return"] = display_bt_df["Actual Forward Return"].map(lambda x: f"{x * 100:.2f}%")

if hasattr(display_bt_df.style, "map"):
    styled_bt_df = display_bt_df.style.map(color_recommendation, subset=["Recommendation"])
else:
    styled_bt_df = display_bt_df.style.applymap(color_recommendation, subset=["Recommendation"])

st.dataframe(styled_bt_df, use_container_width=True)
st.caption("*Backtest is indicative — 10 stocks, single as-of date (2024-07-15).*")

# -------------------------------------------------------------
# Section 8 — Download Report
# -------------------------------------------------------------
st.header("Section 8 — Download Report")
report_profile = profile.lower()
report_text = build_report(prices, report_profile)
today_str = pd.Timestamp.now().strftime("%Y-%m-%d")

st.download_button(
    label="Download Plain-Text Report",
    data=report_text,
    file_name=f"investment_report_{report_profile}_{today_str}.txt",
    mime="text/plain"
)

# Footer disclaimer
st.markdown("---")
st.caption("Educational project. Not investment advice.")
