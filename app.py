import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# Import data loading, analysis, and recommendation modules
from src.data_loader import load_data
from src.analysis import moving_averages, metrics_summary
from src.recommender import recommend
from src.report import build_report

# Page Setup
st.set_page_config(page_title="Investment Recommendation System", layout="wide")
st.title("Investment Recommendation System")
st.caption("Built by Utkarsh Soni · BBA (International Business), MIT-WPU")

# Cache data loading using Streamlit's cache decorator
@st.cache_data
def get_cached_data():
    return load_data()

# Load historical stock prices
prices = get_cached_data()

if prices is None:
    st.error("Could not load price data. Make sure data/stock_prices.csv exists.")
    st.stop()

# Get available stock tickers (excludes the NIFTY 50 index benchmark)
all_stocks = sorted([col for col in prices.columns if col != "^NSEI"])

# Sidebar selections
st.sidebar.header("User Settings")
profile = st.sidebar.radio(
    "Select Risk Profile",
    ["Conservative", "Moderate", "Aggressive"],
    index=1
)
selected_stocks = st.sidebar.multiselect(
    "Select Tracked Stocks",
    all_stocks,
    default=all_stocks
)

if not selected_stocks:
    st.warning("Please select at least one stock to display dashboard metrics.")
    st.stop()

# Pre-calculate common metrics and SMAs to avoid repeating inside display logic
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

# NIFTY 50 chart with overlaid MAs
fig_nifty = go.Figure()
fig_nifty.add_trace(go.Scatter(x=prices.index, y=prices["^NSEI"], name="NIFTY 50 Close", line=dict(color="#1f77b4", width=2)))
fig_nifty.add_trace(go.Scatter(x=prices.index, y=sma_50_df["^NSEI"], name="50-day MA", line=dict(color="#ff7f0e", width=1.5, dash="dash")))
fig_nifty.add_trace(go.Scatter(x=prices.index, y=sma_200_df["^NSEI"], name="200-day MA", line=dict(color="#d62728", width=1.5, dash="dot")))
fig_nifty.update_layout(
    title="NIFTY 50 Index (Benchmark) with Moving Averages",
    xaxis_title="Date",
    yaxis_title="Index Level",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    margin=dict(l=20, r=20, t=50, b=20),
    height=400
)
st.plotly_chart(fig_nifty, use_container_width=True)

# -------------------------------------------------------------
# Section 2 — Recommendations
# -------------------------------------------------------------
st.header("Section 2 — Stock Recommendations")
st.subheader(f"Risk Profile: {profile}")

# Generate recommendation table
rec_df = recommend(prices, profile=profile.lower())
rec_df_filtered = rec_df.loc[selected_stocks]

# Formatting function for recommendation cells
def color_recommendation(val):
    if val == "Buy":
        return "background-color: #d4edda; color: #155724; font-weight: bold;"
    elif val == "Hold":
        return "background-color: #fff3cd; color: #856404; font-weight: bold;"
    elif val == "Sell":
        return "background-color: #f8d7da; color: #721c24; font-weight: bold;"
    return ""

# Display styled DataFrame
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
    # Selected stock metric cards
    stock_metrics = metrics_df.loc[deep_dive_stock]
    m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)
    m_col1.metric("Annualized CAGR", f"{stock_metrics['Annualized Return'] * 100:.2f}%")
    m_col2.metric("Annualized Volatility", f"{stock_metrics['Volatility'] * 100:.2f}%")
    m_col3.metric("Sharpe Ratio", f"{stock_metrics['Sharpe Ratio']:.2f}")
    m_col4.metric("Max Drawdown", f"{stock_metrics['Max Drawdown'] * 100:.2f}%")
    m_col5.metric("CAPM Beta", f"{stock_metrics['Beta']:.2f}")
    
    # Selected stock price history chart
    fig_stock = go.Figure()
    fig_stock.add_trace(go.Scatter(x=prices.index, y=prices[deep_dive_stock], name="Close Price", line=dict(color="#2ca02c", width=2)))
    fig_stock.add_trace(go.Scatter(x=prices.index, y=sma_50_df[deep_dive_stock], name="50-day MA", line=dict(color="#ff7f0e", width=1.5, dash="dash")))
    fig_stock.add_trace(go.Scatter(x=prices.index, y=sma_200_df[deep_dive_stock], name="200-day MA", line=dict(color="#d62728", width=1.5, dash="dot")))
    fig_stock.update_layout(
        title=f"{deep_dive_stock} Historical Price & Moving Averages",
        xaxis_title="Date",
        yaxis_title="Price (INR)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=50, b=20),
        height=400
    )
    st.plotly_chart(fig_stock, use_container_width=True)

# -------------------------------------------------------------
# Section 4 — Comparison
# -------------------------------------------------------------
st.header("Section 4 — Comparative Analysis")
comp_col1, comp_col2 = st.columns(2)

with comp_col1:
    # Normalized performance chart (Rebased to 100)
    normalized_prices = prices[selected_stocks].div(prices[selected_stocks].iloc[0]) * 100.0
    fig_norm = px.line(
        normalized_prices,
        x=normalized_prices.index,
        y=selected_stocks,
        title="Normalized Stock Performance (Rebased to 100 at Start Date)"
    )
    fig_norm.update_layout(
        xaxis_title="Date",
        yaxis_title="Normalized Price Level",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=50, b=20),
        height=400
    )
    st.plotly_chart(fig_norm, use_container_width=True)

with comp_col2:
    # Sharpe ratio comparison bar chart
    fig_sharpe = px.bar(
        metrics_df.loc[selected_stocks],
        y="Sharpe Ratio",
        title="Sharpe Ratio Comparison",
        color="Sharpe Ratio",
        color_continuous_scale=px.colors.diverging.RdYlGn,
        labels={"Stock": "Ticker"}
    )
    fig_sharpe.update_layout(
        xaxis_title="Stock Ticker",
        yaxis_title="Sharpe Ratio",
        margin=dict(l=20, r=20, t=50, b=20),
        height=400
    )
    st.plotly_chart(fig_sharpe, use_container_width=True)

# -------------------------------------------------------------
# Section 5 — Download Report
# -------------------------------------------------------------
st.header("Section 5 — Download Report")
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
