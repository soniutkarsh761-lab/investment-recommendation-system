import streamlit as st
import pandas as pd
from src.data_loader import load_data
from src.analysis import moving_averages, metrics_summary
from src.recommender import recommend
from src.report import build_report
from src.backtest import run_backtest
from src.insights import growth_simulation, correlation_matrix
from src.allocator import allocate, allocation_summary
from src.charts import (
    nifty_chart, stock_chart, normalized_chart, sharpe_chart,
    growth_chart, correlation_heatmap, allocation_chart
)

st.set_page_config(page_title="Investment Recommendation System", layout="wide")

def format_inr(val):
    is_neg = val < 0
    val = abs(val)
    num, dec = f"{val:.2f}".split(".")
    if len(num) <= 3:
        f_num = f"{num}.{dec}"
    else:
        rem = num[:-3]
        groups = [rem[max(i-2, 0):i] for i in range(len(rem), 0, -2)]
        groups.reverse()
        f_num = ",".join(groups) + "," + num[-3:] + "." + dec
    return f"-₹{f_num}" if is_neg else f"₹{f_num}"

@st.cache_data
def get_cached_data(): return load_data()

@st.cache_data
def get_backtest_results(_prices): return run_backtest(_prices, "2024-07-15", holding_days=252)

prices = get_cached_data()
if prices is None:
    st.error("Could not load price data. Make sure data/stock_prices.csv exists.")
    st.stop()

# Header banner & CSS Styling injection
header_html = (
    f'<div style="display:flex;justify-content:space-between;align-items:center;padding:12px 18px;background-color:#1A2332;border-radius:8px;margin-bottom:25px;border-left:5px solid #00C2A8;">'
    f'<div style="display:flex;align-items:center;gap:12px;">'
    f'<span style="background-color:#00C2A8;color:#0E1525;padding:3px 8px;border-radius:4px;font-weight:bold;font-family:monospace;font-size:14px;">US·IRS</span>'
    f'<div><div style="font-weight:bold;font-size:22px;color:#FFFFFF;line-height:1.2;">Investment Recommendation System</div>'
    f'<div style="font-size:12px;color:#8892B0;">by Utkarsh Soni</div></div></div>'
    f'<div style="text-align:right;font-size:12px;color:#8892B0;font-family:monospace;">Data as of:<br>'
    f'<span style="color:#00C2A8;font-weight:bold;">{prices.index[-1].strftime("%Y-%m-%d")}</span></div></div>'
    f'<style>'
    f'[data-testid="stMetric"], [data-testid="metric-container"] {{background-color:#1A2332;border-left:3px solid #00C2A8;padding:10px 15px;border-radius:6px;}}'
    f'button[data-baseweb="tab"] {{font-size:16px !important;color:#8892B0 !important;}}'
    f'button[data-baseweb="tab"][aria-selected="true"] {{color:#00C2A8 !important;border-bottom-color:#00C2A8 !important;}}'
    f'div[data-testid="stTable"] td, div[data-testid="stDataFrame"] td {{padding:6px 12px !important;}}'
    f'</style>'
)
st.markdown(header_html, unsafe_allow_html=True)

all_stocks = sorted([col for col in prices.columns if col != "^NSEI"])

st.sidebar.header("User Settings")
profile = st.sidebar.radio("Select Risk Profile", ["Conservative", "Moderate", "Aggressive"], index=1)
selected_stocks = st.sidebar.multiselect("Select Tracked Stocks", all_stocks, default=all_stocks)

st.sidebar.markdown("---")
st.sidebar.caption("Data: Yahoo Finance · 5-year daily history · 10 NIFTY large-caps")
st.sidebar.caption(f"Data as of: {prices.index[-1].strftime('%Y-%m-%d')}")

sma_50_df, sma_200_df = moving_averages(prices)
metrics_df = metrics_summary(prices)
latest_nifty = prices["^NSEI"].iloc[-1]
nifty_sma_200 = sma_200_df["^NSEI"].iloc[-1]

tabs = st.tabs([
    "📊 Market Overview",
    "🎯 Recommendations",
    "🔍 Stock Analysis",
    "⚖️ Portfolio Insights",
    "✅ Validation"
])

def color_recommendation(val):
    colors = {"Buy": "background-color: #d4edda; color: #155724; font-weight: bold;",
              "Hold": "background-color: #fff3cd; color: #856404; font-weight: bold;",
              "Sell": "background-color: #f8d7da; color: #721c24; font-weight: bold;"}
    return colors.get(val, "")

def show_styled_df(df):
    f = df.style.map if hasattr(df.style, "map") else df.style.applymap
    return f(color_recommendation, subset=["Recommendation"])

# Tab 1 — Market Overview
with tabs[0]:
    st.subheader("Market Overview", divider=True)
    if latest_nifty < nifty_sma_200:
        st.warning("⚠️ **Market Regime: Correction** — NIFTY below its 200-day average. Recommendations reflect defensive conditions.")
    else:
        st.success("📈 **Market Regime: Uptrend** — NIFTY 50 is trending positive.")
        
    t_6m = prices.index[-1] - pd.DateOffset(months=6)
    idx_6m = prices.index.get_indexer([t_6m], method="nearest")[0]
    nifty_return_6m = (latest_nifty - prices["^NSEI"].iloc[idx_6m]) / prices["^NSEI"].iloc[idx_6m]
    
    above_200 = sum(1 for stock in all_stocks if prices[stock].iloc[-1] > sma_200_df[stock].iloc[-1])
    pct_above_200 = (above_200 / len(all_stocks)) * 100
    
    col1, col2, col3 = st.columns(3)
    col1.metric("NIFTY 50 Latest Close", f"{latest_nifty:,.2f}")
    col2.metric("NIFTY 50 6-Month Return", f"{nifty_return_6m * 100:+.2f}%")
    col3.metric("Tracked Stocks > 200-day MA", f"{pct_above_200:.2f}%")
    st.plotly_chart(nifty_chart(prices, sma_50_df["^NSEI"], sma_200_df["^NSEI"]), use_container_width=True)

# Tab 2 — Recommendations
with tabs[1]:
    st.subheader("Stock Recommendations", divider=True)
    st.write(f"**Selected Risk Profile:** {profile}")
    if not selected_stocks:
        st.info("Select at least one stock from the sidebar to view this analysis.")
    else:
        rec_df = recommend(prices, profile=profile.lower())
        st.dataframe(show_styled_df(rec_df.loc[selected_stocks]), use_container_width=True)
        
    st.subheader("Download Recommendation Report", divider=True)
    report_profile = profile.lower()
    report_text = build_report(prices, report_profile)
    today_str = pd.Timestamp.now().strftime("%Y-%m-%d")
    st.download_button("Download Plain-Text Report", report_text, f"investment_report_{report_profile}_{today_str}.txt", "text/plain")

# Tab 3 — Stock Analysis
with tabs[2]:
    st.subheader("Stock Deep-Dive Analysis", divider=True)
    if not selected_stocks:
        st.info("Select at least one stock from the sidebar to view this analysis.")
    else:
        deep_dive_stock = st.selectbox("Select Stock for Deep-Dive Analysis", selected_stocks)
        if deep_dive_stock:
            m = metrics_df.loc[deep_dive_stock]
            m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)
            m_col1.metric("Annualized CAGR", f"{m['Annualized Return'] * 100:.2f}%")
            m_col2.metric("Annualized Volatility", f"{m['Volatility'] * 100:.2f}%")
            m_col3.metric("Sharpe Ratio", f"{m['Sharpe Ratio']:.2f}")
            m_col4.metric("Max Drawdown", f"{m['Max Drawdown'] * 100:.2f}%")
            m_col5.metric("CAPM Beta", f"{m['Beta']:.2f}")
            st.plotly_chart(stock_chart(prices, deep_dive_stock, sma_50_df[deep_dive_stock], sma_200_df[deep_dive_stock]), use_container_width=True)
            
    st.subheader("Wealth Growth Simulator", divider=True)
    if not selected_stocks:
        st.info("Select at least one stock from the sidebar to view this analysis.")
    else:
        sim_col1, sim_col2 = st.columns([1, 3])
        with sim_col1:
            sim_stock = st.selectbox("Select Stock to Simulate", selected_stocks)
            initial_amount = st.number_input("Initial Investment Amount (₹)", min_value=1000.0, value=100000.0, step=5000.0)
        if sim_stock:
            sim_series, final_val, multiple = growth_simulation(prices, sim_stock, initial_amount)
            with sim_col1:
                st.metric("Final Portfolio Value", format_inr(final_val))
                st.metric("Growth Multiple", multiple)
            with sim_col2:
                st.plotly_chart(growth_chart(sim_series, sim_stock), use_container_width=True)

# Tab 4 — Portfolio Insights
with tabs[3]:
    st.subheader("Comparative Analysis", divider=True)
    if not selected_stocks:
        st.info("Select at least one stock from the sidebar to view this analysis.")
    else:
        comp_col1, comp_col2 = st.columns(2)
        with comp_col1:
            normalized_prices = prices[selected_stocks].div(prices[selected_stocks].iloc[0]) * 100.0
            st.plotly_chart(normalized_chart(normalized_prices, selected_stocks), use_container_width=True)
        with comp_col2:
            st.plotly_chart(sharpe_chart(metrics_df, selected_stocks), use_container_width=True)
            
    st.subheader("Diversification Insights", divider=True)
    if not selected_stocks:
        st.info("Select at least one stock from the sidebar to view this analysis.")
    else:
        corr_matrix_df, most_corr, least_corr = correlation_matrix(prices)
        st.plotly_chart(correlation_heatmap(corr_matrix_df.loc[selected_stocks, selected_stocks]), use_container_width=True)
        st.caption(
            f"**Correlations**: Most correlated pair is **{most_corr[0]}** and **{most_corr[1]}** ({most_corr[2]:.2f}). "
            f"Least correlated pair is **{least_corr[0]}** and **{least_corr[1]}** ({least_corr[2]:.2f})."
        )
        
    st.subheader("Suggested Allocation", divider=True)
    if not selected_stocks:
        st.info("Select at least one stock from the sidebar to view this analysis.")
    else:
        alloc_col1, alloc_col2 = st.columns([3, 2])
        with alloc_col1:
            alloc_amount = st.number_input("Portfolio Size to Allocate (₹)", min_value=1000.0, value=100000.0, step=5000.0, key="alloc_amt")
            alloc_df = allocate(prices, profile, alloc_amount)
            display_alloc = alloc_df.copy()
            display_alloc["Weight %"] = display_alloc["Weight %"].map(lambda x: f"{x:.2f}%")
            display_alloc["Rupee Amount"] = display_alloc["Rupee Amount"].map(format_inr)
            st.dataframe(display_alloc, use_container_width=True)
            st.write(f"**Allocation Summary:** {allocation_summary(alloc_df, profile, prices)}")
        with alloc_col2:
            st.plotly_chart(allocation_chart(alloc_df), use_container_width=True)

# Tab 5 — Validation
with tabs[4]:
    st.subheader("Validation (Historical Backtest)", divider=True)
    if not selected_stocks:
        st.info("Select at least one stock from the sidebar to view this analysis.")
    else:
        backtest_df = get_backtest_results(prices)
        backtest_filtered = backtest_df.loc[selected_stocks]
        
        avg_buy_return = backtest_filtered[backtest_filtered["Recommendation"] == "Buy"]["Actual Forward Return"].mean()
        avg_hold_return = backtest_filtered[backtest_filtered["Recommendation"] == "Hold"]["Actual Forward Return"].mean()
        
        idx_bt = prices.index.get_indexer([pd.to_datetime("2024-07-15")], method="nearest")[0]
        nifty_bt_return = (prices["^NSEI"].iloc[idx_bt + 252] - prices["^NSEI"].iloc[idx_bt]) / prices["^NSEI"].iloc[idx_bt]
        
        b_col1, b_col2, b_col3 = st.columns(3)
        b_col1.metric("Average Buy Return (Backtest)", f"{avg_buy_return * 100:+.2f}%" if not pd.isna(avg_buy_return) else "N/A")
        b_col2.metric("Average Hold Return (Backtest)", f"{avg_hold_return * 100:+.2f}%" if not pd.isna(avg_hold_return) else "N/A")
        b_col3.metric("NIFTY 50 Benchmark Return", f"{nifty_bt_return * 100:+.2f}%")
        
        display_bt_df = backtest_filtered.copy()
        display_bt_df["Actual Forward Return"] = display_bt_df["Actual Forward Return"].map(lambda x: f"{x * 100:.2f}%")
        st.dataframe(show_styled_df(display_bt_df), use_container_width=True)
        st.caption("*Backtest is indicative — 10 stocks, single as-of date (2024-07-15).*")

# Footer disclaimer
st.markdown("---")
st.caption("Educational project. Not investment advice.")
