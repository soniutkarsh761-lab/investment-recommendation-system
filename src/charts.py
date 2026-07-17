import plotly.graph_objects as go
import plotly.express as px

def nifty_chart(prices, nifty_sma_50, nifty_sma_200):
    """
    Renders NIFTY 50 line chart with 50-day and 200-day moving averages overlaid.
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=prices.index, y=prices["^NSEI"], name="NIFTY 50 Close", line=dict(color="#1f77b4", width=2)))
    fig.add_trace(go.Scatter(x=prices.index, y=nifty_sma_50, name="50-day MA", line=dict(color="#ff7f0e", width=1.5, dash="dash")))
    fig.add_trace(go.Scatter(x=prices.index, y=nifty_sma_200, name="200-day MA", line=dict(color="#d62728", width=1.5, dash="dot")))
    fig.update_layout(
        title="NIFTY 50 Index (Benchmark) with Moving Averages",
        xaxis_title="Date",
        yaxis_title="Index Level",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=50, b=20),
        height=400
    )
    return fig

def stock_chart(prices, ticker, sma_50, sma_200):
    """
    Renders stock price chart with 50-day and 200-day moving averages overlaid.
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=prices.index, y=prices[ticker], name="Close Price", line=dict(color="#2ca02c", width=2)))
    fig.add_trace(go.Scatter(x=prices.index, y=sma_50, name="50-day MA", line=dict(color="#ff7f0e", width=1.5, dash="dash")))
    fig.add_trace(go.Scatter(x=prices.index, y=sma_200, name="200-day MA", line=dict(color="#d62728", width=1.5, dash="dot")))
    fig.update_layout(
        title=f"{ticker} Historical Price & Moving Averages",
        xaxis_title="Date",
        yaxis_title="Price (INR)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=50, b=20),
        height=400
    )
    return fig

def normalized_chart(normalized_prices, selected_stocks):
    """
    Renders a line chart with normalized prices of the selected stocks rebased to 100 at start.
    """
    fig = px.line(
        normalized_prices,
        x=normalized_prices.index,
        y=selected_stocks,
        title="Normalized Stock Performance (Rebased to 100 at Start Date)"
    )
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Normalized Price Level",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=50, b=20),
        height=400
    )
    return fig

def sharpe_chart(metrics_df, selected_stocks):
    """
    Renders a bar chart comparing Sharpe ratios of the selected stocks.
    """
    fig = px.bar(
        metrics_df.loc[selected_stocks],
        y="Sharpe Ratio",
        title="Sharpe Ratio Comparison",
        color="Sharpe Ratio",
        color_continuous_scale=px.colors.diverging.RdYlGn,
        labels={"Stock": "Ticker"}
    )
    fig.update_layout(
        xaxis_title="Stock Ticker",
        yaxis_title="Sharpe Ratio",
        margin=dict(l=20, r=20, t=50, b=20),
        height=400
    )
    return fig

def growth_chart(simulation_series, ticker):
    """
    Renders a line chart for the wealth growth simulation.
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=simulation_series.index, y=simulation_series, name="Investment Value", line=dict(color="#636efa", width=2)))
    fig.update_layout(
        title=f"Value of Initial ₹100,000 Investment in {ticker} Over Time",
        xaxis_title="Date",
        yaxis_title="Portfolio Value (INR)",
        margin=dict(l=20, r=20, t=50, b=20),
        height=400
    )
    return fig

def correlation_heatmap(corr_df):
    """
    Renders a Pearson correlation heatmap of daily returns for selected stocks.
    """
    fig = px.imshow(
        corr_df,
        text_auto=".2f",
        color_continuous_scale="RdBu",
        color_continuous_midpoint=0.0,
        zmin=-1.0,
        zmax=1.0,
        title="Correlation Matrix of Daily Returns"
    )
    fig.update_layout(
        margin=dict(l=20, r=20, t=50, b=20),
        height=500
    )
    return fig

def allocation_chart(alloc_df):
    """
    Renders a donut chart showing the asset allocation weights.
    """
    df_reset = alloc_df.reset_index()
    df_reset = df_reset[df_reset["Weight %"] > 0]
    
    fig = px.pie(
        df_reset, 
        values="Weight %", 
        names="Asset", 
        title="Asset Allocation Weights",
        hole=0.4,
        color_discrete_sequence=px.colors.qualitative.Plotly
    )
    fig.update_traces(textinfo="percent+label")
    fig.update_layout(
        margin=dict(l=20, r=20, t=50, b=20),
        height=400,
        showlegend=False
    )
    return fig
