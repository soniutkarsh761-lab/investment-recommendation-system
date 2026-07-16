# Investment Recommendation System

The Investment Recommendation System is a modular quantitative investment analysis platform designed to process historical stock price data and output clear, risk-adjusted Buy, Hold, or Sell recommendations. By downloading and cleaning 5 years of daily closing prices for key National Stock Exchange (NSE) stocks and benchmarking them against the NIFTY 50 index, the system dynamically adapts to market trends and risk profiles to support systematic decision-making.

**Built by Utkarsh Soni · BBA (International Business), MIT-WPU — Investment Banking with Finance Capstone Project**

---

## Key Features

- **Market Regime Detection**: Automatically categorizes the broad market as in a *Correction* or *Uptrend* based on NIFTY 50 index performance relative to its 200-day moving average.
- **Composite Scoring Engine**: Evaluates individual stock metrics across four core dimensions (totaling 100 points):
  - **Trend (30 pts)**: Overlays price vs. 200-day SMA and 50-day SMA vs. 200-day SMA.
  - **Risk-Adjusted Return (30 pts)**: Linear scaling of Sharpe ratios.
  - **Momentum (20 pts)**: Linear scaling of 6-month historical returns.
  - **Stability (20 pts)**: Linear scaling of inverse annualized volatility.
- **Three Risk Profiles**:
  - `Conservative`: Penalizes high volatility (>25%) and high beta (>1.1) stocks by up to 15 points.
  - `Moderate`: Uses baseline scoring.
  - `Aggressive`: Adds up to 10 points for high-beta stocks (>1.1) trading in short-term uptrends (above the 50-day SMA).
- **Explainable Recommendations**: Generates plain-English reasoning detailing the specific technical drivers for each Buy, Hold, or Sell recommendation.
- **Backtested Validation**: Includes a dedicated testing module to validate historical performance.

---

## Validation Results

A historical backtest was executed as of **July 15, 2024**, with a holding period of **252 trading days** (1 year) under the Moderate risk profile:

- **Buy-rated stocks** achieved an average actual forward return of **+11.22%**.
- **Hold-rated stocks** achieved an average actual forward return of **-1.58%**.
- **NIFTY 50 Benchmark** yielded a forward return of **+1.55%**.

The system's Buy-rated recommendations outperformed the benchmark index by nearly **10%** over the holding period. 
*(Note: These validation results are indicative, given the 10-stock cohort and single-date backtest sample).*

---

## Project Architecture

```text
investment-recommendation-system/
├── data/
│   ├── stock_prices.csv        # Cleaned 5-year daily closing prices of NSE tickers
│   └── sample_report.txt       # Generated plain-text investment recommendation report
├── src/
│   ├── data_loader.py          # Data ingestion, forward-fill cleaning, and CSV saving/loading
│   ├── analysis.py             # Performance metric formulas (CAGR, Volatility, Sharpe, Drawdown, Beta)
│   ├── recommender.py          # Multi-factor composite scoring and risk-profile adjustment
│   ├── report.py               # Document compiler compiling metrics into plain-text summaries
│   └── backtest.py             # Backtesting engine to run historical simulation tests
├── app.py                      # Interactive Streamlit layout and visualization dashboard
├── requirements.txt            # Project dependencies (yfinance, pandas, streamlit, plotly)
└── .gitignore                  # Git patterns configuration
```

---

## Tech Stack

- **Core Logic**: Python 3
- **Data Manipulation**: Pandas
- **Ingestion**: yfinance
- **Visualization**: Plotly
- **Interface**: Streamlit

---

## How to Run Locally

1. **Install requirements**:
   ```bash
   pip install -r requirements.txt
   ```
2. **Download and clean stock data**:
   ```bash
   python src/data_loader.py
   ```
3. **Launch the Streamlit dashboard**:
   ```bash
   streamlit run app.py
   ```

---

## Disclaimer

*Educational project. Not investment advice.*
