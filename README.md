# Advanced Crypto Portfolio Optimizer

A Streamlit web application designed for sophisticated analysis and optimization of digital asset portfolios, focusing on robust risk management techniques tailored for the cryptocurrency market (EUR-based).

## Key Features

-   **Robust Optimization Models**: Implements advanced portfolio construction methods beyond traditional Mean-Variance:
    -   Hierarchical Risk Parity (HRP): Focuses on diversifying risk contributions across assets based on correlation structures.
    -   Mean-CVaR Optimization: Aims to maximize risk-adjusted returns while explicitly minimizing Conditional Value-at-Risk (tail risk).
-   **Dynamic & Manual Asset Selection**:
    -   **Dynamic Mode**: Automatically selects a 'Top N' portfolio based on recent risk-adjusted performance (Sharpe Ratio) with added cluster-based diversification checks.
    -   **Manual Mode**: Allows users to select their specific assets of interest.
-   **Risk Management via Stablecoin**: Incorporates a EUR Stablecoin/Cash component, allowing the optimizer to adjust overall portfolio risk (e.g., target volatility for HRP).
-   **Backtesting Engine**: Simulate the performance of the selected dynamic strategy (selection + rebalancing + optimization) over historical periods.
-   **Momentum Prediction (Illustrative)**: Provides a simple momentum-based price projection for forward-looking context (use with caution).
-   **Data Integration**: Fetches historical price data from CoinGecko (EUR) and live prices from LiveCoinWatch (EUR).
-   **Interactive Visualizations**: Utilizes Plotly for dynamic charts including:
    -   Portfolio Allocation (Pie/Donut)
    -   Risk Contribution Breakdown (HRP)
    * Historical Performance & Backtesting Results (vs. Benchmark)
    * Predicted Portfolio Trajectory
    * Asset Price Trends
    * Correlation Matrix Heatmap
    * Hierarchical Clustering Dendrogram

## Prerequisites

-   Python 3.9 or higher
-   LiveCoinWatch API key (for live price data)

## Installation

1.  **Clone the repository:**
    ```bash
    git clone [your-repository-url]
    cd [your-repository-folder]
    ```

2.  **Set up environment variables:**
    Create a `.env` file in the project root directory and add your LiveCoinWatch API key:
    ```plaintext
    LIVE_COIN_WATCH_API_KEY=your_livecoinwatch_api_key_here
    ```

3.  **Create and activate a virtual environment** (Recommended):
    Using `venv`:
    ```bash
    python -m venv venv
    # On Windows: .\venv\Scripts\activate
    # On Unix/MacOS: source venv/bin/activate
    ```
    Or using `uv` (if installed):
    ```bash
    uv venv
    # Activate as shown above
    ```

4.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    # OR using uv:
    uv pip install -r requirements.txt
    ```

## Usage

Run the Streamlit application from your terminal:

```bash
streamlit app.py
```

Navigate to the local URL provided by Streamlit (usually http://localhost:8501) in your web browser.

## Core Libraries & Dependencies:
- streamlit: Web application framework.
- riskfolio-lib: Core library for advanced portfolio optimization (HRP, Mean-CVaR, risk metrics, plotting)
- pandas, numpy: Data manipulation and numerical operations
- plotly: Interactive data visualization
- requests: Fetching data from APIs (CoinGecko, LiveCoinWatch)
- scikit-learn, scipy, matplotlib: Used for clustering, statistical functions, and specific plots (dendrogram)
- python-dotenv: Loading environment variables (API keys)

## License
MIT License
