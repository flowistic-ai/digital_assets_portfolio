# Updated Portfolio Optimization Tool Code
# Changes:
# - Replaced ADA, MATIC with SOL, LINK.
# - Removed all stress testing features.
# - Added simple momentum prediction function.
# - Added visualization for predicted portfolio value.
# - Refined comments and structure.
# - Included basic historical backtesting visualization.

import streamlit as st
from web3 import Web3
import os
from dotenv import load_dotenv
import pandas as pd
import numpy as np # Added for calculations
import plotly.express as px
import plotly.graph_objects as go
from pypfopt import EfficientFrontier, risk_models, expected_returns, objective_functions
import requests
import time
from functools import reduce
import datetime # Added for date calculations

# --- Configuration & Setup ---

# Load environment variables (ensure you have .env file with these)
load_dotenv()
infura_url = os.getenv('INFURA_URL')
live_coin_watch_api_key = os.getenv('LIVE_COIN_WATCH_API_KEY')

# --- Blockchain Connection (Keep as is, but note its limitations) ---
# This section connects to an example contract.
# The value from `get_portfolio_value` represents the state of *that specific contract*,
# NOT necessarily the value of the *optimized portfolio* calculated by this tool
# unless trades reflecting the optimization were executed on that contract.
contract_address = '0x46711F6E9Ab3a50fdCf95D0528bfCa814b5276e4' # Example Address
abi = [ # Standard ERC20 ABI Snippets + Custom Functions (Assumed)
    {"inputs": [{"internalType": "address", "name": "_user", "type": "address"}],"name": "addToWhitelist","outputs": [],"stateMutability": "nonpayable","type": "function"},
    {"inputs": [{"internalType": "address", "name": "spender", "type": "address"},{"internalType": "uint256", "name": "value", "type": "uint256"}],"name": "approve","outputs": [{"internalType": "bool", "name": "", "type": "bool"}],"stateMutability": "nonpayable","type": "function"},
    {"inputs": [],"stateMutability": "nonpayable","type": "constructor"},
    {"inputs": [{"internalType": "address", "name": "spender", "type": "address"},{"internalType": "uint256", "name": "allowance", "type": "uint256"},{"internalType": "uint256", "name": "needed", "type": "uint256"}],"name": "ERC20InsufficientAllowance","type": "error"},
    {"inputs": [{"internalType": "address", "name": "sender", "type": "address"},{"internalType": "uint256", "name": "balance", "type": "uint256"},{"internalType": "uint256", "name": "needed", "type": "uint256"}],"name": "ERC20InsufficientBalance","type": "error"},
    {"inputs": [{"internalType": "address", "name": "approver", "type": "address"}],"name": "ERC20InvalidApprover","type": "error"},
    {"inputs": [{"internalType": "address", "name": "receiver", "type": "address"}],"name": "ERC20InvalidReceiver","type": "error"},
    {"inputs": [{"internalType": "address", "name": "sender", "type": "address"}],"name": "ERC20InvalidSender","type": "error"},
    {"inputs": [{"internalType": "address", "name": "spender", "type": "address"}],"name": "ERC20InvalidSpender","type": "error"},
    {"inputs": [{"internalType": "address", "name": "to", "type": "address"},{"internalType": "uint256", "name": "amount", "type": "uint256"}],"name": "mint","outputs": [],"stateMutability": "nonpayable","type": "function"},
    {"inputs": [{"internalType": "address", "name": "owner", "type": "address"}],"name": "OwnableInvalidOwner","type": "error"},
    {"inputs": [{"internalType": "address", "name": "account", "type": "address"}],"name": "OwnableUnauthorizedAccount","type": "error"},
    {"anonymous": False,"inputs": [{"indexed": True, "internalType": "address", "name": "owner", "type": "address"},{"indexed": True, "internalType": "address", "name": "spender", "type": "address"},{"indexed": False, "internalType": "uint256", "name": "value", "type": "uint256"}],"name": "Approval","type": "event"},
    {"anonymous": False,"inputs": [{"indexed": True, "internalType": "address", "name": "previousOwner", "type": "address"},{"indexed": True, "internalType": "address", "name": "newOwner", "type": "address"}],"name": "OwnershipTransferred","type": "event"},
    {"inputs": [],"name": "renounceOwnership","outputs": [],"stateMutability": "nonpayable","type": "function"},
    {"inputs": [{"internalType": "address", "name": "to", "type": "address"},{"internalType": "uint256", "name": "value", "type": "uint256"}],"name": "transfer","outputs": [{"internalType": "bool", "name": "", "type": "bool"}],"stateMutability": "nonpayable","type": "function"},
    {"anonymous": False,"inputs": [{"indexed": True, "internalType": "address", "name": "from", "type": "address"},{"indexed": True, "internalType": "address", "name": "to", "type": "address"},{"indexed": False, "internalType": "uint256", "name": "value", "type": "uint256"}],"name": "Transfer","type": "event"},
    {"inputs": [{"internalType": "address", "name": "from", "type": "address"},{"internalType": "address", "name": "to", "type": "address"},{"internalType": "uint256", "name": "value", "type": "uint256"}],"name": "transferFrom","outputs": [{"internalType": "bool", "name": "", "type": "bool"}],"stateMutability": "nonpayable","type": "function"},
    {"inputs": [{"internalType": "address", "name": "newOwner", "type": "address"}],"name": "transferOwnership","outputs": [],"stateMutability": "nonpayable","type": "function"},
    {"inputs": [{"internalType": "address", "name": "owner", "type": "address"},{"internalType": "address", "name": "spender", "type": "address"}],"name": "allowance","outputs": [{"internalType": "uint256", "name": "", "type": "uint256"}],"stateMutability": "view","type": "function"},
    {"inputs": [{"internalType": "address", "name": "account", "type": "address"}],"name": "balanceOf","outputs": [{"internalType": "uint256", "name": "", "type": "uint256"}],"stateMutability": "view","type": "function"},
    # --- Custom functions assumed below (replace with actual ABI if different) ---
    {"inputs": [],"name": "btcQuantity","outputs": [{"internalType": "uint256", "name": "", "type": "uint256"}],"stateMutability": "view","type": "function"},
    {"inputs": [],"name": "decimals","outputs": [{"internalType": "uint8", "name": "", "type": "uint8"}],"stateMutability": "view","type": "function"},
    {"inputs": [],"name": "ethQuantity","outputs": [{"internalType": "uint256", "name": "", "type": "uint256"}],"stateMutability": "view","type": "function"},
    {"inputs": [],"name": "getPortfolioValue","outputs": [{"internalType": "uint256", "name": "", "type": "uint256"}],"stateMutability": "view","type": "function"},
    {"inputs": [],"name": "name","outputs": [{"internalType": "string", "name": "", "type": "string"}],"stateMutability": "view","type": "function"},
    {"inputs": [],"name": "owner","outputs": [{"internalType": "address", "name": "", "type": "address"}],"stateMutability": "view","type": "function"},
    {"inputs": [],"name": "symbol","outputs": [{"internalType": "string", "name": "", "type": "string"}],"stateMutability": "view","type": "function"},
    {"inputs": [],"name": "totalSupply","outputs": [{"internalType": "uint256", "name": "", "type": "uint256"}],"stateMutability": "view","type": "function"},
    {"inputs": [],"name": "usdcQuantity","outputs": [{"internalType": "uint256", "name": "", "type": "uint256"}],"stateMutability": "view","type": "function"},
    {"inputs": [{"internalType": "address", "name": "", "type": "address"}],"name": "whitelist","outputs": [{"internalType": "bool", "name": "", "type": "bool"}],"stateMutability": "view","type": "function"}
]

# Check Infura URL validity
if not infura_url:
    st.error("INFURA_URL not found in environment variables. Please set it in your .env file.")
    st.stop()

# Connect to Web3
w3 = Web3(Web3.HTTPProvider(infura_url))
if not w3.is_connected():
    st.error("Failed to connect to the blockchain via Infura. Check your Infura URL and network status.")
    st.stop()

# Instantiate contract object
try:
    contract = w3.eth.contract(address=w3.to_checksum_address(contract_address), abi=abi)
except Exception as e:
    st.error(f"Failed to initialize contract. Check address and ABI. Error: {e}")
    st.stop()

# --- Asset Definition ---
# Updated list of assets: ETH, BTC, DOGE, SOL, LINK
# Corresponding CoinGecko IDs
ASSET_IDS = {
    "ETH": "ethereum",
    "BTC": "bitcoin",
    "DOGE": "dogecoin",
    "SOL": "solana",
    "LINK": "chainlink"
}
ASSET_SYMBOLS = list(ASSET_IDS.keys())

# --- Utility Functions ---

def usd_to_eur(prices_df, rate=0.92):
    """Converts a DataFrame of USD prices to EUR using a fixed rate."""
    # Consider using a live FX rate API for more accuracy in a production setting
    return prices_df * rate

def eur_to_usd(value_eur, rate=0.92):
    """Converts a EUR value to USD."""
    if rate == 0: return 0 # Avoid division by zero
    return value_eur / rate

# --- Data Fetching Functions ---

@st.cache_data(ttl=600) # Cache for 10 minutes
def get_onchain_portfolio_value():
    """Fetches portfolio value from the example smart contract."""
    try:
        # Ensure contract object is valid
        if contract:
            # Call the contract function (adjust decimals if needed, assuming 18 here)
            value_wei = contract.functions.getPortfolioValue().call()
            return value_wei / 10**18 # Convert from Wei to Ether (or appropriate unit)
        else:
            st.warning("Contract object not initialized.")
            return 0
    except Exception as e:
        # Log the specific error for debugging
        st.error(f"Error fetching on-chain portfolio value: {e}")
        # Return 0 or None to indicate failure gracefully
        return 0

@st.cache_data(ttl=3600) # Cache for 1 hour
def get_historical_prices(asset_id, days=180):
    """Fetches historical price data for a given asset ID from CoinGecko."""
    url = f"https://api.coingecko.com/api/v3/coins/{asset_id}/market_chart?vs_currency=usd&days={days}&interval=daily"
    max_retries = 3
    backoff_factor = 5 # Time to wait increases with retries

    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=15) # Increased timeout

            # Handle specific HTTP errors
            if response.status_code == 429: # Rate limit
                wait_time = backoff_factor * (attempt + 1)
                st.warning(f"Rate limit hit for {asset_id}. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
                continue # Go to next attempt
            elif response.status_code == 404: # Not found
                 st.error(f"Asset ID '{asset_id}' not found on CoinGecko.")
                 return pd.DataFrame() # Return empty DataFrame
            elif response.status_code != 200: # Other errors
                st.error(f"Failed to fetch data for {asset_id}. Status: {response.status_code}, Response: {response.text}")
                # Optional: Wait before retrying on general errors too
                time.sleep(backoff_factor)
                continue # Go to next attempt

            # Process successful response
            data = response.json()
            if 'prices' not in data or not data['prices']:
                st.warning(f"No price data returned for {asset_id} from CoinGecko.")
                return pd.DataFrame()

            # Create DataFrame
            prices_df = pd.DataFrame(data['prices'], columns=['timestamp', 'price'])
            prices_df['timestamp'] = pd.to_datetime(prices_df['timestamp'], unit='ms').dt.date # Keep only date part
            prices_df.set_index('timestamp', inplace=True)
            prices_df.index = pd.to_datetime(prices_df.index) # Ensure index is DateTimeIndex

            # Basic data validation
            if prices_df['price'].isnull().any():
                st.warning(f"Found missing price values for {asset_id}. Filling forward.")
                prices_df['price'].ffill(inplace=True) # Forward fill missing values

            return prices_df

        except requests.exceptions.RequestException as e:
            st.error(f"Network error fetching data for {asset_id}: {e}")
            if attempt < max_retries - 1:
                 time.sleep(backoff_factor * (attempt + 1)) # Wait before retrying network errors
            # No need to continue loop here, error is terminal for this attempt

        except Exception as e: # Catch other potential errors (e.g., JSON parsing)
            st.error(f"An unexpected error occurred fetching data for {asset_id}: {e}")
            return pd.DataFrame() # Return empty on unexpected errors

    # If loop finishes without success
    st.error(f"Failed to fetch data for {asset_id} after {max_retries} attempts.")
    return pd.DataFrame()

@st.cache_data(ttl=600) # Cache for 10 minutes
def get_live_prices(asset_symbols):
    """Fetches live prices for a list of asset symbols using LiveCoinWatch."""
    if not live_coin_watch_api_key:
        st.error("LIVE_COIN_WATCH_API_KEY not found in environment variables.")
        return {symbol: 0 for symbol in asset_symbols} # Return zeros if no key

    url = "https://api.livecoinwatch.com/coins/list"
    headers = {
        "content-type": "application/json",
        "x-api-key": live_coin_watch_api_key
    }
    # Fetch more coins than needed in case ranks shift
    payload = {
        "currency": "USD",
        "sort": "rank",
        "order": "ascending",
        "offset": 0,
        "limit": 100, # Fetch top 100
        "meta": False
    }
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=20)
            if response.status_code == 200:
                data = response.json()
                # Create a dictionary mapping symbol (code) to rate
                live_rates = {coin['code']: coin['rate'] for coin in data if 'code' in coin and 'rate' in coin}

                # Get prices for our specific assets, defaulting to 0 if not found
                final_prices = {symbol: live_rates.get(symbol, 0) for symbol in asset_symbols}
                return final_prices
            else:
                st.error(f"Failed to fetch live prices from LiveCoinWatch. Status: {response.status_code}, Response: {response.text}")
                time.sleep(5 * (attempt + 1))  # Wait before retrying
        except requests.exceptions.RequestException as e:
            st.error(f"Network error fetching live prices: {e}")
            time.sleep(5 * (attempt + 1))  # Wait before retrying
        except Exception as e:
            st.error(f"Error processing live prices: {e}")
            return {symbol: 0 for symbol in asset_symbols}
    return {symbol: 0 for symbol in asset_symbols}  # Return default if all retries fail

# --- Portfolio Optimization Function ---

def optimize_portfolio(prices_eur, risk_free_rate_eur, risk_aversion_param, max_weight=0.50):
    """
    Optimizes the portfolio based on historical prices (in EUR), risk-free rate,
    risk aversion, and constraints.

    Args:
        prices_eur (pd.DataFrame): DataFrame of historical prices in EUR, indexed by date.
        risk_free_rate_eur (float): Annualized risk-free rate (e.g., 0.02 for 2%).
        risk_aversion_param (float): User's risk aversion (higher means more risk averse).
                                      Used for maximizing quadratic utility.
        max_weight (float): Maximum weight allowed for any single asset (0 to 1).

    Returns:
        tuple: (dict_of_cleaned_weights, tuple_of_performance_metrics)
               Returns (None, (None, None, None)) on failure.
    """
    if prices_eur.empty:
        st.error("Cannot optimize: Price data is empty.")
        return None, (None, None, None)

    try:
        # 1. Calculate Expected Returns (using mean historical return)
        mu = expected_returns.mean_historical_return(prices_eur, frequency=252) # Annualized

        # 2. Calculate Covariance Matrix (using Ledoit-Wolf shrinkage)
        S = risk_models.CovarianceShrinkage(prices_eur, frequency=252).ledoit_wolf()

        # 3. Initialize EfficientFrontier
        ef = EfficientFrontier(mu, S)

        # 4. Add Constraints
        # Max weight per asset
        n_assets = len(prices_eur.columns)
        for i in range(n_assets):
            ef.add_constraint(lambda w, i=i: w[i] <= max_weight)
            ef.add_constraint(lambda w, i=i: w[i] >= 0.00) # Optional: Ensure non-negative weights

        # Add L2 regularization (helps prevent extreme weights)
        ef.add_objective(objective_functions.L2_reg, gamma=0.1) # Increased gamma slightly

        # 5. Define Optimization Objective: Maximize Quadratic Utility
        # This balances expected return and risk according to user's risk aversion.
        # A higher risk_aversion_param penalizes variance more heavily.
        ef.max_quadratic_utility(risk_aversion=risk_aversion_param, market_neutral=False)

        # --- Alternative Objectives (commented out, keep for reference) ---
        # if risk_tolerance <= 0.1: # Very low risk tolerance
        #     ef.min_volatility()
        # elif risk_tolerance >= 0.9: # Very high risk tolerance
        #     ef.max_sharpe(risk_free_rate=risk_free_rate_eur)
        # else: # Intermediate risk tolerance - target a specific risk level
        #     # Calculate target volatility based on tolerance (example mapping)
        #     min_vol = EfficientFrontier(mu, S).min_volatility()
        #     max_ret_portfolio = EfficientFrontier(mu, S) # Need a separate instance for max return portfolio calc
        #     max_ret_portfolio.max_quadratic_utility(risk_aversion=0.01) # Approx max return
        #     max_vol = max_ret_portfolio.portfolio_performance(verbose=False)[1] # Get volatility of max return portfolio
        #     target_vol = min_vol + risk_tolerance * (max_vol - min_vol)
        #     try:
        #         ef.efficient_risk(target_volatility=target_vol)
        #     except ValueError as e_risk:
        #         st.warning(f"Could not target volatility {target_vol:.2f} ({e_risk}). Falling back to max Sharpe.")
        #         ef.max_sharpe(risk_free_rate=risk_free_rate_eur)
        # --- End Alternative Objectives ---

        # 6. Perform Optimization and Clean Weights
        cleaned_weights = ef.clean_weights() # Rounds small weights to zero

        # Check if optimization was successful (weights sum to 1)
        if not np.isclose(sum(cleaned_weights.values()), 1):
             st.warning("Optimization might not have converged properly. Weights do not sum to 1.")
             # Fallback: Try min volatility as a robust option
             ef_fallback = EfficientFrontier(mu, S)
             for i in range(n_assets):
                 ef_fallback.add_constraint(lambda w, i=i: w[i] <= max_weight)
                 ef_fallback.add_constraint(lambda w, i=i: w[i] >= 0.00)
             ef_fallback.min_volatility()
             cleaned_weights = ef_fallback.clean_weights()
             if not np.isclose(sum(cleaned_weights.values()), 1):
                 st.error("Fallback optimization (min volatility) also failed. Cannot proceed.")
                 return None, (None, None, None)
             st.info("Used minimum volatility portfolio as fallback.")


        # 7. Calculate Expected Performance Metrics
        expected_return, annual_volatility, sharpe_ratio = ef.portfolio_performance(
            verbose=False, risk_free_rate=risk_free_rate_eur
        )

        return cleaned_weights, (expected_return, annual_volatility, sharpe_ratio)

    except ValueError as ve:
        st.error(f"Optimization Error: {ve}. This might be due to data issues (e.g., insufficient history, assets with zero variance) or conflicting constraints.")
        return None, (None, None, None)
    except Exception as e:
        st.error(f"An unexpected error occurred during optimization: {e}")
        return None, (None, None, None)

# --- Predictive Modeling Function (Simple Momentum) ---

def predict_momentum(prices_df, lookback_days=30, projection_days=30):
    """
    Predicts future prices based on simple momentum (price change over lookback period).
    WARNING: This is a very basic model and should not be solely relied upon for investment decisions.

    Args:
        prices_df (pd.DataFrame): DataFrame of historical prices (USD or EUR).
        lookback_days (int): Number of days to calculate momentum over.
        projection_days (int): Number of days into the future to project.

    Returns:
        pd.DataFrame: DataFrame with predicted prices for the projection period.
                      Returns empty DataFrame on error or insufficient data.
    """
    if prices_df.empty or len(prices_df) < lookback_days:
        st.warning(f"Insufficient data for momentum prediction (need {lookback_days} days, have {len(prices_df)}).")
        return pd.DataFrame()

    try:
        # Calculate momentum: (Price_today / Price_lookback_days_ago) - 1
        # Using pct_change is simpler: price_today / price_yesterday - 1
        # Let's use the average daily return over the lookback period for projection
        daily_returns = prices_df.pct_change().dropna()
        if len(daily_returns) < lookback_days:
             st.warning(f"Insufficient return data for {lookback_days}-day momentum average.")
             return pd.DataFrame()

        # Calculate average daily return over the lookback period
        avg_daily_return = daily_returns.tail(lookback_days).mean()

        # Get the last known price
        last_price = prices_df.iloc[-1]

        # Project future prices
        future_dates = pd.date_range(prices_df.index[-1] + pd.Timedelta(days=1), periods=projection_days, freq='D')
        predicted_prices = pd.DataFrame(index=future_dates, columns=prices_df.columns)

        # Simple projection: Last Price * (1 + avg_daily_return) ^ n_days
        for i in range(projection_days):
            day_num = i + 1
            predicted_prices.iloc[i] = last_price * ((1 + avg_daily_return) ** day_num)

        return predicted_prices

    except Exception as e:
        st.error(f"Error during momentum prediction: {e}")
        return pd.DataFrame()


# --- Streamlit App Layout ---

st.set_page_config(page_title="Digital Asset Portfolio Optimizer", layout="wide", initial_sidebar_state="expanded")
st.title("🚀 Digital Asset Portfolio Optimizer & Predictor")
st.markdown("Optimize your crypto portfolio based on historical data and explore simple momentum-based predictions.")
st.warning(" This tool performs Mean-Variance Optimization (maximizing Quadratic Utility based on user risk aversion) using mean historical returns and Ledoit-Wolf shrunk covariance, subject to max weight and L2 constraints.")

# --- Sidebar Controls ---
with st.sidebar:
    st.header("⚙️ Portfolio Configuration")
    st.subheader("Initial Capital (EUR)")
    principal_eur = st.number_input("Enter your initial capital in EUR:", value=10000, step=1000, min_value=100)

    st.subheader("Risk & Optimization Parameters")
    # Using Risk Aversion for Quadratic Utility (more intuitive than tolerance for this objective)
    # Scale: 1 (low aversion) to 10 (high aversion) seems reasonable
    risk_aversion = st.slider("Risk Aversion (1=Low, 10=High):", min_value=0.1, max_value=20.0, value=5.0, step=0.1,
                              help="Higher values penalize risk more heavily in the optimization.")

    risk_free_rate_eur = st.slider(
        "Annual Risk-Free Rate (EUR):",
        min_value=0.00,
        max_value=0.10, # 10%
        value=0.02, # Default 2%
        step=0.005,
        format="%.3f",
        help="Represents the return on a 'risk-free' investment (e.g., government bond yield). Used for Sharpe Ratio calculation."
    )

    max_asset_weight = st.slider("Max Single Asset Weight:", 0.1, 1.0, 0.40, 0.05, # Default 40%
                                 help="Maximum percentage of the portfolio allowed in any single cryptocurrency.")

    st.subheader("Prediction Parameters")
    pred_lookback = st.slider("Momentum Lookback (Days):", 10, 90, 30, 5,
                               help="How many past days to use for calculating momentum.")
    pred_projection = st.slider("Prediction Horizon (Days):", 5, 90, 30, 5,
                                help="How many days into the future to predict.")

    st.markdown("---")
    st.info("Blockchain integration (below) shows data from an *example* contract. It's not directly linked to the optimization results unless you execute trades.")
    # Display On-Chain Value (from example contract)
    onchain_value_usd = get_onchain_portfolio_value()
    st.metric("Example On-Chain Value (USD)", f"${onchain_value_usd:,.2f}" if onchain_value_usd is not None else "N/A")


# --- Main Area Tabs ---
tab1, tab2, tab3 = st.tabs(["📈 Optimize & Predict", "📊 Visualizations", "ℹ️ Asset Data"])

# --- Tab 1: Optimize & Predict ---
with tab1:
    st.header("Portfolio Optimization")

    # Fetch historical data for all assets
    asset_data_usd = {}
    all_data_fetched = True
    with st.spinner("Fetching historical price data..."):
        for symbol, asset_id in ASSET_IDS.items():
            df = get_historical_prices(asset_id)
            if df.empty:
                st.error(f"Failed to fetch sufficient data for {symbol}. Cannot proceed with optimization.")
                all_data_fetched = False
                # break # Stop fetching if one fails critical data
            else:
                asset_data_usd[symbol] = df['price'] # Select only the price column

    if all_data_fetched and asset_data_usd:
        # Combine into a single DataFrame, aligning dates
        try:
            prices_usd = pd.concat(asset_data_usd, axis=1).sort_index()
            # Handle potential NaNs after concat (e.g., if assets have different start dates)
            prices_usd.ffill(inplace=True) # Forward fill first
            prices_usd.bfill(inplace=True) # Then backfill remaining NaNs at the start
            prices_usd.dropna(inplace=True) # Drop any rows that still have NaNs (shouldn't happen with ffill/bfill)

            if prices_usd.empty:
                 st.error("No overlapping historical data found for the selected assets after alignment.")
                 st.stop() # Stop execution if no valid data

            # Convert to EUR for optimization
            prices_eur = usd_to_eur(prices_usd)
            st.session_state['prices_usd'] = prices_usd # Store USD prices for later use
            st.session_state['prices_eur'] = prices_eur # Store EUR prices

            st.success("Historical price data loaded successfully.")

            # Optimization Button
            if st.button("🚀 Run Optimization", key="run_opt"):
                with st.spinner("Optimizing portfolio..."):
                    weights, performance = optimize_portfolio(
                        prices_eur,
                        risk_free_rate_eur,
                        risk_aversion, # Use risk aversion parameter
                        max_asset_weight
                    )

                if weights and performance:
                    st.session_state['opt_weights'] = weights
                    st.session_state['opt_performance'] = performance
                    st.success("✅ Optimization Complete!")
                    st.subheader("Optimized Portfolio Allocation (EUR)")
                    weights_df = pd.DataFrame.from_dict(weights, orient='index', columns=['Weight'])
                    weights_df['Value (EUR)'] = weights_df['Weight'] * principal_eur
                    st.dataframe(weights_df.style.format({'Weight': '{:.2%}', 'Value (EUR)': '€{:,.2f}'}))
                    ret, vol, sharpe = performance
                    st.subheader("Expected Portfolio Performance (Annualized, EUR)")
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Expected Return", f"{ret:.2%}")
                    col2.metric("Volatility (Risk)", f"{vol:.2%}")
                    col3.metric("Sharpe Ratio", f"{sharpe:.2f}")

                    with st.expander("Performance Metric Definitions"):
                         st.markdown("- **Expected Annual Return**: The anticipated average return over a year, based on historical data (mean return).")
                         st.markdown("- **Annual Volatility**: The standard deviation of returns (a measure of risk). Higher volatility means price swings are larger.")
                         st.markdown("- **Sharpe Ratio**: Measures risk-adjusted return. Calculated as `(Expected Return - Risk-Free Rate) / Volatility`. Higher is generally better.")

                else:
                    st.error("Optimization failed. Please check the data or parameters.")
                    if 'opt_weights' in st.session_state: del st.session_state['opt_weights'] # Clear previous results
                    if 'opt_performance' in st.session_state: del st.session_state['opt_performance']

        except Exception as e:
             st.error(f"Error processing historical data: {e}")
             st.stop()


    else:
        st.warning("Could not fetch all required asset data. Optimization disabled.")

    st.markdown("---")
    st.header("Momentum-Based Prediction")

    # Prediction Button (only enable if optimization was successful)
    if 'opt_weights' in st.session_state and 'prices_usd' in st.session_state:
        if st.button("🔮 Generate Prediction", key="run_pred"):
            with st.spinner("Calculating momentum and predicting..."):
                prices_usd_hist = st.session_state['prices_usd']
                predicted_prices_usd = predict_momentum(prices_usd_hist, pred_lookback, pred_projection)

            if not predicted_prices_usd.empty:
                st.session_state['predicted_prices_usd'] = predicted_prices_usd
                st.success(f"✅ Prediction generated for the next {pred_projection} days.")

                # Calculate predicted portfolio value
                opt_w = st.session_state['opt_weights']
                initial_value_usd = eur_to_usd(principal_eur) # Convert principal to USD for comparison

                # Calculate initial USD allocation per asset
                initial_alloc_usd = {asset: opt_w.get(asset, 0) * initial_value_usd for asset in prices_usd_hist.columns}

                # Calculate number of units of each asset initially bought (approximate)
                # Use the last *historical* price for this calculation
                last_hist_prices = prices_usd_hist.iloc[-1]
                units = {asset: initial_alloc_usd[asset] / last_hist_prices[asset]
                         if last_hist_prices[asset] > 0 else 0
                         for asset in initial_alloc_usd}

                # Calculate predicted portfolio value for each future day
                predicted_portfolio_values = pd.Series(index=predicted_prices_usd.index, dtype=float)
                for date in predicted_prices_usd.index:
                    daily_value = 0
                    for asset in predicted_prices_usd.columns:
                        daily_value += units[asset] * predicted_prices_usd.loc[date, asset]
                    predicted_portfolio_values[date] = daily_value

                st.session_state['predicted_portfolio_values_usd'] = predicted_portfolio_values

                # Display summary
                final_predicted_value = predicted_portfolio_values.iloc[-1]
                predicted_profit_loss = final_predicted_value - initial_value_usd
                predicted_profit_loss_pct = (predicted_profit_loss / initial_value_usd) * 100 if initial_value_usd else 0

                st.subheader(f"Predicted Portfolio Value after {pred_projection} Days (USD)")
                st.metric("Predicted Value", f"${final_predicted_value:,.2f}",
                          delta=f"${predicted_profit_loss:,.2f} ({predicted_profit_loss_pct:+.2f}%) vs Initial")
                st.caption(f"Based on simple momentum from the past {pred_lookback} days. **This is highly speculative.**")

            else:
                st.error("Prediction failed. Insufficient data or error during calculation.")
                if 'predicted_prices_usd' in st.session_state: del st.session_state['predicted_prices_usd']
                if 'predicted_portfolio_values_usd' in st.session_state: del st.session_state['predicted_portfolio_values_usd']
    else:
        st.info("Run optimization first to enable predictions.")


# --- Tab 2: Visualizations ---
with tab2:
    st.header("Portfolio Visualizations")

    # 1. Optimized Allocation Pie Chart
    st.subheader("Optimized Portfolio Allocation")
    if 'opt_weights' in st.session_state:
        opt_w = st.session_state['opt_weights']
        # Filter out zero weights for cleaner pie chart
        non_zero_weights = {k: v for k, v in opt_w.items() if v > 0.001} # Threshold for display
        if non_zero_weights:
            fig_pie = px.pie(
                names=list(non_zero_weights.keys()),
                values=list(non_zero_weights.values()),
                title="Optimized Asset Weights (EUR)",
                hole=0.3, # Make it a donut chart
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            fig_pie.update_traces(textinfo='percent+label', pull=[0.05]*len(non_zero_weights)) # Explode slices slightly
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info("Optimization resulted in zero allocation (or weights too small to display).")
    else:
        st.info("Run optimization on the 'Optimize & Predict' tab to see the allocation chart.")

    st.markdown("---")

    # 2. Historical Performance Backtest (Simple)
    st.subheader("Historical Portfolio Performance (Backtest)")
    if 'opt_weights' in st.session_state and 'prices_usd' in st.session_state:
        opt_w = st.session_state['opt_weights']
        prices_usd_hist = st.session_state['prices_usd']
        initial_value_usd = eur_to_usd(principal_eur)

        # Calculate historical portfolio value based on fixed initial weights
        # Calculate daily returns of assets
        returns = prices_usd_hist.pct_change().dropna()

        # Calculate weighted daily portfolio returns
        # Ensure weights align with returns columns
        aligned_weights = pd.Series(opt_w).reindex(returns.columns).fillna(0)
        portfolio_returns = returns.dot(aligned_weights)

        # Calculate cumulative portfolio value
        cumulative_returns = (1 + portfolio_returns).cumprod()
        historical_portfolio_value = initial_value_usd * cumulative_returns

        # Add the initial value at the start date
        start_date = historical_portfolio_value.index.min() - pd.Timedelta(days=1) # Approx start
        historical_portfolio_value = pd.concat([pd.Series({start_date: initial_value_usd}), historical_portfolio_value])
        historical_portfolio_value.index = pd.to_datetime(historical_portfolio_value.index)


        fig_hist = go.Figure()
        fig_hist.add_trace(go.Scatter(x=historical_portfolio_value.index, y=historical_portfolio_value.values,
                                      mode='lines', name='Optimized Portfolio (Backtest)',
                                      line=dict(color='royalblue', width=2)))

        fig_hist.update_layout(
            title="Portfolio Value Over Time (USD) - Based on Initial Optimized Weights",
            xaxis_title="Date",
            yaxis_title="Portfolio Value (USD)",
            yaxis_tickprefix="$",
            hovermode="x unified"
        )
        st.plotly_chart(fig_hist, use_container_width=True)
        st.caption("Note: This backtest assumes the initial optimized weights were held constant throughout the historical period.")

    else:
        st.info("Run optimization to see the historical performance backtest.")

    st.markdown("---")

    # 3. Prediction Visualization
    st.subheader("Predicted Portfolio Trajectory (Momentum-Based)")
    if 'predicted_portfolio_values_usd' in st.session_state and 'historical_portfolio_value' in locals():
        pred_values = st.session_state['predicted_portfolio_values_usd']

        # Combine historical and predicted data for plotting
        plot_df_hist = historical_portfolio_value.reset_index()
        plot_df_hist.columns = ['Date', 'Value']
        plot_df_hist['Type'] = 'Historical'

        plot_df_pred = pred_values.reset_index()
        plot_df_pred.columns = ['Date', 'Value']
        plot_df_pred['Type'] = 'Predicted'

        # Add the last historical point to the start of the prediction for continuity
        last_hist_point = plot_df_hist.iloc[[-1]].copy()
        last_hist_point['Type'] = 'Predicted' # Treat as start of prediction line
        plot_df_combined = pd.concat([plot_df_hist, last_hist_point, plot_df_pred], ignore_index=True)


        fig_pred = px.line(plot_df_combined, x='Date', y='Value', color='Type',
                           title=f"Portfolio Value: Historical & Predicted ({pred_projection} Days)",
                           labels={"Value": "Portfolio Value (USD)"},
                           color_discrete_map={'Historical': 'royalblue', 'Predicted': 'lightcoral'}) # Custom colors

        fig_pred.update_traces(line=dict(dash='dash'), selector=dict(name='Predicted')) # Dashed line for prediction
        fig_pred.update_layout(
            yaxis_tickprefix="$",
            hovermode="x unified",
            legend_title_text='Period'
        )
        st.plotly_chart(fig_pred, use_container_width=True)
        st.warning("Prediction is based on simple momentum and is highly speculative. Do not treat as financial advice.")

    elif 'opt_weights' in st.session_state:
         st.info("Generate prediction on the 'Optimize & Predict' tab to see the forecast chart.")
    else:
         st.info("Run optimization and generate prediction to see the forecast chart.")


# --- Tab 3: Asset Data ---
with tab3:
    st.header("Live & Historical Asset Data")

    # Fetch and display live prices
    st.subheader("Live Prices (USD)")
    if st.button("Refresh Live Prices", key="refresh_live"):
        live_p = get_live_prices(ASSET_SYMBOLS)
        st.session_state['live_prices_data'] = {
            'prices': live_p,
            'timestamp': time.time()
        }
    elif 'live_prices_data' not in st.session_state:
         # Fetch on first load if not in state
        live_p = get_live_prices(ASSET_SYMBOLS)
        st.session_state['live_prices_data'] = {
            'prices': live_p,
            'timestamp': time.time()
        }


    if 'live_prices_data' in st.session_state:
        data_dict = st.session_state['live_prices_data']['prices']
        ts = st.session_state['live_prices_data']['timestamp']
        last_upd = datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")
        st.caption(f"Last updated: {last_upd}")

        cols = st.columns(len(ASSET_SYMBOLS))
        for i, symbol in enumerate(ASSET_SYMBOLS):
            price = data_dict.get(symbol, 0)
            cols[i].metric(f"{symbol} Price", f"${price:,.2f}" if price else "N/A")
    else:
        st.warning("Live prices could not be fetched.")

    st.markdown("---")

    # Display historical price trends
    st.subheader("Historical Price Trends (USD)")
    if 'prices_usd' in st.session_state:
        prices_usd_hist = st.session_state['prices_usd']
        if not prices_usd_hist.empty:
            # Melt dataframe for Plotly Express
            prices_melted = prices_usd_hist.reset_index().melt(id_vars=['timestamp'], var_name='Asset', value_name='Price')

            fig_trends = px.line(
                prices_melted,
                x='timestamp',
                y='Price',
                color='Asset',
                title="Historical Price Trends (USD)",
                labels={"timestamp": "Date", "Price": "Price (USD)"},
                color_discrete_sequence=px.colors.qualitative.Vivid # Use a different color sequence
            )
            fig_trends.update_layout(
                 yaxis_tickprefix="$",
                 hovermode="x unified"
            )
            st.plotly_chart(fig_trends, use_container_width=True)
        else:
            st.warning("Historical price data is empty.")
    else:
        st.info("Historical data will be loaded when you navigate to the 'Optimize & Predict' tab.")

# --- Footer ---
st.markdown("---")
st.caption("Developed using Streamlit, PyPortfolioOpt, Web3.py, Plotly, and CoinGecko/LiveCoinWatch APIs.")

