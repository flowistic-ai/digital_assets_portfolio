import streamlit as st
from web3 import Web3
import os
from dotenv import load_dotenv
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pypfopt import EfficientFrontier, risk_models, expected_returns
import requests
import time
from functools import reduce

# Load environment variables
load_dotenv()
infura_url = os.getenv('INFURA_URL')
live_coin_watch_api_key = os.getenv('LIVE_COIN_WATCH_API_KEY')

# Contract details
contract_address = '0x46711F6E9Ab3a50fdCf95D0528bfCa814b5276e4'
abi = [
    {
        "inputs": [{"internalType": "address", "name": "_user", "type": "address"}],
        "name": "addToWhitelist",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function"
    },
    {
        "inputs": [
            {"internalType": "address", "name": "spender", "type": "address"},
            {"internalType": "uint256", "name": "value", "type": "uint256"}
        ],
        "name": "approve",
        "outputs": [{"internalType": "bool", "name": "", "type": "bool"}],
        "stateMutability": "nonpayable",
        "type": "function"
    },
    {
        "inputs": [],
        "stateMutability": "nonpayable",
        "type": "constructor"
    },
    {
        "inputs": [
            {"internalType": "address", "name": "spender", "type": "address"},
            {"internalType": "uint256", "name": "allowance", "type": "uint256"},
            {"internalType": "uint256", "name": "needed", "type": "uint256"}
        ],
        "name": "ERC20InsufficientAllowance",
        "type": "error"
    },
    {
        "inputs": [
            {"internalType": "address", "name": "sender", "type": "address"},
            {"internalType": "uint256", "name": "balance", "type": "uint256"},
            {"internalType": "uint256", "name": "needed", "type": "uint256"}
        ],
        "name": "ERC20InsufficientBalance",
        "type": "error"
    },
    {
        "inputs": [{"internalType": "address", "name": "approver", "type": "address"}],
        "name": "ERC20InvalidApprover",
        "type": "error"
    },
    {
        "inputs": [{"internalType": "address", "name": "receiver", "type": "address"}],
        "name": "ERC20InvalidReceiver",
        "type": "error"
    },
    {
        "inputs": [{"internalType": "address", "name": "sender", "type": "address"}],
        "name": "ERC20InvalidSender",
        "type": "error"
    },
    {
        "inputs": [{"internalType": "address", "name": "spender", "type": "address"}],
        "name": "ERC20InvalidSpender",
        "type": "error"
    },
    {
        "inputs": [
            {"internalType": "address", "name": "to", "type": "address"},
            {"internalType": "uint256", "name": "amount", "type": "uint256"}
        ],
        "name": "mint",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function"
    },
    {
        "inputs": [{"internalType": "address", "name": "owner", "type": "address"}],
        "name": "OwnableInvalidOwner",
        "type": "error"
    },
    {
        "inputs": [{"internalType": "address", "name": "account", "type": "address"}],
        "name": "OwnableUnauthorizedAccount",
        "type": "error"
    },
    {
        "anonymous": False,
        "inputs": [
            {"indexed": True, "internalType": "address", "name": "owner", "type": "address"},
            {"indexed": True, "internalType": "address", "name": "spender", "type": "address"},
            {"indexed": False, "internalType": "uint256", "name": "value", "type": "uint256"}
        ],
        "name": "Approval",
        "type": "event"
    },
    {
        "anonymous": False,
        "inputs": [
            {"indexed": True, "internalType": "address", "name": "previousOwner", "type": "address"},
            {"indexed": True, "internalType": "address", "name": "newOwner", "type": "address"}
        ],
        "name": "OwnershipTransferred",
        "type": "event"
    },
    {
        "inputs": [],
        "name": "renounceOwnership",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function"
    },
    {
        "inputs": [
            {"internalType": "address", "name": "to", "type": "address"},
            {"internalType": "uint256", "name": "value", "type": "uint256"}
        ],
        "name": "transfer",
        "outputs": [{"internalType": "bool", "name": "", "type": "bool"}],
        "stateMutability": "nonpayable",
        "type": "function"
    },
    {
        "anonymous": False,
        "inputs": [
            {"indexed": True, "internalType": "address", "name": "from", "type": "address"},
            {"indexed": True, "internalType": "address", "name": "to", "type": "address"},
            {"indexed": False, "internalType": "uint256", "name": "value", "type": "uint256"}
        ],
        "name": "Transfer",
        "type": "event"
    },
    {
        "inputs": [
            {"internalType": "address", "name": "from", "type": "address"},
            {"internalType": "address", "name": "to", "type": "address"},
            {"internalType": "uint256", "name": "value", "type": "uint256"}
        ],
        "name": "transferFrom",
        "outputs": [{"internalType": "bool", "name": "", "type": "bool"}],
        "stateMutability": "nonpayable",
        "type": "function"
    },
    {
        "inputs": [{"internalType": "address", "name": "newOwner", "type": "address"}],
        "name": "transferOwnership",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function"
    },
    {
        "inputs": [
            {"internalType": "address", "name": "owner", "type": "address"},
            {"internalType": "address", "name": "spender", "type": "address"}
        ],
        "name": "allowance",
        "outputs": [{"internalType": "uint256", "name": "", "type": "uint256"}],
        "stateMutability": "view",
        "type": "function"
    },
    {
        "inputs": [{"internalType": "address", "name": "account", "type": "address"}],
        "name": "balanceOf",
        "outputs": [{"internalType": "uint256", "name": "", "type": "uint256"}],
        "stateMutability": "view",
        "type": "function"
    },
    {
        "inputs": [],
        "name": "btcQuantity",
        "outputs": [{"internalType": "uint256", "name": "", "type": "uint256"}],
        "stateMutability": "view",
        "type": "function"
    },
    {
        "inputs": [],
        "name": "decimals",
        "outputs": [{"internalType": "uint8", "name": "", "type": "uint8"}],
        "stateMutability": "view",
        "type": "function"
    },
    {
        "inputs": [],
        "name": "ethQuantity",
        "outputs": [{"internalType": "uint256", "name": "", "type": "uint256"}],
        "stateMutability": "view",
        "type": "function"
    },
    {
        "inputs": [],
        "name": "getPortfolioValue",
        "outputs": [{"internalType": "uint256", "name": "", "type": "uint256"}],
        "stateMutability": "view",
        "type": "function"
    },
    {
        "inputs": [],
        "name": "name",
        "outputs": [{"internalType": "string", "name": "", "type": "string"}],
        "stateMutability": "view",
        "type": "function"
    },
    {
        "inputs": [],
        "name": "owner",
        "outputs": [{"internalType": "address", "name": "", "type": "address"}],
        "stateMutability": "view",
        "type": "function"
    },
    {
        "inputs": [],
        "name": "symbol",
        "outputs": [{"internalType": "string", "name": "", "type": "string"}],
        "stateMutability": "view",
        "type": "function"
    },
    {
        "inputs": [],
        "name": "totalSupply",
        "outputs": [{"internalType": "uint256", "name": "", "type": "uint256"}],
        "stateMutability": "view",
        "type": "function"
    },
    {
        "inputs": [],
        "name": "usdcQuantity",
        "outputs": [{"internalType": "uint256", "name": "", "type": "uint256"}],
        "stateMutability": "view",
        "type": "function"
    },
    {
        "inputs": [{"internalType": "address", "name": "", "type": "address"}],
        "name": "whitelist",
        "outputs": [{"internalType": "bool", "name": "", "type": "bool"}],
        "stateMutability": "view",
        "type": "function"
    }
]

# Connect to Sepolia testnet
w3 = Web3(Web3.HTTPProvider(infura_url))
if not w3.is_connected():
    st.error("Failed to connect to the blockchain. Check your Infura URL.")
    st.stop()

# Load the contract
contract = w3.eth.contract(address=contract_address, abi=abi)

# Function to fetch portfolio value from the contract
@st.cache_data(ttl=60)
def get_portfolio_value():
    try:
        value = contract.functions.getPortfolioValue().call()
        return value / 10**18  # Adjust for 18 decimals
    except Exception as e:
        st.error(f"Error fetching portfolio value: {e}")
        return 0  # Default to 0 if fetching fails

# Function to check if an address is whitelisted
def check_compliance(address):
    try:
        return contract.functions.whitelist(address).call()
    except Exception as e:
        st.error(f"Error checking compliance: {e}")
        return None

# Function to fetch historical prices from CoinGecko with caching and retry logic
@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_historical_prices(asset, days=90):
    url = f"https://api.coingecko.com/api/v3/coins/{asset}/market_chart?vs_currency=usd&days={days}"
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 429:  # Rate limit exceeded
                wait_time = 10 * (attempt + 1)  # Exponential backoff
                st.warning(f"Rate limit hit for {asset}. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
                continue
            if response.status_code != 200:
                st.error(f"Failed to fetch data for {asset}. Status code: {response.status_code}")
                return pd.DataFrame()
            data = response.json()
            if 'prices' not in data or len(data['prices']) == 0:
                st.error(f"No price data returned for {asset}.")
                return pd.DataFrame()
            prices = pd.DataFrame(data['prices'], columns=['timestamp', 'price'])
            prices['timestamp'] = pd.to_datetime(prices['timestamp'], unit='ms')
            prices.set_index('timestamp', inplace=True)
            return prices
        except Exception as e:
            st.error(f"Error fetching data for {asset}: {str(e)}")
            return pd.DataFrame()
    st.error(f"Failed to fetch data for {asset} after {max_retries} attempts.")
    return pd.DataFrame()

# Function to fetch live prices from Live Coin Watch
def get_live_prices():
    url = "https://api.livecoinwatch.com/coins/list"
    headers = {
        "content-type": "application/json",
        "x-api-key": live_coin_watch_api_key
    }
    payload = {
        "currency": "USD",
        "sort": "rank",
        "order": "ascending",
        "offset": 0,
        "limit": 50,
        "meta": False
    }
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=10)
        if response.status_code != 200:
            st.error(f"Failed to fetch live prices. Status code: {response.status_code}")
            return {'ETH': 0, 'BTC': 0, 'USDC': 0}  # Default values
        data = response.json()
        prices = {coin['code']: coin['rate'] for coin in data if coin['code'] in ['ETH', 'BTC', 'USDC']}
        # Ensure all required keys are present
        default_prices = {'ETH': 0, 'BTC': 0, 'USDC': 0}
        default_prices.update(prices)
        return default_prices
    except Exception as e:
        st.error(f"Error fetching live prices: {e}")
        return {'ETH': 0, 'BTC': 0, 'USDC': 0}  # Default values

# Optimized portfolio function
def optimize_portfolio():
    assets = ['ethereum', 'bitcoin', 'usd-coin']
    prices_list = [get_historical_prices(asset) for asset in assets]
    
    if any(df.empty for df in prices_list):
        st.error("One or more assets have no price data. Check your connection or asset IDs.")
        return None, None, None
    
    start = max(df.index.min() for df in prices_list)
    end = min(df.index.max() for df in prices_list)
    
    if start >= end:
        st.error("No overlapping time range across the assets.")
        return None, None, None
    
    common_index = pd.date_range(start=start, end=end, freq='H')
    prices_list_aligned = [df.reindex(common_index, method='ffill') for df in prices_list]
    prices = pd.concat(prices_list_aligned, axis=1)
    prices.columns = ['ETH', 'BTC', 'USDC']
    
    prices = prices.apply(pd.to_numeric, errors='coerce')
    prices.dropna(inplace=True)
    
    if prices.empty:
        st.error("No valid price data available after alignment and cleaning.")
        return None, None, None
    
    mu = expected_returns.mean_historical_return(prices)
    S = risk_models.CovarianceShrinkage(prices).ledoit_wolf()
    ef = EfficientFrontier(mu, S)
    risk_free_rate = -0.01  # -1%
    weights = ef.max_sharpe(risk_free_rate=risk_free_rate)
    cleaned_weights = ef.clean_weights()
    performance = ef.portfolio_performance(verbose=False, risk_free_rate=risk_free_rate)
    
    return cleaned_weights, performance, prices

# Streamlit app configuration
st.set_page_config(page_title="Advanced Digital Assets Portfolio Tool", layout="wide", initial_sidebar_state="expanded")
st.title("🌐 Advanced Digital Assets Portfolio Management")
st.markdown("Optimize, analyze, and stress-test your portfolio with blockchain integration.")

# Sidebar for stress testing and custom weights
with st.sidebar:
    st.header("🛠️ Portfolio Controls")
    st.subheader("Stress Testing Parameters")
    eth_change = st.slider("ETH % Change", -50.0, 50.0, 0.0, step=1.0)
    btc_change = st.slider("BTC % Change", -50.0, 50.0, 0.0, step=1.0)
    usdc_change = st.slider("USDC % Change", -50.0, 50.0, 0.0, step=1.0)
    
    st.subheader("Custom Portfolio Weights")
    eth_weight_custom = st.number_input("ETH Weight (%)", 0.0, 100.0, 33.33, step=1.0)
    btc_weight_custom = st.number_input("BTC Weight (%)", 0.0, 100.0, 33.33, step=1.0)
    usdc_weight_custom = st.number_input("USDC Weight (%)", 0.0, 100.0, 33.34, step=1.0)
    total = eth_weight_custom + btc_weight_custom + usdc_weight_custom
    if total != 100.0:
        st.error(f"Total weights must sum to 100%. Current total: {total:.2f}%")
    else:
        custom_weights = {
            'ETH': eth_weight_custom / 100,
            'BTC': btc_weight_custom / 100,
            'USDC': usdc_weight_custom / 100
        }

# Main layout with tabs
tab1, tab2, tab3, tab4 = st.tabs(["📊 Portfolio", "🔍 Compliance", "📈 Visualizations", "ℹ️ Asset Insights"])

with tab1:
    st.subheader("Current Portfolio Value")
    portfolio_value = get_portfolio_value()
    st.metric(label="Portfolio Value", value=f"{portfolio_value:.2f} USD", delta="Live")

    # Optimized Portfolio
    st.subheader("Optimized Portfolio")
    if st.button("Optimize Portfolio"):
        with st.spinner("Calculating optimal allocation..."):
            result = optimize_portfolio()
            if result[0] is not None:
                weights, (ret, vol, sharpe), prices = result
                st.session_state['opt_weights'] = weights
                st.session_state['prices'] = prices
                st.success("Optimal Portfolio Allocation:")
                for asset, weight in weights.items():
                    st.write(f"- {asset}: {weight*100:.2f}%")
                st.write(f"Expected Annual Return: {ret*100:.2f}%")
                st.write(f"Annual Volatility: {vol*100:.2f}%")
                st.write(f"Sharpe Ratio: {sharpe:.2f}")
            else:
                st.error("Optimization failed. Check the error messages above.")

    # Custom Portfolio
    st.subheader("Custom Portfolio")
    if total == 100.0 and st.button("Save Custom Portfolio"):
        st.session_state['custom_weights'] = custom_weights
        # Calculate custom portfolio value using live prices
        live_prices = get_live_prices()
        custom_value = (
            live_prices['ETH'] * custom_weights['ETH'] * 100000 +  # Assuming 100,000 USD initial investment
            live_prices['BTC'] * custom_weights['BTC'] * 100000 +
            live_prices['USDC'] * custom_weights['USDC'] * 100000
        )
        st.session_state['custom_value'] = custom_value
        st.success(f"Custom portfolio saved! Value: ${custom_value:.2f}")

with tab2:
    st.subheader("KYC/AML Compliance Check")
    user_address = st.text_input("Enter Wallet Address")
    if st.button("Check Compliance"):
        if user_address:
            with st.spinner("Checking compliance..."):
                status = check_compliance(user_address)
                if status is True:
                    st.success("✅ Compliant")
                elif status is False:
                    st.warning("⚠️ Not Compliant")
                else:
                    st.error("Unable to check compliance.")
        else:
            st.error("Enter an address first!")

with tab3:
    st.subheader("Portfolio Visualizations")
    if 'opt_weights' in st.session_state or 'custom_weights' in st.session_state:
        prices = st.session_state.get('prices', None)
        if prices is not None:
            # Optimal Portfolio Pie Chart
            if 'opt_weights' in st.session_state:
                opt_weights = st.session_state['opt_weights']
                fig_opt_pie = px.pie(
                    names=list(opt_weights.keys()),
                    values=[w * 100 for w in opt_weights.values()],
                    title="Optimized Portfolio Allocation",
                    color_discrete_sequence=px.colors.sequential.Viridis
                )
                fig_opt_pie.update_traces(textinfo='percent+label')
                st.plotly_chart(fig_opt_pie, use_container_width=True)

            # Custom Portfolio Pie Chart
            if 'custom_weights' in st.session_state:
                custom_weights = st.session_state['custom_weights']
                fig_custom_pie = px.pie(
                    names=list(custom_weights.keys()),
                    values=[w * 100 for w in custom_weights.values()],
                    title="Custom Portfolio Allocation",
                    color_discrete_sequence=px.colors.sequential.Plasma
                )
                fig_custom_pie.update_traces(textinfo='percent+label')
                st.plotly_chart(fig_custom_pie, use_container_width=True)

            # Historical Performance Comparison
            if 'opt_weights' in st.session_state and 'custom_weights' in st.session_state:
                live_prices = get_live_prices()
                initial_investment = 100000  # USD
                opt_units = {asset: weight * initial_investment / live_prices[asset] for asset, weight in opt_weights.items()}
                custom_units = {asset: weight * initial_investment / live_prices[asset] for asset, weight in custom_weights.items()}
                opt_history = pd.DataFrame(index=prices.index)
                custom_history = pd.DataFrame(index=prices.index)
                for asset in ['ETH', 'BTC', 'USDC']:
                    opt_history[asset] = prices[asset] * opt_units[asset]
                    custom_history[asset] = prices[asset] * custom_units[asset]
                opt_total = opt_history.sum(axis=1)
                custom_total = custom_history.sum(axis=1)
                df_comparison = pd.DataFrame({
                    'Optimized': opt_total,
                    'Custom': custom_total
                })
                fig_comparison = px.line(
                    df_comparison,
                    title="Optimized vs Custom Portfolio Performance",
                    labels={"value": "Portfolio Value (USD)"}
                )
                st.plotly_chart(fig_comparison, use_container_width=True)

    # Stress Test Section
    st.subheader("Stress Test Results")
    if st.button("Run Stress Test"):
        if portfolio_value != 0 and ('opt_weights' in st.session_state or 'custom_weights' in st.session_state):
            fig_stress = go.Figure()
            if 'opt_weights' in st.session_state:
                opt_weights = st.session_state['opt_weights']
                opt_adjusted = (
                    portfolio_value * (1 + eth_change / 100) * opt_weights['ETH'] +
                    portfolio_value * (1 + btc_change / 100) * opt_weights['BTC'] +
                    portfolio_value * (1 + usdc_change / 100) * opt_weights['USDC']
                )
                st.success(f"**Optimized Adjusted Value:** {opt_adjusted:.2f} USD")
                fig_stress.add_trace(go.Bar(
                    x=['Original', 'Stressed'],
                    y=[portfolio_value, opt_adjusted],
                    name='Optimized',
                    marker_color='#1f77b4'
                ))
            if 'custom_weights' in st.session_state:
                custom_weights = st.session_state['custom_weights']
                custom_value = st.session_state.get('custom_value', portfolio_value)
                custom_adjusted = (
                    custom_value * (1 + eth_change / 100) * custom_weights['ETH'] +
                    custom_value * (1 + btc_change / 100) * custom_weights['BTC'] +
                    custom_value * (1 + usdc_change / 100) * custom_weights['USDC']
                )
                st.success(f"**Custom Adjusted Value:** {custom_adjusted:.2f} USD")
                fig_stress.add_trace(go.Bar(
                    x=['Original', 'Stressed'],
                    y=[custom_value, custom_adjusted],
                    name='Custom',
                    marker_color='#ff7f0e'
                ))
            fig_stress.update_layout(
                title="Portfolio Value Under Stress",
                yaxis_title="Value (USD)",
                barmode='group'
            )
            st.plotly_chart(fig_stress, use_container_width=True)
        else:
            st.error("Run portfolio optimization or save custom weights first!")

with tab4:
    st.subheader("Asset Insights")
    
    # Live Prices Section with Refresh Button
    if 'live_prices_data' not in st.session_state or st.button("Refresh Live Prices"):
        prices = get_live_prices()
        st.session_state['live_prices_data'] = {
            'prices': prices,
            'timestamp': time.time()
        }
    
    if 'live_prices_data' in st.session_state:
        live_prices = st.session_state['live_prices_data']['prices']
        timestamp = st.session_state['live_prices_data']['timestamp']
        last_updated = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp))
        st.write(f"Last updated: {last_updated}")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("ETH Price", f"${live_prices['ETH']:.2f}")
        with col2:
            st.metric("BTC Price", f"${live_prices['BTC']:.2f}")
        with col3:
            st.metric("USDC Price", f"${live_prices['USDC']:.2f}")
    else:
        st.warning("Live prices not available. Click 'Refresh Live Prices' to fetch.")
    
    # Historical Trends
    assets = ['ethereum', 'bitcoin', 'usd-coin']
    prices_list = [get_historical_prices(asset) for asset in assets]
    if not any(df.empty for df in prices_list):
        start = max(df.index.min() for df in prices_list)
        end = min(df.index.max() for df in prices_list)
        common_index = pd.date_range(start=start, end=end, freq='H')
        prices_list_aligned = [df.reindex(common_index, method='ffill') for df in prices_list]
        prices = pd.concat(prices_list_aligned, axis=1)
        prices.columns = ['ETH', 'BTC', 'USDC']
        fig_trends = px.line(
            prices,
            title="Historical Price Trends",
            labels={"value": "Price (USD)"}
        )
        st.plotly_chart(fig_trends, use_container_width=True)

# Periodic refresh for portfolio value
def update_portfolio_value():
    portfolio_value = get_portfolio_value()
    with st.sidebar:
        st.metric(label="Live Portfolio Value", value=f"{portfolio_value:.2f} USD", delta="Refreshed Now")

if 'last_update' not in st.session_state or (time.time() - st.session_state['last_update']) > 60:
    st.session_state['last_update'] = time.time()
    update_portfolio_value()