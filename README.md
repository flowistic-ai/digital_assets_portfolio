# Digital Assets Portfolio Manager

A sophisticated web application for managing and optimizing digital asset portfolios with blockchain integration. Built with Streamlit and Web3, this tool provides real-time portfolio analytics, optimization strategies, and blockchain compliance checking.

## Features

- **Real-time Portfolio Tracking**: Monitor your digital asset portfolio value in real-time
- **Portfolio Optimization**: Utilize modern portfolio theory to optimize asset allocation
- **Blockchain Integration**: Direct integration with Sepolia testnet for token management
- **Market Data**: Live price feeds from Live Coin Watch API
- **Advanced Analytics**: 
  - Historical price analysis
  - Portfolio performance metrics
  - Risk assessment
  - Stress testing
- **Compliance Checking**: Whitelist verification for addresses
- **Interactive Visualizations**: Dynamic charts and graphs for portfolio analysis

## Prerequisites

- Python 3.11 or higher
- Infura API key (for Ethereum network access)
- Live Coin Watch API key

## Installation

### Using uv (Recommended)

1. Install uv:
```bash
pip install uv
```

2. Clone the repository:
```bash
git clone [repository-url]
cd digital-assets-portfolio
```

3. Set up your environment variables in `.env`:
```
INFURA_URL=your_infura_url
LIVE_COIN_WATCH_API_KEY=your_api_key
```

4. Create and activate a virtual environment with uv:
```bash
uv venv
.venv/Scripts/activate  # On Windows
source .venv/bin/activate  # On Unix/MacOS
```

5. Install dependencies with uv:
```bash
uv pip install -e .
```

### Using pip (Alternative)

1. Clone the repository:
```bash
git clone [repository-url]
cd digital-assets-portfolio
```

2. Set up your environment variables in `.env`:
```
INFURA_URL=your_infura_url
LIVE_COIN_WATCH_API_KEY=your_api_key
```

3. Install dependencies:
```bash
pip install .
```

## Usage

Run the Streamlit application:
```bash
streamlit run app.py
```

The application will be available at `http://localhost:8501`

## Dependencies

- `pandas`: Data manipulation and analysis
- `plotly`: Interactive visualizations
- `pyportfolioopt`: Portfolio optimization
- `python-dotenv`: Environment variable management
- `requests`: HTTP requests for API integration
- `scikit-learn`: Machine learning utilities
- `streamlit`: Web application framework
- `web3`: Ethereum blockchain interaction

## Smart Contract Integration

The application integrates with a custom ERC20 token contract deployed on the Sepolia testnet at:
`0x46711F6E9Ab3a50fdCf95D0528bfCa814b5276e4`

Features include:
- Token balance checking
- Whitelist management
- Transfer functionality
- Portfolio value calculation

## License

[Your chosen license]

## Contributing

[Your contribution guidelines]