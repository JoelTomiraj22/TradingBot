"""
Configuration module for the Crypto Trading Bot.
Loads API keys from .env, connects to Binance Futures (demo or live).
"""

import os
import ccxt
from dotenv import load_dotenv

load_dotenv()

# ─── Testnet Toggle ─────────────────────────────────────────────
USE_TESTNET = os.getenv("USE_TESTNET", "True").lower() in ("true", "1", "yes")

# ─── API Keys ───────────────────────────────────────────────────
if USE_TESTNET:
    API_KEY = os.getenv("BINANCE_TESTNET_API_KEY", "")
    API_SECRET = os.getenv("BINANCE_TESTNET_API_SECRET", "")
else:
    API_KEY = os.getenv("BINANCE_API_KEY", "")
    API_SECRET = os.getenv("BINANCE_API_SECRET", "")

# ─── Default Trading Parameters ────────────────────────────────
DEFAULT_PAIR = "BTC/USDT"
DEFAULT_TIMEFRAME = "5m"
DEFAULT_LEVERAGE = 10
RISK_PER_TRADE = 0.02          # 2% of account balance
MIN_RR_RATIO = 2.2             # Minimum reward:risk ratio (after fees + slippage)
CAPITAL_PER_TRADE = 10.0       # $8–$10 per trade
SCAN_INTERVAL_MINUTES = 15     # Scanner runs every 15 min

# ─── Re-analysis whipsaw guard ──────────────────────────────────
REANALYSIS_COOLDOWN_MINUTES = 10   # Min time between applied SL/TP changes per position
MIN_LEVEL_CHANGE_PCT = 0.10        # Ignore SL/TP "moves" smaller than this % of price

# ─── Scanner liquidity filter ───────────────────────────────────
MIN_DOLLAR_VOLUME = 50_000     # Min avg $ volume per 5m candle (~ $14M/day) to consider a coin

# ─── Scanner AI batch size cap ──────────────────────────────────
# Caps how many pre-filtered coins get a full multi-TF fetch + sent to the
# AI in one scan prompt, ranked by liquidity. Keeps the scan prompt (and
# exchange API load) bounded regardless of how many coins pass pre-filter.
MAX_SCAN_CANDIDATES = 12

# ─── Volatility-spike guard ─────────────────────────────────────
VOLATILITY_SPIKE_ATR_MULT = 2.5   # Last candle range > this x ATR = volatility spike

# ─── Exchange Connection ────────────────────────────────────────
def get_exchange():
    """Create and return a ccxt Binance USD-M Futures exchange instance."""
    exchange = ccxt.binanceusdm({
        "apiKey": API_KEY,
        "secret": API_SECRET,
        "enableRateLimit": True,
        "options": {
            "adjustForTimeDifference": True,
            "fetchCurrencies": False,
        },
    })

    if USE_TESTNET:
        # Redirect all fapi endpoints to Binance Demo
        demo = "https://demo-fapi.binance.com"
        for key in list(exchange.urls["api"].keys()):
            url = exchange.urls["api"][key]
            if "fapi.binance.com" in url:
                exchange.urls["api"][key] = url.replace(
                    "https://fapi.binance.com", demo
                )
            elif "api.binance.com" in url:
                # Neuter spot/sapi endpoints so they don't get called
                exchange.urls["api"][key] = url.replace(
                    "https://api.binance.com", demo
                )

    return exchange


def print_config():
    """Print current config (never prints keys)."""
    mode = "TESTNET (Demo)" if USE_TESTNET else "LIVE"
    key_preview = API_KEY[:6] + "..." if len(API_KEY) > 6 else "(not set)"
    print(f"Mode:            {mode}")
    print(f"API Key:         {key_preview}")
    print(f"Default Pair:    {DEFAULT_PAIR}")
    print(f"Timeframe:       {DEFAULT_TIMEFRAME}")
    print(f"Leverage:        {DEFAULT_LEVERAGE}x")
    print(f"Risk/Trade:      {RISK_PER_TRADE * 100}%")
    print(f"Min R:R:         {MIN_RR_RATIO}")
    print(f"Capital/Trade:   ${CAPITAL_PER_TRADE}")


if __name__ == "__main__":
    print_config()
    exchange = get_exchange()
    print(f"\nExchange:        {exchange.id}")
    print(f"Demo Mode:       {USE_TESTNET}")
    try:
        ticker = exchange.fetch_ticker("BTC/USDT:USDT")
        print(f"BTC/USDT Price:  ${ticker['last']:,.2f}")
    except Exception as e:
        print(f"Ticker test:     {e}")
    try:
        balance = exchange.fetch_balance()
        usdt = balance.get("USDT", {})
        print(f"USDT Balance:    {usdt.get('total', 'N/A')}")
    except Exception as e:
        print(f"Balance test:    {e}")
