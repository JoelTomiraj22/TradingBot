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
# 5 min (was 10): on a 5-30 min scalp, a 10-min cooldown blocked every
# justified SL tighten for most of the trade's life.
REANALYSIS_COOLDOWN_MINUTES = 5    # Min time between applied SL/TP changes per position
MIN_LEVEL_CHANGE_PCT = 0.10        # Ignore SL/TP "moves" smaller than this % of price

# ─── Automatic AI re-analysis of open positions ─────────────────
# Every N minutes the monitor re-checks each open trade with fresh
# multi-timeframe data + AI. Tighten-only SL moves are auto-applied;
# CLOSE_NOW verdicts only ALERT (never auto-close). 0 = disabled.
AUTO_REANALYZE_MINUTES = 2

# ─── Scanner liquidity filter ───────────────────────────────────
MIN_DOLLAR_VOLUME = 50_000     # Min avg $ volume per 5m candle (~ $14M/day) to consider a coin

# ─── Scanner AI batch size cap ──────────────────────────────────
# Caps how many pre-filtered coins get a full multi-TF fetch + sent to the
# AI in one scan prompt, ranked by liquidity. Keeps the scan prompt (and
# exchange API load) bounded regardless of how many coins pass pre-filter.
MAX_SCAN_CANDIDATES = 12

# ─── allCoins scan: dynamic universe + multi-TF pre-filter ──────
# Top N gainers + top N losers (24h) merged into the static ALL_COINS
# watchlist for scan_all_coins(), each filtered to this min 24h $ volume
# (matches scanner.py's top-gainers threshold).
TOP_MOVERS_LIMIT = 10
TOP_MOVERS_MIN_VOLUME = 5_000_000

# Thread pool size for concurrent multi-TF pre-filter fetches in
# scan_all_coins(). Each worker uses its own exchange instance.
PREFILTER_WORKERS = 6

# Min weighted-timeframe agreement fraction (0-1) for a coin's dominant
# trend to qualify it as a scan candidate.
TREND_ALIGNMENT_MIN = 0.5

# ─── No-chase policy (hard) ─────────────────────────────────────
# If market price has drifted more than this % from the AI entry,
# market execution is DISABLED — the only option is a limit order at
# the AI entry level. We never chase; price comes to us.
MAX_CHASE_PCT = 0.15

# ─── Volatility-spike guard ─────────────────────────────────────
VOLATILITY_SPIKE_ATR_MULT = 2.5   # Last candle range > this x ATR = volatility spike

# ─── Mandatory SL/TP fallback ───────────────────────────────────
# If a trade's SL/TP ever come back inverted/missing relative to its
# direction (e.g. a broken AI verdict), the position monitor substitutes
# these default % distances from entry so every position is ALWAYS
# protected by a valid stop-loss and take-profit.
DEFAULT_SL_PCT = 1.0     # fallback stop-loss distance from entry, %
DEFAULT_TP_PCT = 2.2     # fallback take-profit distance from entry, % (matches MIN_RR_RATIO)

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
