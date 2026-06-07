"""
Market data fetcher for Binance Futures.
Fetches OHLCV candles, current price, and top gainers.
"""

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import time
import pandas as pd
from config import get_exchange


def fetch_ohlcv(symbol: str, timeframe: str = "5m", limit: int = 200, exchange=None) -> pd.DataFrame:
    """
    Fetch OHLCV candlestick data from Binance Futures.

    Args:
        symbol: Trading pair (e.g. "BTC/USDT")
        timeframe: Candle interval ("1m", "5m", "15m", "1h")
        limit: Number of candles to fetch (max 1500)
        exchange: ccxt exchange instance (creates one if None)

    Returns:
        DataFrame with columns: timestamp, open, high, low, close, volume
    """
    if exchange is None:
        exchange = get_exchange()

    max_retries = 3
    for attempt in range(max_retries):
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
            df["timestamp"] = pd.to_datetime(df["timestamp"].astype("int64"), unit="ms")
            df = df.astype({
                "open": float, "high": float, "low": float,
                "close": float, "volume": float,
            })
            return df
        except Exception as e:
            # Don't retry if the symbol doesn't exist
            if "does not have market symbol" in str(e):
                raise
            if attempt < max_retries - 1:
                wait = 2 ** attempt
                time.sleep(wait)
            else:
                print(f"[fetch_ohlcv] Failed after {max_retries} attempts: {e}")
                raise


def get_current_price(symbol: str, exchange=None) -> float:
    """Get the latest price for a symbol."""
    if exchange is None:
        exchange = get_exchange()
    try:
        ticker = exchange.fetch_ticker(symbol)
        price = ticker.get("last")
        if price is None:
            raise ValueError(f"{symbol} — no price data (coin may not be on Futures or Demo)")
        return float(price)
    except Exception as e:
        err = str(e)
        if "does not have market symbol" in err or "not found" in err.lower():
            print(f"[Price] {symbol} is not available on Binance Futures Demo.")
            print(f"[Price] It may exist on live Futures but not on the demo environment.")
        else:
            print(f"[Price] Error fetching {symbol}: {e}")
        raise


def get_top_gainers(limit: int = 10, exchange=None) -> pd.DataFrame:
    """
    Fetch top gainers from Binance Futures (USDT-M perpetual) by 24h % change.

    Returns:
        DataFrame with columns: symbol, price, change_pct, volume_24h
        Sorted by change_pct descending, top `limit` results.
    """
    if exchange is None:
        exchange = get_exchange()

    max_retries = 3
    for attempt in range(max_retries):
        try:
            tickers = exchange.fetch_tickers()
            rows = []
            for sym, t in tickers.items():
                # Only USDT perpetual pairs
                if not sym.endswith("/USDT") or ":" not in sym:
                    # On futures, symbols are like "BTC/USDT:USDT"
                    pass
                if t.get("percentage") is None:
                    continue
                rows.append({
                    "symbol": sym.replace(":USDT", ""),
                    "price": float(t.get("last", 0)),
                    "change_pct": float(t.get("percentage", 0)),
                    "volume_24h": float(t.get("quoteVolume", 0)),
                })

            df = pd.DataFrame(rows)
            if df.empty:
                return df

            # Filter to USDT pairs only
            df = df[df["symbol"].str.endswith("/USDT")]
            df = df.sort_values("change_pct", ascending=False).head(limit).reset_index(drop=True)
            return df

        except Exception as e:
            if attempt < max_retries - 1:
                wait = 2 ** attempt
                print(f"[get_top_gainers] Retry {attempt + 1}/{max_retries} in {wait}s: {e}")
                time.sleep(wait)
            else:
                print(f"[get_top_gainers] Failed after {max_retries} attempts: {e}")
                raise


if __name__ == "__main__":
    from tabulate import tabulate

    exchange = get_exchange()

    # Test current price
    price = get_current_price("BTC/USDT", exchange)
    print(f"BTC/USDT Price: ${price:,.2f}\n")

    # Test OHLCV
    df = fetch_ohlcv("BTC/USDT", "5m", 5, exchange)
    print("Last 5 candles (5m):")
    print(tabulate(df, headers="keys", tablefmt="simple", showindex=False))

    # Test top gainers
    print("\nTop 10 Gainers (24h):")
    gainers = get_top_gainers(10, exchange)
    if not gainers.empty:
        gainers["change_pct"] = gainers["change_pct"].apply(lambda x: f"{x:+.2f}%")
        gainers["volume_24h"] = gainers["volume_24h"].apply(lambda x: f"${x:,.0f}")
        print(tabulate(gainers, headers="keys", tablefmt="simple", showindex=False))
    else:
        print("No data returned.")
