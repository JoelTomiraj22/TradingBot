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


def _fetch_ticker_rows(exchange, max_retries: int = 3, label: str = "fetch_tickers") -> pd.DataFrame:
    """
    Fetch all USDT-M perpetual tickers and return a DataFrame with columns:
    symbol, price, change_pct, volume_24h. Used by get_top_gainers() and
    get_top_movers().
    """
    for attempt in range(max_retries):
        try:
            tickers = exchange.fetch_tickers()
            rows = []
            for sym, t in tickers.items():
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
            return df.reset_index(drop=True)

        except Exception as e:
            if attempt < max_retries - 1:
                wait = 2 ** attempt
                print(f"[{label}] Retry {attempt + 1}/{max_retries} in {wait}s: {e}")
                time.sleep(wait)
            else:
                print(f"[{label}] Failed after {max_retries} attempts: {e}")
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

    df = _fetch_ticker_rows(exchange, label="get_top_gainers")
    if df.empty:
        return df

    return df.sort_values("change_pct", ascending=False).head(limit).reset_index(drop=True)


def get_top_movers(limit_each: int = 10, min_volume: float = 5_000_000, exchange=None) -> pd.DataFrame:
    """
    Fetch the top gainers AND top losers (24h % change) from Binance Futures
    (USDT-M perpetual), filtered to a minimum 24h $ volume.

    Returns:
        DataFrame with columns: symbol, price, change_pct, volume_24h, mover_type
        ("GAINER" or "LOSER"), top `limit_each` of each, combined.
    """
    if exchange is None:
        exchange = get_exchange()

    df = _fetch_ticker_rows(exchange, label="get_top_movers")
    if df.empty:
        return df

    df = df[df["volume_24h"] >= min_volume]
    if df.empty:
        return df

    gainers = df.sort_values("change_pct", ascending=False).head(limit_each).copy()
    gainers["mover_type"] = "GAINER"

    losers = df.sort_values("change_pct", ascending=True).head(limit_each).copy()
    losers["mover_type"] = "LOSER"

    combined = pd.concat([gainers, losers], ignore_index=True)
    combined = combined.drop_duplicates(subset="symbol", keep="first")
    return combined.reset_index(drop=True)


def get_market_context(symbol: str, exchange=None) -> str:
    """
    Build a short market-microstructure context block for AI analysis:
    funding rate (crowded-side signal) and order-book imbalance/spread.
    Returns "" if nothing could be fetched — always safe to call.
    """
    if exchange is None:
        exchange = get_exchange()

    parts = []

    # Funding rate — positive = longs pay shorts (long-crowded), negative = short-crowded
    try:
        fr = exchange.fetch_funding_rate(symbol)
        rate = fr.get("fundingRate")
        if rate is not None:
            rate_pct = float(rate) * 100
            crowd = "longs pay shorts (long-crowded)" if rate_pct > 0 else "shorts pay longs (short-crowded)"
            parts.append(f"Funding rate: {rate_pct:+.4f}% per 8h — {crowd}")
    except Exception:
        pass

    # Order book imbalance + spread (top 20 levels)
    try:
        ob = exchange.fetch_order_book(symbol, limit=20)
        bids, asks = ob.get("bids") or [], ob.get("asks") or []
        if bids and asks:
            bid_vol = sum(level[1] for level in bids)
            ask_vol = sum(level[1] for level in asks)
            total = bid_vol + ask_vol
            if total > 0:
                bid_pct = bid_vol / total * 100
                mid = (bids[0][0] + asks[0][0]) / 2
                spread_pct = (asks[0][0] - bids[0][0]) / mid * 100 if mid > 0 else 0
                parts.append(
                    f"Order book (top 20 levels): {bid_pct:.0f}% bid / {100 - bid_pct:.0f}% ask"
                    f" | Spread: {spread_pct:.4f}%"
                )
    except Exception:
        pass

    if not parts:
        return ""

    return (
        "MARKET MICROSTRUCTURE (live):\n  "
        + "\n  ".join(parts)
        + "\nUse as secondary context: heavy one-sided funding often precedes squeezes "
        "against the crowded side; strong book imbalance supports the heavier side short-term."
    )


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
