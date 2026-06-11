"""
Market scanner.
Scans top Binance Futures gainers, filters, and ranks by strategy confidence.
"""

from tabulate import tabulate
from fetch_data import fetch_ohlcv, get_top_gainers
from indicators import (add_all_indicators, add_higher_tf_indicators,
                        estimate_eta_minutes, check_sl_hunt_risk)
from strategy import evaluate_signal
from config import get_exchange, DEFAULT_TIMEFRAME

# Entry TF -> confirmation TF for multi-timeframe alignment
HTF_MAP = {"1m": "5m", "5m": "15m", "15m": "1h", "1h": "4h"}
TF_MINUTES = {"1m": 1, "3m": 3, "5m": 5, "15m": 15, "30m": 30, "1h": 60}


def scan_top_gainers(timeframe: str = None, limit: int = 10, exchange=None) -> list:
    """
    Scan top gainers for trade opportunities.

    1. Fetch top gainers
    2. Filter by volume (>$5M) and extension (<50%)
    3. Run strategy on each
    4. Return sorted by confidence

    Returns:
        List of dicts with signal info, sorted by confidence descending.
    """
    if exchange is None:
        exchange = get_exchange()
    if timeframe is None:
        timeframe = DEFAULT_TIMEFRAME

    # Fetch top gainers
    gainers = get_top_gainers(limit=20, exchange=exchange)
    if gainers.empty:
        print("[Scanner] No gainers data returned.")
        return []

    # Filters
    gainers = gainers[gainers["volume_24h"] >= 5_000_000]  # Min $5M volume
    gainers = gainers[gainers["change_pct"] <= 50]          # Not too extended

    if gainers.empty:
        print("[Scanner] All coins filtered out.")
        return []

    results = []
    htf = HTF_MAP.get(timeframe, "1h")
    for _, row in gainers.iterrows():
        symbol = row["symbol"]
        try:
            df = fetch_ohlcv(symbol, timeframe, 200, exchange)
            df = add_all_indicators(df)

            # Higher-timeframe confirmation (same boost/penalty as manual analysis)
            try:
                df_htf = fetch_ohlcv(symbol, htf, 100, exchange)
                htf_data = add_higher_tf_indicators(df_htf)
            except Exception:
                htf_data = {"htf_trend": "NEUTRAL", "htf_valid": False}

            signal = evaluate_signal(df, htf_data)

            # Time-to-TP estimate + stop-hunt check for actionable signals
            eta_range = None
            sl_hunt = None
            if signal["direction"] in ("LONG", "SHORT") and signal["entry"] and signal["take_profit"]:
                eta_range = estimate_eta_minutes(
                    df, signal["entry"], signal["take_profit"],
                    TF_MINUTES.get(timeframe, 5),
                )
                sl_hunt = check_sl_hunt_risk(
                    df, signal["direction"], signal["stop_loss"], signal["entry"],
                )

            results.append({
                "coin": symbol,
                "price": row["price"],
                "change_24h": f"{row['change_pct']:+.2f}%",
                "volume_24h": row["volume_24h"],
                "signal": signal["direction"],
                "confidence": signal["confidence"],
                "entry": signal["entry"],
                "stop_loss": signal["stop_loss"],
                "take_profit": signal["take_profit"],
                "leverage": signal["leverage"],
                "rr_ratio": None,
                "eta_range": eta_range,
                "sl_hunt": sl_hunt,
                "reasons": signal["reasons"],
            })

            # Calculate R:R if we have entry and SL
            if signal["entry"] and signal["stop_loss"] and signal["take_profit"]:
                sl_dist = abs(signal["entry"] - signal["stop_loss"])
                tp_dist = abs(signal["take_profit"] - signal["entry"])
                if sl_dist > 0:
                    results[-1]["rr_ratio"] = round(tp_dist / sl_dist, 1)

        except Exception as e:
            print(f"[Scanner] Error analyzing {symbol}: {e}")

    # Sort by confidence descending
    results.sort(key=lambda x: x["confidence"], reverse=True)
    return results


def print_scan_results(results: list):
    """Print scan results as a clean table."""
    if not results:
        print("\nNo trade opportunities found.\n")
        return

    table_data = []
    for r in results:
        table_data.append({
            "Coin": r["coin"].replace("/USDT", ""),
            "Price": f"${r['price']:,.4f}" if r["price"] < 1 else f"${r['price']:,.2f}",
            "24h%": r["change_24h"],
            "Vol($M)": f"{r['volume_24h'] / 1_000_000:.1f}",
            "Signal": r["signal"],
            "Conf": f"{r['confidence']}/10",
            "Entry": f"${r['entry']:,.2f}" if r["entry"] else "-",
            "SL": f"${r['stop_loss']:,.2f}" if r["stop_loss"] else "-",
            "TP": f"${r['take_profit']:,.2f}" if r["take_profit"] else "-",
            "R:R": f"{r['rr_ratio']}:1" if r["rr_ratio"] else "-",
            "ETA": f"{r['eta_range'][0]}-{r['eta_range'][1]}m" if r.get("eta_range") else "-",
            "Lev": f"{r['leverage']}x" if r["leverage"] else "-",
        })

    print(f"\n{'='*100}")
    print(f"  SCANNER RESULTS — Top Opportunities")
    print(f"{'='*100}")
    print(tabulate(table_data, headers="keys", tablefmt="simple"))
    print()


if __name__ == "__main__":
    print("Scanning Binance Futures for opportunities...\n")
    results = scan_top_gainers()
    print_scan_results(results)

    # Show details for top signal
    if results and results[0]["confidence"] >= 5:
        top = results[0]
        print(f"Top Signal: {top['coin']} — {top['signal']} (Confidence {top['confidence']}/10)")
        print(f"Reasons:")
        for r in top["reasons"]:
            print(f"  - {r}")
