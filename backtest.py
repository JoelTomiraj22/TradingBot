"""
Strategy backtester.
Runs the strategy on historical data candle-by-candle, simulating trades.
"""

import pandas as pd
from tabulate import tabulate
from fetch_data import fetch_ohlcv
from indicators import add_all_indicators
from strategy import evaluate_signal, check_exit
from risk_manager import get_leverage_for_confidence, calculate_breakeven
from config import CAPITAL_PER_TRADE, get_exchange


def backtest(symbol: str, timeframe: str = "5m", days: int = 7,
             capital_per_trade: float = None, exchange=None) -> dict:
    """
    Backtest the strategy on historical data.

    Args:
        symbol: Trading pair
        timeframe: Candle timeframe
        days: Number of days of history
        capital_per_trade: Capital per trade (default from config)
        exchange: ccxt exchange instance

    Returns:
        dict with trades list, stats, and summary
    """
    if capital_per_trade is None:
        capital_per_trade = CAPITAL_PER_TRADE
    if exchange is None:
        exchange = get_exchange()

    # Calculate how many candles we need
    tf_minutes = {"1m": 1, "5m": 5, "15m": 15, "1h": 60}
    minutes = tf_minutes.get(timeframe, 5)
    candles_needed = min((days * 24 * 60) // minutes, 1500)

    print(f"Fetching {candles_needed} candles of {symbol} ({timeframe})...")
    df = fetch_ohlcv(symbol, timeframe, candles_needed, exchange)
    df = add_all_indicators(df)

    trades = []
    open_trade = None
    lookback = 200  # Minimum candles for indicator warmup

    for i in range(lookback, len(df)):
        window = df.iloc[:i + 1].copy()
        row = df.iloc[i]

        if open_trade is None:
            # Look for entry
            signal = evaluate_signal(window)
            if signal["direction"] in ("LONG", "SHORT") and signal["confidence"] >= 5:
                leverage = get_leverage_for_confidence(signal["confidence"])
                open_trade = {
                    "entry_idx": i,
                    "entry_time": row["timestamp"],
                    "coin": symbol,
                    "direction": signal["direction"],
                    "entry": signal["entry"],
                    "stop_loss": signal["stop_loss"],
                    "take_profit": signal["take_profit"],
                    "leverage": leverage,
                    "confidence": signal["confidence"],
                    "capital": capital_per_trade,
                    "breakeven": calculate_breakeven(signal["entry"], signal["direction"]),
                }
        else:
            # Check for exit
            exit_check = check_exit(
                window,
                open_trade["direction"],
                open_trade["entry"],
                open_trade["stop_loss"],
                open_trade["take_profit"],
            )

            if exit_check["should_exit"]:
                exit_price = exit_check["current_price"]
                direction = open_trade["direction"]
                entry = open_trade["entry"]
                lev = open_trade["leverage"]

                if direction == "LONG":
                    pnl_pct = ((exit_price - entry) / entry) * lev * 100
                else:
                    pnl_pct = ((entry - exit_price) / entry) * lev * 100

                pnl_dollar = capital_per_trade * (pnl_pct / 100)

                trades.append({
                    "entry_time": str(open_trade["entry_time"]),
                    "exit_time": str(row["timestamp"]),
                    "coin": symbol,
                    "direction": direction,
                    "entry": round(entry, 6),
                    "exit": round(exit_price, 6),
                    "sl": round(open_trade["stop_loss"], 6),
                    "tp": round(open_trade["take_profit"], 6),
                    "leverage": lev,
                    "confidence": open_trade["confidence"],
                    "pnl_pct": round(pnl_pct, 2),
                    "pnl_dollar": round(pnl_dollar, 2),
                    "result": "WIN" if pnl_dollar > 0 else "LOSS",
                    "exit_reason": exit_check["reason"],
                    "candles_held": i - open_trade["entry_idx"],
                })
                open_trade = None

    # Calculate stats
    stats = _calculate_stats(trades, capital_per_trade)
    return {"trades": trades, "stats": stats}


def _calculate_stats(trades: list, capital: float) -> dict:
    """Calculate backtest statistics."""
    if not trades:
        return {"total_trades": 0, "message": "No trades triggered"}

    total = len(trades)
    wins = sum(1 for t in trades if t["result"] == "WIN")
    losses = total - wins
    win_rate = (wins / total * 100) if total > 0 else 0

    pnl_list = [t["pnl_dollar"] for t in trades]
    total_pnl = sum(pnl_list)
    avg_pnl = total_pnl / total

    # Max drawdown
    cumulative = []
    running = 0
    peak = 0
    max_dd = 0
    for pnl in pnl_list:
        running += pnl
        cumulative.append(running)
        if running > peak:
            peak = running
        dd = peak - running
        if dd > max_dd:
            max_dd = dd

    # Average R:R
    win_pnls = [t["pnl_dollar"] for t in trades if t["result"] == "WIN"]
    loss_pnls = [abs(t["pnl_dollar"]) for t in trades if t["result"] == "LOSS"]
    avg_win = sum(win_pnls) / len(win_pnls) if win_pnls else 0
    avg_loss = sum(loss_pnls) / len(loss_pnls) if loss_pnls else 0
    avg_rr = avg_win / avg_loss if avg_loss > 0 else 0

    total_return = (total_pnl / capital) * 100

    return {
        "total_trades": total,
        "wins": wins,
        "losses": losses,
        "win_rate": round(win_rate, 1),
        "total_pnl": round(total_pnl, 2),
        "avg_pnl": round(avg_pnl, 2),
        "best_trade": round(max(pnl_list), 2),
        "worst_trade": round(min(pnl_list), 2),
        "max_drawdown": round(max_dd, 2),
        "avg_rr": round(avg_rr, 2),
        "total_return_pct": round(total_return, 1),
    }


def print_backtest_results(result: dict):
    """Print backtest results as tables."""
    stats = result["stats"]
    trades = result["trades"]

    print(f"\n{'='*60}")
    print(f"  BACKTEST RESULTS")
    print(f"{'='*60}")

    if stats["total_trades"] == 0:
        print("  No trades triggered during this period.")
        return

    print(f"  Total Trades:    {stats['total_trades']}")
    print(f"  Win Rate:        {stats['win_rate']}%  ({stats['wins']}W / {stats['losses']}L)")
    print(f"  Total P&L:       ${stats['total_pnl']:+.2f}")
    print(f"  Avg P&L/Trade:   ${stats['avg_pnl']:+.2f}")
    print(f"  Best Trade:      ${stats['best_trade']:+.2f}")
    print(f"  Worst Trade:     ${stats['worst_trade']:+.2f}")
    print(f"  Max Drawdown:    ${stats['max_drawdown']:.2f}")
    print(f"  Avg R:R:         {stats['avg_rr']}")
    print(f"  Total Return:    {stats['total_return_pct']:+.1f}%")
    print(f"{'='*60}")

    # Trade table
    if trades:
        table_data = []
        for t in trades:
            table_data.append({
                "Time": t["entry_time"][:16],
                "Dir": t["direction"],
                "Entry": t["entry"],
                "Exit": t["exit"],
                "Lev": f"{t['leverage']}x",
                "P&L$": f"${t['pnl_dollar']:+.2f}",
                "P&L%": f"{t['pnl_pct']:+.1f}%",
                "Result": t["result"],
                "Reason": t["exit_reason"],
                "Candles": t["candles_held"],
            })
        print(f"\nTrade Log:")
        print(tabulate(table_data, headers="keys", tablefmt="simple"))
    print()


if __name__ == "__main__":
    result = backtest("BTC/USDT", "15m", days=7)
    print_backtest_results(result)
