"""
Trade tracker.
Saves trades to CSV and calculates running statistics.
"""

import os
import csv
from datetime import datetime
import pandas as pd
from tabulate import tabulate

TRADES_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trades.csv")

COLUMNS = [
    "date", "coin", "direction", "entry", "exit", "sl", "tp",
    "leverage", "capital", "pnl_dollar", "pnl_pct", "result",
    "pattern", "confidence", "notes",
]


def _ensure_file():
    """Create trades.csv with headers if it doesn't exist."""
    if not os.path.exists(TRADES_FILE):
        with open(TRADES_FILE, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(COLUMNS)


def log_trade(coin: str, direction: str, entry: float, exit_price: float,
              sl: float, tp: float, leverage: int, capital: float,
              confidence: int, pattern: str = "", notes: str = "") -> dict:
    """
    Save a completed trade to trades.csv.

    Returns:
        dict with pnl_dollar, pnl_pct, result
    """
    _ensure_file()

    if direction == "LONG":
        pnl_pct = ((exit_price - entry) / entry) * leverage * 100
    else:
        pnl_pct = ((entry - exit_price) / entry) * leverage * 100

    pnl_dollar = capital * (pnl_pct / 100)
    result = "WIN" if pnl_dollar > 0 else "LOSS"

    row = [
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        coin, direction,
        round(entry, 6), round(exit_price, 6),
        round(sl, 6), round(tp, 6),
        leverage, round(capital, 2),
        round(pnl_dollar, 2), round(pnl_pct, 2),
        result, pattern, confidence, notes,
    ]

    with open(TRADES_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)

    return {
        "pnl_dollar": round(pnl_dollar, 2),
        "pnl_pct": round(pnl_pct, 2),
        "result": result,
    }


def get_trades() -> pd.DataFrame:
    """Load all trades from CSV."""
    _ensure_file()
    try:
        df = pd.read_csv(TRADES_FILE)
        return df
    except Exception:
        return pd.DataFrame(columns=COLUMNS)


def get_stats() -> dict:
    """Calculate running statistics from trade history."""
    df = get_trades()
    if df.empty:
        return {"total_trades": 0, "message": "No trades recorded yet."}

    total = len(df)
    wins = len(df[df["result"] == "WIN"])
    losses = len(df[df["result"] == "LOSS"])
    win_rate = (wins / total * 100) if total > 0 else 0

    total_pnl = df["pnl_dollar"].sum()
    avg_pnl = df["pnl_dollar"].mean()
    best = df["pnl_dollar"].max()
    worst = df["pnl_dollar"].min()

    avg_rr = 0
    if losses > 0 and wins > 0:
        avg_win = df[df["result"] == "WIN"]["pnl_dollar"].mean()
        avg_loss = abs(df[df["result"] == "LOSS"]["pnl_dollar"].mean())
        avg_rr = avg_win / avg_loss if avg_loss > 0 else 0

    return {
        "total_trades": total,
        "wins": wins,
        "losses": losses,
        "win_rate": round(win_rate, 1),
        "total_pnl": round(total_pnl, 2),
        "avg_pnl": round(avg_pnl, 2),
        "best_trade": round(best, 2),
        "worst_trade": round(worst, 2),
        "avg_rr": round(avg_rr, 2),
    }


def print_stats():
    """Print a summary of trading stats."""
    stats = get_stats()
    if stats["total_trades"] == 0:
        print("No trades recorded yet.")
        return

    print(f"\n{'='*40}")
    print(f"  TRADING STATISTICS")
    print(f"{'='*40}")
    print(f"  Total Trades:  {stats['total_trades']}")
    print(f"  Wins:          {stats['wins']}")
    print(f"  Losses:        {stats['losses']}")
    print(f"  Win Rate:      {stats['win_rate']}%")
    print(f"  Total P&L:     ${stats['total_pnl']:+.2f}")
    print(f"  Avg P&L:       ${stats['avg_pnl']:+.2f}")
    print(f"  Best Trade:    ${stats['best_trade']:+.2f}")
    print(f"  Worst Trade:   ${stats['worst_trade']:+.2f}")
    print(f"  Avg R:R:       {stats['avg_rr']}")
    print()


def print_recent_trades(n: int = 10):
    """Print last N trades in table format."""
    df = get_trades()
    if df.empty:
        print("No trades recorded yet.")
        return

    recent = df.tail(n).copy()
    display_cols = ["date", "coin", "direction", "entry", "exit",
                    "leverage", "pnl_dollar", "pnl_pct", "result", "confidence"]
    display = recent[display_cols]
    print(f"\nLast {min(n, len(display))} Trades:")
    print(tabulate(display, headers="keys", tablefmt="simple", showindex=False))
    print()


if __name__ == "__main__":
    print_stats()
    print_recent_trades()
