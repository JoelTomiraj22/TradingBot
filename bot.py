"""
Main Bot Runner — Entry point for the Crypto Trading Bot.

Features:
- Startup: connects to Binance, prints balance, sets leverage
- Runs scanner every 15 minutes (configurable)
- Manual analysis mode
- Confirmation required before every trade (v1)
- Emergency close-all command
- Graceful shutdown with Ctrl+C
"""

import signal
import sys
import time
from datetime import datetime

from tabulate import tabulate

from config import (
    get_exchange, print_config, DEFAULT_PAIR, DEFAULT_LEVERAGE,
    SCAN_INTERVAL_MINUTES,
)
from fetch_data import get_current_price
from strategy import evaluate_signal, evaluate_with_mtf
from risk_manager import validate_trade
from order_executor import OrderExecutor
from scanner import scan_top_gainers, print_scan_results
from trade_tracker import log_trade, print_stats, print_recent_trades
from position_monitor import PositionMonitor
from ai_review import get_claude_analysis
from logger_setup import get_logger

logger = get_logger("bot")
executor = None
exchange = None
monitor = None
running = True
consecutive_losses = 0
CIRCUIT_BREAKER_LIMIT = 3  # Pause after this many consecutive losses


# ─── Colors ────────────────────────────────────────────────────

class C:
    """ANSI color codes for terminal output."""
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RESET = "\033[0m"

    @staticmethod
    def red(text): return f"\033[91m{text}\033[0m"
    @staticmethod
    def green(text): return f"\033[92m{text}\033[0m"
    @staticmethod
    def yellow(text): return f"\033[93m{text}\033[0m"
    @staticmethod
    def blue(text): return f"\033[94m{text}\033[0m"
    @staticmethod
    def cyan(text): return f"\033[96m{text}\033[0m"
    @staticmethod
    def bold(text): return f"\033[1m{text}\033[0m"
    @staticmethod
    def dim(text): return f"\033[2m{text}\033[0m"


# ─── Graceful shutdown ─────────────────────────────────────────

def shutdown_handler(signum, frame):
    global running
    print(f"\n\n{C.yellow('Shutting down gracefully...')}")
    logger.info("Shutdown signal received")
    running = False
    if monitor:
        monitor.stop()
    if executor:
        try:
            positions = executor.get_open_positions()
            if positions:
                print(f"{C.yellow(f'You have {len(positions)} open position(s).')}")
                resp = input("Cancel all open orders? (y/n): ").strip().lower()
                if resp == "y":
                    for p in positions:
                        executor.cancel_all_orders(p["symbol"])
                    logger.info("All open orders cancelled on shutdown")
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
    print(C.green("Bot stopped. Goodbye!"))
    sys.exit(0)


signal.signal(signal.SIGINT, shutdown_handler)
signal.signal(signal.SIGTERM, shutdown_handler)


# ─── Display helpers ───────────────────────────────────────────

def print_trade_table(signal_data: dict, validation: dict, symbol: str):
    """Print the trade confirmation table."""
    direction = signal_data["direction"]
    dir_color = C.green if direction == "LONG" else C.red

    print(f"\n{C.bold('=' * 55)}")
    print(f"  {C.bold('TRADE SIGNAL')} — {C.cyan(symbol)}")
    print(f"{C.bold('=' * 55)}")

    # Show pattern analysis FIRST
    print(f"\n  {C.bold(C.yellow('PATTERN ANALYSIS & CONFIRMATIONS:'))}")
    for i, r in enumerate(signal_data["reasons"], 1):
        if "WARNING" in r or "REJECTED" in r:
            print(f"    {C.red(f'[{i}]')} {C.red(r)}")
        else:
            print(f"    {C.green(f'[{i}]')} {r}")

    print(f"\n  {C.bold('TRADE DETAILS:')}")
    rows = [
        [C.bold("Direction"), dir_color(direction)],
        [C.bold("Confidence"), C.yellow(f"{signal_data['confidence']}/10")],
        ["Entry", f"${validation['entry']:,.6f}"],
        ["Stop Loss", C.red(f"${validation['stop_loss']:,.6f}")],
        ["Take Profit", C.green(f"${validation['take_profit']:,.6f}")],
        ["Breakeven", f"${validation['breakeven']:,.6f}"],
        ["Leverage", C.yellow(f"{validation['leverage']}x")],
        ["Position Size", f"${validation['position_size']:,.2f}"],
        ["Quantity", f"{validation['quantity']:.6f}"],
        ["Risk Amount", C.red(f"${validation['risk_amount']:,.2f}")],
        ["Expected Profit", C.green(f"${validation['expected_profit']:+,.2f}")],
        ["Expected Loss", C.red(f"${validation['expected_loss']:+,.2f}")],
        ["Fees (round-trip)", C.dim(f"${validation.get('fees', 0):,.2f}")],
        ["R:R (after fees)", C.cyan(f"{validation['rr_ratio']}:1")],
        ["Max Hold Time", C.dim(f"{validation.get('max_hold_hours', 4)}h")],
    ]
    print(tabulate(rows, tablefmt="simple"))
    print(C.bold('=' * 55))


def print_trade_result(result: dict, symbol: str, direction: str):
    """Print a clean trade execution result."""
    entry = result.get("entry", {})
    sl = result.get("stop_loss", {})
    tp = result.get("take_profit", {})

    has_entry = "error" not in entry
    has_sl = "error" not in sl
    has_tp = "error" not in tp

    print(f"\n{C.bold('─' * 55)}")
    print(f"  {C.bold('TRADE EXECUTION RESULT')} — {C.cyan(symbol)}")
    print(f"{C.bold('─' * 55)}")

    # Entry
    if has_entry:
        fill_price = entry.get("average") or entry.get("price") or 0
        order_id = entry.get("id", "N/A")
        qty = entry.get("filled") or entry.get("amount") or 0
        cost = entry.get("cost") or 0
        dir_color = C.green if direction == "LONG" else C.red
        print(f"  {C.green('ENTRY')}     {dir_color(direction)} filled")
        print(f"            Price:    ${float(fill_price):,.4f}")
        print(f"            Qty:      {qty}")
        print(f"            Cost:     ${float(cost):,.2f}" if cost else "            Cost:     N/A")
        print(f"            Order ID: {C.dim(str(order_id))}")
    else:
        print(f"  {C.red('ENTRY')}     {C.red('FAILED')}: {entry.get('error', 'Unknown')}")

    print()

    # Stop Loss
    if has_sl:
        sl_price = sl.get("triggerPrice") or sl.get("stopPrice") or sl.get("price", "N/A")
        print(f"  {C.green('STOP LOSS')} Set @ {C.red(f'${float(sl_price):,.2f}')}")
        print(f"            Order ID: {C.dim(str(sl.get('id', 'N/A')))}")
    else:
        print(f"  {C.red('STOP LOSS')} {C.red('FAILED')}: {C.red(sl.get('error', 'Unknown'))}")

    print()

    # Take Profit
    if has_tp:
        tp_price = tp.get("triggerPrice") or tp.get("stopPrice") or tp.get("price", "N/A")
        print(f"  {C.green('TAKE PROF')} Set @ {C.green(f'${float(tp_price):,.2f}')}")
        print(f"            Order ID: {C.dim(str(tp.get('id', 'N/A')))}")
    else:
        print(f"  {C.red('TAKE PROF')} {C.red('FAILED')}: {C.red(tp.get('error', 'Unknown'))}")

    print(f"{C.bold('─' * 55)}")

    # Summary line
    if has_entry and has_sl and has_tp:
        print(f"\n  {C.green(C.bold('ALL ORDERS PLACED SUCCESSFULLY'))}")
    elif has_entry:
        warnings = []
        if not has_sl:
            warnings.append("Stop Loss")
        if not has_tp:
            warnings.append("Take Profit")
        print(f"\n  {C.yellow(C.bold('WARNING'))}: Entry filled but {C.red(', '.join(warnings))} failed!")
        print(f"  {C.yellow('Monitor this position manually or close it.')}")
    else:
        print(f"\n  {C.red(C.bold('TRADE FAILED — No position opened.'))}")
    print()


def print_menu():
    """Print the bot menu."""
    print(f"\n{C.dim('─' * 40)}")
    print(f"  {C.bold('Commands:')}")
    print(f"  {C.cyan('allcoins')}  — Scan ALL coins, show safest trade")
    print(f"  {C.cyan('scan')}      — Run scanner (top gainers only)")
    print(f"  {C.cyan('analyze')}   — Analyze a specific coin")
    print(f"  {C.cyan('balance')}   — Check account balance")
    print(f"  {C.cyan('positions')} — View open positions")
    print(f"  {C.cyan('stats')}     — Trading statistics")
    print(f"  {C.cyan('trades')}    — Recent trade history")
    print(f"  {C.cyan('backtest')}  — Run backtest on a coin")
    print(f"  {C.cyan('close')}     — Close a specific position")
    print(f"  {C.cyan('monitored')} — View SL/TP monitored positions")
    print(f"  {C.red('closeall')}  — Emergency close ALL positions")
    print(f"  {C.dim('quit')}      — Shut down bot")
    print(f"  {C.dim('Or type a coin name: btc, eth, sol...')}")
    print(f"{C.dim('─' * 40)}")


# ─── Core functions ────────────────────────────────────────────

def analyze_coin(symbol: str):
    """Analyze a single coin and prompt for trade if signal found."""
    global exchange, executor

    global consecutive_losses

    # Circuit breaker
    if consecutive_losses >= CIRCUIT_BREAKER_LIMIT:
        print(f"\n{C.red(C.bold('CIRCUIT BREAKER ACTIVE'))}")
        print(f"  {C.red(f'{consecutive_losses} consecutive losses. Trading paused.')}")
        print(f"  {C.dim('Type reset to clear the circuit breaker and resume.')}")
        return

    print(f"\n{C.cyan(f'Analyzing {symbol}...')}")
    try:
        price = get_current_price(symbol, exchange)
        print(f"Current Price: {C.bold(f'${price:,.2f}')}")

        signal_data = evaluate_with_mtf(symbol, "5m", exchange)

        if signal_data["direction"] == "NO TRADE":
            print(f"Signal: {C.yellow('NO TRADE')}")
            for r in signal_data["reasons"]:
                print(f"  {C.dim('•')} {r}")
            return

        # Validate trade
        balance = executor.get_total_balance()
        validation = validate_trade(
            account_balance=balance,
            entry_price=signal_data["entry"],
            stop_loss=signal_data["stop_loss"],
            take_profit=signal_data["take_profit"],
            confidence=signal_data["confidence"],
            direction=signal_data["direction"],
        )

        if not validation["approved"]:
            print(f"{C.red('Trade REJECTED')}: {C.red(validation['reason'])}")
            return

        # Show trade table
        print_trade_table(signal_data, validation, symbol)

        # If strategy suggests waiting
        entry_type = signal_data.get("entry_type", "MARKET")
        wait_price = signal_data.get("wait_price")
        if entry_type == "WAIT" and wait_price:
            wp = f"${wait_price:,.4f}"
            print(f"\n  {C.yellow(C.bold('SUGGESTION: WAIT FOR BETTER ENTRY'))}")
            print(f"  {C.dim('Price is not at a key level. Consider limit order at')} {C.cyan(wp)}")

        # ─── STEP 1: Mandatory Capital & Leverage ──────────────
        print(f"\n{C.bold(C.cyan('STEP 1: Set your trade size'))}")
        cap_input = input(f"  Capital $ (required): ").strip()
        if not cap_input:
            print(C.yellow("No capital entered. Trade cancelled."))
            return
        try:
            user_capital = float(cap_input)
        except ValueError:
            print(C.red("Invalid amount. Trade cancelled."))
            return
        if user_capital <= 0:
            print(C.red("Capital must be > 0. Trade cancelled."))
            return

        lev_input = input(f"  Leverage (required, max 20): ").strip()
        if not lev_input:
            print(C.yellow("No leverage entered. Trade cancelled."))
            return
        try:
            user_leverage = int(lev_input)
        except ValueError:
            print(C.red("Invalid leverage. Trade cancelled."))
            return
        if user_leverage <= 0 or user_leverage > 20:
            print(C.red("Leverage must be 1-20. Trade cancelled."))
            return

        # Optional: custom SL/TP
        custom_sl = input(f"  Stop Loss (Enter to keep ${validation['stop_loss']:,.4f}): ").strip()
        custom_tp = input(f"  Take Profit (Enter to keep ${validation['take_profit']:,.4f}): ").strip()
        if custom_sl:
            validation["stop_loss"] = float(custom_sl)
        if custom_tp:
            validation["take_profit"] = float(custom_tp)

        # Recalculate with user's values
        position_size = user_capital * user_leverage
        quantity = position_size / validation["entry"]
        validation["leverage"] = user_leverage
        validation["quantity"] = quantity
        validation["position_size"] = round(position_size, 2)

        # Recalculate R:R
        sl_dist = abs(validation["entry"] - validation["stop_loss"])
        tp_dist = abs(validation["take_profit"] - validation["entry"])
        validation["rr_ratio"] = round(tp_dist / sl_dist, 1) if sl_dist > 0 else 0
        validation["fees"] = round(position_size * 0.0008, 2)

        pos_str = f"${position_size:,.2f}"
        margin_str = f"${user_capital:,.2f}"
        print(f"\n  {C.bold('Your setup:')} {margin_str} x {user_leverage}x = {C.cyan(pos_str)} position")

        # ─── STEP 2: Claude AI Review ──────────────────────────
        print(f"\n{C.bold(C.cyan('STEP 2: AI Second Opinion'))}")
        ai_result = get_claude_analysis(
            symbol=symbol,
            signal=signal_data,
            capital=user_capital,
            leverage=user_leverage,
            balance=balance,
        )

        if not ai_result["approved"]:
            print(f"\n  {C.red(C.bold('Claude says NO.'))}")
            override = input(f"  Override AI rejection? (yes/no): ").strip().lower()
            if override not in ("yes", "y"):
                print(C.yellow("Trade cancelled based on AI review."))
                logger.info(f"Trade cancelled by AI review: {symbol}")
                return
            print(f"  {C.yellow('Overriding AI — proceeding at your own risk.')}")

        # Apply AI adjustments if any
        if ai_result.get("adjustments"):
            adj = ai_result["adjustments"]
            if "stop_loss" in adj:
                ai_sl = adj["stop_loss"]
                cur_sl = validation["stop_loss"]
                print(f"  {C.yellow(f'AI suggests SL: ${ai_sl}')} (yours: ${cur_sl:,.4f})")
                use_ai_sl = input(f"  Use AI's SL? (yes/no): ").strip().lower()
                if use_ai_sl in ("yes", "y"):
                    validation["stop_loss"] = float(ai_sl)
            if "take_profit" in adj:
                ai_tp = adj["take_profit"]
                cur_tp = validation["take_profit"]
                print(f"  {C.yellow(f'AI suggests TP: ${ai_tp}')} (yours: ${cur_tp:,.4f})")
                use_ai_tp = input(f"  Use AI's TP? (yes/no): ").strip().lower()
                if use_ai_tp in ("yes", "y"):
                    validation["take_profit"] = float(ai_tp)
            if "leverage" in adj:
                ai_lev = int(adj["leverage"])
                print(f"  {C.yellow(f'AI suggests {ai_lev}x leverage')} (yours: {user_leverage}x)")
                use_ai_lev = input(f"  Use AI's leverage? (yes/no): ").strip().lower()
                if use_ai_lev in ("yes", "y"):
                    user_leverage = ai_lev
                    validation["leverage"] = ai_lev
                    position_size = user_capital * user_leverage
                    validation["quantity"] = position_size / validation["entry"]
                    validation["position_size"] = round(position_size, 2)

        # ─── STEP 3: Final Confirmation ────────────────────────
        print(f"\n{C.bold(C.cyan('STEP 3: Final Confirmation'))}")
        print_trade_table(signal_data, validation, symbol)
        confirm = input(f"\n{C.bold('EXECUTE NOW? (yes/no/wait):')} ").strip().lower()

        if confirm in ("wait", "w"):
            # Place limit order
            if wait_price:
                limit_price = wait_price
            else:
                lp_input = input(f"  Limit price: ").strip()
                if not lp_input:
                    print(C.yellow("Cancelled."))
                    return
                limit_price = float(lp_input)

            executor.set_leverage(symbol, user_leverage)
            side = "buy" if signal_data["direction"] == "LONG" else "sell"
            order = executor.place_limit_order(symbol, side, validation["quantity"], limit_price)
            if "error" in order:
                print(f"  {C.red('ERROR')}: {C.red(order['error'])}")
            else:
                lp_str = f"${limit_price:,.4f}"
                print(f"  {C.green(C.bold('LIMIT ORDER PLACED'))}")
                print(f"  Waiting for price to reach {C.cyan(lp_str)}")
            return

        if confirm in ("yes", "y"):
            logger.info(f"Trade confirmed: {signal_data['direction']} {symbol} | Capital: ${user_capital} | Leverage: {user_leverage}x")
            print(f"\n{C.yellow('Placing orders...')}")

            result = executor.execute_trade(
                symbol=symbol,
                direction=signal_data["direction"],
                quantity=validation["quantity"],
                leverage=validation["leverage"],
                stop_loss=validation["stop_loss"],
                take_profit=validation["take_profit"],
            )

            if "error" in result:
                logger.error(f"Trade execution failed: {result['error']}")
                print(f"\n{C.red(C.bold('ERROR'))}: {C.red(result['error'])}")
            else:
                logger.info(f"Trade executed on {symbol}")
                try:
                    print_trade_result(result, symbol, signal_data["direction"])
                except Exception as e:
                    print(f"{C.green('Trade placed!')} (display error: {e})")

                # Start position monitor with trailing SL
                if monitor:
                    try:
                        entry_price = float(result["entry"].get("average") or result["entry"].get("price") or validation["entry"])
                        filled_qty = float(result["entry"].get("filled") or result["entry"].get("amount") or validation["quantity"])
                    except (TypeError, ValueError):
                        entry_price = validation["entry"]
                        filled_qty = validation["quantity"]
                    monitor.add_position(
                        symbol=symbol,
                        direction=signal_data["direction"],
                        entry_price=entry_price,
                        stop_loss=validation["stop_loss"],
                        take_profit=validation["take_profit"],
                        quantity=filled_qty,
                        leverage=validation["leverage"],
                        confidence=signal_data["confidence"],
                    )
        else:
            print(C.yellow("Trade skipped."))
            logger.info(f"Trade skipped by user: {signal_data['direction']} {symbol}")

    except Exception as e:
        logger.error(f"Error analyzing {symbol}: {e}")
        print(f"{C.red(C.bold('ERROR'))}: {C.red(str(e))}")


def scan_all_coins():
    """
    Scan ALL major Binance Futures coins, run full pattern analysis
    with multi-timeframe on each, and show only the safest trades.
    """
    global exchange

    # Coins confirmed available on Binance Futures Demo
    ALL_COINS = [
        "BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT",
        "DOGE/USDT", "ADA/USDT", "AVAX/USDT", "LINK/USDT", "DOT/USDT",
        "UNI/USDT", "ATOM/USDT", "LTC/USDT", "FIL/USDT",
        "APT/USDT", "ARB/USDT", "OP/USDT", "SUI/USDT", "NEAR/USDT",
        "INJ/USDT", "SEI/USDT", "TIA/USDT", "FET/USDT",
        "WIF/USDT", "RENDER/USDT", "AAVE/USDT", "MKR/USDT", "ETC/USDT",
        "BCH/USDT", "ALGO/USDT", "SAND/USDT", "MANA/USDT", "CRV/USDT",
        "RUNE/USDT", "GALA/USDT", "IMX/USDT", "ENS/USDT", "DYDX/USDT",
    ]

    ts = datetime.now().strftime("%H:%M:%S")
    print(f"\n{C.bold(C.cyan(f'[{ts}] SCANNING ALL {len(ALL_COINS)} COINS...'))}")
    print(f"{C.dim('This may take 1-2 minutes — analyzing patterns + multi-timeframe on each coin')}\n")
    logger.info(f"allCoins scan started ({len(ALL_COINS)} coins)")

    results = []
    for i, symbol in enumerate(ALL_COINS):
        short_name = symbol.replace("/USDT", "")
        progress = f"[{i+1}/{len(ALL_COINS)}]"
        print(f"  {C.dim(progress)} Analyzing {short_name}...", end="", flush=True)

        try:
            signal = evaluate_with_mtf(symbol, "5m", exchange)

            if signal["direction"] != "NO TRADE" and signal["confidence"] >= 6:
                results.append({
                    "symbol": symbol,
                    "signal": signal,
                })
                dir_color = C.green if signal["direction"] == "LONG" else C.red
                print(f" {dir_color(signal['direction'])} (conf {signal['confidence']}/10)")
            else:
                print(f" {C.dim('no setup')}")
        except Exception as e:
            print(f" {C.red('error')}")
            logger.error(f"allCoins error on {symbol}: {e}")

    # Sort by confidence (highest first)
    results.sort(key=lambda x: x["signal"]["confidence"], reverse=True)

    print(f"\n{C.bold('=' * 65)}")
    print(f"  {C.bold(C.cyan('ALL COINS SCAN RESULTS'))}")
    print(f"{C.bold('=' * 65)}")

    if not results:
        print(f"\n  {C.yellow(C.bold('NO SAFE TRADES FOUND'))}")
        print(f"  {C.dim('All coins failed the quality gate or had low confidence.')}")
        print(f"  {C.dim('This is normal — a good strategy waits for the right setup.')}")
        print(f"{C.bold('=' * 65)}\n")
        logger.info("allCoins scan complete — no trades found")
        return

    # Show results table
    table_data = []
    for r in results:
        sig = r["signal"]
        sym = r["symbol"].replace("/USDT", "")
        dir_color = C.green if sig["direction"] == "LONG" else C.red
        htf_trend = sig.get("htf_trend", "?")
        htf_color = C.green if htf_trend == "BULLISH" else (C.red if htf_trend == "BEARISH" else C.dim)

        # Count quality indicators
        reasons = sig["reasons"]
        has_candle = "Y" if any(k in r2 for r2 in reasons for k in ["Engulfing", "Hammer", "Pin Bar", "Star"]) else "-"
        has_pullback = "Y" if any("Pullback" in r2 for r2 in reasons) else "-"

        table_data.append([
            sym,
            dir_color(sig["direction"]),
            C.yellow(f"{sig['confidence']}/10"),
            f"${sig['entry']:,.2f}",
            C.red(f"${sig['stop_loss']:,.2f}"),
            C.green(f"${sig['take_profit']:,.2f}"),
            htf_color(htf_trend),
            has_candle,
            has_pullback,
        ])

    print(tabulate(table_data, headers=[
        "Coin", "Signal", "Conf", "Entry", "SL", "TP", "HTF Trend", "Candle", "Pullback"
    ], tablefmt="simple"))

    print(f"\n  {C.green(f'Found {len(results)} trade setup(s)')}")
    print(f"{C.bold('=' * 65)}")

    # Show detailed analysis for the best trade
    best = results[0]
    best_sig = best["signal"]
    best_sym = best["symbol"]

    print(f"\n{C.bold(C.cyan(f'BEST SETUP: {best_sym}'))}")
    print(f"{C.dim('─' * 55)}")
    for i, r in enumerate(best_sig["reasons"], 1):
        if "WARNING" in r or "REJECTED" in r:
            print(f"  {C.red(f'[{i}]')} {C.red(r)}")
        else:
            print(f"  {C.green(f'[{i}]')} {r}")
    print(f"{C.dim('─' * 55)}")

    # Ask if user wants to trade the best one
    print(f"\n{C.dim('Type the coin name to analyze and trade it, or press Enter to skip.')}")


def run_scanner():
    """Run the scanner and process results."""
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"\n{C.cyan(f'[{ts}] Running scanner...')}")
    logger.info("Scanner started")

    balance = executor.get_total_balance()
    logger.info(f"Account balance: ${balance:.2f}")
    print(f"Account Balance: {C.bold(f'${balance:.2f}')}")

    results = scan_top_gainers(exchange=exchange)
    print_scan_results(results)

    # Process signals with confidence >= 5
    actionable = [r for r in results if r["confidence"] >= 5]
    if not actionable:
        print(C.yellow("No actionable signals found."))
        logger.info("Scanner complete — no actionable signals")
        return

    print(f"{C.green(f'Found {len(actionable)} actionable signal(s).')}\n")
    for r in actionable:
        symbol = r["coin"]
        dir_color = C.green if r["signal"] == "LONG" else C.red
        conf = r["confidence"]
        sig = r["signal"]
        print(f"\n{C.bold('---')} {C.cyan(symbol)} ({dir_color(sig)}, Confidence {C.yellow(f'{conf}/10')}) {C.bold('---')}")

        validation = validate_trade(
            account_balance=balance,
            entry_price=r["entry"],
            stop_loss=r["stop_loss"],
            take_profit=r["take_profit"],
            confidence=r["confidence"],
            direction=r["signal"],
        )

        if not validation["approved"]:
            print(f"  {C.red('REJECTED')}: {C.red(validation['reason'])}")
            continue

        # Build signal_data dict for display
        signal_data = {
            "direction": r["signal"],
            "confidence": r["confidence"],
            "reasons": r["reasons"],
        }
        print_trade_table(signal_data, validation, symbol)

        prompt_text = f"Execute {sig} on {symbol}? (yes/no/skip-all):"
        confirm = input(f"\n{C.bold(prompt_text)} ").strip().lower()

        if confirm in ("yes", "y"):
            print(f"\n{C.yellow('Placing orders...')}")
            result = executor.execute_trade(
                symbol=symbol,
                direction=r["signal"],
                quantity=validation["quantity"],
                leverage=validation["leverage"],
                stop_loss=validation["stop_loss"],
                take_profit=validation["take_profit"],
            )
            if "error" in result:
                print(f"\n{C.red(C.bold('ERROR'))}: {C.red(result['error'])}")
                logger.error(f"Trade failed on {symbol}: {result['error']}")
            else:
                try:
                    print_trade_result(result, symbol, sig)
                except Exception as e:
                    print(f"{C.green('Trade placed!')} (display error: {e})")
                logger.info(f"Trade executed on {symbol}")
                if monitor:
                    try:
                        entry_price = float(result["entry"].get("average") or result["entry"].get("price") or validation["entry"])
                        filled_qty = float(result["entry"].get("filled") or result["entry"].get("amount") or validation["quantity"])
                    except (TypeError, ValueError):
                        entry_price = validation["entry"]
                        filled_qty = validation["quantity"]
                    monitor.add_position(
                        symbol=symbol, direction=sig,
                        entry_price=entry_price,
                        stop_loss=validation["stop_loss"],
                        take_profit=validation["take_profit"],
                        quantity=filled_qty, leverage=validation["leverage"],
                        confidence=conf,
                    )
        elif confirm == "skip-all":
            print(C.yellow("Skipping remaining signals."))
            break
        else:
            print(C.yellow("Skipped."))

    logger.info("Scanner cycle complete")


# ─── Main loop ─────────────────────────────────────────────────

def main():
    global exchange, executor, monitor, running

    print(f"\n{C.BOLD}{C.CYAN}{'=' * 50}")
    print(f"  CRYPTO TRADING BOT v1.0")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'=' * 50}{C.RESET}\n")

    # Initialize
    print_config()
    print()

    exchange = get_exchange()
    executor = OrderExecutor(exchange)
    monitor = PositionMonitor(exchange, executor)

    # Check connection
    try:
        balance = executor.get_total_balance()
        print(f"{C.green('Connected!')} USDT Balance: {C.bold(f'${balance:.2f}')}")
        logger.info(f"Bot started. Balance: ${balance:.2f}")
    except Exception as e:
        print(f"{C.red('Connection failed')}: {C.red(str(e))}")
        logger.error(f"Connection failed: {e}")
        return

    # Set default leverage
    try:
        executor.set_leverage(DEFAULT_PAIR, DEFAULT_LEVERAGE)
    except Exception:
        pass

    print_menu()

    last_scan = 0
    scan_interval = SCAN_INTERVAL_MINUTES * 60

    while running:
        try:
            # Auto-scan check
            now = time.time()
            if now - last_scan >= scan_interval:
                if last_scan > 0:  # Skip auto-scan on first loop
                    run_scanner()
                last_scan = now

            # Wait for user input
            cmd = input(f"\n{C.BOLD}{C.CYAN}>{C.RESET} ").strip().lower()

            if cmd == "allcoins":
                scan_all_coins()

            elif cmd == "scan":
                run_scanner()
                last_scan = time.time()

            elif cmd == "analyze":
                coin = input(f"Enter coin (e.g. BTC/USDT): ").strip().upper()
                if "/" not in coin:
                    coin = coin + "/USDT"
                analyze_coin(coin)

            elif cmd == "balance":
                free = executor.get_balance()
                total = executor.get_total_balance()
                print(f"Available: {C.green(f'${free:.2f}')}  |  Total: {C.bold(f'${total:.2f}')}")

            elif cmd == "positions":
                positions = executor.get_open_positions()
                if positions:
                    rows = []
                    for p in positions:
                        side = p.get("side", "?").upper()
                        contracts = p.get("contracts", 0)
                        pnl = float(p.get("unrealizedPnl", 0))
                        entry_price = float(p.get("entryPrice", 0))
                        pnl_color = C.green if pnl >= 0 else C.red
                        side_color = C.green if side == "LONG" else C.red
                        rows.append([
                            p["symbol"].replace(":USDT", ""),
                            side_color(side),
                            contracts,
                            f"${entry_price:,.2f}",
                            pnl_color(f"${pnl:+.2f}"),
                        ])
                    print(f"\n{C.bold('Open Positions:')}")
                    print(tabulate(rows, headers=["Symbol", "Side", "Qty", "Entry", "PnL"], tablefmt="simple"))
                else:
                    print(C.dim("No open positions."))

            elif cmd == "stats":
                print_stats()

            elif cmd == "trades":
                print_recent_trades()

            elif cmd == "monitored":
                tracked = monitor.get_tracked() if monitor else {}
                if tracked:
                    print(f"\n{C.bold('Monitored Positions (Trailing SL/TP):')}")
                    rows = []
                    for sym, t in tracked.items():
                        dir_color = C.green if t["direction"] == "LONG" else C.red
                        stage_colors = {"INITIAL": C.dim, "BREAKEVEN": C.yellow, "LOCK PROFIT": C.cyan, "TRAILING": C.green}
                        stage_fn = stage_colors.get(t.get("sl_stage", "INITIAL"), C.dim)
                        rows.append([
                            sym.replace(":USDT", "").replace("/USDT", ""),
                            dir_color(t["direction"]),
                            f"${t['entry_price']:,.2f}",
                            C.red(f"${t['stop_loss']:,.2f}"),
                            C.green(f"${t['take_profit']:,.2f}"),
                            stage_fn(t.get("sl_stage", "INITIAL")),
                            t["opened_at"],
                        ])
                    print(tabulate(rows, headers=["Symbol", "Side", "Entry", "SL", "TP", "SL Stage", "Opened"], tablefmt="simple"))
                else:
                    print(C.dim("No positions being monitored."))

            elif cmd == "backtest":
                from backtest import backtest as run_backtest, print_backtest_results
                coin = input("Coin (e.g. BTC/USDT): ").strip().upper()
                if "/" not in coin:
                    coin = coin + "/USDT"
                days = input("Days of history (default 7): ").strip()
                days = int(days) if days else 7
                result = run_backtest(coin, "15m", days, exchange=exchange)
                print_backtest_results(result)

            elif cmd == "close":
                coin = input("Coin to close (e.g. BTC/USDT): ").strip().upper()
                if "/" not in coin:
                    coin = coin + "/USDT"
                confirm = input(f"{C.yellow(f'Close position on {coin}? (yes/no):')} ").strip().lower()
                if confirm in ("yes", "y"):
                    result = executor.close_position(coin)
                    if "error" in result:
                        print(f"{C.red('Error')}: {C.red(result['error'])}")
                    elif "info" in result:
                        print(C.yellow(result["info"]))
                    elif result.get("closed"):
                        pnl = result["pnl"]
                        pnl_color = C.green if pnl >= 0 else C.red
                        side_color = C.green if result["side"] == "long" else C.red
                        print(f"\n{C.bold('─' * 45)}")
                        print(f"  {C.bold('POSITION CLOSED')} — {C.cyan(coin)}")
                        print(f"{C.bold('─' * 45)}")
                        print(f"  Side:        {side_color(result['side'].upper())}")
                        print(f"  Entry:       ${result['entry_price']:,.2f}")
                        print(f"  Close:       ${result['close_price']:,.2f}")
                        print(f"  Quantity:    {result['quantity']}")
                        print(f"  P&L:         {pnl_color(f'${pnl:+.2f}')}")
                        print(f"  Result:      {C.green('WIN') if pnl >= 0 else C.red('LOSS')}")
                        print(f"{C.bold('─' * 45)}\n")

            elif cmd == "closeall":
                confirm = input(f"{C.red(C.bold('CLOSE ALL POSITIONS? (type confirm):'))} ").strip().lower()
                if confirm == "confirm":
                    results = executor.close_all_positions()
                    for r in results:
                        sym = r["symbol"].replace(":USDT", "")
                        if "error" in r["result"]:
                            print(f"  {C.red('FAIL')} {sym}: {r['result']['error']}")
                        else:
                            print(f"  {C.green('CLOSED')} {sym}")
                    logger.warning("Emergency close-all executed")
                else:
                    print(C.yellow("Cancelled."))

            elif cmd in ("quit", "exit", "q"):
                shutdown_handler(None, None)

            elif cmd == "reset":
                consecutive_losses = 0
                print(C.green("Circuit breaker reset. Trading resumed."))

            elif cmd == "help":
                print_menu()

            elif cmd == "":
                continue

            else:
                # Try as a coin name
                coin = cmd.upper()
                if "/" not in coin:
                    coin = coin + "/USDT"
                try:
                    analyze_coin(coin)
                except Exception:
                    print(f"{C.red('Unknown command')}: {cmd} — type {C.cyan('help')} for options")

        except EOFError:
            shutdown_handler(None, None)
        except KeyboardInterrupt:
            shutdown_handler(None, None)
        except Exception as e:
            logger.error(f"Main loop error: {e}")
            print(f"{C.red(C.bold('ERROR'))}: {C.red(str(e))}")


if __name__ == "__main__":
    main()
