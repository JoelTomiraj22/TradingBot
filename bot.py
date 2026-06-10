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
    SCAN_INTERVAL_MINUTES, CAPITAL_PER_TRADE,
    REANALYSIS_COOLDOWN_MINUTES, MIN_LEVEL_CHANGE_PCT,
    MIN_DOLLAR_VOLUME, VOLATILITY_SPIKE_ATR_MULT, MAX_SCAN_CANDIDATES,
)
from fetch_data import fetch_ohlcv, get_current_price
from strategy import evaluate_signal
from risk_manager import validate_trade, calculate_take_profit, MIN_RR_BY_TYPE
from config import MIN_RR_RATIO
from order_executor import OrderExecutor
from scanner import scan_top_gainers, print_scan_results
from trade_tracker import log_trade, print_stats, print_recent_trades
from position_monitor import PositionMonitor
from multi_ai_verifier import analyze_coin_ai, scan_coins_ai, build_indicator_snapshot, reanalyze_position_ai, build_btc_context
from indicators import add_all_indicators, check_liquidity, check_volatility_spike
from logger_setup import get_logger

logger = get_logger("bot")
executor = None
exchange = None
monitor = None
running = True
consecutive_losses = 0
daily_pnl = 0.0                # Track daily realized P&L
session_start_balance = 0.0    # Balance at bot start (for daily loss limit)
CIRCUIT_BREAKER_LIMIT = 3      # Pause after this many consecutive losses
MAX_DAILY_LOSS_PCT = 0.05      # 5% max daily drawdown — stop trading


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

def print_trade_table(signal_data: dict, validation: dict, symbol: str, current_price: float = 0):
    """Print the trade confirmation table."""
    direction = signal_data["direction"]
    dir_color = C.green if direction == "LONG" else C.red
    entry = validation.get("entry", 0)

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

    # Chase alert — warn if market has moved away from AI entry
    chase_line = None
    if current_price and entry:
        drift_pct = abs(current_price - entry) / entry * 100
        if drift_pct >= 0.3:
            label = "above" if current_price > entry else "below"
            chase_line = (f"\n  {C.red(C.bold('⚠  DO NOT CHASE'))}"
                          f" — market is {drift_pct:.2f}% {label} AI entry"
                          f" | AI entry: ${entry:,.6f} | Now: ${current_price:,.6f}")

    print(f"\n  {C.bold('TRADE DETAILS:')}")
    price_display = C.cyan(f"${current_price:,.6f}") if current_price else C.dim("N/A")
    rows = [
        [C.bold("Direction"), dir_color(direction)],
        [C.bold("Confidence"), C.yellow(f"{signal_data['confidence']}/10")],
        [C.bold("Market Price"), price_display],
        ["AI Entry", f"${entry:,.6f}"],
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
    if chase_line:
        print(chase_line)
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
        failed = []
        if not has_sl:
            failed.append("SL")
        if not has_tp:
            failed.append("TP")
        failed_str = " & ".join(failed)
        print(f"\n  {C.yellow(C.bold(f'Exchange {failed_str} orders failed — software trailing stop is active.'))}")
        print(f"  {C.dim('Position is being monitored automatically. No action needed.')}")
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
    print(f"  {C.cyan('reanalyse')} — Re-analyze an active trade, AI may suggest moving SL/TP")
    print(f"  {C.red('closeall')}  — Emergency close ALL positions")
    print(f"  {C.dim('quit')}      — Shut down bot")
    print(f"  {C.dim('Or type a coin name: btc, eth, sol...')}")
    print(f"{C.dim('─' * 40)}")


# ─── Core functions ────────────────────────────────────────────

def analyze_coin(symbol: str):
    """
    AI-driven analysis of a single coin:
    1. Bot fetches OHLCV data + calculates indicators
    2. AI analyzes ALL data — decides direction, confidence, SL/TP
    3. Risk manager validates
    4. User enters capital + leverage
    5. User confirms → Execute
    """
    global exchange, executor, consecutive_losses, daily_pnl, session_start_balance

    # Daily loss limit
    if session_start_balance > 0:
        daily_loss_pct = abs(min(daily_pnl, 0)) / session_start_balance
        if daily_loss_pct >= MAX_DAILY_LOSS_PCT:
            print(f"\n{C.red(C.bold('DAILY LOSS LIMIT REACHED'))}")
            print(f"  {C.red(f'Daily P&L: ${daily_pnl:+.2f} ({daily_loss_pct*100:.1f}% drawdown)')}")
            print(f"  {C.dim('Trading paused for today. Type reset to override.')}")
            return

    # Circuit breaker
    if consecutive_losses >= CIRCUIT_BREAKER_LIMIT:
        print(f"\n{C.red(C.bold('CIRCUIT BREAKER ACTIVE'))}")
        print(f"  {C.red(f'{consecutive_losses} consecutive losses. Trading paused.')}")
        print(f"  {C.dim('Type reset to clear the circuit breaker and resume.')}")
        return

    print(f"\n{C.cyan(f'Fetching multi-timeframe data for {symbol}...')}")
    try:
        price = get_current_price(symbol, exchange)
        print(f"Current Price: {C.bold(f'${price:,.4f}')}")

        # ─── Tell the AI what you have and what you want ───────
        balance = executor.get_total_balance()
        print(f"\n{C.bold(C.cyan('Trade preferences'))} {C.dim(f'(Balance: ${balance:,.2f})')}")

        cap_input = input(f"  Capital $ to use (Enter for ${CAPITAL_PER_TRADE:,.2f}): ").strip()
        if cap_input:
            try:
                user_capital = float(cap_input)
                if user_capital <= 0:
                    raise ValueError
            except ValueError:
                print(C.red("Invalid amount."))
                return
        else:
            user_capital = CAPITAL_PER_TRADE

        mode_input = input(f"  Mode — scalp/intraday/swing (Enter = AI picks best): ").strip().upper()
        mode_pref = mode_input if mode_input in ("SCALP", "INTRADAY", "SWING") else "ANY"

        lev_input = input(f"  Preferred leverage 1-25 (Enter = AI decides): ").strip()
        if lev_input:
            try:
                lev_pref = int(lev_input)
                if not (1 <= lev_pref <= 25):
                    raise ValueError
            except ValueError:
                print(C.red("Leverage must be 1-25."))
                return
        else:
            lev_pref = None

        profit_target = input(f"  Profit target $ or % (Enter = AI decides): ").strip()
        max_loss = input(f"  Max loss you can take $ or % (Enter = standard risk mgmt): ").strip()

        user_prefs = {
            "capital": user_capital,
            "mode": mode_pref,
            "leverage": lev_pref,
            "profit_target": profit_target or None,
            "max_loss": max_loss or None,
        }

        # Fetch ALL timeframes and compute indicators
        from multi_ai_verifier import ALL_TIMEFRAMES
        tf_data = {}
        for tf in ALL_TIMEFRAMES:
            try:
                limit = 200 if tf in ["1m", "3m", "5m"] else 100
                df_tf = fetch_ohlcv(symbol, tf, limit, exchange)
                df_tf = add_all_indicators(df_tf)
                tf_data[tf] = df_tf
                print(f"  {C.dim(f'✓ {tf} — {len(df_tf)} candles')}")
            except Exception as e:
                print(f"  {C.dim(f'✗ {tf} — {e}')}")

        if not tf_data:
            print(f"{C.red('Failed to fetch any timeframe data.')}")
            return

        # ─── BTC market regime context (skip if analyzing BTC itself) ──
        btc_context = None
        if symbol != "BTC/USDT":
            try:
                btc_tf_data = {}
                for tf in ["1h", "1d"]:
                    df_btc = fetch_ohlcv("BTC/USDT", tf, 100, exchange)
                    btc_tf_data[tf] = add_all_indicators(df_btc)
                btc_context = build_btc_context(btc_tf_data)
            except Exception as e:
                print(f"  {C.dim(f'(BTC regime context unavailable: {e})')}")

        # ─── AI ANALYSIS — AI makes ALL trade decisions ────────
        ai_signal = analyze_coin_ai(symbol, tf_data, user_prefs=user_prefs, btc_context=btc_context)

        direction = ai_signal.get("direction", "NO_TRADE")
        confidence = ai_signal.get("confidence", 0)

        if direction == "WAIT":
            wait_dir = ai_signal.get("wait_direction", "?")
            wait_cond = ai_signal.get("wait_condition", "")
            entry = ai_signal.get("entry")
            sl = ai_signal.get("stop_loss")
            tp = ai_signal.get("take_profit")
            ai_lev = ai_signal.get("leverage", 5)
            dir_color = C.green if wait_dir == "LONG" else C.red
            print(f"Signal: {C.yellow('WAITING FOR SETUP')}")
            print(f"  {C.bold('Bias:')} {dir_color(wait_dir)} | Confidence: {confidence}/10")
            if wait_cond:
                print(f"  {C.bold('Trigger:')} {wait_cond}")
            if entry and sl and tp:
                sl_d = abs(entry - sl) / entry * 100
                tp_d = abs(tp - entry) / entry * 100
                rr = tp_d / sl_d if sl_d > 0 else 0
                print(f"  {C.bold('Target entry:')} ${entry:,.6f} | SL: ${sl:,.6f} ({sl_d:.2f}%) | TP: ${tp:,.6f} ({tp_d:.2f}%) | R:R: {rr:.2f}:1 | Lev: {ai_lev}x")
            for r in ai_signal.get("reasons", []):
                print(f"  {C.dim('•')} {r}")
            if ai_signal.get("advice"):
                print(f"  {C.cyan('Advice:')} {ai_signal['advice']}")

            # Offer to place a limit order at the AI's target entry
            if entry and sl and tp and monitor:
                user_leverage = lev_pref if lev_pref else min(ai_lev, 25)
                print(f"\n  {C.cyan(C.bold('Place limit order at AI target entry?'))}")
                print(f"  {C.dim('When filled, SL/TP + trailing stop activate automatically.')}")
                print(f"  {C.dim(f'Capital: ${user_capital:,.2f} | Leverage: {user_leverage}x')}")
                confirm = input(f"  Place limit order? (yes/no): ").strip().lower()
                if confirm not in ("yes", "y"):
                    return

                position_size = user_capital * user_leverage
                quantity = position_size / entry

                executor.set_leverage(symbol, user_leverage)
                order_side = "buy" if wait_dir == "LONG" else "sell"
                order = executor.place_limit_order(symbol, order_side, quantity, entry)
                if "error" in order:
                    print(f"  {C.red('Limit order failed')}: {order['error']}")
                    return

                monitor.add_pending_order(
                    symbol=symbol,
                    order_id=str(order["id"]),
                    direction=wait_dir,
                    stop_loss=sl,
                    take_profit=tp,
                    quantity=quantity,
                    leverage=user_leverage,
                    confidence=confidence,
                    limit_price=entry,
                )
                logger.info(f"Limit order placed for WAIT signal: {wait_dir} {symbol} @ ${entry} | Order {order['id']}")
            return

        if direction == "NO_TRADE" or confidence < 7:
            print(f"Signal: {C.yellow('NO TRADE')}")
            if confidence > 0:
                print(f"  {C.dim(f'Confidence: {confidence}/10 (need 7+)')}")
            for r in ai_signal.get("reasons", []):
                print(f"  {C.dim('•')} {r}")
            return

        # Show AI's chosen timeframe and trade type
        trade_type = ai_signal.get("trade_type", "SCALP")
        ai_timeframe = ai_signal.get("timeframe", "5m")
        hold_time = ai_signal.get("hold_time", "")
        print(f"  {C.cyan(f'Trade type: {trade_type} on {ai_timeframe}')}")
        if hold_time:
            print(f"  {C.dim(f'Expected hold: {hold_time}')}")

        # Volatility-spike guard — recent candle range vs ATR (proxy for news/liquidation event)
        spike_df = tf_data.get(ai_timeframe)
        if spike_df is None:
            spike_df = tf_data.get("5m")
        vol_spike = check_volatility_spike(spike_df, VOLATILITY_SPIKE_ATR_MULT)
        if vol_spike["spike"]:
            ratio = vol_spike["ratio"]
            print(f"\n  {C.red(C.bold(f'⚠  VOLATILITY SPIKE — last {ai_timeframe} candle is {ratio}x ATR'))}")
            print(f"  {C.dim('Possible news/liquidation event. SL/TP may be unreliable.')}")
            spike_confirm = input(f"\n{C.bold('Continue anyway? (yes/no):')} ").strip().lower()
            if spike_confirm not in ("yes", "y"):
                print(C.yellow("Trade cancelled."))
                return

        entry = ai_signal.get("entry") or price
        sl = ai_signal.get("stop_loss")
        tp = ai_signal.get("take_profit")

        if not sl or not tp:
            print(f"{C.red('AI did not provide SL/TP — cannot trade.')}")
            return

        # Validate with risk manager
        balance = executor.get_total_balance()
        validation = validate_trade(
            account_balance=balance,
            entry_price=entry,
            stop_loss=sl,
            take_profit=tp,
            confidence=confidence,
            direction=direction,
            trade_type=trade_type,
        )

        if not validation["approved"]:
            print(f"{C.red('Trade REJECTED by risk manager')}: {C.red(validation['reason'])}")
            # Show what TP would be needed to meet minimum R:R
            if "R:R" in validation["reason"] and sl:
                required_tp = calculate_take_profit(entry, sl, direction, MIN_RR_BY_TYPE.get(trade_type, MIN_RR_RATIO))
                sl_dist = abs(entry - sl)
                tp_dist = abs(required_tp - entry)
                needed_rr = tp_dist / sl_dist if sl_dist > 0 else 0
                tp_pct = tp_dist / entry * 100
                print(f"  {C.yellow('To pass:')} TP must be ${required_tp:,.6f} ({tp_pct:.2f}% from entry) for {needed_rr:.1f}:1 R:R")
                print(f"  {C.dim('AI set TP at')} ${tp:,.6f} — {C.dim('too close to entry. Wait for a deeper target level.')}")

                override = input(f"\n  {C.yellow('Override and proceed with the AI TP anyway? (yes/no):')} ").strip().lower()
                if override not in ("yes", "y"):
                    print(C.yellow("Trade cancelled."))
                    return

                validation = validate_trade(
                    account_balance=balance,
                    entry_price=entry,
                    stop_loss=sl,
                    take_profit=tp,
                    confidence=confidence,
                    direction=direction,
                    trade_type=trade_type,
                    allow_low_rr=True,
                )
                if not validation["approved"]:
                    print(f"{C.red('Trade REJECTED by risk manager')}: {C.red(validation['reason'])}")
                    return
                rr = validation["rr_ratio"]
                print(f"  {C.yellow(C.bold(f'Manual override — proceeding at {rr:.2f}:1 R:R'))}")
            else:
                return

        # ─── Correlation check — warn on stacking same-direction risk ──
        if monitor:
            same_direction = [
                s for s, t in monitor.get_tracked().items()
                if t.get("direction") == direction and s != symbol
            ]
            if len(same_direction) >= 2:
                names = ", ".join(s.replace("/USDT", "") for s in same_direction)
                print(f"\n  {C.yellow(C.bold('CORRELATION WARNING'))}")
                print(f"  {C.yellow(f'Already {len(same_direction)} other {direction} positions open: {names}')}")
                print(f"  {C.dim('Altcoins tend to move together with BTC — stacking same-direction')}")
                print(f"  {C.dim('positions multiplies your exposure to a single market move.')}")

        # ─── STEP 1: Capital & Leverage ────────────────────────
        ai_lev = ai_signal.get("leverage", 5)
        user_leverage = lev_pref if lev_pref else min(ai_lev, 25)
        print(f"\n{C.bold(C.cyan('Your setup'))}")
        print(f"  {C.dim(f'Capital: ${user_capital:,.2f} | Leverage: {user_leverage}x (AI suggested {ai_lev}x)')}")

        # Recalculate with user's values
        position_size = user_capital * user_leverage
        quantity = position_size / entry
        validation["leverage"] = user_leverage
        validation["quantity"] = quantity
        validation["position_size"] = round(position_size, 2)

        sl_dist = abs(entry - sl)
        tp_dist = abs(tp - entry)
        validation["rr_ratio"] = round(tp_dist / sl_dist, 1) if sl_dist > 0 else 0
        validation["fees"] = round(position_size * 0.0018, 2)

        pos_str = f"${position_size:,.2f}"
        margin_str = f"${user_capital:,.2f}"
        print(f"\n  {C.bold('Your setup:')} {margin_str} x {user_leverage}x = {C.cyan(pos_str)} position")

        # Optional: custom SL/TP override
        custom_sl = input(f"  Custom Stop Loss? (Enter to keep ${sl:,.6f}): ").strip()
        custom_tp = input(f"  Custom Take Profit? (Enter to keep ${tp:,.6f}): ").strip()
        if custom_sl:
            sl = float(custom_sl)
            validation["stop_loss"] = sl
        if custom_tp:
            tp = float(custom_tp)
            validation["take_profit"] = tp

        if custom_sl or custom_tp:
            sl_dist = abs(entry - sl)
            tp_dist = abs(tp - entry)
            validation["rr_ratio"] = round(tp_dist / sl_dist, 1) if sl_dist > 0 else 0

        # Build signal_data for display
        signal_data = {
            "direction": direction,
            "confidence": confidence,
            "reasons": ai_signal.get("reasons", []),
        }

        # ─── Show final trade table & confirm ──────────────────
        # Refresh price just before confirm to catch any drift since analysis
        try:
            price = get_current_price(symbol, exchange)
        except Exception:
            pass
        print_trade_table(signal_data, validation, symbol, current_price=price)

        # Anti-chase guard — warn hard if market has run away from AI entry
        drift_pct = abs(price - entry) / entry * 100 if entry else 0
        if drift_pct >= 0.5:
            chase_dir = "above" if price > entry else "below"
            print(f"\n  {C.red(C.bold('DO NOT CHASE — price moved {:.2f}% {} since AI entry'.format(drift_pct, chase_dir)))}")
            print(f"  {C.dim('Use wait to set a limit order at the original entry instead.')}")
            confirm = input(f"\n{C.bold('Override chase & execute anyway? (yes to force / wait for limit / no):')} ").strip().lower()
            if confirm not in ("yes", "y", "wait", "w"):
                print(C.yellow("Smart call. Trade cancelled."))
                return
        else:
            confirm = input(f"\n{C.bold('EXECUTE NOW? (yes/no/wait):')} ").strip().lower()

        if confirm in ("wait", "w"):
            ai_entry = ai_signal.get("entry")
            if ai_entry and abs(ai_entry - price) > 0.001:
                default_limit = ai_entry
                print(f"  {C.dim(f'AI suggested entry: ${default_limit:,.6f}')}")
            else:
                default_limit = None

            if default_limit:
                lp_input = input(f"  Limit price (Enter for ${default_limit:,.6f}): ").strip()
                limit_price = float(lp_input) if lp_input else default_limit
            else:
                lp_input = input(f"  Limit price: ").strip()
                if not lp_input:
                    print(C.yellow("Cancelled."))
                    return
                limit_price = float(lp_input)

            executor.set_leverage(symbol, user_leverage)
            side = "buy" if direction == "LONG" else "sell"
            order = executor.place_limit_order(symbol, side, quantity, limit_price)
            if "error" in order:
                print(f"  {C.red('ERROR')}: {C.red(order['error'])}")
            else:
                lp_str = f"${limit_price:,.6f}"
                print(f"  {C.green(C.bold('LIMIT ORDER PLACED'))}")
                print(f"  Waiting for price to reach {C.cyan(lp_str)}")
            return

        if confirm in ("yes", "y"):
            logger.info(f"Trade confirmed: {direction} {symbol} | Capital: ${user_capital} | Leverage: {user_leverage}x")
            print(f"\n{C.yellow('Placing orders...')}")

            result = executor.execute_trade(
                symbol=symbol,
                direction=direction,
                quantity=quantity,
                leverage=user_leverage,
                stop_loss=sl,
                take_profit=tp,
            )

            if "error" in result:
                logger.error(f"Trade execution failed: {result['error']}")
                print(f"\n{C.red(C.bold('ERROR'))}: {C.red(result['error'])}")
            else:
                logger.info(f"Trade executed on {symbol}")
                try:
                    print_trade_result(result, symbol, direction)
                except Exception as e:
                    print(f"{C.green('Trade placed!')} (display error: {e})")

                # Start position monitor with trailing SL
                if monitor:
                    try:
                        entry_price = float(result["entry"].get("average") or result["entry"].get("price") or entry)
                        filled_qty = float(result["entry"].get("filled") or result["entry"].get("amount") or quantity)
                    except (TypeError, ValueError):
                        entry_price = entry
                        filled_qty = quantity

                    sl_result = result.get("stop_loss", {})
                    tp_result = result.get("take_profit", {})
                    sl_order_id = str(sl_result["id"]) if "id" in sl_result and "error" not in sl_result else None
                    tp_order_id = str(tp_result["id"]) if "id" in tp_result and "error" not in tp_result else None

                    # Set max hold based on AI trade type
                    hold_map = {"SCALP": 1, "INTRADAY": 4, "SWING": 24}
                    max_hold_h = hold_map.get(trade_type, 4)

                    monitor.add_position(
                        symbol=symbol,
                        direction=direction,
                        entry_price=entry_price,
                        stop_loss=sl,
                        take_profit=tp,
                        quantity=filled_qty,
                        leverage=user_leverage,
                        confidence=confidence,
                        sl_order_id=sl_order_id,
                        tp_order_id=tp_order_id,
                    )
                    # Update max hold time based on trade type
                    if symbol in monitor.tracked:
                        monitor.tracked[symbol]["max_hold_seconds"] = max_hold_h * 3600
        else:
            print(C.yellow("Trade skipped."))
            logger.info(f"Trade skipped by user: {direction} {symbol}")

    except Exception as e:
        logger.error(f"Error analyzing {symbol}: {e}")
        print(f"{C.red(C.bold('ERROR'))}: {C.red(str(e))}")


def reanalyze_position(symbol: str):
    """
    Re-analyze an ALREADY-OPEN, monitored position with fresh multi-timeframe
    data and the trade's current state. AI suggests HOLD / MOVE_SL / MOVE_TP /
    MOVE_BOTH / CLOSE_NOW. SL/TP changes and closes require user confirmation
    before being applied to the live trade.
    """
    global exchange, executor, monitor

    if not monitor:
        print(C.red("Position monitor not running."))
        return

    trade = monitor.get_tracked().get(symbol)

    if not trade:
        # Not tracked in-memory (e.g. bot was restarted) — check if it's an
        # open position on the exchange and offer to start monitoring it.
        positions = executor.get_open_positions()
        exch_pos = next(
            (p for p in positions if p.get("symbol", "").split(":")[0] == symbol
             and float(p.get("contracts", 0)) > 0),
            None,
        )
        if not exch_pos:
            print(f"{C.yellow(f'{symbol} is not an active position.')}")
            print(f"  {C.dim('Use')} {C.cyan('monitored')} {C.dim('or')} {C.cyan('positions')} {C.dim('to see active positions.')}")
            return

        direction = "LONG" if exch_pos.get("side") == "long" else "SHORT"
        entry_price = float(exch_pos.get("entryPrice", 0))
        quantity = float(exch_pos.get("contracts", 0))
        leverage = int(float(exch_pos.get("leverage", 1) or 1))

        print(f"\n{C.yellow(f'{symbol} is open on the exchange but not being monitored')} {C.dim('(bot restarted?)')}")
        print(f"  {C.dim(f'{direction} | Entry: ${entry_price:,.6f} | Qty: {quantity} | Leverage: {leverage}x')}")
        print(f"  {C.dim('Enter the current SL/TP so the AI can re-analyze and live monitoring can resume.')}")

        sl_input = input(f"  Current Stop Loss price: ").strip()
        tp_input = input(f"  Current Take Profit price: ").strip()
        try:
            stop_loss = float(sl_input)
            take_profit = float(tp_input)
        except ValueError:
            print(C.red("Invalid SL/TP. Cancelled."))
            return

        if not monitor.add_position(symbol, direction, entry_price, stop_loss, take_profit, quantity, leverage):
            print(C.red("Could not start monitoring this position."))
            return

        trade = monitor.get_tracked().get(symbol)
        if not trade:
            print(C.red("Could not start monitoring this position."))
            return

    print(f"\n{C.cyan(f'Fetching fresh multi-timeframe data for {symbol}...')}")
    try:
        price = get_current_price(symbol, exchange)
        print(f"Current Price: {C.bold(f'${price:,.4f}')}")

        from multi_ai_verifier import ALL_TIMEFRAMES
        tf_data = {}
        for tf in ALL_TIMEFRAMES:
            try:
                limit = 200 if tf in ["1m", "3m", "5m"] else 100
                df_tf = fetch_ohlcv(symbol, tf, limit, exchange)
                df_tf = add_all_indicators(df_tf)
                tf_data[tf] = df_tf
                print(f"  {C.dim(f'✓ {tf} — {len(df_tf)} candles')}")
            except Exception as e:
                print(f"  {C.dim(f'✗ {tf} — {e}')}")

        if not tf_data:
            print(f"{C.red('Failed to fetch any timeframe data.')}")
            return

        direction = trade["direction"]
        entry = trade["entry_price"]
        leverage = trade["leverage"]
        if direction == "LONG":
            pnl_pct = ((price - entry) / entry) * leverage * 100
        else:
            pnl_pct = ((entry - price) / entry) * leverage * 100

        elapsed = time.time() - trade.get("opened_ts", time.time())
        hours, rem_secs = divmod(int(elapsed), 3600)
        minutes = rem_secs // 60
        hold_time = f"{hours}h {minutes}m" if hours else f"{minutes}m"

        position_info = {
            "direction": direction,
            "entry_price": entry,
            "current_price": price,
            "stop_loss": trade["stop_loss"],
            "take_profit": trade["take_profit"],
            "leverage": leverage,
            "pnl_pct": pnl_pct,
            "sl_stage": trade.get("sl_stage", "INITIAL"),
            "hold_time": hold_time,
        }

        verdict = reanalyze_position_ai(symbol, tf_data, position_info)

        action = verdict.get("action", "HOLD")
        new_sl = verdict.get("new_stop_loss")
        new_tp = verdict.get("new_take_profit")

        if action == "HOLD":
            print(C.green("AI recommends holding the position as-is. No changes made."))
            return

        if action == "CLOSE_NOW":
            confirm = input(f"\n{C.red(C.bold('AI recommends CLOSING NOW. Close position? (yes/no):'))} ").strip().lower()
            if confirm in ("yes", "y"):
                result = executor.close_position(symbol)
                if "error" in result:
                    print(f"{C.red('Error')}: {C.red(result['error'])}")
                elif "info" in result:
                    print(C.yellow(result["info"]))
                elif result.get("closed"):
                    monitor.remove_position(symbol)
                    pnl = result["pnl"]
                    pnl_color = C.green if pnl >= 0 else C.red
                    print(f"\n{C.bold('─' * 45)}")
                    print(f"  {C.bold('POSITION CLOSED')} — {C.cyan(symbol)}")
                    print(f"{C.bold('─' * 45)}")
                    print(f"  Entry:       ${result['entry_price']:,.2f}")
                    print(f"  Close:       ${result['close_price']:,.2f}")
                    print(f"  P&L:         {pnl_color(f'${pnl:+.2f}')}")
                    print(f"{C.bold('─' * 45)}\n")
                    logger.info(f"Position {symbol} closed via reanalyze CLOSE_NOW")
            else:
                print(C.yellow("Position kept open."))
            return

        # ─── Reanalysis cooldown (anti-whipsaw) ────────────────
        last_reanalysis_ts = trade.get("last_reanalysis_ts", trade.get("opened_ts", time.time()))
        cooldown_secs = REANALYSIS_COOLDOWN_MINUTES * 60
        elapsed_since = time.time() - last_reanalysis_ts
        if elapsed_since < cooldown_secs:
            remaining_min = int((cooldown_secs - elapsed_since) // 60) + 1
            print(f"\n  {C.yellow(C.bold('REANALYSIS COOLDOWN ACTIVE'))}")
            print(f"  {C.dim(f'Last SL/TP change was {int(elapsed_since // 60)}m ago — wait {remaining_min}m more before applying another move.')}")
            print(f"  {C.dim('(AI suggestion above is for reference only — not applied.)')}")
            return

        # ─── Minimum meaningful-change threshold (anti-whipsaw) ─
        if action in ("MOVE_SL", "MOVE_BOTH") and new_sl:
            if abs(new_sl - trade["stop_loss"]) / price * 100 < MIN_LEVEL_CHANGE_PCT:
                new_sl = None
        if action in ("MOVE_TP", "MOVE_BOTH") and new_tp:
            if abs(new_tp - trade["take_profit"]) / price * 100 < MIN_LEVEL_CHANGE_PCT:
                new_tp = None
        if not new_sl and not new_tp:
            print(C.yellow(f"AI suggested a change smaller than the {MIN_LEVEL_CHANGE_PCT}% minimum — treating as HOLD."))
            return

        # MOVE_SL / MOVE_TP / MOVE_BOTH
        MIN_SL_GAP_PCT = 0.3  # below this, the new SL sits within 1-2 candle wicks — stop-hunt risk

        changes = []
        sl_warning = False
        if action in ("MOVE_SL", "MOVE_BOTH") and new_sl:
            gap_pct = abs(new_sl - price) / price * 100
            changes.append(f"  SL: ${trade['stop_loss']:,.6f} -> {C.cyan(f'${new_sl:,.6f}')} {C.dim(f'(gap from current price: {gap_pct:.2f}%)')}")
            if gap_pct < MIN_SL_GAP_PCT:
                sl_warning = True
        if action in ("MOVE_TP", "MOVE_BOTH") and new_tp:
            changes.append(f"  TP: ${trade['take_profit']:,.6f} -> {C.cyan(f'${new_tp:,.6f}')}")

        if not changes:
            print(C.yellow("AI suggested an adjustment but did not provide valid new levels. No changes made."))
            return

        print(f"\n{C.bold('Suggested changes:')}")
        for c in changes:
            print(c)

        if sl_warning:
            print(f"\n  {C.red(C.bold('WARNING:'))} {C.red(f'New SL is within {MIN_SL_GAP_PCT}% of current price')} {C.dim(f'(${price:,.6f})')}")
            print(f"  {C.red('— a normal wick could trigger it. Consider rejecting or adjusting manually.')}")

        confirm = input(f"\n{C.cyan(C.bold('Apply these changes to the live trade? (yes/no):'))} ").strip().lower()
        if confirm not in ("yes", "y"):
            print(C.yellow("No changes applied."))
            return

        sl_arg = new_sl if action in ("MOVE_SL", "MOVE_BOTH") and new_sl else None
        tp_arg = new_tp if action in ("MOVE_TP", "MOVE_BOTH") and new_tp else None

        result = monitor.update_levels(symbol, new_sl=sl_arg, new_tp=tp_arg)
        if "error" in result:
            print(f"{C.red('Error')}: {C.red(result['error'])}")
            return

        if result.get("sl_updated"):
            print(f"  {C.green('SL updated to')} ${result['new_sl']:,.6f}")
        if result.get("tp_updated"):
            print(f"  {C.green('TP updated to')} ${result['new_tp']:,.6f}")

        if result.get("sl_updated") or result.get("tp_updated"):
            monitor.mark_reanalyzed(symbol)

        logger.info(f"Position {symbol} levels updated via reanalyze: {result}")

    except Exception as e:
        logger.error(f"Error reanalyzing {symbol}: {e}")
        print(f"{C.red(C.bold('ERROR'))}: {C.red(str(e))}")


def scan_all_coins():
    """
    AI-driven scan of ALL major Binance Futures coins:
    1. Bot fetches data + indicators for all coins
    2. Bot pre-filters (basic trend + volume check)
    3. Sends pre-filtered candidates to AI in ONE batch prompt
    4. AI picks the best 3 setups with full SL/TP
    """
    global exchange

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
    print(f"{C.dim('Phase 1: Fetching data + pre-filtering...')}\n")
    logger.info(f"allCoins scan started ({len(ALL_COINS)} coins)")

    # Phase 1: Fetch 5m data and pre-filter
    from multi_ai_verifier import ALL_TIMEFRAMES
    pre_filtered = []
    for i, symbol in enumerate(ALL_COINS):
        short_name = symbol.replace("/USDT", "")
        progress = f"[{i+1}/{len(ALL_COINS)}]"
        print(f"  {C.dim(progress)} {short_name}...", end="", flush=True)

        try:
            df = fetch_ohlcv(symbol, "5m", 200, exchange)
            df = add_all_indicators(df)

            if len(df) < 50:
                print(f" {C.dim('insufficient data')}")
                continue

            row = df.iloc[-1]

            # Liquidity filter — skip thin coins before spending AI budget on them
            liquidity = check_liquidity(df, MIN_DOLLAR_VOLUME)
            if not liquidity["liquid"]:
                dv = liquidity["dollar_volume"]
                print(f" {C.dim(f'illiquid (~${dv:,.0f}/candle)')}")
                continue

            # Volatility-spike guard — skip coins mid-spike (proxy for news/liquidation event)
            vol_spike = check_volatility_spike(df, VOLATILITY_SPIKE_ATR_MULT)
            if vol_spike["spike"]:
                ratio = vol_spike["ratio"]
                print(f" {C.dim(f'volatility spike ({ratio}x ATR)')}")
                continue

            # Basic pre-filter: must have trend + volume + not overextended
            has_trend = (row["ema_9"] != row["ema_21"])  # EMAs not equal
            has_volume = row["vol_sma_20"] > 0 and row["volume"] > 0.5 * row["vol_sma_20"]
            not_overextended = not row.get("overextended", False)
            rsi_ok = 25 <= row.get("rsi", 50) <= 75

            if not (has_trend and has_volume and not_overextended and rsi_ok):
                print(f" {C.dim('filtered out')}")
                continue

            trend_dir = "BULL" if row["ema_9"] > row["ema_21"] else "BEAR"
            pre_filtered.append({
                "symbol": symbol, "trend": trend_dir,
                "dollar_volume": liquidity["dollar_volume"],
            })
            print(f" {C.yellow(f'candidate ({trend_dir})')}")

        except Exception as e:
            print(f" {C.red('error')}")
            logger.error(f"allCoins error on {symbol}: {e}")

    print(f"\n{C.bold(f'Pre-filter: {len(pre_filtered)}/{len(ALL_COINS)} coins passed')}")

    if not pre_filtered:
        print(f"\n  {C.yellow(C.bold('NO CANDIDATES'))}")
        print(f"  {C.dim('All coins filtered out. No clear trends or volume.')}")
        logger.info("allCoins scan — no candidates passed pre-filter")
        return

    # Cap the AI batch size — rank by liquidity so the busiest coins win,
    # bounding both the scan prompt size and the multi-TF fetch below.
    if len(pre_filtered) > MAX_SCAN_CANDIDATES:
        pre_filtered.sort(key=lambda x: x["dollar_volume"], reverse=True)
        print(f"{C.dim(f'Capping to top {MAX_SCAN_CANDIDATES} by liquidity (of {len(pre_filtered)})')}")
        pre_filtered = pre_filtered[:MAX_SCAN_CANDIDATES]

    # Phase 1.5: Fetch all TFs for candidates only
    print(f"\n{C.dim('Fetching multi-timeframe data for candidates...')}")
    candidates = []
    for item in pre_filtered:
        symbol = item["symbol"]
        short_name = symbol.replace("/USDT", "")
        print(f"  {C.dim(short_name)}:", end="", flush=True)

        tf_data = {}
        for tf in ALL_TIMEFRAMES:
            try:
                limit = 200 if tf in ["1m", "3m", "5m"] else 100
                df_tf = fetch_ohlcv(symbol, tf, limit, exchange)
                df_tf = add_all_indicators(df_tf)
                tf_data[tf] = df_tf
                print(f" {C.dim(tf)}", end="", flush=True)
            except Exception:
                pass

        if tf_data:
            snapshot = build_indicator_snapshot(tf_data, symbol)
            candidates.append({"symbol": symbol, "snapshot": snapshot})
            print(f" {C.green('✓')}")
        else:
            print(f" {C.red('✗')}")

    print(f"\n{C.bold(f'Pre-filter: {len(candidates)}/{len(ALL_COINS)} coins passed')}")

    if not candidates:
        print(f"\n  {C.yellow(C.bold('NO CANDIDATES'))}")
        print(f"  {C.dim('All coins filtered out. No clear trends or volume.')}")
        logger.info("allCoins scan — no candidates passed pre-filter")
        return

    # Phase 2: Send to AI
    print(f"\n{C.dim('Phase 2: Sending candidates to AI for analysis...')}")
    ai_picks = scan_coins_ai(candidates)

    if not ai_picks:
        print(f"\n  {C.yellow(C.bold('AI found NO safe trade setups.'))}")
        print(f"  {C.dim('This is normal — a good strategy waits for the right setup.')}")
        logger.info("allCoins scan — AI found no setups")
        return

    # Display AI picks
    print(f"\n{C.bold('=' * 65)}")
    print(f"  {C.bold(C.cyan('AI TRADE PICKS'))}")
    print(f"{C.bold('=' * 65)}")

    table_data = []
    for p in ai_picks:
        sym = p["symbol"].replace("/USDT", "")
        dir_color = C.green if p["direction"] == "LONG" else C.red
        entry = p.get("entry", 0)
        sl = p.get("stop_loss", 0)
        tp = p.get("take_profit", 0)
        risk = p.get("risk_score", "?")
        r_color = C.green if risk == "LOW" else (C.yellow if risk == "MEDIUM" else C.red)

        # Calculate R:R
        if entry and sl and tp:
            sl_d = abs(entry - sl)
            tp_d = abs(tp - entry)
            rr = f"{tp_d/sl_d:.1f}:1" if sl_d > 0 else "?"
        else:
            rr = "?"

        tt = p.get("trade_type", "SCALP")
        ai_tf = p.get("timeframe", "5m")

        table_data.append([
            sym,
            dir_color(p["direction"]),
            C.yellow(f"{p['confidence']}/10"),
            f"{tt}/{ai_tf}",
            f"${entry:,.4f}" if entry else "-",
            C.red(f"${sl:,.4f}") if sl else "-",
            C.green(f"${tp:,.4f}") if tp else "-",
            rr,
            r_color(risk),
        ])

    print(tabulate(table_data, headers=[
        "Coin", "Signal", "Conf", "Type/TF", "Entry", "SL", "TP", "R:R", "Risk"
    ], tablefmt="simple"))

    print(f"\n  {C.green(f'AI selected {len(ai_picks)} setup(s)')}")
    print(f"{C.bold('=' * 65)}")

    # Show reasons for the best pick
    best = ai_picks[0]
    best_sym = best["symbol"]
    print(f"\n{C.bold(C.cyan(f'BEST SETUP: {best_sym}'))}")
    print(f"{C.dim('─' * 55)}")
    for i, r in enumerate(best.get("reasons", []), 1):
        if any(kw in r.upper() for kw in ["WARNING", "RISK", "DANGER"]):
            print(f"  {C.red(f'[{i}]')} {C.red(r)}")
        else:
            print(f"  {C.green(f'[{i}]')} {r}")
    advice = best.get("advice", "")
    if advice:
        print(f"  {C.cyan(f'Advice: {advice}')}")
    print(f"{C.dim('─' * 55)}")

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

    # Process signals with confidence >= 7
    actionable = [r for r in results if r["confidence"] >= 7]
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
            trade_type=r.get("trade_type", "SCALP"),
        )

        if not validation["approved"]:
            print(f"  {C.red('REJECTED')}: {C.red(validation['reason'])}")
            if "R:R" in validation["reason"]:
                tt = r.get("trade_type", "SCALP")
                required_tp = calculate_take_profit(r["entry"], r["stop_loss"], r["signal"], MIN_RR_BY_TYPE.get(tt, MIN_RR_RATIO))
                sl_dist = abs(r["entry"] - r["stop_loss"])
                tp_dist = abs(required_tp - r["entry"])
                needed_rr = tp_dist / sl_dist if sl_dist > 0 else 0
                tp_pct = tp_dist / r["entry"] * 100
                print(f"  {C.yellow('To pass:')} TP must be ${required_tp:,.6f} ({tp_pct:.2f}% from entry) for {needed_rr:.1f}:1 R:R")

                override = input(f"  {C.yellow(f'Override and proceed with the AI TP for {symbol} anyway? (yes/no):')} ").strip().lower()
                if override not in ("yes", "y"):
                    print(C.yellow("Skipped."))
                    continue

                validation = validate_trade(
                    account_balance=balance,
                    entry_price=r["entry"],
                    stop_loss=r["stop_loss"],
                    take_profit=r["take_profit"],
                    confidence=r["confidence"],
                    direction=r["signal"],
                    trade_type=tt,
                    allow_low_rr=True,
                )
                if not validation["approved"]:
                    print(f"  {C.red('REJECTED')}: {C.red(validation['reason'])}")
                    continue
                rr = validation["rr_ratio"]
                print(f"  {C.yellow(C.bold(f'Manual override — proceeding at {rr:.2f}:1 R:R'))}")
            else:
                continue

        # Build signal_data dict for display
        signal_data = {
            "direction": r["signal"],
            "confidence": r["confidence"],
            "reasons": r["reasons"],
        }
        scan_price = r.get("entry", 0)
        print_trade_table(signal_data, validation, symbol, current_price=scan_price)

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
                    sl_result = result.get("stop_loss", {})
                    tp_result = result.get("take_profit", {})
                    sl_order_id = str(sl_result["id"]) if "id" in sl_result and "error" not in sl_result else None
                    tp_order_id = str(tp_result["id"]) if "id" in tp_result and "error" not in tp_result else None
                    monitor.add_position(
                        symbol=symbol, direction=sig,
                        entry_price=entry_price,
                        stop_loss=validation["stop_loss"],
                        take_profit=validation["take_profit"],
                        quantity=filled_qty, leverage=validation["leverage"],
                        confidence=conf,
                        sl_order_id=sl_order_id,
                        tp_order_id=tp_order_id,
                    )
        elif confirm == "skip-all":
            print(C.yellow("Skipping remaining signals."))
            break
        else:
            print(C.yellow("Skipped."))

    logger.info("Scanner cycle complete")


# ─── Main loop ─────────────────────────────────────────────────

def main():
    global exchange, executor, monitor, running, session_start_balance

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
        session_start_balance = balance
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

            elif cmd in ("reanalyse", "reanalyze") or cmd.startswith(("reanalyse ", "reanalyze ")):
                if " " in cmd:
                    coin = cmd.split(" ", 1)[1].strip().upper()
                else:
                    coin = input(f"Enter active trade's coin (e.g. BTC/USDT): ").strip().upper()
                if coin and "/" not in coin:
                    coin = coin + "/USDT"
                if coin:
                    reanalyze_position(coin)

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
                daily_pnl = 0.0
                print(C.green("Circuit breaker & daily loss limit reset. Trading resumed."))

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
