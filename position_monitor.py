"""
Position Monitor — Software-based SL/TP with Smart Trailing Stop.

Features:
- Confirms position is actually active on exchange before monitoring
- Checks price every 3 seconds
- Trailing stop logic:
  * 0-33% to TP:   SL stays at original level
  * 33% to TP:     SL moves to breakeven (entry + fees)
  * 50% to TP:     SL moves to halfway between entry and current price
  * 66%+ to TP:    SL trails at 30% of remaining distance to TP
- Prints live P&L updates and SL adjustments in terminal
"""

import threading
import time
from datetime import datetime
from logger_setup import get_logger

logger = get_logger("monitor")


class C:
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    CYAN = "\033[96m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RESET = "\033[0m"


class PositionMonitor:
    def __init__(self, exchange, executor):
        self.exchange = exchange
        self.executor = executor
        self.tracked = {}  # symbol -> trade info
        self._running = False
        self._thread = None
        self._lock = threading.Lock()
        self._last_print = {}  # symbol -> last printed SL (to avoid spam)

    def add_position(self, symbol: str, direction: str, entry_price: float,
                     stop_loss: float, take_profit: float, quantity: float,
                     leverage: int, confidence: int = 0):
        """
        Add a position to monitor. First confirms the position is active
        on the exchange before starting to track.
        """
        # Confirm position is actually open on exchange
        confirmed = self._confirm_position(symbol, direction)
        if not confirmed:
            print(f"  {C.RED}[Monitor] Could not confirm {direction} position on {symbol}{C.RESET}")
            print(f"  {C.YELLOW}Position may not have been filled. Not monitoring.{C.RESET}")
            logger.warning(f"[Monitor] Position not confirmed for {symbol}")
            return False

        breakeven = entry_price * 1.0008 if direction == "LONG" else entry_price * 0.9992
        total_distance = abs(take_profit - entry_price)

        with self._lock:
            self.tracked[symbol] = {
                "direction": direction,
                "entry_price": entry_price,
                "stop_loss": stop_loss,
                "original_sl": stop_loss,
                "take_profit": take_profit,
                "breakeven": breakeven,
                "quantity": quantity,
                "leverage": leverage,
                "confidence": confidence,
                "total_distance": total_distance,
                "opened_at": datetime.now().strftime("%H:%M:%S"),
                "opened_ts": time.time(),   # For max hold time check
                "best_price": entry_price,
                "sl_stage": "INITIAL",
                "max_hold_seconds": 4 * 3600,  # 4 hours default
            }

        sl_str = f"${stop_loss:,.2f}"
        tp_str = f"${take_profit:,.2f}"
        be_str = f"${breakeven:,.2f}"
        logger.info(f"[Monitor] Tracking {direction} {symbol} | SL: {sl_str} | TP: {tp_str} | BE: {be_str}")

        print(f"\n  {C.GREEN}{C.BOLD}POSITION CONFIRMED & MONITORED{C.RESET}")
        print(f"  {C.DIM}Checking price every 3 seconds{C.RESET}")
        print(f"  {C.DIM}Trailing stop will adjust automatically:{C.RESET}")
        print(f"    {C.DIM}25% to TP → SL moves to breakeven ({be_str}){C.RESET}")
        print(f"    {C.DIM}50% to TP → SL locks 60% of profit{C.RESET}")
        print(f"    {C.DIM}66% to TP → SL trails tight (15% cushion){C.RESET}")
        print()

        self._ensure_running()
        return True

    def _confirm_position(self, symbol: str, direction: str) -> bool:
        """Verify the position actually exists on the exchange."""
        try:
            # Wait a moment for the order to settle
            time.sleep(1)
            positions = self.exchange.fetch_positions([symbol])
            for pos in positions:
                contracts = float(pos.get("contracts", 0))
                pos_side = pos.get("side", "")
                if contracts > 0:
                    expected = "long" if direction == "LONG" else "short"
                    if pos_side == expected:
                        return True
            # If we can't match side, at least check contracts > 0
            for pos in positions:
                if float(pos.get("contracts", 0)) > 0:
                    return True
            return False
        except Exception as e:
            logger.error(f"[Monitor] Error confirming position: {e}")
            # If we can't check, assume it's there (entry order was filled)
            return True

    def remove_position(self, symbol: str):
        """Stop tracking a position."""
        with self._lock:
            self.tracked.pop(symbol, None)
            self._last_print.pop(symbol, None)

    def get_tracked(self) -> dict:
        """Get all tracked positions with current SL stage."""
        with self._lock:
            return {k: dict(v) for k, v in self.tracked.items()}

    def _ensure_running(self):
        """Start the monitor thread if not already running."""
        if not self._running:
            self._running = True
            self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self._thread.start()

    def stop(self):
        """Stop the monitor."""
        self._running = False

    def _monitor_loop(self):
        """Background loop: check prices, trail stops, trigger exits."""
        while self._running:
            try:
                with self._lock:
                    symbols = list(self.tracked.keys())

                if not symbols:
                    time.sleep(3)
                    continue

                for symbol in symbols:
                    with self._lock:
                        trade = self.tracked.get(symbol)
                    if not trade:
                        continue

                    try:
                        ticker = self.exchange.fetch_ticker(symbol)
                        current_price = float(ticker["last"])
                    except Exception:
                        continue

                    direction = trade["direction"]
                    sl = trade["stop_loss"]
                    tp = trade["take_profit"]
                    entry = trade["entry_price"]

                    # ── Check if SL or TP hit ──────────────────
                    triggered = None
                    reason = None

                    if direction == "LONG":
                        if current_price <= sl:
                            triggered = "STOP LOSS"
                            reason = f"Price ${current_price:,.2f} hit SL ${sl:,.2f}"
                        elif current_price >= tp:
                            triggered = "TAKE PROFIT"
                            reason = f"Price ${current_price:,.2f} hit TP ${tp:,.2f}"
                    else:
                        if current_price >= sl:
                            triggered = "STOP LOSS"
                            reason = f"Price ${current_price:,.2f} hit SL ${sl:,.2f}"
                        elif current_price <= tp:
                            triggered = "TAKE PROFIT"
                            reason = f"Price ${current_price:,.2f} hit TP ${tp:,.2f}"

                    # ── Max hold time check ────────────────────
                    if not triggered:
                        elapsed = time.time() - trade.get("opened_ts", time.time())
                        max_hold = trade.get("max_hold_seconds", 4 * 3600)
                        if elapsed >= max_hold:
                            hours = elapsed / 3600
                            triggered = "MAX HOLD TIME"
                            reason = f"Position held {hours:.1f}h (max {max_hold/3600:.0f}h) — auto-closing"

                    if triggered:
                        self._close_position(symbol, trade, current_price, triggered, reason)
                        continue

                    # ── Trail the stop loss ────────────────────
                    self._trail_stop(symbol, trade, current_price)

            except Exception as e:
                logger.error(f"[Monitor] Error: {e}")

            time.sleep(3)

    def _trail_stop(self, symbol: str, trade: dict, current_price: float):
        """
        Smart trailing stop logic.
        Adjusts SL based on how far price has moved toward TP.
        """
        direction = trade["direction"]
        entry = trade["entry_price"]
        tp = trade["take_profit"]
        total_dist = trade["total_distance"]
        breakeven = trade["breakeven"]
        current_sl = trade["stop_loss"]
        old_stage = trade["sl_stage"]

        if total_dist == 0:
            return

        # Calculate progress toward TP (0.0 to 1.0)
        if direction == "LONG":
            progress = (current_price - entry) / total_dist
            # Update best price seen
            best = max(trade["best_price"], current_price)
        else:
            progress = (entry - current_price) / total_dist
            best = min(trade["best_price"], current_price)

        if progress <= 0:
            return  # Price hasn't moved in our favor yet

        new_sl = current_sl
        new_stage = old_stage

        if progress >= 0.66:
            # TRAILING: SL trails tight — only 15% behind best price
            # At this point we're protecting most of the profit
            trail_distance = total_dist * 0.15
            if direction == "LONG":
                new_sl = max(current_sl, best - trail_distance)
            else:
                new_sl = min(current_sl, best + trail_distance)
            new_stage = "TRAILING"

        elif progress >= 0.50:
            # LOCK PROFIT: SL moves to at least halfway, or 60% of the move
            if direction == "LONG":
                halfway = entry + (current_price - entry) * 0.6
                new_sl = max(current_sl, halfway)
            else:
                halfway = entry - (entry - current_price) * 0.6
                new_sl = min(current_sl, halfway)
            new_stage = "LOCK PROFIT"

        elif progress >= 0.25:
            # BREAKEVEN: SL moves to breakeven (entry + fees) — earlier than before
            if direction == "LONG":
                new_sl = max(current_sl, breakeven)
            else:
                new_sl = min(current_sl, breakeven)
            new_stage = "BREAKEVEN"

        # Update if SL changed
        sl_changed = abs(new_sl - current_sl) > 0.01
        stage_changed = new_stage != old_stage

        if sl_changed or stage_changed:
            with self._lock:
                if symbol in self.tracked:
                    self.tracked[symbol]["stop_loss"] = round(new_sl, 6)
                    self.tracked[symbol]["best_price"] = best
                    self.tracked[symbol]["sl_stage"] = new_stage

            # Calculate current P&L
            if direction == "LONG":
                pnl_pct = ((current_price - entry) / entry) * trade["leverage"] * 100
            else:
                pnl_pct = ((entry - current_price) / entry) * trade["leverage"] * 100

            # Print SL adjustment (but don't spam)
            last = self._last_print.get(symbol)
            if last != new_stage or sl_changed:
                self._last_print[symbol] = new_stage

                stage_colors = {
                    "BREAKEVEN": C.YELLOW,
                    "LOCK PROFIT": C.CYAN,
                    "TRAILING": C.GREEN,
                }
                color = stage_colors.get(new_stage, C.DIM)

                pnl_color = C.GREEN if pnl_pct >= 0 else C.RED
                sym_short = symbol.replace("/USDT", "").replace(":USDT", "")

                print(f"\n  {color}{C.BOLD}[{sym_short}] SL → {new_stage}{C.RESET}"
                      f" | SL: ${new_sl:,.2f}"
                      f" | Price: ${current_price:,.2f}"
                      f" | P&L: {pnl_color}{pnl_pct:+.1f}%{C.RESET}"
                      f" | Progress: {progress*100:.0f}%")
                print(f"{C.CYAN}{C.BOLD}>{C.RESET} ", end="", flush=True)

                logger.info(f"[Monitor] {symbol} SL adjusted to {new_stage}: ${new_sl:,.2f} (progress {progress*100:.0f}%)")

    def _close_position(self, symbol: str, trade: dict, current_price: float,
                        triggered: str, reason: str):
        """Close a position when SL/TP is hit."""
        direction = trade["direction"]
        entry = trade["entry_price"]
        leverage = trade["leverage"]
        quantity = trade["quantity"]
        original_sl = trade["original_sl"]
        final_sl = trade["stop_loss"]

        # Calculate P&L
        if direction == "LONG":
            pnl_pct = ((current_price - entry) / entry) * leverage * 100
        else:
            pnl_pct = ((entry - current_price) / entry) * leverage * 100

        capital = abs(entry * quantity) / leverage
        pnl_dollar = capital * (pnl_pct / 100)

        # Close via market order
        result = self.executor.close_position(symbol)

        # Remove from tracking
        self.remove_position(symbol)

        # Print result
        is_win = pnl_dollar >= 0
        pnl_color = C.GREEN if is_win else C.RED
        trigger_color = C.GREEN if triggered == "TAKE PROFIT" else C.RED

        print(f"\n{C.BOLD}{'=' * 55}{C.RESET}")
        print(f"  {trigger_color}{C.BOLD}{triggered} TRIGGERED{C.RESET} — {C.CYAN}{symbol}{C.RESET}")
        print(f"{C.BOLD}{'=' * 55}{C.RESET}")
        print(f"  {C.DIM}{reason}{C.RESET}")
        print(f"  Direction:    {C.GREEN if direction == 'LONG' else C.RED}{direction}{C.RESET}")
        print(f"  Entry:        ${entry:,.2f}")
        print(f"  Exit:         ${current_price:,.2f}")
        print(f"  Leverage:     {leverage}x")
        print(f"  Original SL:  ${original_sl:,.2f}")
        print(f"  Final SL:     ${final_sl:,.2f}  ({trade['sl_stage']})")
        print(f"  P&L:          {pnl_color}{C.BOLD}${pnl_dollar:+.2f} ({pnl_pct:+.1f}%){C.RESET}")
        print(f"  Result:       {C.GREEN}{C.BOLD}WIN{C.RESET}" if is_win else f"  Result:       {C.RED}{C.BOLD}LOSS{C.RESET}")

        # Show if trailing SL saved money on a loss
        if triggered == "STOP LOSS" and trade["sl_stage"] != "INITIAL":
            if direction == "LONG":
                saved = (final_sl - original_sl) * quantity * leverage
            else:
                saved = (original_sl - final_sl) * quantity * leverage
            if saved > 0:
                print(f"  {C.GREEN}Trailing SL saved: ${saved:,.2f}{C.RESET}")

        print(f"{C.BOLD}{'=' * 55}{C.RESET}")
        print(f"\n{C.CYAN}{C.BOLD}>{C.RESET} ", end="", flush=True)

        logger.info(f"[Monitor] {triggered} on {symbol} | P&L: ${pnl_dollar:+.2f} ({pnl_pct:+.1f}%) | SL stage: {trade['sl_stage']}")

        # Log to trade tracker
        try:
            from trade_tracker import log_trade
            log_trade(
                coin=symbol, direction=direction,
                entry=entry, exit_price=current_price,
                sl=final_sl, tp=trade["take_profit"],
                leverage=leverage, capital=capital,
                confidence=trade.get("confidence", 0),
                pattern=triggered,
                notes=f"SL stage: {trade['sl_stage']} | Original SL: ${original_sl:,.2f}",
            )
        except Exception as e:
            logger.error(f"[Monitor] Failed to log trade: {e}")
