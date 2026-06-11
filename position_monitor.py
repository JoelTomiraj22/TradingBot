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

import json
import os
import threading
import time
from datetime import datetime
from logger_setup import get_logger
from config import (REANALYSIS_COOLDOWN_MINUTES, MIN_LEVEL_CHANGE_PCT,
                    AUTO_REANALYZE_MINUTES, DEFAULT_SL_PCT, DEFAULT_TP_PCT)
from risk_manager import TAKER_FEE_PCT

logger = get_logger("monitor")

STATE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "monitor_state.json")

# Below this distance from the reference price, an SL/TP level is inside the
# fee/friction zone and unusable — same threshold as the AI sanity gate.
_MIN_LEVEL_DIST_PCT = 0.30


def _validate_levels(direction: str, ref_price: float, sl: float, tp: float):
    """
    Guarantee a usable, correctly-ordered (SL, TP) pair for `direction` so
    that every tracked position is ALWAYS protected — regardless of what the
    AI (or a manual override) provided.

    For LONG, valid levels are SL < ref_price < TP; for SHORT, TP < ref_price
    < SL. If the given levels are exactly inverted (a known AI-verdict bug),
    swapping them fixes it. If they're unusable even after swapping (missing,
    on the wrong side, or too close to ref_price), fall back to default %
    distances from ref_price so the position is never left unprotected.

    Returns (sl, tp, note) — note is "" if the input levels were fine, or a
    human-readable description of the correction that was applied.
    """
    def _dist_ok(level):
        if not level or not ref_price:
            return False
        return abs(level - ref_price) / ref_price * 100 >= _MIN_LEVEL_DIST_PCT

    if direction == "LONG":
        if sl and tp and sl < ref_price < tp and _dist_ok(sl) and _dist_ok(tp):
            return sl, tp, ""
        if sl and tp and tp < ref_price < sl and _dist_ok(sl) and _dist_ok(tp):
            return tp, sl, (f"levels were inverted for LONG — swapped SL/TP "
                            f"(SL ${tp:,.6f} / TP ${sl:,.6f})")
        new_sl = ref_price * (1 - DEFAULT_SL_PCT / 100)
        new_tp = ref_price * (1 + DEFAULT_TP_PCT / 100)
        return new_sl, new_tp, (f"levels unusable for LONG (SL={sl}, TP={tp}, ref=${ref_price:,.6f}) "
                                 f"— using default {DEFAULT_SL_PCT}%/{DEFAULT_TP_PCT}%")
    else:  # SHORT
        if sl and tp and tp < ref_price < sl and _dist_ok(sl) and _dist_ok(tp):
            return sl, tp, ""
        if sl and tp and sl < ref_price < tp and _dist_ok(sl) and _dist_ok(tp):
            return tp, sl, (f"levels were inverted for SHORT — swapped SL/TP "
                            f"(SL ${tp:,.6f} / TP ${sl:,.6f})")
        new_sl = ref_price * (1 + DEFAULT_SL_PCT / 100)
        new_tp = ref_price * (1 - DEFAULT_TP_PCT / 100)
        return new_sl, new_tp, (f"levels unusable for SHORT (SL={sl}, TP={tp}, ref=${ref_price:,.6f}) "
                                 f"— using default {DEFAULT_SL_PCT}%/{DEFAULT_TP_PCT}%")


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
        self.tracked = {}        # symbol -> active trade info
        self.pending = {}        # symbol -> pending limit order info
        self.on_trade_closed = None  # callback(pnl_dollar) — set by bot.py for circuit breaker / daily loss tracking
        self._running = False
        self._thread = None
        self._lock = threading.RLock()
        self._last_print = {}  # symbol -> last printed SL (to avoid spam)
        self._load_state()

    def add_position(self, symbol: str, direction: str, entry_price: float,
                     stop_loss: float, take_profit: float, quantity: float,
                     leverage: int, confidence: int = 0,
                     sl_order_id: str = None, tp_order_id: str = None):
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

        # Mandatory SL/TP: an open position must always have a valid,
        # correctly-ordered SL/TP — fix or replace them before tracking.
        stop_loss, take_profit, fix_note = _validate_levels(direction, entry_price, stop_loss, take_profit)
        if fix_note:
            print(f"  {C.YELLOW}{C.BOLD}[Monitor] {symbol}: {fix_note}{C.RESET}")
            logger.warning(f"[Monitor] {symbol}: {fix_note}")

        # Breakeven cushion: taker fees (0.10% round-trip) + slippage between
        # the 3s price trigger and the actual market fill. 0.18% proved too
        # thin in practice (a "breakeven" exit landed net negative) — 0.25%.
        breakeven = entry_price * 1.0025 if direction == "LONG" else entry_price * 0.9975
        total_distance = abs(take_profit - entry_price)
        exit_side = "sell" if direction == "LONG" else "buy"

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
                "last_reanalysis_ts": time.time(),  # Cooldown anchor for AI re-analysis SL/TP changes
                "last_auto_reana_ts": time.time(),  # Scheduler anchor for automatic AI re-checks
                "best_price": entry_price,
                "sl_stage": "INITIAL",
                "max_hold_seconds": 4 * 3600,  # 4 hours default
                "sl_order_id": sl_order_id,
                "tp_order_id": tp_order_id,
                "exit_side": exit_side,
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
        if AUTO_REANALYZE_MINUTES > 0:
            print(f"  {C.CYAN}AI auto re-check every {AUTO_REANALYZE_MINUTES} min{C.RESET}"
                  f"{C.DIM} — tightens SL when justified, alerts on reversal (never auto-closes){C.RESET}")
        print()

        self._save_state()
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

    def add_pending_order(self, symbol: str, order_id: str, direction: str,
                          stop_loss: float, take_profit: float, quantity: float,
                          leverage: int, confidence: int = 0, limit_price: float = 0):
        """
        Track a limit order that hasn't filled yet. When it fills, the monitor
        will automatically place SL/TP and start tracking the position.
        """
        # Mandatory SL/TP: validate against the limit price (the entry once
        # filled) so a pending order never carries inverted/unusable levels.
        if limit_price:
            stop_loss, take_profit, fix_note = _validate_levels(direction, limit_price, stop_loss, take_profit)
            if fix_note:
                print(f"  {C.YELLOW}{C.BOLD}[Monitor] {symbol}: {fix_note}{C.RESET}")
                logger.warning(f"[Monitor] {symbol} (pending): {fix_note}")

        with self._lock:
            self.pending[symbol] = {
                "order_id": str(order_id),
                "direction": direction,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "quantity": quantity,
                "leverage": leverage,
                "confidence": confidence,
                "limit_price": limit_price,
                "placed_at": datetime.now().strftime("%H:%M:%S"),
                "placed_ts": time.time(),
            }

        sym_short = symbol.replace("/USDT", "").replace(":USDT", "")
        dir_color = C.GREEN if direction == "LONG" else C.RED
        print(f"\n  {C.GREEN}{C.BOLD}LIMIT ORDER PENDING{C.RESET} — {C.CYAN}{sym_short}{C.RESET}")
        print(f"  {C.DIM}Waiting for fill @ ${limit_price:,.6f}{C.RESET}")
        print(f"  {C.DIM}Direction: {dir_color}{direction}{C.RESET} | SL: {C.RED}${stop_loss:,.6f}{C.RESET} | TP: {C.GREEN}${take_profit:,.6f}{C.RESET}")
        print(f"  {C.DIM}When filled → SL/TP placed automatically + trailing stop starts{C.RESET}\n")
        logger.info(f"[Monitor] Pending limit order {order_id} for {direction} {symbol} @ ${limit_price}")

        self._save_state()
        self._ensure_running()

    def remove_position(self, symbol: str):
        """Stop tracking a position."""
        with self._lock:
            self.tracked.pop(symbol, None)
            self._last_print.pop(symbol, None)
        self._save_state()

    def get_tracked(self) -> dict:
        """Get all tracked positions with current SL stage."""
        with self._lock:
            return {k: dict(v) for k, v in self.tracked.items()}

    def mark_reanalyzed(self, symbol: str):
        """Record that an AI re-analysis SL/TP change was just applied —
        anchors the re-analysis cooldown window."""
        with self._lock:
            if symbol in self.tracked:
                self.tracked[symbol]["last_reanalysis_ts"] = time.time()
        self._save_state()

    def _load_state(self):
        """Restore tracked/pending positions from disk so SL/TP survive a bot restart."""
        if not os.path.exists(STATE_FILE):
            return
        try:
            with open(STATE_FILE) as f:
                data = json.load(f)

            tracked = data.get("tracked", {})
            pending = data.get("pending", {})

            # Drop tracked entries whose position no longer exists on the exchange
            open_symbols = None
            try:
                open_positions = self.executor.get_open_positions()
                open_symbols = {
                    p["symbol"].split(":")[0]: p.get("side")
                    for p in open_positions if float(p.get("contracts", 0)) > 0
                }
            except Exception:
                pass

            restored = 0
            for symbol, trade in tracked.items():
                if open_symbols is not None:
                    expected_side = "long" if trade.get("direction") == "LONG" else "short"
                    if open_symbols.get(symbol) != expected_side:
                        continue
                self.tracked[symbol] = trade
                restored += 1

            self.pending = pending

            if self.tracked or self.pending:
                dropped = len(tracked) - restored
                msg = f"[Monitor] Restored {restored} tracked position(s) from {STATE_FILE}"
                if dropped:
                    msg += f" ({dropped} stale entries dropped — position no longer open)"
                logger.info(msg)
                self._ensure_running()
        except Exception as e:
            logger.error(f"[Monitor] Failed to load state from {STATE_FILE}: {e}")

    def _save_state(self):
        """Persist tracked/pending positions to disk."""
        try:
            with self._lock:
                data = {"tracked": self.tracked, "pending": self.pending}
            with open(STATE_FILE, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"[Monitor] Failed to save state to {STATE_FILE}: {e}")

    def update_levels(self, symbol: str, new_sl: float = None, new_tp: float = None) -> dict:
        """
        Manually update SL and/or TP for a tracked position (e.g. after an
        AI re-analysis the user confirmed). Updates self.tracked and tries
        to move the live exchange order; falls back cleanly to software-only
        monitoring (e.g. on Demo) if the exchange call is skipped/fails.
        """
        with self._lock:
            trade = self.tracked.get(symbol)
            if not trade:
                return {"error": f"{symbol} is not being monitored"}
            trade = dict(trade)

        exit_side = trade.get("exit_side", "sell" if trade["direction"] == "LONG" else "buy")
        quantity = trade["quantity"]
        result = {"sl_updated": False, "tp_updated": False}

        if new_sl is not None:
            new_sl = round(new_sl, 6)
            old_sl_order_id = trade.get("sl_order_id")
            new_sl_order_id = old_sl_order_id
            try:
                order = self.executor.update_stop_loss(symbol, exit_side, quantity, new_sl, old_sl_order_id)
                if "error" not in order:
                    new_sl_order_id = str(order.get("id", old_sl_order_id))
                    logger.info(f"[Monitor] {symbol} exchange SL manually updated to ${new_sl:,.6f} (order {new_sl_order_id})")
                else:
                    logger.warning(f"[Monitor] {symbol} exchange SL update skipped/failed: {order['error']}")
            except Exception as e:
                logger.error(f"[Monitor] {symbol} exchange SL update error: {e}")

            with self._lock:
                if symbol in self.tracked:
                    self.tracked[symbol]["stop_loss"] = new_sl
                    self.tracked[symbol]["sl_order_id"] = new_sl_order_id
            result["sl_updated"] = True
            result["new_sl"] = new_sl

        if new_tp is not None:
            new_tp = round(new_tp, 6)
            old_tp_order_id = trade.get("tp_order_id")
            new_tp_order_id = old_tp_order_id
            try:
                order = self.executor.update_take_profit(symbol, exit_side, quantity, new_tp, old_tp_order_id)
                if "error" not in order:
                    new_tp_order_id = str(order.get("id", old_tp_order_id))
                    logger.info(f"[Monitor] {symbol} exchange TP manually updated to ${new_tp:,.6f} (order {new_tp_order_id})")
                else:
                    logger.warning(f"[Monitor] {symbol} exchange TP update skipped/failed: {order['error']}")
            except Exception as e:
                logger.error(f"[Monitor] {symbol} exchange TP update error: {e}")

            with self._lock:
                if symbol in self.tracked:
                    self.tracked[symbol]["take_profit"] = new_tp
                    self.tracked[symbol]["tp_order_id"] = new_tp_order_id
                    # Recompute progress baseline so trailing logic stays consistent
                    entry = self.tracked[symbol]["entry_price"]
                    self.tracked[symbol]["total_distance"] = abs(new_tp - entry)
            result["tp_updated"] = True
            result["new_tp"] = new_tp

        self._save_state()
        logger.info(f"[Monitor] {symbol} levels manually updated via reanalysis: {result}")
        return result

    def _ensure_running(self):
        """Start the monitor (and auto re-analysis) threads if not running."""
        if not self._running:
            self._running = True
            self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self._thread.start()
            if AUTO_REANALYZE_MINUTES > 0:
                self._reana_thread = threading.Thread(target=self._reanalysis_loop, daemon=True)
                self._reana_thread.start()

    def stop(self):
        """Stop the monitor."""
        self._running = False

    # ─── Automatic AI re-analysis ──────────────────────────────

    @staticmethod
    def _sl_auto_apply_check(direction: str, current_sl: float, new_sl: float,
                             price: float, min_gap_pct: float = 0.3,
                             min_change_pct: float = MIN_LEVEL_CHANGE_PCT):
        """
        Decide whether an AI-suggested SL may be applied AUTOMATICALLY.
        Policy: tighten-only (lock profit / cut risk), never loosen, never
        cross current price, never within the stop-hunt buffer of price,
        ignore sub-noise changes. Returns (ok: bool, reason: str).
        """
        if not new_sl or not price or not current_sl:
            return False, "no level"
        tighten = new_sl > current_sl if direction == "LONG" else new_sl < current_sl
        if not tighten:
            return False, "would loosen SL — never auto-applied"
        if direction == "LONG" and new_sl >= price:
            return False, "SL would be above current price"
        if direction == "SHORT" and new_sl <= price:
            return False, "SL would be below current price"
        if abs(new_sl - price) / price * 100 < min_gap_pct:
            return False, f"within {min_gap_pct}% of price (stop-hunt zone)"
        if abs(new_sl - current_sl) / price * 100 < min_change_pct:
            return False, "change too small to matter"
        return True, "ok"

    def _reanalysis_loop(self):
        """Background scheduler: AI re-check each open trade every N minutes."""
        interval = AUTO_REANALYZE_MINUTES * 60
        while self._running:
            time.sleep(5)
            try:
                now = time.time()
                with self._lock:
                    due = [
                        s for s, t in self.tracked.items()
                        if now - t.get("last_auto_reana_ts", t.get("opened_ts", now)) >= interval
                    ]
                for symbol in due:
                    self._auto_reanalyze(symbol)
            except Exception as e:
                logger.error(f"[Monitor] Auto re-analysis loop error: {e}")

    def _auto_reanalyze(self, symbol: str):
        """One automatic AI re-check of an open trade (runs off the price loop)."""
        with self._lock:
            trade = self.tracked.get(symbol)
            if not trade:
                return
            # Stamp first so a failing provider doesn't retrigger every 5s
            self.tracked[symbol]["last_auto_reana_ts"] = time.time()
            trade = dict(trade)

        from fetch_data import fetch_ohlcv
        from indicators import add_all_indicators
        from multi_ai_verifier import reanalyze_position_ai, ALL_TIMEFRAMES

        try:
            price = float(self.exchange.fetch_ticker(symbol)["last"])
        except Exception as e:
            logger.error(f"[AutoReana] {symbol} price fetch failed: {e}")
            return

        tf_data = {}
        for tf in ALL_TIMEFRAMES:
            try:
                limit = 200 if tf in ("1m", "3m", "5m") else 100
                df = fetch_ohlcv(symbol, tf, limit, self.exchange)
                tf_data[tf] = add_all_indicators(df)
            except Exception:
                pass
        if not tf_data:
            logger.error(f"[AutoReana] {symbol} — no timeframe data, skipping")
            return

        direction = trade["direction"]
        entry = trade["entry_price"]
        leverage = trade["leverage"]
        if direction == "LONG":
            pnl_pct = ((price - entry) / entry) * leverage * 100
        else:
            pnl_pct = ((entry - price) / entry) * leverage * 100

        elapsed = int(time.time() - trade.get("opened_ts", time.time()))
        hold_time = f"{elapsed // 3600}h {elapsed % 3600 // 60}m" if elapsed >= 3600 else f"{elapsed // 60}m"

        verdict = reanalyze_position_ai(symbol, tf_data, {
            "direction": direction,
            "entry_price": entry,
            "current_price": price,
            "stop_loss": trade["stop_loss"],
            "take_profit": trade["take_profit"],
            "leverage": leverage,
            "pnl_pct": pnl_pct,
            "sl_stage": trade.get("sl_stage", "INITIAL"),
            "hold_time": hold_time,
        }, quiet=True)

        action = verdict.get("action", "HOLD")
        reasons = verdict.get("reasons", [])
        first_reason = reasons[0][:90] if reasons else ""
        sym_short = symbol.replace("/USDT", "").replace(":USDT", "")
        pnl_color = C.GREEN if pnl_pct >= 0 else C.RED
        head = (f"\n  {C.CYAN}{C.BOLD}[AUTO RE-CHECK] {sym_short}{C.RESET} "
                f"{C.DIM}{direction} | P&L: {C.RESET}{pnl_color}{pnl_pct:+.1f}%{C.RESET}")

        if action == "CLOSE_NOW":
            print(head)
            print(f"  {C.RED}{C.BOLD}⚠  AI RECOMMENDS CLOSING — {verdict.get('urgency', '?')} urgency{C.RESET}")
            for r in reasons[:3]:
                print(f"  {C.RED}• {r}{C.RESET}")
            print(f"  {C.YELLOW}Not auto-closing. Review now: type{C.RESET} {C.CYAN}close{C.RESET} "
                  f"{C.YELLOW}or{C.RESET} {C.CYAN}reanalyse {sym_short.lower()}{C.RESET}")
            print(f"{C.CYAN}{C.BOLD}>{C.RESET} ", end="", flush=True)
            logger.warning(f"[AutoReana] {symbol}: AI recommends CLOSE_NOW ({verdict.get('urgency')}) — alert only")
            return

        new_sl = verdict.get("new_stop_loss")
        new_tp = verdict.get("new_take_profit")

        if action == "HOLD" or (not new_sl and not new_tp):
            # Stay quiet-ish: one line so you stay aware without spam
            print(f"{head} {C.DIM}| AI: HOLD — {first_reason}{C.RESET}")
            print(f"{C.CYAN}{C.BOLD}>{C.RESET} ", end="", flush=True)
            return

        printed_head = False

        # SL: tighten-only auto-apply, respecting cooldown + guards
        if action in ("MOVE_SL", "MOVE_BOTH") and new_sl:
            ok, why = self._sl_auto_apply_check(direction, trade["stop_loss"], new_sl, price)
            cooldown = time.time() - trade.get("last_reanalysis_ts", 0) < REANALYSIS_COOLDOWN_MINUTES * 60
            print(head)
            printed_head = True
            if ok and not cooldown:
                result = self.update_levels(symbol, new_sl=new_sl)
                if result.get("sl_updated"):
                    self.mark_reanalyzed(symbol)
                    print(f"  {C.GREEN}SL auto-tightened: ${trade['stop_loss']:,.6f} → ${new_sl:,.6f}{C.RESET} "
                          f"{C.DIM}({first_reason}){C.RESET}")
                    logger.info(f"[AutoReana] {symbol} SL auto-tightened to ${new_sl:,.6f}")
            else:
                skip_why = "cooldown active" if cooldown else why
                print(f"  {C.DIM}AI suggested SL ${new_sl:,.6f} — not applied ({skip_why}){C.RESET}")

        # TP: never auto-changed — suggestion only
        if action in ("MOVE_TP", "MOVE_BOTH") and new_tp:
            if not printed_head:
                print(head)
                printed_head = True
            print(f"  {C.YELLOW}AI suggests TP ${trade['take_profit']:,.6f} → ${new_tp:,.6f}{C.RESET} "
                  f"{C.DIM}— review with: reanalyse {sym_short.lower()}{C.RESET}")

        if printed_head:
            print(f"{C.CYAN}{C.BOLD}>{C.RESET} ", end="", flush=True)

    def _check_pending_orders(self):
        """Poll pending limit orders; auto-activate on fill."""
        with self._lock:
            pending_symbols = list(self.pending.keys())

        for symbol in pending_symbols:
            with self._lock:
                pend = self.pending.get(symbol)
            if not pend:
                continue

            try:
                order = self.exchange.fetch_order(pend["order_id"], symbol)
                status = order.get("status", "")
            except Exception as e:
                logger.error(f"[Monitor] Error fetching pending order {pend['order_id']}: {e}")
                continue

            if status == "closed":
                # Order filled — auto-activate
                with self._lock:
                    self.pending.pop(symbol, None)

                filled_price = float(order.get("average") or order.get("price") or pend["limit_price"])
                filled_qty = float(order.get("filled") or pend["quantity"])
                direction = pend["direction"]
                sl = pend["stop_loss"]
                tp = pend["take_profit"]
                leverage = pend["leverage"]
                confidence = pend["confidence"]
                exit_side = "sell" if direction == "LONG" else "buy"

                sym_short = symbol.replace("/USDT", "").replace(":USDT", "")
                dir_color = C.GREEN if direction == "LONG" else C.RED
                print(f"\n  {C.GREEN}{C.BOLD}[AI] LIMIT ORDER FILLED — {sym_short}{C.RESET}")
                print(f"  {dir_color}{direction}{C.RESET} filled @ ${filled_price:,.6f}")
                print(f"  {C.DIM}Placing SL/TP automatically...{C.RESET}")
                logger.info(f"[Monitor] Limit order filled: {direction} {symbol} @ ${filled_price}")

                sl_order = self.executor.place_stop_loss(symbol, exit_side, filled_qty, sl)
                tp_order = self.executor.place_take_profit(symbol, exit_side, filled_qty, tp)

                sl_order_id = str(sl_order["id"]) if "id" in sl_order and "error" not in sl_order else None
                tp_order_id = str(tp_order["id"]) if "id" in tp_order and "error" not in tp_order else None

                if "error" not in sl_order:
                    print(f"  {C.GREEN}SL placed @ ${sl:,.6f}{C.RESET}")
                else:
                    print(f"  {C.YELLOW}SL order failed (software monitor active): ${sl:,.6f}{C.RESET}")

                if "error" not in tp_order:
                    print(f"  {C.GREEN}TP placed @ ${tp:,.6f}{C.RESET}")
                else:
                    print(f"  {C.YELLOW}TP order failed (software monitor active): ${tp:,.6f}{C.RESET}")

                self.add_position(
                    symbol=symbol,
                    direction=direction,
                    entry_price=filled_price,
                    stop_loss=sl,
                    take_profit=tp,
                    quantity=filled_qty,
                    leverage=leverage,
                    confidence=confidence,
                    sl_order_id=sl_order_id,
                    tp_order_id=tp_order_id,
                )
                print(f"{C.CYAN}{C.BOLD}>{C.RESET} ", end="", flush=True)

            elif status == "canceled":
                with self._lock:
                    self.pending.pop(symbol, None)
                self._save_state()
                sym_short = symbol.replace("/USDT", "").replace(":USDT", "")
                print(f"\n  {C.YELLOW}[Monitor] Pending order for {sym_short} was cancelled.{C.RESET}")
                print(f"{C.CYAN}{C.BOLD}>{C.RESET} ", end="", flush=True)
                logger.info(f"[Monitor] Pending order {pend['order_id']} for {symbol} was cancelled")

    def _monitor_loop(self):
        """Background loop: check prices, trail stops, trigger exits."""
        while self._running:
            try:
                with self._lock:
                    symbols = list(self.tracked.keys())
                    has_pending = bool(self.pending)

                if not symbols and not has_pending:
                    time.sleep(3)
                    continue

                if has_pending:
                    self._check_pending_orders()

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

        # Update if SL changed — RELATIVE threshold (0.05% of price), so the
        # trailing stop also works on sub-dollar coins (DOGE, XRP, ...) where
        # a fixed $0.01 threshold would never trigger an update.
        sl_changed = abs(new_sl - current_sl) > current_price * 0.0005
        stage_changed = new_stage != old_stage

        if sl_changed or stage_changed:
            old_sl_order_id = trade.get("sl_order_id")
            exit_side = trade.get("exit_side", "sell" if direction == "LONG" else "buy")
            quantity = trade["quantity"]

            # Auto-update the live exchange SL order
            new_sl_order_id = old_sl_order_id
            if sl_changed and self.executor:
                try:
                    new_order = self.executor.update_stop_loss(
                        symbol, exit_side, quantity, round(new_sl, 6), old_sl_order_id
                    )
                    if "error" not in new_order:
                        new_sl_order_id = str(new_order.get("id", old_sl_order_id))
                        logger.info(f"[Monitor] {symbol} exchange SL updated to ${new_sl:,.2f} (order {new_sl_order_id})")
                    else:
                        logger.warning(f"[Monitor] {symbol} exchange SL update failed: {new_order['error']}")
                except Exception as e:
                    logger.error(f"[Monitor] {symbol} exchange SL update error: {e}")

            with self._lock:
                if symbol in self.tracked:
                    self.tracked[symbol]["stop_loss"] = round(new_sl, 6)
                    self.tracked[symbol]["best_price"] = best
                    self.tracked[symbol]["sl_stage"] = new_stage
                    self.tracked[symbol]["sl_order_id"] = new_sl_order_id
            self._save_state()

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

                print(f"\n  {color}{C.BOLD}[AI] [{sym_short}] SL → {new_stage}{C.RESET}"
                      f" | SL: ${new_sl:,.2f}"
                      f" | Price: ${current_price:,.2f}"
                      f" | P&L: {pnl_color}{pnl_pct:+.1f}%{C.RESET}"
                      f" | Progress: {progress*100:.0f}%")
                print(f"{C.CYAN}{C.BOLD}>{C.RESET} ", end="", flush=True)

                logger.info(f"[Monitor] {symbol} SL adjusted to {new_stage}: ${new_sl:,.2f} (progress {progress*100:.0f}%)")

    @staticmethod
    def _net_pnl(direction: str, entry: float, exit_price: float,
                 quantity: float, leverage: int):
        """
        Honest P&L from ACTUAL fill prices, net of taker fees on both sides.
        Returns (pnl_dollar_net, pnl_pct_net_on_margin, fees).
        """
        if direction == "LONG":
            gross = (exit_price - entry) * quantity
        else:
            gross = (entry - exit_price) * quantity
        fees = (entry + exit_price) * quantity * TAKER_FEE_PCT
        net = gross - fees
        capital = abs(entry * quantity) / max(leverage, 1)
        pct = (net / capital * 100) if capital > 0 else 0.0
        return net, pct, fees

    def _close_position(self, symbol: str, trade: dict, current_price: float,
                        triggered: str, reason: str):
        """Close a position when SL/TP is hit."""
        direction = trade["direction"]
        entry = trade["entry_price"]
        leverage = trade["leverage"]
        quantity = trade["quantity"]
        original_sl = trade["original_sl"]
        final_sl = trade["stop_loss"]

        # Close via market order FIRST, then compute P&L from the ACTUAL fill
        # (the old display used the trigger price and ignored fees, so a
        # "breakeven WIN" could actually be a small net loss on the balance).
        result = self.executor.close_position(symbol)

        exit_price = current_price
        if isinstance(result, dict) and result.get("close_price"):
            try:
                fill = float(result["close_price"])
                if fill > 0:
                    exit_price = fill
            except (TypeError, ValueError):
                pass

        capital = abs(entry * quantity) / max(leverage, 1)
        pnl_dollar, pnl_pct, fees = self._net_pnl(direction, entry, exit_price, quantity, leverage)

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
        print(f"  Entry (fill): ${entry:,.4f}")
        print(f"  Exit (fill):  ${exit_price:,.4f}  {C.DIM}(trigger was ${current_price:,.4f}){C.RESET}")
        print(f"  Leverage:     {leverage}x")
        print(f"  Original SL:  ${original_sl:,.4f}")
        print(f"  Final SL:     ${final_sl:,.4f}  ({trade['sl_stage']})")
        print(f"  Fees (taker): {C.DIM}-${fees:,.2f}{C.RESET}")
        print(f"  P&L (net):    {pnl_color}{C.BOLD}${pnl_dollar:+.2f} ({pnl_pct:+.1f}%){C.RESET}")
        print(f"  Result:       {C.GREEN}{C.BOLD}WIN{C.RESET}" if is_win else f"  Result:       {C.RED}{C.BOLD}LOSS{C.RESET}")

        # Show if trailing SL saved money on a loss
        if triggered == "STOP LOSS" and trade["sl_stage"] != "INITIAL":
            if direction == "LONG":
                saved = (final_sl - original_sl) * quantity * leverage
            else:
                saved = (original_sl - final_sl) * quantity * leverage
            if saved > 0:
                print(f"  {C.GREEN}Trailing SL saved: ${saved:,.2f}{C.RESET}")

        # Fetch updated balance after close
        try:
            new_balance = self.executor.get_total_balance()
            balance_color = C.GREEN if is_win else C.RED
            print(f"  Account Balance: {balance_color}{C.BOLD}${new_balance:,.2f} USDT{C.RESET}")
        except Exception:
            pass

        print(f"{C.BOLD}{'=' * 55}{C.RESET}")
        print(f"\n{C.CYAN}{C.BOLD}>{C.RESET} ", end="", flush=True)

        logger.info(f"[Monitor] {triggered} on {symbol} | P&L: ${pnl_dollar:+.2f} ({pnl_pct:+.1f}%) | SL stage: {trade['sl_stage']}")

        # Notify the bot so circuit breaker / daily loss limit actually update
        if self.on_trade_closed:
            try:
                self.on_trade_closed(pnl_dollar)
            except Exception as e:
                logger.error(f"[Monitor] on_trade_closed callback error: {e}")

        # Log to trade tracker
        try:
            from trade_tracker import log_trade
            log_trade(
                coin=symbol, direction=direction,
                entry=entry, exit_price=exit_price,
                sl=final_sl, tp=trade["take_profit"],
                leverage=leverage, capital=capital,
                confidence=trade.get("confidence", 0),
                pattern=triggered,
                notes=f"SL stage: {trade['sl_stage']} | Original SL: ${original_sl:,.4f} | Fees: ${fees:,.2f}",
            )
        except Exception as e:
            logger.error(f"[Monitor] Failed to log trade: {e}")
