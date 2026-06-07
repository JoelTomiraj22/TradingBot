"""
Multi-confirmation trading strategy with pattern analysis.
Uses VWAP, S/R levels, candlestick patterns, multi-timeframe,
pullback entries, and volume profile — never chases price.
"""

import pandas as pd
from indicators import add_all_indicators, add_higher_tf_indicators
from fetch_data import fetch_ohlcv


def evaluate_signal(df: pd.DataFrame, htf_data: dict = None) -> dict:
    """
    Evaluate the latest candle for a trade signal using full pattern analysis.

    Args:
        df: OHLCV DataFrame with indicators already added.
        htf_data: Higher timeframe indicator data (from add_higher_tf_indicators)

    Returns:
        dict with: direction, confidence, entry, stop_loss, take_profit,
                   breakeven, leverage, reasons, patterns_found
    """
    if len(df) < 50:
        return _no_trade("Insufficient data (need 50+ candles)")

    row = df.iloc[-1]
    prev = df.iloc[-2]

    # Check for NaN in critical indicators
    required = ["ema_9", "ema_21", "ema_50", "rsi", "macd_histogram", "vol_sma_20", "atr", "vwap"]
    for col in required:
        if pd.isna(row.get(col)):
            return _no_trade(f"Indicator {col} is NaN")

    # ─── SAFETY CHECKS (reject before scoring) ────────────────
    rejection = _safety_checks(row, prev)
    if rejection:
        return rejection

    # ─── SCORE LONG AND SHORT ─────────────────────────────────
    long_score, long_reasons = _check_long(row, prev, htf_data)
    short_score, short_reasons = _check_short(row, prev, htf_data)

    # Pick the stronger signal
    if long_score >= short_score and long_score > 0:
        direction = "LONG"
        confidence = min(long_score, 10)
        reasons = long_reasons
    elif short_score > long_score and short_score > 0:
        direction = "SHORT"
        confidence = min(short_score, 10)
        reasons = short_reasons
    else:
        return _no_trade("No confirmations met")

    if confidence < 6:
        return _no_trade(f"Confidence too low ({confidence}/10) — DON'T TRADE. Need 6+ for a valid setup.", reasons=reasons)

    # ─── QUALITY GATE: Require at least one strong confirmation ──
    # A trade must have at least one of:
    # - A candlestick pattern (engulfing, hammer, pin bar, etc.)
    # - A pullback entry (at EMA or VWAP, not chasing)
    # - Multi-timeframe alignment
    has_candle = any(kw in r for r in reasons for kw in ["Engulfing", "Hammer", "Pin Bar", "Star", "Shooting"])
    has_pullback = any("Pullback" in r for r in reasons)
    has_mtf = any("aligned" in r.lower() for r in reasons)
    has_vwap = any("VWAP" in r for r in reasons if "WARNING" not in r)
    has_sr = any("support" in r.lower() or "resistance" in r.lower() for r in reasons if "WARNING" not in r)

    quality_checks = sum([has_candle, has_pullback, has_mtf, has_vwap, has_sr])
    if quality_checks < 2:
        missing = []
        if not has_candle:
            missing.append("candle pattern")
        if not has_pullback:
            missing.append("pullback entry")
        if not has_mtf:
            missing.append("multi-TF alignment")
        if not has_vwap:
            missing.append("VWAP confirmation")
        reasons.append(f"QUALITY GATE FAILED: Only {quality_checks}/2 quality confirmations. Missing: {', '.join(missing)}")
        return _no_trade(f"Not enough quality confirmations ({quality_checks}/2) — NO TRADE", reasons=reasons)

    entry = float(row["close"])
    atr = float(row["atr"])

    # ─── SCALPING SL/TP — wide enough to avoid wicks ────────────
    # SL: 1.5x ATR (gives room to breathe, avoids wick stops)
    # TP: 3x ATR (2:1 R:R after accounting for fees)
    if direction == "LONG":
        stop_loss = entry - (1.5 * atr)
        take_profit = entry + (3.0 * atr)
    else:
        stop_loss = entry + (1.5 * atr)
        take_profit = entry - (3.0 * atr)

    breakeven = entry * 1.0008
    leverage = _confidence_to_leverage(confidence)

    # ─── SCALPING GATE: Must be at a key level ─────────────────
    # For scalping, never enter in no-man's land.
    # Price must be at a pullback level, S/R zone, or VWAP.
    at_key_level = has_pullback or has_sr or has_vwap
    if not at_key_level:
        reasons.append("REJECTED: Price is not at a key level (no pullback, no S/R, no VWAP). Wait for price to come to you.")
        return _no_trade("Not at a key level — wait for better entry", reasons=reasons)

    return {
        "direction": direction,
        "confidence": confidence,
        "entry": round(entry, 6),
        "stop_loss": round(stop_loss, 6),
        "take_profit": round(take_profit, 6),
        "breakeven": round(breakeven, 6),
        "leverage": leverage,
        "atr": round(atr, 6),
        "reasons": reasons,
    }


def evaluate_with_mtf(symbol: str, timeframe: str = "5m", exchange=None) -> dict:
    """
    Full analysis with multi-timeframe confirmation.
    Fetches both the entry timeframe and a higher timeframe.
    """
    # Entry timeframe data
    df = fetch_ohlcv(symbol, timeframe, 200, exchange)
    df = add_all_indicators(df)

    # Higher timeframe data
    htf_map = {"1m": "5m", "5m": "15m", "15m": "1h", "1h": "4h"}
    htf = htf_map.get(timeframe, "1h")

    try:
        df_htf = fetch_ohlcv(symbol, htf, 200, exchange)
        htf_data = add_higher_tf_indicators(df_htf)
    except Exception:
        htf_data = {"htf_trend": "NEUTRAL", "htf_valid": False}

    signal = evaluate_signal(df, htf_data)
    signal["htf_timeframe"] = htf
    signal["htf_trend"] = htf_data.get("htf_trend", "NEUTRAL")

    return signal


# ─── Safety Checks ─────────────────────────────────────────────

def _safety_checks(row, prev) -> dict:
    """Reject trades that are chasing or overextended."""

    # Don't chase: price overextended from EMA 21
    if row.get("overextended", False):
        return _no_trade(
            "Price overextended from EMA 21 (>2 ATR away) — would be chasing",
            reasons=["REJECTED: Price is too far from EMA 21, likely chasing the move"]
        )

    # Don't enter during a streak of 4+ candles in one direction
    if row.get("chasing_up", False):
        return _no_trade(
            "4+ consecutive bullish candles — chasing upward move",
            reasons=["REJECTED: 4+ green candles in a row — wait for pullback"]
        )
    if row.get("chasing_down", False):
        return _no_trade(
            "4+ consecutive bearish candles — chasing downward move",
            reasons=["REJECTED: 4+ red candles in a row — wait for pullback"]
        )

    return None


# ─── Long Scoring ──────────────────────────────────────────────

def _check_long(row, prev, htf_data=None) -> tuple:
    """Check long entry conditions with full pattern analysis."""
    score = 0
    reasons = []

    # ── 1. TREND (EMA alignment) ───────────────────────────────
    if row.get("ema_cross_up", False):
        score += 2
        reasons.append("EMA 9 crossed above EMA 21 (bullish crossover)")
    elif row["ema_9"] > row["ema_21"]:
        score += 1
        reasons.append("EMA 9 above EMA 21 (trend aligned)")

    if row["close"] > row["ema_50"]:
        score += 1
        reasons.append("Price above EMA 50 (bigger trend bullish)")

    # ── 2. MOMENTUM (RSI + MACD) ───────────────────────────────
    rsi = row["rsi"]
    if 40 <= rsi <= 65:
        score += 1
        reasons.append(f"RSI in sweet spot ({rsi:.1f}) — room to run")

    if row["macd_histogram"] > 0:
        score += 1
        reasons.append("MACD histogram positive (bullish momentum)")
    elif row["macd_histogram"] > prev.get("macd_histogram", 0):
        score += 0.5
        reasons.append("MACD histogram turning positive")

    # ── 3. VOLUME ──────────────────────────────────────────────
    if row["vol_sma_20"] > 0 and row["volume"] > 1.5 * row["vol_sma_20"]:
        vol_ratio = row["volume"] / row["vol_sma_20"]
        score += 1
        reasons.append(f"Volume spike ({vol_ratio:.1f}x average)")

    # ── 4. VWAP CONFIRMATION ───────────────────────────────────
    if row["close"] > row["vwap"]:
        score += 1
        reasons.append(f"Price above VWAP (${row['vwap']:,.2f}) — buyers in control")

    # ── 5. SUPPORT/RESISTANCE ──────────────────────────────────
    if row.get("near_support", False):
        score += 1
        reasons.append(f"Price near support level (${row['nearest_support']:,.2f})")

    if row.get("near_resistance", False):
        score -= 1
        reasons.append(f"WARNING: Near resistance (${row['nearest_resistance']:,.2f}) — risky entry")

    # ── 6. CANDLESTICK PATTERNS ────────────────────────────────
    candle_found = False
    if row.get("bullish_engulfing", False):
        score += 1.5
        reasons.append("Bullish Engulfing candle (strong reversal signal)")
        candle_found = True
    if row.get("hammer", False):
        score += 1.5
        reasons.append("Hammer candle (buyers rejected lower prices)")
        candle_found = True
    if row.get("pin_bar_bull", False) and not candle_found:
        score += 1
        reasons.append("Bullish Pin Bar (long lower wick rejection)")
        candle_found = True
    if row.get("morning_star", False) and not candle_found:
        score += 1.5
        reasons.append("Morning Star pattern (3-candle bullish reversal)")
        candle_found = True

    # Bearish candle patterns penalize long
    if row.get("bearish_engulfing", False) or row.get("shooting_star", False):
        score -= 1
        reasons.append("WARNING: Bearish candle pattern detected — conflicting signal")

    # ── 7. PULLBACK ENTRY ──────────────────────────────────────
    if row.get("pullback_to_ema21", False) or row.get("pullback_to_vwap", False):
        score += 1
        level = "EMA 21" if row.get("pullback_to_ema21") else "VWAP"
        reasons.append(f"Pullback entry at {level} — not chasing")

    # ── 8. VOLUME PROFILE ──────────────────────────────────────
    if row.get("high_volume_node", False):
        score += 0.5
        reasons.append(f"At high volume node (POC: ${row['poc_price']:,.2f}) — strong level")

    # ── 9. MULTI-TIMEFRAME ─────────────────────────────────────
    if htf_data and htf_data.get("htf_valid"):
        htf_trend = htf_data["htf_trend"]
        if htf_trend == "BULLISH":
            score += 1.5
            reasons.append(f"Higher TF trend BULLISH (aligned)")
        elif htf_trend == "BEARISH":
            score -= 2
            reasons.append(f"WARNING: Higher TF trend BEARISH — trading against bigger trend!")
        else:
            reasons.append(f"Higher TF trend NEUTRAL")

    return int(score), reasons


# ─── Short Scoring ─────────────────────────────────────────────

def _check_short(row, prev, htf_data=None) -> tuple:
    """Check short entry conditions with full pattern analysis."""
    score = 0
    reasons = []

    # ── 1. TREND ───────────────────────────────────────────────
    if row.get("ema_cross_down", False):
        score += 2
        reasons.append("EMA 9 crossed below EMA 21 (bearish crossover)")
    elif row["ema_9"] < row["ema_21"]:
        score += 1
        reasons.append("EMA 9 below EMA 21 (trend aligned)")

    if row["close"] < row["ema_50"]:
        score += 1
        reasons.append("Price below EMA 50 (bigger trend bearish)")

    # ── 2. MOMENTUM ────────────────────────────────────────────
    rsi = row["rsi"]
    if 35 <= rsi <= 60:
        score += 1
        reasons.append(f"RSI in sweet spot ({rsi:.1f}) — room to drop")

    if row["macd_histogram"] < 0:
        score += 1
        reasons.append("MACD histogram negative (bearish momentum)")
    elif row["macd_histogram"] < prev.get("macd_histogram", 0):
        score += 0.5
        reasons.append("MACD histogram turning negative")

    # ── 3. VOLUME ──────────────────────────────────────────────
    if row["vol_sma_20"] > 0 and row["volume"] > 1.5 * row["vol_sma_20"]:
        vol_ratio = row["volume"] / row["vol_sma_20"]
        score += 1
        reasons.append(f"Volume spike ({vol_ratio:.1f}x average)")

    # ── 4. VWAP ────────────────────────────────────────────────
    if row["close"] < row["vwap"]:
        score += 1
        reasons.append(f"Price below VWAP (${row['vwap']:,.2f}) — sellers in control")

    # ── 5. SUPPORT/RESISTANCE ──────────────────────────────────
    if row.get("near_resistance", False):
        score += 1
        reasons.append(f"Price near resistance (${row['nearest_resistance']:,.2f}) — rejection zone")

    if row.get("near_support", False):
        score -= 1
        reasons.append(f"WARNING: Near support (${row['nearest_support']:,.2f}) — risky short")

    # ── 6. CANDLESTICK PATTERNS ────────────────────────────────
    candle_found = False
    if row.get("bearish_engulfing", False):
        score += 1.5
        reasons.append("Bearish Engulfing candle (strong reversal signal)")
        candle_found = True
    if row.get("shooting_star", False):
        score += 1.5
        reasons.append("Shooting Star candle (sellers rejected higher prices)")
        candle_found = True
    if row.get("pin_bar_bear", False) and not candle_found:
        score += 1
        reasons.append("Bearish Pin Bar (long upper wick rejection)")
        candle_found = True
    if row.get("evening_star", False) and not candle_found:
        score += 1.5
        reasons.append("Evening Star pattern (3-candle bearish reversal)")
        candle_found = True

    if row.get("bullish_engulfing", False) or row.get("hammer", False):
        score -= 1
        reasons.append("WARNING: Bullish candle pattern detected — conflicting signal")

    # ── 7. PULLBACK ENTRY ──────────────────────────────────────
    if row.get("pullback_to_ema21", False) or row.get("pullback_to_vwap", False):
        score += 1
        level = "EMA 21" if row.get("pullback_to_ema21") else "VWAP"
        reasons.append(f"Pullback entry at {level} — not chasing")

    # ── 8. VOLUME PROFILE ──────────────────────────────────────
    if row.get("high_volume_node", False):
        score += 0.5
        reasons.append(f"At high volume node (POC: ${row['poc_price']:,.2f}) — strong level")

    # ── 9. MULTI-TIMEFRAME ─────────────────────────────────────
    if htf_data and htf_data.get("htf_valid"):
        htf_trend = htf_data["htf_trend"]
        if htf_trend == "BEARISH":
            score += 1.5
            reasons.append(f"Higher TF trend BEARISH (aligned)")
        elif htf_trend == "BULLISH":
            score -= 2
            reasons.append(f"WARNING: Higher TF trend BULLISH — trading against bigger trend!")
        else:
            reasons.append(f"Higher TF trend NEUTRAL")

    return int(score), reasons


# ─── Exit Check ────────────────────────────────────────────────

def check_exit(df: pd.DataFrame, direction: str, entry_price: float,
               stop_loss: float, take_profit: float) -> dict:
    """Check if any exit condition is met for an open position."""
    row = df.iloc[-1]
    price = float(row["close"])

    if direction == "LONG" and price <= stop_loss:
        return {"should_exit": True, "reason": "Stop loss hit", "current_price": price}
    if direction == "SHORT" and price >= stop_loss:
        return {"should_exit": True, "reason": "Stop loss hit", "current_price": price}
    if direction == "LONG" and price >= take_profit:
        return {"should_exit": True, "reason": "Take profit hit", "current_price": price}
    if direction == "SHORT" and price <= take_profit:
        return {"should_exit": True, "reason": "Take profit hit", "current_price": price}

    if direction == "LONG" and row.get("ema_cross_down", False):
        return {"should_exit": True, "reason": "EMA 9/21 bearish crossover", "current_price": price}
    if direction == "SHORT" and row.get("ema_cross_up", False):
        return {"should_exit": True, "reason": "EMA 9/21 bullish crossover", "current_price": price}

    rsi = row.get("rsi")
    if rsi is not None and not pd.isna(rsi):
        if direction == "LONG" and rsi > 75:
            return {"should_exit": True, "reason": f"RSI overbought ({rsi:.1f})", "current_price": price}
        if direction == "SHORT" and rsi < 25:
            return {"should_exit": True, "reason": f"RSI oversold ({rsi:.1f})", "current_price": price}

    return {"should_exit": False, "reason": None, "current_price": price}


# ─── Helpers ───────────────────────────────────────────────────

def _confidence_to_leverage(confidence: int) -> int:
    """Must match risk_manager.get_leverage_for_confidence."""
    if confidence <= 5:
        return 0
    elif confidence == 6:
        return 5
    elif confidence == 7:
        return 10
    elif confidence == 8:
        return 15
    else:
        return 20  # Hard cap


def _no_trade(reason: str, reasons: list = None) -> dict:
    return {
        "direction": "NO TRADE",
        "confidence": 0,
        "entry": None,
        "stop_loss": None,
        "take_profit": None,
        "breakeven": None,
        "leverage": 0,
        "atr": None,
        "reasons": reasons or [reason],
    }


if __name__ == "__main__":
    signal = evaluate_with_mtf("BTC/USDT", "5m")
    print(f"\nSignal: {signal['direction']}  |  Confidence: {signal['confidence']}/10")
    if signal.get("htf_trend"):
        print(f"Higher TF ({signal.get('htf_timeframe', '?')}): {signal['htf_trend']}")
    print(f"\nAnalysis:")
    for r in signal["reasons"]:
        print(f"  {'*' if 'WARNING' in r else '+'} {r}")
