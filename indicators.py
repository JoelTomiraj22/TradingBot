"""
Technical indicators module.
Adds trend, momentum, volatility, VWAP, S/R levels, candlestick patterns,
and volume profile to OHLCV DataFrames.
"""

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import pandas as pd
import numpy as np
import ta


def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add all technical indicators to an OHLCV DataFrame."""
    df = df.copy()
    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    # ─── EMA (9, 21, 50, 200) ──────────────────────────────────
    df.loc[:, "ema_9"] = ta.trend.ema_indicator(close, window=9)
    df.loc[:, "ema_21"] = ta.trend.ema_indicator(close, window=21)
    df.loc[:, "ema_50"] = ta.trend.ema_indicator(close, window=50)
    df.loc[:, "ema_200"] = ta.trend.ema_indicator(close, window=200)

    # ─── RSI (14) ───────────────────────────────────────────────
    df.loc[:, "rsi"] = ta.momentum.rsi(close, window=14)

    # ─── MACD (12, 26, 9) ──────────────────────────────────────
    macd = ta.trend.MACD(close, window_slow=26, window_fast=12, window_sign=9)
    df.loc[:, "macd"] = macd.macd()
    df.loc[:, "macd_signal"] = macd.macd_signal()
    df.loc[:, "macd_histogram"] = macd.macd_diff()

    # ─── Bollinger Bands (20, 2) ────────────────────────────────
    bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
    df.loc[:, "bb_upper"] = bb.bollinger_hband()
    df.loc[:, "bb_middle"] = bb.bollinger_mavg()
    df.loc[:, "bb_lower"] = bb.bollinger_lband()

    # ─── Volume SMA (20) ───────────────────────────────────────
    df.loc[:, "vol_sma_20"] = volume.rolling(window=20).mean()

    # ─── ATR (14) ──────────────────────────────────────────────
    df.loc[:, "atr"] = ta.volatility.average_true_range(high, low, close, window=14)

    # ─── VWAP (Volume Weighted Average Price) ──────────────────
    df.loc[:, "vwap"] = _calculate_vwap(df)

    # ─── EMA Crossover helpers ─────────────────────────────────
    df.loc[:, "ema_9_prev"] = df["ema_9"].shift(1)
    df.loc[:, "ema_21_prev"] = df["ema_21"].shift(1)
    df.loc[:, "ema_cross_up"] = (df["ema_9_prev"] < df["ema_21_prev"]) & (df["ema_9"] > df["ema_21"])
    df.loc[:, "ema_cross_down"] = (df["ema_9_prev"] > df["ema_21_prev"]) & (df["ema_9"] < df["ema_21"])

    # ─── Candlestick Patterns ──────────────────────────────────
    _add_candle_patterns(df)

    # ─── Support / Resistance Levels ───────────────────────────
    _add_support_resistance(df)

    # ─── Volume Profile ────────────────────────────────────────
    _add_volume_profile(df)

    # ─── Pullback Detection ────────────────────────────────────
    _add_pullback_detection(df)

    # ─── Compression bars (Inside Bar / NR7) ───────────────────
    rng = df["high"] - df["low"]
    df.loc[:, "inside_bar"] = (df["high"] <= df["high"].shift(1)) & (df["low"] >= df["low"].shift(1))
    df.loc[:, "nr7"] = rng <= rng.rolling(7).min()

    # ─── Three White Soldiers / Three Black Crows ──────────────
    body = (df["close"] - df["open"])
    strong = body.abs() >= (rng * 0.55)
    green = body > 0
    red = body < 0
    higher_close = df["close"] > df["close"].shift(1)
    lower_close = df["close"] < df["close"].shift(1)
    df.loc[:, "three_white_soldiers"] = (
        green & green.shift(1) & green.shift(2)
        & strong & strong.shift(1) & strong.shift(2)
        & higher_close & higher_close.shift(1)
    )
    df.loc[:, "three_black_crows"] = (
        red & red.shift(1) & red.shift(2)
        & strong & strong.shift(1) & strong.shift(2)
        & lower_close & lower_close.shift(1)
    )

    return df


def add_higher_tf_indicators(df_htf: pd.DataFrame) -> dict:
    """
    Calculate key indicators on a higher timeframe (15m or 1h).
    Returns a dict of the latest values for multi-timeframe confirmation.
    """
    if len(df_htf) < 50:
        return {"htf_trend": "NEUTRAL", "htf_valid": False}

    close = df_htf["close"]
    ema_21 = ta.trend.ema_indicator(close, window=21)
    ema_50 = ta.trend.ema_indicator(close, window=50)
    rsi = ta.momentum.rsi(close, window=14)

    latest = df_htf.iloc[-1]
    ema21_val = ema_21.iloc[-1]
    ema50_val = ema_50.iloc[-1]
    rsi_val = rsi.iloc[-1]

    # Determine higher timeframe trend
    if latest["close"] > ema21_val > ema50_val:
        htf_trend = "BULLISH"
    elif latest["close"] < ema21_val < ema50_val:
        htf_trend = "BEARISH"
    else:
        htf_trend = "NEUTRAL"

    return {
        "htf_trend": htf_trend,
        "htf_ema_21": round(ema21_val, 6),
        "htf_ema_50": round(ema50_val, 6),
        "htf_rsi": round(rsi_val, 2),
        "htf_valid": True,
    }


def classify_trend(row) -> str:
    """
    Classify a single row's trend from EMA 9/21/50 alignment:
    "STRONG BULL" / "BULL" / "STRONG BEAR" / "BEAR" / "NEUTRAL".
    """
    ema9 = row.get("ema_9")
    ema21 = row.get("ema_21")
    ema50 = row.get("ema_50")
    if ema9 is None or ema21 is None or ema50 is None:
        return "NEUTRAL"
    if pd.isna(ema9) or pd.isna(ema21) or pd.isna(ema50):
        return "NEUTRAL"

    if ema9 > ema21 > ema50:
        return "STRONG BULL"
    if ema9 > ema21:
        return "BULL"
    if ema9 < ema21 < ema50:
        return "STRONG BEAR"
    if ema9 < ema21:
        return "BEAR"
    return "NEUTRAL"


# Timeframes considered for the scan_all_coins() pre-filter, with weights
# favoring higher TFs (a 1h trend is more meaningful than a 1m blip).
PREFILTER_TFS = ["1m", "3m", "5m", "15m", "30m", "1h"]
PREFILTER_TF_WEIGHTS = {"1h": 3, "30m": 2.5, "15m": 2, "5m": 1.5, "3m": 1, "1m": 1}


def score_multi_tf_setup(tf_data: dict) -> dict:
    """
    Score a coin's setup across multiple timeframes for scan_all_coins()'s
    pre-filter:

    1. Weighted trend alignment across PREFILTER_TFS (higher TFs weigh more).
    2. Pattern/structure score on the 5m entry TF (candle patterns, S/R,
       VWAP/POC, pullback, streaks, RSI), directional to the dominant trend.

    Args:
        tf_data: dict of {timeframe: DataFrame}, each with indicators already
                 added via add_all_indicators().

    Returns:
        {"dominant": "BULL"/"BEAR"/"NEUTRAL", "alignment_ratio": float,
         "pattern_score": float, "tf_trends": {tf: label}}
    """
    tf_trends = {}
    bull_weight = 0.0
    bear_weight = 0.0
    total_weight = 0.0

    for tf in PREFILTER_TFS:
        df = tf_data.get(tf)
        if df is None or len(df) < 1:
            continue
        trend = classify_trend(df.iloc[-1])
        tf_trends[tf] = trend

        weight = PREFILTER_TF_WEIGHTS[tf]
        total_weight += weight
        if trend in ("BULL", "STRONG BULL"):
            bull_weight += weight
        elif trend in ("BEAR", "STRONG BEAR"):
            bear_weight += weight

    if total_weight == 0:
        return {"dominant": "NEUTRAL", "alignment_ratio": 0.0, "pattern_score": 0.0, "tf_trends": tf_trends}

    if bull_weight > bear_weight:
        dominant = "BULL"
        alignment_ratio = bull_weight / total_weight
    elif bear_weight > bull_weight:
        dominant = "BEAR"
        alignment_ratio = bear_weight / total_weight
    else:
        dominant = "NEUTRAL"
        alignment_ratio = bull_weight / total_weight

    # Pattern/structure score on the 5m entry TF, directional to `dominant`.
    pattern_score = 0.0
    df_5m = tf_data.get("5m")
    if dominant != "NEUTRAL" and df_5m is not None and len(df_5m) > 0:
        row = df_5m.iloc[-1]

        bull_patterns = ("bullish_engulfing", "hammer", "pin_bar_bull", "morning_star")
        bear_patterns = ("bearish_engulfing", "shooting_star", "pin_bar_bear", "evening_star")
        patterns = bull_patterns if dominant == "BULL" else bear_patterns
        if any(row.get(p, False) for p in patterns):
            pattern_score += 1

        if dominant == "BULL" and row.get("near_support", False):
            pattern_score += 1
        elif dominant == "BEAR" and row.get("near_resistance", False):
            pattern_score += 1

        if row.get("pullback_to_ema21", False) or row.get("pullback_to_vwap", False):
            pattern_score += 1

        if row.get("high_volume_node", False):
            pattern_score += 0.5

        streak = row.get("bull_streak", 0) if dominant == "BULL" else row.get("bear_streak", 0)
        if not pd.isna(streak) and streak >= 3:
            pattern_score += 0.5

        if row.get("overextended", False):
            pattern_score -= 1

        rsi = row.get("rsi", 50)
        if not pd.isna(rsi):
            pattern_score += 0.5 if 25 <= rsi <= 75 else -1

    return {
        "dominant": dominant,
        "alignment_ratio": round(alignment_ratio, 3),
        "pattern_score": pattern_score,
        "tf_trends": tf_trends,
    }


def check_volatility_spike(df: pd.DataFrame, multiplier: float = 2.5) -> dict:
    """
    Detect an abnormal volatility spike on the most recent candle by comparing
    its high-low range to the rolling ATR. Used as a proxy for "something just
    happened" (news/liquidation cascade) when no news-calendar API is available.

    Returns: {"spike": bool, "ratio": float}
    """
    if df is None or len(df) < 20:
        return {"spike": False, "ratio": 0.0}

    row = df.iloc[-1]
    atr = row.get("atr", 0)
    if atr is None or pd.isna(atr) or atr <= 0:
        return {"spike": False, "ratio": 0.0}

    candle_range = row["high"] - row["low"]
    ratio = candle_range / atr

    return {"spike": ratio >= multiplier, "ratio": round(ratio, 2)}


# ─── Setup detection (scalping/intraday pattern playbook) ─────
# Deterministic detectors for the tiered playbook: flags (S), S/R flip (S),
# double bottom/top (A), volatility contraction (A), dead cat bounce (A),
# inside bar/NR7 squeeze (B), VWAP reclaim/rejection (B), soldiers/crows.
# All operate on the LAST bar using only past data — safe for backtests.

def detect_setups(df: pd.DataFrame) -> dict:
    """Detect multi-bar trade setups as of the latest candle."""
    out = {
        "bull_flag": False, "bear_flag": False, "flag_breakout_level": None,
        "vol_contraction": False,
        "sr_flip_support": False, "sr_flip_resistance": False,
        "double_bottom": False, "double_top": False, "neckline": None,
        "vwap_reclaim": False, "vwap_rejection": False,
        "dead_cat_bounce": False, "dcb_bounce_high": None,
        "three_white_soldiers": False, "three_black_crows": False,
        "inside_bar": False, "nr7": False,
        "volume_confirmed": False,
    }
    if df is None or len(df) < 60:
        return out

    row = df.iloc[-1]
    atr = row.get("atr")
    if atr is None or pd.isna(atr) or atr <= 0:
        return out

    tail = df.iloc[-120:] if len(df) > 120 else df
    closes = tail["close"].to_numpy()
    highs = tail["high"].to_numpy()
    lows = tail["low"].to_numpy()
    vols = tail["volume"].to_numpy()
    n = len(closes)
    vol_sma = row.get("vol_sma_20", 0)
    if pd.isna(vol_sma):
        vol_sma = 0

    # Simple last-bar flags from precomputed columns
    for col in ("inside_bar", "nr7", "three_white_soldiers", "three_black_crows"):
        v = row.get(col, False)
        out[col] = bool(v) if not pd.isna(v) else False
    out["volume_confirmed"] = bool(vol_sma > 0 and row["volume"] > 1.3 * vol_sma)

    # ── Bull/Bear flag: impulse pole + tight low-volume consolidation ──
    for cons_len in range(3, 9):
        if n < cons_len + 10:
            break
        cons_h = highs[-cons_len:]
        cons_l = lows[-cons_len:]
        cons_v = vols[-cons_len:]
        pole_close_end = closes[-cons_len - 1]
        pole_close_start = closes[-cons_len - 9]
        pole_v = vols[-cons_len - 9:-cons_len]
        pole_move = pole_close_end - pole_close_start
        if abs(pole_move) < 2.2 * atr:
            continue
        cons_range = cons_h.max() - cons_l.min()
        if cons_range > 0.45 * abs(pole_move):
            continue
        if pole_v.mean() <= 0 or cons_v.mean() >= 0.85 * pole_v.mean():
            continue
        # retrace must stay shallow (≤ 50% of pole)
        if pole_move > 0:
            if closes[-1] < pole_close_end - 0.5 * pole_move:
                continue
            out["bull_flag"] = True
            out["flag_breakout_level"] = float(cons_h.max())
        else:
            if closes[-1] > pole_close_end - 0.5 * pole_move:
                continue
            out["bear_flag"] = True
            out["flag_breakout_level"] = float(cons_l.min())
        break

    # ── Volatility contraction (VCP-lite): ranges + volume drying up ──
    rng = highs - lows
    if n >= 35:
        r5, r15, r35 = rng[-5:].mean(), rng[-15:-5].mean(), rng[-35:-15].mean()
        v5, v20 = vols[-5:].mean(), vols[-20:].mean()
        out["vol_contraction"] = bool(r5 < 0.75 * r15 < 0.75 * 1.34 * r35 and v5 < 0.85 * v20)

    # ── S/R flip: broken level retested from the other side ──
    swing_high = tail.get("swing_high")
    swing_low = tail.get("swing_low")
    price = closes[-1]
    if swing_high is not None and swing_low is not None:
        sh = swing_high.to_numpy()
        sl_ = swing_low.to_numpy()
        # resistance levels confirmed at j+10 (window half = 10)
        half = 10
        for j in range(n - half - 2, max(0, n - 60), -1):
            if j + half >= n - 1:
                continue
            if sh[j]:
                level = highs[j]
                broke = (closes[j + half:n - 1] > level + 0.2 * atr).any()
                retest = lows[-1] <= level + 0.6 * atr and price > level
                if broke and retest and abs(price - level) <= 1.2 * atr:
                    out["sr_flip_support"] = True
                    break
        for j in range(n - half - 2, max(0, n - 60), -1):
            if j + half >= n - 1:
                continue
            if sl_[j]:
                level = lows[j]
                broke = (closes[j + half:n - 1] < level - 0.2 * atr).any()
                retest = highs[-1] >= level - 0.6 * atr and price < level
                if broke and retest and abs(price - level) <= 1.2 * atr:
                    out["sr_flip_resistance"] = True
                    break

        # ── Double bottom / top from confirmed swings ──
        sw_lows = [(j, lows[j]) for j in range(max(0, n - 70), n - half) if sl_[j]]
        if len(sw_lows) >= 2:
            (j1, l1), (j2, l2) = sw_lows[-2], sw_lows[-1]
            if j2 - j1 >= 5 and abs(l2 - l1) / max(l1, 1e-12) <= 0.006 and l2 >= l1 * 0.999:
                neckline = float(highs[j1:j2 + 1].max())
                if price > l2 and price >= neckline - 1.0 * atr:
                    out["double_bottom"] = True
                    out["neckline"] = neckline
        sw_highs = [(j, highs[j]) for j in range(max(0, n - 70), n - half) if sh[j]]
        if len(sw_highs) >= 2:
            (j1, h1), (j2, h2) = sw_highs[-2], sw_highs[-1]
            if j2 - j1 >= 5 and abs(h2 - h1) / max(h1, 1e-12) <= 0.006 and h2 <= h1 * 1.001:
                neckline = float(lows[j1:j2 + 1].min())
                if price < h2 and price <= neckline + 1.0 * atr:
                    out["double_top"] = True
                    out["neckline"] = neckline

    # ── VWAP reclaim / rejection (with RSI 50 filter) ──
    vwap = row.get("vwap")
    rsi = row.get("rsi", 50)
    if vwap is not None and not pd.isna(vwap) and not pd.isna(rsi):
        recent_vwaps = tail["vwap"].to_numpy()[-7:-1]
        recent_closes = closes[-7:-1]
        # Require a MEANINGFUL excursion (>=0.3 ATR beyond VWAP) before the
        # reclaim/rejection — noise wiggles around VWAP are not a setup.
        was_below = (recent_closes < recent_vwaps - 0.3 * atr).any()
        was_above = (recent_closes > recent_vwaps + 0.3 * atr).any()
        if was_below and price > vwap + 0.05 * atr and rsi > 50:
            out["vwap_reclaim"] = True
        if was_above and price < vwap - 0.05 * atr and rsi < 50:
            out["vwap_rejection"] = True

    # ── Dead cat bounce: high-volume dump, weak low-volume bounce ──
    if n >= 40 and vol_sma > 0:
        trough_i = int(lows[-15:].argmin()) + (n - 15)
        if trough_i < n - 1:  # bounce must have started
            pre = closes[max(0, trough_i - 10):trough_i]
            if len(pre) >= 3:
                peak = float(pre.max())
                trough = float(lows[trough_i])
                drop = peak - trough
                if drop >= 3 * atr:
                    drop_vol = vols[max(0, trough_i - 10):trough_i + 1].mean()
                    bounce_vol = vols[trough_i + 1:].mean() if trough_i + 1 < n else 0
                    retrace = (price - trough) / drop if drop > 0 else 0
                    if (drop_vol > 1.1 * vol_sma and bounce_vol < 0.7 * drop_vol
                            and 0.15 <= retrace <= 0.6 and rsi < 50):
                        out["dead_cat_bounce"] = True
                        out["dcb_bounce_high"] = float(highs[trough_i + 1:].max())

    return out


def estimate_eta_minutes(df: pd.DataFrame, entry: float, take_profit: float,
                         timeframe_minutes: int):
    """
    Deterministic time-to-TP estimate band, independent of the AI:
    candles_needed ~= TP distance / ATR, optimistic bound at 1x ATR per
    candle of favorable movement, conservative bound at 2.5x (price rarely
    moves straight to target).

    Returns (min_minutes, max_minutes) or None if not computable.
    """
    if df is None or len(df) < 20 or not entry or not take_profit:
        return None

    atr = df["atr"].iloc[-1]
    if atr is None or pd.isna(atr) or atr <= 0:
        return None

    dist = abs(take_profit - entry)
    if dist <= 0:
        return None

    candles = max(1.0, dist / atr)
    lo = max(timeframe_minutes, int(round(candles * timeframe_minutes)))
    hi = int(round(candles * 2.5 * timeframe_minutes))
    return (lo, max(hi, lo))


def _is_round_number(price: float, tol_pct: float = 0.05) -> bool:
    """True if price sits within tol_pct of a 1–2 significant-digit round
    level (e.g. 50000, 49500, 0.10, 3.5) — classic liquidity magnets."""
    import math
    if not price or price <= 0:
        return False
    exp = math.floor(math.log10(price))
    for digits in (1, 2):
        scale = 10 ** (exp - digits + 1)
        rounded = round(price / scale) * scale
        if rounded > 0 and abs(price - rounded) / price * 100 <= tol_pct:
            return True
    return False


def check_sl_hunt_risk(df: pd.DataFrame, direction: str, stop_loss: float,
                       entry: float, lookback: int = 15,
                       buffer_pct: float = 0.15) -> dict:
    """
    Assess whether a proposed stop loss is likely to get stop-hunted:
    1. SL sits INSIDE the recent wick cluster — if recent candles' wicks
       already reached the SL level, ordinary noise will tag it again.
    2. SL parked at/near a round number — a classic liquidity target.

    Returns {"risky": bool, "reasons": [...], "suggested_sl": float}
    where suggested_sl sits beyond the recent wick extreme with a buffer.
    """
    out = {"risky": False, "reasons": [], "suggested_sl": stop_loss}
    if df is None or len(df) < lookback or not stop_loss or not entry:
        return out

    recent = df.iloc[-lookback:]
    buf = buffer_pct / 100

    if direction == "LONG":
        wick_extreme = float(recent["low"].min())
        pierced = int((recent["low"] <= stop_loss).sum())
        if pierced > 0:
            out["risky"] = True
            out["reasons"].append(
                f"{pierced} of the last {lookback} candles wicked to/below the proposed SL "
                f"— it sits inside the liquidity zone"
            )
            out["suggested_sl"] = round(wick_extreme * (1 - buf), 6)
    else:
        wick_extreme = float(recent["high"].max())
        pierced = int((recent["high"] >= stop_loss).sum())
        if pierced > 0:
            out["risky"] = True
            out["reasons"].append(
                f"{pierced} of the last {lookback} candles wicked to/above the proposed SL "
                f"— it sits inside the liquidity zone"
            )
            out["suggested_sl"] = round(wick_extreme * (1 + buf), 6)

    if _is_round_number(stop_loss):
        out["risky"] = True
        out["reasons"].append("SL is at/near a round-number level — classic stop-hunt target")
        # Nudge beyond the round number if the wick check didn't already widen it
        if direction == "LONG":
            out["suggested_sl"] = round(min(out["suggested_sl"], stop_loss * (1 - buf)), 6)
        else:
            out["suggested_sl"] = round(max(out["suggested_sl"], stop_loss * (1 + buf)), 6)

    return out


def check_liquidity(df: pd.DataFrame, min_dollar_volume: float) -> dict:
    """
    Estimate average $ volume per candle from vol_sma_20 * close, used to
    filter out illiquid coins before sending them to the AI / trading them.

    Returns: {"liquid": bool, "dollar_volume": float}
    """
    if df is None or len(df) < 20:
        return {"liquid": False, "dollar_volume": 0.0}

    row = df.iloc[-1]
    vol_sma = row.get("vol_sma_20", 0)
    close = row.get("close", 0)
    if vol_sma is None or pd.isna(vol_sma) or close <= 0:
        return {"liquid": False, "dollar_volume": 0.0}

    dollar_volume = vol_sma * close
    return {"liquid": dollar_volume >= min_dollar_volume, "dollar_volume": round(dollar_volume, 2)}


# ─── VWAP ──────────────────────────────────────────────────────

def _calculate_vwap(df: pd.DataFrame) -> pd.Series:
    """Calculate VWAP (resets each session, approximated with rolling 78 bars for 5m = ~6.5h)."""
    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    tp_volume = typical_price * df["volume"]

    # Rolling VWAP over ~6.5 hours (78 bars on 5m)
    window = min(78, len(df))
    cum_tp_vol = tp_volume.rolling(window=window, min_periods=1).sum()
    cum_vol = df["volume"].rolling(window=window, min_periods=1).sum()

    vwap = cum_tp_vol / cum_vol
    return vwap


# ─── Candlestick Patterns ─────────────────────────────────────

def _add_candle_patterns(df: pd.DataFrame):
    """Detect key candlestick patterns."""
    o = df["open"]
    h = df["high"]
    l = df["low"]
    c = df["close"]
    body = abs(c - o)
    candle_range = h - l

    # Prevent division by zero
    safe_range = candle_range.replace(0, np.nan)

    # Bullish Engulfing: current green candle fully engulfs previous red candle
    prev_bearish = (o.shift(1) > c.shift(1))
    curr_bullish = (c > o)
    engulfs = (o <= c.shift(1)) & (c >= o.shift(1))
    df.loc[:, "bullish_engulfing"] = prev_bearish & curr_bullish & engulfs

    # Bearish Engulfing: current red candle fully engulfs previous green candle
    prev_bullish = (c.shift(1) > o.shift(1))
    curr_bearish = (o > c)
    engulfs_bear = (o >= c.shift(1)) & (c <= o.shift(1))
    df.loc[:, "bearish_engulfing"] = prev_bullish & curr_bearish & engulfs_bear

    # Hammer (bullish): small body at top, long lower wick (2x+ body)
    lower_wick = pd.concat([o, c], axis=1).min(axis=1) - l
    upper_wick = h - pd.concat([o, c], axis=1).max(axis=1)
    df.loc[:, "hammer"] = (lower_wick > 2 * body) & (upper_wick < body * 0.5) & (body > 0)

    # Shooting Star (bearish): small body at bottom, long upper wick
    df.loc[:, "shooting_star"] = (upper_wick > 2 * body) & (lower_wick < body * 0.5) & (body > 0)

    # Doji: very small body relative to range
    df.loc[:, "doji"] = body < (safe_range * 0.1)

    # Pin Bar Bullish: long lower wick, close near high
    df.loc[:, "pin_bar_bull"] = (lower_wick > 2.5 * body) & (c > (h + l) / 2)

    # Pin Bar Bearish: long upper wick, close near low
    df.loc[:, "pin_bar_bear"] = (upper_wick > 2.5 * body) & (c < (h + l) / 2)

    # Morning Star (3-candle bullish reversal)
    candle1_bear = o.shift(2) > c.shift(2)
    candle2_small = body.shift(1) < body.shift(2) * 0.3
    candle3_bull = (c > o) & (c > (o.shift(2) + c.shift(2)) / 2)
    df.loc[:, "morning_star"] = candle1_bear & candle2_small & candle3_bull

    # Evening Star (3-candle bearish reversal)
    candle1_bull = c.shift(2) > o.shift(2)
    candle2_small_e = body.shift(1) < body.shift(2) * 0.3
    candle3_bear = (o > c) & (c < (o.shift(2) + c.shift(2)) / 2)
    df.loc[:, "evening_star"] = candle1_bull & candle2_small_e & candle3_bear


# ─── Support / Resistance ─────────────────────────────────────

def _add_support_resistance(df: pd.DataFrame, window: int = 20):
    """
    Detect support and resistance levels using swing highs/lows.

    Point-in-time correct: each row's nearest support/resistance uses only
    swings CONFIRMED by that row (a swing at bar j needs `half` bars on each
    side, so it's only known at bar j + half). This removes the look-ahead
    bias that previously broadcast the last candle's levels to every row,
    making backtest results match what the live bot could actually know.
    """
    import bisect

    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()
    closes = df["close"].to_numpy()
    atrs = df["atr"].to_numpy()
    n = len(df)
    half = window // 2

    swing_high = np.zeros(n, dtype=bool)
    swing_low = np.zeros(n, dtype=bool)
    for i in range(half, n - half):
        if highs[i] == highs[i - half:i + half + 1].max():
            swing_high[i] = True
        if lows[i] == lows[i - half:i + half + 1].min():
            swing_low[i] = True

    df.loc[:, "swing_high"] = swing_high
    df.loc[:, "swing_low"] = swing_low

    nearest_support = np.empty(n)
    nearest_resistance = np.empty(n)
    near_support = np.zeros(n, dtype=bool)
    near_resistance = np.zeros(n, dtype=bool)

    res_levels: list = []   # sorted swing-high values confirmed so far
    sup_levels: list = []   # sorted swing-low values confirmed so far

    for i in range(n):
        # A swing centered at j = i - half is confirmed exactly at bar i
        j = i - half
        if j >= 0:
            if swing_high[j]:
                bisect.insort(res_levels, highs[j])
            if swing_low[j]:
                bisect.insort(sup_levels, lows[j])

        price = closes[i]
        atr = atrs[i] if not np.isnan(atrs[i]) else price * 0.01

        # Nearest resistance strictly above price
        k = bisect.bisect_right(res_levels, price)
        nearest_resistance[i] = res_levels[k] if k < len(res_levels) else price * 1.05

        # Nearest support strictly below price
        k = bisect.bisect_left(sup_levels, price)
        nearest_support[i] = sup_levels[k - 1] if k > 0 else price * 0.95

        near_support[i] = abs(price - nearest_support[i]) < atr
        near_resistance[i] = abs(price - nearest_resistance[i]) < atr

    df.loc[:, "near_support"] = near_support
    df.loc[:, "near_resistance"] = near_resistance
    df.loc[:, "nearest_support"] = nearest_support
    df.loc[:, "nearest_resistance"] = nearest_resistance


# ─── Volume Profile ───────────────────────────────────────────

def _add_volume_profile(df: pd.DataFrame, bins: int = 20, lookback: int = 100):
    """
    Simple volume profile: find the price level with the most volume (POC).
    High volume at current price = strong level. Low volume = weak, may break.

    Point-in-time correct: each row's POC is computed from the `lookback`
    bars ENDING at that row (no future data), so backtests see the same
    levels the live bot would have seen.
    """
    closes = df["close"].to_numpy()
    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()
    volumes = df["volume"].to_numpy()
    atrs = df["atr"].to_numpy()
    n = len(df)

    poc_price = np.empty(n)
    high_volume_node = np.zeros(n, dtype=bool)

    for i in range(n):
        start = max(0, i - lookback + 1)
        if i - start + 1 < 10:
            poc_price[i] = closes[i]
            continue

        lo = lows[start:i + 1].min()
        hi = highs[start:i + 1].max()
        if hi <= lo:
            poc_price[i] = closes[i]
            continue

        hist, edges = np.histogram(
            closes[start:i + 1], bins=bins, range=(lo, hi),
            weights=volumes[start:i + 1],
        )
        k = int(hist.argmax())
        poc = (edges[k] + edges[k + 1]) / 2
        poc_price[i] = poc

        atr = atrs[i] if not np.isnan(atrs[i]) else closes[i] * 0.01
        high_volume_node[i] = abs(closes[i] - poc) < atr

    df.loc[:, "poc_price"] = poc_price
    df.loc[:, "high_volume_node"] = high_volume_node


# ─── Pullback Detection ───────────────────────────────────────

def _add_pullback_detection(df: pd.DataFrame):
    """
    Detect if price is pulling back to a key level (EMA/VWAP)
    rather than chasing a move.
    """
    close = df["close"]
    atr = df["atr"]

    # Price pulling back to EMA 21 (within 0.5 ATR)
    dist_to_ema21 = abs(close - df["ema_21"])
    df.loc[:, "pullback_to_ema21"] = dist_to_ema21 < (atr * 0.5)

    # Price pulling back to VWAP (within 0.5 ATR)
    dist_to_vwap = abs(close - df["vwap"])
    df.loc[:, "pullback_to_vwap"] = dist_to_vwap < (atr * 0.5)

    # Is price overextended from EMA 21? (more than 2 ATR away)
    df.loc[:, "overextended"] = dist_to_ema21 > (atr * 2)

    # Consecutive candles in same direction (chasing detection)
    bullish = (close > df["open"]).astype(int)
    bearish = (close < df["open"]).astype(int)

    # Count consecutive bullish/bearish candles
    bull_streak = bullish.groupby((bullish != bullish.shift()).cumsum()).cumsum()
    bear_streak = bearish.groupby((bearish != bearish.shift()).cumsum()).cumsum()
    df.loc[:, "bull_streak"] = bull_streak
    df.loc[:, "bear_streak"] = bear_streak

    # Chasing = 4+ consecutive candles in one direction
    df.loc[:, "chasing_up"] = bull_streak >= 4
    df.loc[:, "chasing_down"] = bear_streak >= 4


if __name__ == "__main__":
    from fetch_data import fetch_ohlcv

    df = fetch_ohlcv("BTC/USDT", "5m", 200)
    df = add_all_indicators(df)

    row = df.iloc[-1]
    print("Latest Indicators:")
    print(f"  EMA 9/21:        {row['ema_9']:.2f} / {row['ema_21']:.2f}")
    print(f"  RSI:             {row['rsi']:.1f}")
    print(f"  VWAP:            {row['vwap']:.2f}")
    print(f"  Near Support:    {row['near_support']}")
    print(f"  Near Resistance: {row['near_resistance']}")
    print(f"  S/R Levels:      S={row['nearest_support']:.2f}  R={row['nearest_resistance']:.2f}")
    print(f"  POC (Vol Prof):  {row['poc_price']:.2f}")
    print(f"  Pullback EMA21:  {row['pullback_to_ema21']}")
    print(f"  Pullback VWAP:   {row['pullback_to_vwap']}")
    print(f"  Overextended:    {row['overextended']}")
    print(f"  Bull Engulfing:  {row['bullish_engulfing']}")
    print(f"  Bear Engulfing:  {row['bearish_engulfing']}")
    print(f"  Hammer:          {row['hammer']}")
    print(f"  Shooting Star:   {row['shooting_star']}")
