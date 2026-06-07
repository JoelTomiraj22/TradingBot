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
    Also checks if current price is near a key level.
    """
    highs = df["high"]
    lows = df["low"]

    # Swing highs: higher than `window` bars on each side
    half = window // 2
    swing_high = pd.Series(False, index=df.index)
    swing_low = pd.Series(False, index=df.index)

    for i in range(half, len(df) - half):
        window_highs = highs.iloc[i - half:i + half + 1]
        window_lows = lows.iloc[i - half:i + half + 1]

        if highs.iloc[i] == window_highs.max():
            swing_high.iloc[i] = True
        if lows.iloc[i] == window_lows.min():
            swing_low.iloc[i] = True

    df.loc[:, "swing_high"] = swing_high
    df.loc[:, "swing_low"] = swing_low

    # Find nearest S/R levels to current price
    last_price = df["close"].iloc[-1]
    atr = df["atr"].iloc[-1] if pd.notna(df["atr"].iloc[-1]) else last_price * 0.01

    resistance_levels = highs[swing_high].values
    support_levels = lows[swing_low].values

    # Nearest resistance (above price)
    above = resistance_levels[resistance_levels > last_price]
    nearest_resistance = float(above.min()) if len(above) > 0 else last_price * 1.05

    # Nearest support (below price)
    below = support_levels[support_levels < last_price]
    nearest_support = float(below.max()) if len(below) > 0 else last_price * 0.95

    # Is price near support or resistance? (within 1 ATR)
    df.loc[:, "near_support"] = abs(last_price - nearest_support) < atr
    df.loc[:, "near_resistance"] = abs(last_price - nearest_resistance) < atr
    df.loc[:, "nearest_support"] = nearest_support
    df.loc[:, "nearest_resistance"] = nearest_resistance


# ─── Volume Profile ───────────────────────────────────────────

def _add_volume_profile(df: pd.DataFrame, bins: int = 20):
    """
    Simple volume profile: find the price level with the most volume (POC).
    High volume at current price = strong level. Low volume = weak, may break.
    """
    recent = df.tail(100)
    if len(recent) < 10:
        df.loc[:, "poc_price"] = df["close"].iloc[-1]
        df.loc[:, "high_volume_node"] = False
        return

    price_range = recent["high"].max() - recent["low"].min()
    if price_range == 0:
        df.loc[:, "poc_price"] = df["close"].iloc[-1]
        df.loc[:, "high_volume_node"] = False
        return

    bin_size = price_range / bins
    vol_profile = {}

    for _, row in recent.iterrows():
        price_bin = round((row["close"] - recent["low"].min()) / bin_size) * bin_size + recent["low"].min()
        vol_profile[price_bin] = vol_profile.get(price_bin, 0) + row["volume"]

    # Point of Control (POC) = price level with highest volume
    poc = max(vol_profile, key=vol_profile.get)
    df.loc[:, "poc_price"] = poc

    # Is current price at a high volume node?
    last_price = df["close"].iloc[-1]
    atr = df["atr"].iloc[-1] if pd.notna(df["atr"].iloc[-1]) else last_price * 0.01
    df.loc[:, "high_volume_node"] = abs(last_price - poc) < atr


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
    from tabulate import tabulate

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
