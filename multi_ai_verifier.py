"""
Multi-AI Trade Analyzer — AI makes ALL trade decisions.

Flow:
  1. Check which AI providers are reachable (parallel ping)
  2. Bot fetches OHLCV data + calculates indicators
  3. Bot sends raw indicator snapshot to the best available AIs
  4. AI analyzes patterns, determines direction, confidence, SL/TP
  5. Risk manager validates AI's output
  6. User confirms → Bot executes

Provider priority (falls back automatically):
  1. DeepSeek V4 Pro (NVIDIA NIM) — primary reasoning analyst
     (model fallbacks: Llama 3.3 70B, Nemotron Ultra 253B)
  2. Qwen 3.5 397B (NVIDIA NIM)   — fallback
     (model fallbacks: Nemotron Ultra 253B, Llama 3.3 70B)
  3. Gemini 2.5 Pro -> Flash      — key rotation across GEMINI_PRO_API_KEYS + GEMINI_API_KEY
  4. OpenRouter free pool         — deepseek-r1:free / deepseek-v3:free / llama-3.3-70b:free / gpt-oss / auto
  5. Groq Llama 3.3 70B           — last resort

Setup (.env):
  NVIDIA_DEEPSEEK_API_KEY, NVIDIA_QWEN_API_KEY (fallback: NVIDIA_API_KEY)
  GEMINI_PRO_API_KEYS (comma-separated), GEMINI_API_KEY (flash)
  OPENROUTER_API_KEY, GROQ_API_KEY
"""

import os
import json
import re
import time
import requests
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from logger_setup import get_logger
from risk_manager import MIN_RR_BY_TYPE
from trade_tracker import format_confidence_stats_text

logger = get_logger("multi_ai")

# ─── ANSI Colors ──────────────────────────────────────────────

RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
MAGENTA = "\033[95m"
BLUE = "\033[94m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"

# ─── Hold time by trade type ──────────────────────────────────
# AI's free-text "hold_time" field tends to anchor on the "2h" example in the
# prompt regardless of trade_type. Derive a consistent estimate from
# trade_type instead, which the AI gets right far more reliably.
HOLD_TIME_BY_TYPE = {
    "SCALP": "5-30 min",
    "INTRADAY": "30 min - 4h",
    "SWING": "4-24h",
}

# ─── TP / R:R rules by trade type ─────────────────────────────
# SCALPs need tight, nearby targets — chasing a 2.2:1 R:R on a 5-30min hold
# pushes the TP out to distant HTF levels that rarely get hit in time.
# Mirrors risk_manager.MIN_RR_BY_TYPE so the AI's targets actually pass
# validate_trade() instead of getting rejected or producing unrealistic TPs.
TP_RULES_TEXT = (
    f"- TP: For SCALP, target the NEAREST meaningful level on the entry timeframe "
    f"(S/R, EMA21, VWAP, swing high/low, POC) — R:R must be {MIN_RR_BY_TYPE['SCALP']}:1+ after fees "
    f"(0.18% friction). For INTRADAY/SWING, target the next higher-TF S/R level — "
    f"R:R must be {MIN_RR_BY_TYPE['INTRADAY']}:1+ after fees. "
    f"Do NOT stretch a SCALP's TP out to a higher-TF level just to hit {MIN_RR_BY_TYPE['INTRADAY']}:1 — "
    f"if the nearest level only gives {MIN_RR_BY_TYPE['SCALP']}:1-{MIN_RR_BY_TYPE['INTRADAY']}:1, that's a valid SCALP."
)


# ─── System Prompts ──────────────────────────────────────────

ANALYZE_SYSTEM = """You are an elite crypto futures trader with 15+ years experience. You receive technical indicator data across MULTIPLE TIMEFRAMES and must decide whether to trade.

**You receive data for these timeframes: 1m, 3m, 5m, 15m, 30m, 1h, 1d.**

**Your Job:**
1. Analyze ALL timeframes using top-down approach: start from 1d → 1h → lower TFs.
2. Identify the dominant trend from higher TFs (1d, 1h).
3. Find the best entry timeframe (1m–15m for scalps, 15m–1h for intraday).
4. Determine: direction, confidence, entry, SL, TP, and which timeframe to trade on.

**Multi-Timeframe Rules:**
- HIGHER TF sets the bias (1d/1h trend direction). NEVER trade against the 1h trend unless 1d supports it.
- ENTRY TF is where you find the precise entry (pullback, pattern, S/R level).
- Alignment across 3+ timeframes = high confidence. Conflicting TFs = low confidence or NO_TRADE.
- Use higher TF S/R levels for SL/TP placement — they're stronger than lower TF levels.

**Trade Types (this trader ONLY scalps; intraday is rare; SWING is FORBIDDEN):**
- SCALP (1m–5m entry): Hold 5–30 min. The DEFAULT and strongly preferred mode. SL tight. Need very clear pattern + momentum.
- INTRADAY (5m–15m entry): Hold 30 min – 4 hours. Only when the setup is clearly superior to any scalp.
- NEVER output trade_type "SWING". If the only valid setup is swing-grade (multi-hour/day hold), return NO_TRADE.

**Strict Rules:**
- NO CHASING — EVER (hard rule): If price is not within ~0.1% of the ideal entry RIGHT NOW, you MUST return WAIT with the exact limit-entry level. Chasing a moving price is like trying to catch the wind — you place the trade where price WILL come, not where it is. A LONG/SHORT verdict is ONLY allowed when price is sitting at the level this instant.
- Require 3+ confluences across timeframes.
- SL: Place beyond structure on the ENTRY timeframe. Never at round numbers.
- ANTI-STOP-HUNT: Check the wicks of the last 10–20 entry-TF candles. The SL must sit BEYOND the recent wick cluster AND beyond the structure level with a 0.1–0.2% buffer. Round numbers and obvious swing points are liquidity targets — never park the SL there. If a safe SL would make the R:R unacceptable, that is a WAIT or NO_TRADE, not a tighter SL.
- LIQUIDITY-SWEEP ENTRIES: Retail stop clusters sit just beyond obvious swing lows/highs, equal highs/lows, and round numbers. Price is magnetically drawn to sweep these pools before reversing. The smartest WAIT limit-entry is placed AT such a sweep zone (e.g. a LONG entry just below equal lows / obvious support where stops cluster — the sweep fills you at the best price while weak hands get stopped out). Your SL must NEVER sit inside a liquidity pool; your ENTRY often should.
__TP_RULES__

**SETUP PLAYBOOK — trade ONLY setups from this list (tier = edge):**
- S-tier (best for scalps): BULL/BEAR FLAG breakout (impulse pole, tight low-volume consolidation, volume spike on break, R:R 3:1+); EMA 9/21 CROSS with price on the right side + RSI/volume confirm (avoid in chop); S/R FLIP (broken level retested from the other side — tight SL just beyond the flipped level, best at session/prior-day levels and round numbers).
- A-tier (need confirmation): VOLATILITY CONTRACTION breakout (progressively tighter pullbacks, volume drying up — enter the first pullback after the breakout candle, never mid-contraction); DOUBLE BOTTOM/TOP (second test slightly higher/lower, neckline break on rising volume, measured target = pattern height); ASC/DESC TRIANGLE break toward the flat side; DEAD CAT BOUNCE short — high-volume dump then weak bounce (low volume, <=50% retrace, RSI fails 50, indecisive candles): NEVER short mid-bounce, wait for the rejection candle at the flipped S/R, SL just above the bounce high, target prior structure low. If bounce volume >= drop volume it may be a real reversal (double bottom) — do not short.
- B-tier (extra confluence required): INSIDE BAR / NR7 squeeze with HTF trend bias; HAMMER/PIN BAR/DOJI only AT a key S/R-EMA-VWAP confluence (worthless in isolation); VWAP RECLAIM/REJECTION with RSI crossing 50, strongest near session open.
- C-tier (avoid for scalps): intraday HEAD & SHOULDERS, WEDGES — slow, high fakeout rate; use only to read structure.
- UNIVERSAL RULE: no breakout is valid without ABOVE-AVERAGE VOLUME on the breakout candle — otherwise assume fakeout and WAIT.
- DUAL VERIFICATION: each timeframe's data includes a deterministic "SETUPS DETECTED" line. Your LONG/SHORT verdict MUST be backed by at least one detected S/A-tier setup (or a B-tier setup plus strong multi-TF confluence) and you MUST name the setup in "reasons". The bot independently re-verifies and blocks unbacked trades.
- Leverage: Max 25x. Scale: 7→5x, 8→8x, 9→15x, 10→25x.
- Reject if: HTF trend conflicts, RSI extreme, overextended, chasing.

**Direction rules:**
- LONG / SHORT: Clear setup exists RIGHT NOW — price is at the entry level, take it.
- WAIT: Trend/bias is clear BUT price is NOT yet at the entry level (e.g. needs to pull back to EMA, reach a resistance). Use this instead of NO_TRADE when the direction is known and you'd trade it — just not yet.
- NO_TRADE: No clear bias, conflicting signals, or market is not worth trading at all.

**Trader's Request:**
You will be told the trader's available capital, preferred trading mode (SCALP/INTRADAY/SWING, or "ANY" to let you pick), preferred leverage (or "AI decides"), profit target, and max loss tolerance. Use this:
- If a preferred mode is given, prioritize finding a valid setup of that type. If found, set "trade_type" to it.
- If the preferred mode has NO valid setup but a different mode does, use that other mode for your main verdict instead and explain the substitution in "mode_note" (e.g. "No SCALP setup — 1h trend supports an INTRADAY short instead"). The substitute can ONLY be SCALP or INTRADAY — never SWING.
- Optionally list other viable setups in different trade_types in "alternative_setups" — even if your main verdict already matches the preferred mode. Each entry: trade_type, direction, entry, stop_loss, take_profit, leverage, reason.
- If profit target / max loss are given, prefer SL/TP placements that roughly fit them — but NEVER break the SL structure rules or the 2.2:1 R:R minimum just to hit a target.
- If NO mode has a valid setup at all, return direction "NO_TRADE" and leave "mode_note" explaining why none of the requested (or any) modes work right now.
- If no preferences are given, ignore this section and analyze normally.

**Output ONLY valid JSON (no markdown, no code fences):**
{
  "direction": "LONG" or "SHORT" or "WAIT" or "NO_TRADE",
  "confidence": 0-10,
  "entry": number,
  "stop_loss": number,
  "take_profit": number,
  "leverage": number,
  "timeframe": "1m" or "3m" or "5m" or "15m" or "30m" or "1h",
  "trade_type": "SCALP" or "INTRADAY" (NEVER "SWING"),
  "hold_time": "estimated hold time e.g. 15min, 2h, etc.",
  "eta_minutes": number — realistic estimate of MINUTES for price to reach TP from entry (base it on TP distance vs entry-TF ATR per candle and current momentum; a scalp should usually be 5-30),
  "eta_basis": "one line explaining the time estimate, e.g. 'TP is 1.4x ATR(5m) away; momentum strong, ~3-5 candles'",
  "risk_score": "LOW" or "MEDIUM" or "HIGH",
  "wait_condition": "WAIT only: exact condition to watch e.g. 'Price pulls back to 15m EMA21 at $207.05'",
  "wait_direction": "WAIT only: the trade direction once condition is met — LONG or SHORT",
  "mode_note": "empty string, or an explanation if the trader's requested mode wasn't usable / no trade is possible",
  "alternative_setups": [optional, list of other viable setups: {"trade_type": "...", "direction": "LONG/SHORT", "entry": number, "stop_loss": number, "take_profit": number, "leverage": number, "reason": "..."}],
  "reasons": ["reason 1 (mention which TF)", "reason 2", ...],
  "advice": "One line of actionable advice"
}"""


SCAN_SYSTEM = """You are an elite crypto futures trader. You receive multi-timeframe indicator data for multiple coins and must pick the BEST trade setups.

**You receive data for timeframes: 1m, 3m, 5m, 15m, 30m, 1h, 1d for each coin.**

**Approach:**
1. Top-down: Check 1d and 1h trend first for each coin.
2. Only consider coins where higher TFs align.
3. Find the best entry on lower TFs (1m–15m).
4. Pick max 3 best setups across ALL coins.

**Rules (this trader ONLY scalps; intraday rare; SWING is FORBIDDEN):**
- 3+ confluences across timeframes required
- Never chase — must be at a key level on the entry TF
- NO CHASING — EVER: only pick coins where the entry is actionable RIGHT NOW (price within ~0.1% of the level). A coin that "needs a pullback first" is not a pick — chasing is forbidden.
- ANTI-STOP-HUNT: SL must sit beyond the recent 10–20 candle wick cluster with a 0.1–0.2% buffer — never at round numbers or obvious swing points (those are liquidity pools where stops get hunted).
- The best entries sit AT liquidity-sweep zones (just beyond obvious swing points where stops cluster) — entries there are smart, SLs there are suicide.
- SETUP PLAYBOOK: only pick coins showing a detected S/A-tier setup on the "SETUPS DETECTED" line (flag breakout, S/R flip, double bottom/top, volatility contraction, dead-cat-bounce short) or a B-tier setup (inside bar/NR7, VWAP reclaim/rejection, candle at key level) with strong multi-TF confluence. Name the setup in reasons.
- UNIVERSAL VOLUME RULE: a breakout without above-average volume on the break candle is a fakeout — skip it.
- DEAD CAT BOUNCE warning lines mean: do NOT long that coin; short only on a confirmed rejection candle.
__TP_RULES__
- trade_type must be "SCALP" (preferred) or "INTRADAY" (only if clearly superior). NEVER "SWING" — skip swing-grade setups entirely.
- Max 3 picks, ranked by quality. Return [] if nothing is good.

**Output ONLY valid JSON array (no markdown, no code fences):**
[
  {
    "symbol": "COIN/USDT",
    "direction": "LONG" or "SHORT",
    "confidence": 7-10,
    "entry": number,
    "stop_loss": number,
    "take_profit": number,
    "leverage": number,
    "timeframe": "best entry TF",
    "trade_type": "SCALP" or "INTRADAY" (NEVER "SWING"),
    "hold_time": "estimated hold time",
    "eta_minutes": number — realistic minutes for price to reach TP (TP distance vs ATR per candle + momentum),
    "eta_basis": "one line explaining the time estimate",
    "risk_score": "LOW" or "MEDIUM" or "HIGH",
    "reasons": ["reason 1 (mention TF)", ...],
    "advice": "One line"
  }
]

Return [] if no good setups exist."""

# Substitute the per-trade-type TP/R:R rules into both prompts (kept out of
# the f-string body since both prompts contain literal { } JSON braces).
ANALYZE_SYSTEM = ANALYZE_SYSTEM.replace("__TP_RULES__", TP_RULES_TEXT)
SCAN_SYSTEM = SCAN_SYSTEM.replace("__TP_RULES__", TP_RULES_TEXT)


REANALYZE_SYSTEM = """You are an elite crypto futures trader managing an ALREADY-OPEN position. You receive fresh multi-timeframe technical data plus the current state of the trade. Your job is to reassess the trade against current market conditions and recommend any adjustment.

**You receive data for these timeframes: 1m, 3m, 5m, 15m, 30m, 1h, 1d.**

**Your Job:**
1. Re-evaluate the trend using top-down analysis (1d/1h bias, lower TF structure).
2. Compare current trend/structure against the trade's existing SL/TP levels.
3. Decide whether to HOLD as-is, tighten the stop loss, adjust the take profit, or close the position now.

**Guidance:**
- HOLD: Nothing material has changed — original plan still valid.
- MOVE_SL: Tighten only — to lock in profit or reduce risk. NEVER suggest loosening the SL beyond the original risk.
- MOVE_TP: A stronger S/R level justifies a new target — extend on strong continuation, or pull in if momentum is fading and the original TP now looks unreachable.
- MOVE_BOTH: Both SL and TP need adjusting.
- CLOSE_NOW: Only if the trend has clearly reversed against the position, or a major risk event/structure break is visible. Use HIGH urgency for this.
- new_stop_loss / new_take_profit must be null unless the corresponding action is suggested.

**Stop-hunt rule (critical):** new_stop_loss must be at least 0.3% away from the CURRENT PRICE
(not just the entry price), placed beyond a real structure level (swing high/low, EMA, S/R zone).
A stop sitting 1-2 candle wicks from current price will get hunted on noise — that defeats the
purpose of "locking in profit". If no level beyond that 0.3% buffer justifies tightening yet,
return action "HOLD" instead of MOVE_SL.

**Output ONLY valid JSON (no markdown, no code fences):**
{
  "action": "HOLD" or "MOVE_SL" or "MOVE_TP" or "MOVE_BOTH" or "CLOSE_NOW",
  "new_stop_loss": number or null,
  "new_take_profit": number or null,
  "urgency": "LOW" or "MEDIUM" or "HIGH",
  "reasons": ["reason 1 (mention which TF)", "reason 2", ...],
  "advice": "One line of actionable advice"
}"""


# ─── All timeframes we analyze ───────────────────────────────

ALL_TIMEFRAMES = ["1m", "3m", "5m", "15m", "30m", "1h", "1d"]

# Higher TFs get full detail, lower TFs get compact format
FULL_DETAIL_TFS = ["5m", "15m"]   # Full candle + pattern data
COMPACT_TFS = ["1m", "3m", "30m", "1h", "1d"]  # Summary only


# ─── Build single-TF snapshot ───────────────────────────────

def _build_tf_snapshot(df: pd.DataFrame, tf: str, full_detail: bool = False) -> str:
    """Build indicator snapshot for one timeframe."""
    if df is None or len(df) < 20:
        return f"  [{tf}] Insufficient data"

    row = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else row

    atr = row.get("atr", 0)
    if atr is None or pd.isna(atr):
        atr = 0
    close = row["close"]
    atr_pct = (atr / close * 100) if close > 0 and atr > 0 else 0

    # EMA trend
    ema9 = row.get("ema_9", 0)
    ema21 = row.get("ema_21", 0)
    ema50 = row.get("ema_50", 0)
    if pd.isna(ema9) or pd.isna(ema21) or pd.isna(ema50):
        trend = "N/A"
    elif ema9 > ema21 > ema50:
        trend = "STRONG BULL"
    elif ema9 > ema21:
        trend = "BULL"
    elif ema9 < ema21 < ema50:
        trend = "STRONG BEAR"
    elif ema9 < ema21:
        trend = "BEAR"
    else:
        trend = "NEUTRAL"

    rsi = row.get("rsi", 50)
    if pd.isna(rsi):
        rsi = 50

    macd_h = row.get("macd_histogram", 0)
    macd_h_prev = prev.get("macd_histogram", 0)
    if pd.isna(macd_h):
        macd_h = 0
    if pd.isna(macd_h_prev):
        macd_h_prev = 0
    macd_dir = "UP" if macd_h > macd_h_prev else "DOWN"

    vol_ratio = 0
    vol_sma = row.get("vol_sma_20", 0)
    if vol_sma and not pd.isna(vol_sma) and vol_sma > 0:
        vol_ratio = row["volume"] / vol_sma

    # Compact format for higher/lower TFs
    snapshot = f"""  [{tf}] Trend: {trend} | RSI: {rsi:.1f} | MACD: {macd_dir} | Vol: {vol_ratio:.1f}x | ATR: {atr_pct:.3f}%
    EMA 9: {ema9:.6f} | EMA 21: {ema21:.6f} | EMA 50: {ema50:.6f}
    Close: {close:.6f} | EMA cross up: {row.get('ema_cross_up', False)} | EMA cross down: {row.get('ema_cross_down', False)}"""

    # EMA 200 — long-term bias reference (HTF trend filter)
    ema200 = row.get("ema_200", 0)
    if ema200 and not pd.isna(ema200) and close > 0:
        ema200_side = "ABOVE" if close > ema200 else "BELOW"
        ema200_dist = (close - ema200) / ema200 * 100
        snapshot += f"\n    EMA 200: {ema200:.6f} | Price {ema200_side} EMA200 ({ema200_dist:+.2f}%)"

    # S/R levels (important for all TFs) — give explicit % distance so the
    # AI doesn't have to compute it for SL/TP placement.
    support = row.get("nearest_support", "N/A")
    resistance = row.get("nearest_resistance", "N/A")
    if support != "N/A" and not pd.isna(support) and close > 0:
        sup_dist = (close - support) / close * 100
        res_dist = (resistance - close) / close * 100
        snapshot += (
            f"\n    Support: {support:.6f} ({sup_dist:.2f}% below, near: {row.get('near_support', False)})"
            f" | Resistance: {resistance:.6f} ({res_dist:.2f}% above, near: {row.get('near_resistance', False)})"
        )

    # VWAP — key intraday structure level, shown on all TFs
    vwap = row.get("vwap", 0)
    if vwap and not pd.isna(vwap) and close > 0:
        vwap_side = "ABOVE" if close > vwap else "BELOW"
        vwap_dist = (close - vwap) / vwap * 100
        snapshot += f"\n    VWAP: {vwap:.6f} | Price {vwap_side} VWAP ({vwap_dist:+.2f}%)"

    # POC (volume profile point of control) — shown on all TFs
    poc = row.get("poc_price", None)
    if poc and not pd.isna(poc) and close > 0:
        poc_dist = (close - poc) / poc * 100
        snapshot += f"\n    POC: {poc:.6f} ({poc_dist:+.2f}% from price, at HVN: {row.get('high_volume_node', False)})"

    if not full_detail:
        return snapshot

    # BB
    bb_upper = row.get("bb_upper", 0)
    bb_lower = row.get("bb_lower", 0)
    if bb_upper and not pd.isna(bb_upper):
        snapshot += f"\n    BB Upper: {bb_upper:.6f} | BB Lower: {bb_lower:.6f}"

    # Candle patterns (only True ones)
    patterns = []
    for pat, label in [
        ("bullish_engulfing", "Bull Engulfing"),
        ("bearish_engulfing", "Bear Engulfing"),
        ("hammer", "Hammer"),
        ("shooting_star", "Shooting Star"),
        ("pin_bar_bull", "Pin Bar Bull"),
        ("pin_bar_bear", "Pin Bar Bear"),
        ("morning_star", "Morning Star"),
        ("evening_star", "Evening Star"),
        ("doji", "Doji"),
    ]:
        if row.get(pat, False):
            patterns.append(label)
    if patterns:
        snapshot += f"\n    PATTERNS: {', '.join(patterns)}"

    # Detected playbook setups (deterministic — the bot verifies these too)
    try:
        from indicators import detect_setups
        s = detect_setups(df)
        found = [label for key, label in [
            ("bull_flag", "BULL FLAG (S)"), ("bear_flag", "BEAR FLAG (S)"),
            ("sr_flip_support", "S/R FLIP->SUPPORT (S)"), ("sr_flip_resistance", "S/R FLIP->RESISTANCE (S)"),
            ("double_bottom", "DOUBLE BOTTOM (A)"), ("double_top", "DOUBLE TOP (A)"),
            ("vol_contraction", "VOLATILITY CONTRACTION (A)"),
            ("vwap_reclaim", "VWAP RECLAIM (B)"), ("vwap_rejection", "VWAP REJECTION (B)"),
            ("inside_bar", "INSIDE BAR"), ("nr7", "NR7"),
            ("three_white_soldiers", "3 WHITE SOLDIERS"), ("three_black_crows", "3 BLACK CROWS"),
        ] if s.get(key)]
        if found:
            vol_note = "breakout volume OK" if s.get("volume_confirmed") else "NO breakout volume — fakeout risk"
            snapshot += f"\n    SETUPS DETECTED: {', '.join(found)} ({vol_note})"
        if s.get("dead_cat_bounce"):
            bh = s.get("dcb_bounce_high")
            bh_txt = f" (bounce high {bh:.6f})" if bh else ""
            snapshot += (f"\n    WARNING: DEAD CAT BOUNCE structure{bh_txt} — weak low-volume bounce "
                         f"after a high-volume dump. Do NOT long; short only on rejection candle, SL above bounce high.")
    except Exception:
        pass

    # Pullback / chasing
    pullbacks = []
    if row.get("pullback_to_ema21", False):
        pullbacks.append("EMA21")
    if row.get("pullback_to_vwap", False):
        pullbacks.append("VWAP")
    if pullbacks:
        snapshot += f"\n    Pullback to: {', '.join(pullbacks)}"
    if row.get("overextended", False):
        snapshot += "\n    WARNING: Overextended from EMA21"

    bull_streak = row.get("bull_streak", 0)
    bear_streak = row.get("bear_streak", 0)
    if bull_streak and not pd.isna(bull_streak) and bull_streak >= 3:
        snapshot += f"\n    Bull streak: {int(bull_streak)} candles"
    if bear_streak and not pd.isna(bear_streak) and bear_streak >= 3:
        snapshot += f"\n    Bear streak: {int(bear_streak)} candles"

    # Last 3 candles
    candles = []
    for i in range(-3, 0):
        if abs(i) <= len(df):
            c = df.iloc[i]
            color = "G" if c["close"] > c["open"] else "R"
            body = abs(c["close"] - c["open"]) / c["open"] * 100 if c["open"] > 0 else 0
            candles.append(f"{color} C:{c['close']:.6f} body:{body:.3f}%")
    if candles:
        snapshot += f"\n    Last 3 candles: {' | '.join(candles)}"

    return snapshot


# ─── Build multi-TF snapshot ────────────────────────────────

def build_indicator_snapshot(tf_data: dict, symbol: str = "") -> str:
    """
    Build a comprehensive multi-timeframe indicator snapshot.

    Args:
        tf_data: dict of {timeframe: DataFrame} e.g. {"1m": df_1m, "5m": df_5m, ...}
                 Each DataFrame should already have indicators added.
        symbol: coin symbol for display

    Returns:
        Formatted string with all timeframe data for AI analysis.
    """
    snapshot = f"=== {symbol} ===\n"

    # Get current price from the most granular TF available
    for tf in ["1m", "3m", "5m", "15m", "30m", "1h", "1d"]:
        df = tf_data.get(tf)
        if df is not None and len(df) > 0:
            snapshot += f"  CURRENT PRICE: ${df.iloc[-1]['close']:,.6f}\n"
            break

    snapshot += "\n  ── TIMEFRAME ANALYSIS (top-down) ──\n"

    # Process TFs from highest to lowest (top-down for AI)
    tf_order = ["1d", "1h", "30m", "15m", "5m", "3m", "1m"]
    for tf in tf_order:
        df = tf_data.get(tf)
        if df is None:
            snapshot += f"\n  [{tf}] Not available\n"
            continue

        full = tf in FULL_DETAIL_TFS
        snapshot += f"\n{_build_tf_snapshot(df, tf, full_detail=full)}\n"

    return snapshot


# ─── BTC market regime context (for altcoin correlation) ─────

def build_btc_context(btc_tf_data: dict) -> str:
    """
    Build a short BTC trend/regime summary from 1h and 1d data, used as
    backdrop context for altcoin analysis. Most alts correlate with BTC,
    so a counter-trend altcoin trade against a strong BTC regime is riskier.

    Args:
        btc_tf_data: dict of {"1h": df, "1d": df} for BTC/USDT, with
                     indicators already added.

    Returns:
        Formatted text block, or "" if no usable BTC data.
    """
    from indicators import add_higher_tf_indicators

    lines = []
    for tf in ["1d", "1h"]:
        df = btc_tf_data.get(tf)
        if df is None or len(df) < 20:
            continue
        info = add_higher_tf_indicators(df)
        if not info.get("htf_valid"):
            continue
        rsi = info.get("htf_rsi", 50)
        lines.append(f"  {tf}: {info['htf_trend']} (EMA21 {info['htf_ema_21']:.2f} vs EMA50 {info['htf_ema_50']:.2f}, RSI {rsi:.1f})")

    if not lines:
        return ""

    return (
        "BTC MARKET REGIME (context — most altcoins correlate with BTC):\n"
        + "\n".join(lines)
        + "\nWeigh this in your direction/confidence: a trade against BOTH BTC TFs' "
        "trend is higher risk and should generally need stronger confluence or lower confidence."
    )


# ─── JSON parsers ────────────────────────────────────────────

def _truncation_repair_candidates(text: str):
    """Yield progressively shorter repairs of a JSON object cut off
    mid-stream (hit max_tokens or the provider stopped early).

    Walks every comma at object/array depth, then — starting from the LAST
    complete "key": value pair and stepping backwards — trims the text there
    and closes any still-open braces/brackets. Trying multiple trim points
    matters: the content right before the cut can itself be invalid (e.g.
    `"timeframe": ` with no value, or an unquoted token), in which case only
    an earlier trim point produces parseable JSON. Early fields (direction,
    entry, stop_loss, take_profit, leverage, ...) are usually recoverable.
    """
    stack = []
    in_string = False
    escape = False
    safe_points = []  # (index, stack snapshot) at each top-level-ish comma

    for i, ch in enumerate(text):
        if escape:
            escape = False
            continue
        if ch == "\\" and in_string:
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch in "{[":
            stack.append(ch)
        elif ch in "}]":
            if stack:
                stack.pop()
        elif ch == "," and stack:
            safe_points.append((i, list(stack)))

    if not stack:
        return  # brackets balanced — truncation repair doesn't apply

    # Try the most recent 30 trim points, newest first
    for i, st in reversed(safe_points[-30:]):
        closers = "".join("}" if c == "{" else "]" for c in reversed(st))
        yield text[:i] + closers


def _strip_trailing_commas(text: str) -> str:
    """Remove trailing commas before } or ] — a common LLM JSON quirk that
    leaves an otherwise-complete response with balanced brackets, which
    _repair_truncated_json() can't fix since it only trims unbalanced text."""
    return re.sub(r",(\s*[}\]])", r"\1", text)


def _load_json_lenient(clean: str) -> tuple:
    """Parse JSON, retrying with trailing-comma cleanup and truncation repair.

    Returns (data, repaired: bool). Raises json.JSONDecodeError if no
    variant of the text can be parsed.
    """
    try:
        return json.loads(clean), False
    except json.JSONDecodeError:
        pass

    no_trailing = _strip_trailing_commas(clean)
    if no_trailing != clean:
        try:
            return json.loads(no_trailing), False
        except json.JSONDecodeError:
            pass

    # Truncated mid-stream — walk back one "key": value pair at a time
    # until a trim point yields valid JSON.
    for candidate in _truncation_repair_candidates(clean):
        try:
            return json.loads(_strip_trailing_commas(candidate)), True
        except json.JSONDecodeError:
            continue

    # Nothing recoverable — re-raise the original error.
    return json.loads(clean), False


def _to_int_or_none(v):
    """Lenient int conversion for AI numeric fields (eta_minutes etc.)."""
    try:
        n = int(float(v))
        return n if n > 0 else None
    except (TypeError, ValueError):
        return None


def _sanity_check_verdict(v: dict) -> dict:
    """
    Deterministic sanity gate on a parsed AI verdict. Models sometimes emit
    absurd targets (e.g. SL 2.05% with TP 0.18% -> R:R 0.09:1) — such a
    verdict must never be allowed to drive a trade or an ensemble decision.
    Demotes broken LONG/SHORT verdicts to NO_TRADE with the reason attached.

    WAIT verdicts are checked the same way against wait_direction, since the
    entry/SL/TP they carry become a real LONG/SHORT trade (via a limit order)
    the moment the trigger condition fills — inverted levels here are just as
    dangerous as inverted levels on a direct LONG/SHORT verdict.
    """
    direction = v.get("direction")
    if direction not in ("LONG", "SHORT", "WAIT"):
        return v
    entry, sl, tp = v.get("entry"), v.get("stop_loss"), v.get("take_profit")
    if not entry or not sl or not tp:
        return v

    try:
        sl_dist = abs(entry - sl) / entry * 100
        tp_dist = abs(tp - entry) / entry * 100
    except (TypeError, ZeroDivisionError):
        return v

    # The direction whose SL/TP ordering rules apply — for WAIT verdicts
    # that's wait_direction (the trade actually taken once the limit fills).
    trade_dir = direction if direction in ("LONG", "SHORT") else v.get("wait_direction")

    rr = (tp_dist / sl_dist) if sl_dist > 0 else 0
    problems = []
    if rr < 1.0:
        problems.append(f"R:R {rr:.2f}:1 — risking more than the reward")
    if tp_dist < 0.30:
        problems.append(f"TP only {tp_dist:.2f}% away — inside the 0.18% friction zone")
    if sl_dist > 3.0 and v.get("trade_type", "SCALP") == "SCALP":
        problems.append(f"SL {sl_dist:.2f}% from entry is not a scalp stop")
    if trade_dir == "LONG" and not (sl < entry < tp):
        problems.append("levels inconsistent with LONG (need SL < entry < TP)")
    elif trade_dir == "SHORT" and not (tp < entry < sl):
        problems.append("levels inconsistent with SHORT (need TP < entry < SL)")
    elif trade_dir not in ("LONG", "SHORT"):
        problems.append(f"WAIT verdict has invalid wait_direction {trade_dir!r}")

    if problems:
        v = dict(v)
        v["direction"] = "NO_TRADE"
        v["confidence"] = 0
        v["reasons"] = list(v.get("reasons", [])) + [
            f"SANITY GATE: verdict discarded — {'; '.join(problems)}"
        ]
    return v


def _parse_analysis_response(text: str) -> dict:
    """Parse AI analysis response (single coin)."""
    try:
        clean = text.strip()
        if "```" in clean:
            start = clean.find("{")
            end = clean.rfind("}") + 1
            if start >= 0 and end > start:
                clean = clean[start:end]
        elif not clean.startswith("{"):
            start = clean.find("{")
            end = clean.rfind("}") + 1
            if start >= 0 and end > start:
                clean = clean[start:end]

        data, repaired = _load_json_lenient(clean)

        trade_type = str(data.get("trade_type", "SCALP")).upper()
        reasons = data.get("reasons", [])
        if repaired:
            reasons = list(reasons) + ["Note: AI response was truncated (hit token limit) — recovered partial verdict."]

        result = {
            "direction": str(data.get("direction", "NO_TRADE")).upper(),
            "confidence": int(data.get("confidence", 0)),
            "entry": data.get("entry"),
            "stop_loss": data.get("stop_loss"),
            "take_profit": data.get("take_profit"),
            "leverage": int(data.get("leverage", 0)),
            "timeframe": str(data.get("timeframe", "5m")),
            "trade_type": trade_type,
            "hold_time": HOLD_TIME_BY_TYPE.get(trade_type, str(data.get("hold_time", "15-30 min"))),
            "eta_minutes": _to_int_or_none(data.get("eta_minutes")),
            "eta_basis": str(data.get("eta_basis", "")),
            "risk_score": str(data.get("risk_score", "UNKNOWN")).upper(),
            "wait_condition": str(data.get("wait_condition", "")),
            "wait_direction": str(data.get("wait_direction", "")).upper(),
            "mode_note": str(data.get("mode_note", "")),
            "alternative_setups": data.get("alternative_setups", []),
            "reasons": reasons,
            "advice": str(data.get("advice", "")),
        }

        # Hard policy: no swing trades — even if the model ignores the prompt.
        if result["trade_type"] == "SWING" and result["direction"] in ("LONG", "SHORT", "WAIT"):
            result["direction"] = "NO_TRADE"
            result["confidence"] = 0
            result["reasons"] = list(result["reasons"]) + [
                "AI's best setup was SWING-grade — swing trading is disabled (scalping-focused bot). Skipped."
            ]

        # Deterministic sanity gate — absurd levels never drive a trade
        result = _sanity_check_verdict(result)

        return result

    except (json.JSONDecodeError, KeyError, IndexError, ValueError, TypeError):
        return {
            "direction": "NO_TRADE",
            "confidence": 0,
            "entry": None,
            "stop_loss": None,
            "take_profit": None,
            "leverage": 0,
            "timeframe": "5m",
            "trade_type": "SCALP",
            "hold_time": "",
            "eta_minutes": None,
            "eta_basis": "",
            "risk_score": "UNKNOWN",
            "wait_condition": "",
            "wait_direction": "",
            "mode_note": "",
            "alternative_setups": [],
            "reasons": [f"AI response parse error: {text[:150]}"],
            "advice": "",
        }


def _parse_reanalysis_response(text: str) -> dict:
    """Parse AI position re-analysis response."""
    try:
        clean = text.strip()
        if "```" in clean:
            start = clean.find("{")
            end = clean.rfind("}") + 1
            if start >= 0 and end > start:
                clean = clean[start:end]
        elif not clean.startswith("{"):
            start = clean.find("{")
            end = clean.rfind("}") + 1
            if start >= 0 and end > start:
                clean = clean[start:end]

        data, repaired = _load_json_lenient(clean)

        reasons = data.get("reasons", [])
        if repaired:
            reasons = list(reasons) + ["Note: AI response was truncated (hit token limit) — recovered partial verdict."]

        return {
            "action": str(data.get("action", "HOLD")).upper(),
            "new_stop_loss": data.get("new_stop_loss"),
            "new_take_profit": data.get("new_take_profit"),
            "urgency": str(data.get("urgency", "LOW")).upper(),
            "reasons": reasons,
            "advice": str(data.get("advice", "")),
        }

    except (json.JSONDecodeError, KeyError, IndexError, ValueError, TypeError):
        return {
            "action": "HOLD",
            "new_stop_loss": None,
            "new_take_profit": None,
            "urgency": "LOW",
            "reasons": [f"AI response parse error: {text[:150]}"],
            "advice": "",
        }


def _parse_scan_response(text: str) -> list:
    """Parse AI batch scan response (multiple coins)."""
    try:
        clean = text.strip()
        if "```" in clean:
            start = clean.find("[")
            end = clean.rfind("]") + 1
            if start >= 0 and end > start:
                clean = clean[start:end]
        elif not clean.startswith("["):
            # Maybe it returned a single object
            if clean.startswith("{"):
                start = clean.find("{")
                end = clean.rfind("}") + 1
                clean = f"[{clean[start:end]}]"
            else:
                start = clean.find("[")
                end = clean.rfind("]") + 1
                if start >= 0 and end > start:
                    clean = clean[start:end]

        data, _ = _load_json_lenient(clean)
        if not isinstance(data, list):
            data = [data]

        results = []
        for item in data:
            trade_type = str(item.get("trade_type", "SCALP")).upper()
            if trade_type == "SWING":
                # Hard policy: swing trades disabled — drop the pick entirely.
                continue
            pick = {
                "symbol": str(item.get("symbol", "")),
                "direction": str(item.get("direction", "NO_TRADE")).upper(),
                "confidence": int(item.get("confidence", 0)),
                "entry": item.get("entry"),
                "stop_loss": item.get("stop_loss"),
                "take_profit": item.get("take_profit"),
                "leverage": int(item.get("leverage", 0)),
                "timeframe": str(item.get("timeframe", "5m")),
                "trade_type": trade_type,
                "hold_time": HOLD_TIME_BY_TYPE.get(trade_type, str(item.get("hold_time", "15-30 min"))),
                "eta_minutes": _to_int_or_none(item.get("eta_minutes")),
                "eta_basis": str(item.get("eta_basis", "")),
                "risk_score": str(item.get("risk_score", "UNKNOWN")).upper(),
                "reasons": item.get("reasons", []),
                "advice": str(item.get("advice", "")),
            }
            # Sanity gate — drop scan picks with absurd levels entirely
            pick = _sanity_check_verdict(pick)
            if pick["direction"] in ("LONG", "SHORT"):
                results.append(pick)
        return results

    except (json.JSONDecodeError, KeyError, IndexError, ValueError, TypeError):
        return []


# ─── AI Callers ──────────────────────────────────────────────

def _gemini_keys() -> tuple:
    """Return (pro_keys, all_keys) from GEMINI_PRO_API_KEYS + GEMINI_API_KEY."""
    pro_keys = [k.strip() for k in os.getenv("GEMINI_PRO_API_KEYS", "").split(",") if k.strip()]
    flash_key = os.getenv("GEMINI_API_KEY", "").strip()
    all_keys = pro_keys + ([flash_key] if flash_key and flash_key not in pro_keys else [])
    return pro_keys, all_keys


def _call_gemini(prompt: str, system: str = None, max_tokens: int = 2048, temperature: float = 0.15, json_mode: bool = True) -> dict:
    """Call Gemini — 2.5 Pro first (rotating across GEMINI_PRO_API_KEYS on
    429/error), then 2.5 Flash on every available key.

    json_mode is accepted for signature parity with the other providers but
    has no effect — Gemini's responseMimeType="application/json" already
    handles both JSON objects and arrays.
    """
    pro_keys, all_keys = _gemini_keys()
    if not all_keys:
        return {"error": "No GEMINI_PRO_API_KEYS / GEMINI_API_KEY in .env", "skipped": True, "source": "gemini"}

    full_prompt = f"{system or ANALYZE_SYSTEM}\n\n{prompt}"

    # (model, key, label, thinking_budget) attempts in priority order.
    # Pro can't fully disable thinking — bound it; Flash: disable thinking
    # so it can't starve the JSON output.
    attempts = [("gemini-2.5-pro", k, "Gemini 2.5 Pro", 1024) for k in pro_keys]
    attempts += [("gemini-2.5-flash", k, "Gemini 2.5 Flash", 0) for k in all_keys]

    last_error = None
    for model, api_key, label, think_budget in attempts:
        # Key routing: "AQ."-prefixed keys are Vertex AI EXPRESS keys and
        # only work on aiplatform.googleapis.com; classic "AIza" keys use
        # generativelanguage.googleapis.com. Try the matching endpoint
        # first, the other as backup.
        vertex_url = f"https://aiplatform.googleapis.com/v1/publishers/google/models/{model}:generateContent"
        genlang_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        urls = [vertex_url, genlang_url] if api_key.startswith("AQ.") else [genlang_url, vertex_url]

        payload = {
            "contents": [{"parts": [{"text": full_prompt}]}],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
                "responseMimeType": "application/json",
                "thinkingConfig": {"thinkingBudget": think_budget},
            },
        }
        headers = {"Content-Type": "application/json", "X-goog-api-key": api_key}

        for url in urls:
            try:
                start = time.time()
                resp = requests.post(url, json=payload, headers=headers, timeout=60)
                # Some Google gateways only accept the key as a query param
                # (?key=...) instead of the X-goog-api-key header — retry once.
                if resp.status_code in (400, 401, 403):
                    resp = requests.post(f"{url}?key={api_key}", json=payload,
                                         headers={"Content-Type": "application/json"}, timeout=60)
                elapsed = time.time() - start

                if resp.status_code != 200:
                    last_error = f"{label} HTTP {resp.status_code}: {resp.text[:150]}"
                    continue  # try the other endpoint, then next key/model

                data = resp.json()
                candidate = data["candidates"][0]
                if candidate.get("finishReason") == "MAX_TOKENS":
                    last_error = f"{label}: truncated (hit max output tokens)"
                    continue
                text = candidate["content"]["parts"][0]["text"]
                return {
                    "raw_text": text,
                    "source": "gemini",
                    "model": label,
                    "time": round(elapsed, 1),
                    "skipped": False,
                }
            except Exception as e:
                last_error = f"{label}: {e}"
                continue

    return {"error": last_error or "All Gemini attempts failed", "source": "gemini", "skipped": True}


def _call_groq(prompt: str, system: str = None, max_tokens: int = 2048, temperature: float = 0.15, json_mode: bool = True) -> dict:
    """Call Llama 3.3 70B via Groq.

    json_mode=True requests Groq's structured JSON object output mode —
    only valid when the response schema is a JSON object (not an array,
    e.g. NOT for SCAN_SYSTEM which returns a JSON array).
    """
    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key:
        return {"error": "No GROQ_API_KEY in .env", "skipped": True, "source": "groq"}

    try:
        from groq import Groq
    except ImportError:
        return {"error": "groq package not installed (pip install groq)", "skipped": True, "source": "groq"}

    try:
        start = time.time()
        client = Groq(api_key=api_key)
        kwargs = dict(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system or ANALYZE_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        chat = client.chat.completions.create(**kwargs)
        elapsed = time.time() - start

        text = chat.choices[0].message.content
        return {
            "raw_text": text,
            "source": "groq",
            "model": "Llama 3.3 70B (Groq)",
            "time": round(elapsed, 1),
            "skipped": False,
        }

    except Exception as e:
        return {"error": str(e), "source": "groq", "skipped": True}


def _strip_think(text: str) -> str:
    """Strip <think>...</think> reasoning blocks (DeepSeek R1, Qwen3)."""
    if "<think>" in text:
        after = text.find("</think>")
        return text[after + 8:].strip() if after >= 0 else text
    return text


_NVIDIA_BASE_URLS = [
    "https://integrate.api.nvidia.com/v1/chat/completions",
    "https://ai.api.nvidia.com/v1/chat/completions",
]


def _nvidia_keys(key_envs: list) -> list:
    """All distinct non-empty keys from the env list (dedicated first,
    generic NVIDIA_API_KEY fallback second)."""
    keys = []
    for e in key_envs:
        k = os.getenv(e, "").strip()
        if k and k not in keys:
            keys.append(k)
    return keys


# NVIDIA NIM model chains: primary model first, then shared fallbacks so a
# delisted/renamed primary never kills the whole provider.
_DEEPSEEK_MODELS = [
    ("deepseek-ai/deepseek-v4-pro", "DeepSeek V4 Pro (NVIDIA)"),
    ("meta/llama-3.3-70b-instruct", "Llama 3.3 70B (NVIDIA)"),
    ("nvidia/llama-3.1-nemotron-ultra-253b-v1", "Nemotron Ultra 253B (NVIDIA)"),
]
_QWEN_MODELS = [
    ("qwen/qwen3.5-397b-a17b", "Qwen 3.5 397B (NVIDIA)"),
    ("nvidia/llama-3.1-nemotron-ultra-253b-v1", "Nemotron Ultra 253B (NVIDIA)"),
    ("meta/llama-3.3-70b-instruct", "Llama 3.3 70B (NVIDIA)"),
]


def _call_nvidia_model(source: str, models: list, key_envs: list,
                       prompt: str, system: str, max_tokens: int,
                       temperature: float) -> dict:
    """Shared NVIDIA NIM caller (REST). Tries each model in the chain across
    every available key and gateway — a per-key 'Public API Endpoints' issue
    (plain '404 page not found') or a delisted model shouldn't kill the
    provider. Reasoning models think in <think> blocks that consume output
    tokens — enforce a generous floor so the JSON answer survives."""
    keys = _nvidia_keys(key_envs)
    if not keys:
        return {"error": f"No {key_envs[0]} in .env", "skipped": True, "source": source}

    last_error = None
    for model_id, label in models:
        payload = {
            "model": model_id,
            "messages": [{"role": "user", "content": f"{system or ANALYZE_SYSTEM}\n\n{prompt}"}],
            "max_tokens": max(max_tokens, 8192),  # room for <think> + JSON
            "temperature": temperature,
        }
        for api_key in keys:
            headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
            for url in _NVIDIA_BASE_URLS:
                try:
                    start = time.time()
                    resp = requests.post(url, json=payload, headers=headers, timeout=120)
                    elapsed = time.time() - start
                    if resp.status_code != 200:
                        last_error = f"{model_id}: HTTP {resp.status_code}: {resp.text[:100]}"
                        continue
                    text = _strip_think(resp.json()["choices"][0]["message"]["content"] or "")
                    if not text.strip():
                        last_error = f"{model_id}: empty response"
                        continue
                    return {
                        "raw_text": text,
                        "source": source,
                        "model": label,
                        "time": round(elapsed, 1),
                        "skipped": False,
                    }
                except Exception as e:
                    last_error = f"{model_id}: {e}"
                    continue

    return {"error": last_error or "All NVIDIA attempts failed", "skipped": True, "source": source}


def _call_deepseek(prompt: str, system: str = None, max_tokens: int = 2048, temperature: float = 0.15, json_mode: bool = True) -> dict:
    """DeepSeek V4 Pro via NVIDIA NIM — primary reasoning analyst."""
    return _call_nvidia_model(
        "deepseek", _DEEPSEEK_MODELS,
        ["NVIDIA_DEEPSEEK_API_KEY", "NVIDIA_API_KEY"],
        prompt, system, max_tokens, temperature,
    )


def _call_qwen3(prompt: str, system: str = None, max_tokens: int = 2048, temperature: float = 0.15, json_mode: bool = True) -> dict:
    """Qwen 3.5 397B via NVIDIA NIM — first fallback."""
    return _call_nvidia_model(
        "qwen3", _QWEN_MODELS,
        ["NVIDIA_QWEN_API_KEY", "NVIDIA_API_KEY"],
        prompt, system, max_tokens, temperature,
    )


def _call_openrouter(prompt: str, system: str = None, max_tokens: int = 2048, temperature: float = 0.15, json_mode: bool = True) -> dict:
    """OpenRouter free pool — deepseek-r1:free, then llama-3.3-70b:free.
    Free tier is heavily rate-limited (~20 req/min, 50-200/day) — used as a
    backup pool, not a primary."""
    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        return {"error": "No OPENROUTER_API_KEY in .env", "skipped": True, "source": "openrouter"}

    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    # Free pool is congested at peak times — try several free models, then
    # the auto-router (picks any currently-live free model) as catch-all.
    models = [
        ("deepseek/deepseek-r1:free", "DeepSeek R1 (OpenRouter free)"),
        ("deepseek/deepseek-chat-v3-0324:free", "DeepSeek V3 (OpenRouter free)"),
        ("meta-llama/llama-3.3-70b-instruct:free", "Llama 3.3 70B (OpenRouter free)"),
        ("openai/gpt-oss-120b:free", "GPT-OSS 120B (OpenRouter free)"),
        ("openrouter/free", "Auto free model (OpenRouter)"),
    ]

    last_error = None
    for model_id, label in models:
        payload = {
            "model": model_id,
            "messages": [
                {"role": "system", "content": system or ANALYZE_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": max(max_tokens, 8192) if "r1" in model_id else max_tokens,
            "temperature": temperature,
        }
        try:
            start = time.time()
            resp = requests.post(url, json=payload, headers=headers, timeout=120)
            elapsed = time.time() - start
            if resp.status_code != 200:
                last_error = f"{model_id} HTTP {resp.status_code}: {resp.text[:120]}"
                continue
            text = _strip_think(resp.json()["choices"][0]["message"]["content"] or "")
            if not text.strip():
                last_error = f"{model_id}: empty response"
                continue
            return {
                "raw_text": text,
                "source": "openrouter",
                "model": label,
                "time": round(elapsed, 1),
                "skipped": False,
            }
        except Exception as e:
            last_error = str(e)
            continue

    return {"error": last_error or "All OpenRouter models failed", "skipped": True, "source": "openrouter"}


# ─── Provider availability check (cached) ───────────────────

_availability_cache: dict = {}
_availability_checked_at: float = 0.0
_AVAILABILITY_TTL = 300  # re-check every 5 minutes

# Which env vars give a provider its key(s)
_KEY_ENVS = {
    "deepseek":   ["NVIDIA_DEEPSEEK_API_KEY", "NVIDIA_API_KEY"],
    "qwen3":      ["NVIDIA_QWEN_API_KEY", "NVIDIA_API_KEY"],
    "gemini":     ["GEMINI_PRO_API_KEYS", "GEMINI_API_KEY"],
    "openrouter": ["OPENROUTER_API_KEY"],
    "groq":       ["GROQ_API_KEY"],
}


def _has_key(provider: str) -> bool:
    return any(os.getenv(e, "").strip() for e in _KEY_ENVS.get(provider, []))


def _check_availability(timeout: int = 8, force: bool = False) -> dict:
    """
    Ping all configured AI providers in parallel with a tiny request.
    Result is cached for 5 minutes — only runs once per session unless
    a provider fails mid-analysis (force=True re-runs immediately).
    Pings are minimal (max_tokens<=5 or auth-only) to protect quotas.
    """
    global _availability_cache, _availability_checked_at

    if not force and _availability_cache and (time.time() - _availability_checked_at < _AVAILABILITY_TTL):
        return _availability_cache

    ping = "READY?"

    def _test_nvidia_key(key_envs, models):
        for model_id, _label in models:
            for api_key in _nvidia_keys(key_envs):
                for url in _NVIDIA_BASE_URLS:
                    try:
                        r = requests.post(
                            url,
                            json={"model": model_id,
                                  "messages": [{"role": "user", "content": ping}],
                                  "max_tokens": 5},
                            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                            timeout=timeout,
                        )
                        if r.status_code == 200:
                            return True
                    except Exception:
                        continue
        return False

    def _test_deepseek():
        return _test_nvidia_key(_KEY_ENVS["deepseek"], _DEEPSEEK_MODELS)

    def _test_qwen3():
        return _test_nvidia_key(_KEY_ENVS["qwen3"], _QWEN_MODELS)

    def _test_gemini():
        _, all_keys = _gemini_keys()
        vertex = "https://aiplatform.googleapis.com/v1/publishers/google/models/gemini-2.5-flash:generateContent"
        genlang = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"
        body = {"contents": [{"parts": [{"text": ping}]}],
                "generationConfig": {"maxOutputTokens": 5}}
        for api_key in all_keys:
            url = vertex if api_key.startswith("AQ.") else genlang
            # header auth, then ?key= auth — some gateways accept only one
            for kwargs in (
                {"headers": {"X-goog-api-key": api_key}},
                {"params": {"key": api_key}},
            ):
                try:
                    r = requests.post(url, json=body, timeout=timeout, **kwargs)
                    if r.status_code == 200:
                        return True
                except Exception:
                    continue
        return False

    def _test_openrouter():
        api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
        if not api_key:
            return False
        try:
            # Auth-only check — costs zero tokens and zero daily requests
            r = requests.get(
                "https://openrouter.ai/api/v1/key",
                headers={"Authorization": f"Bearer {api_key}"}, timeout=timeout,
            )
            return r.status_code == 200
        except Exception:
            return False

    def _test_groq():
        api_key = os.getenv("GROQ_API_KEY", "")
        if not api_key:
            return False
        try:
            from groq import Groq
            client = Groq(api_key=api_key)
            r = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": ping}],
                max_tokens=5,
            )
            return bool(r.choices[0].message.content)
        except Exception:
            return False

    testers = {
        "deepseek": _test_deepseek,
        "qwen3": _test_qwen3,
        "gemini": _test_gemini,
        "openrouter": _test_openrouter,
        "groq": _test_groq,
    }

    available = {name: False for name in testers}
    with ThreadPoolExecutor(max_workers=len(testers)) as pool:
        fs = {pool.submit(fn): name for name, fn in testers.items()}
        try:
            # Each tester may try several key/endpoint combos sequentially —
            # give the overall window enough room (was timeout+2, which cut
            # multi-attempt testers off and falsely marked them unreachable).
            for f in as_completed(fs, timeout=timeout * 4 + 5):
                name = fs[f]
                try:
                    available[name] = f.result()
                except Exception:
                    available[name] = False
        except Exception:
            # TimeoutError — mark any still-pending futures as offline
            for future, name in fs.items():
                if not future.done():
                    available[name] = False

    _availability_cache = available
    _availability_checked_at = time.time()
    return available


# Provider priority order and display metadata
_PROVIDERS = [
    ("deepseek",   "DeepSeek V4 Pro (NVIDIA)",  CYAN,    _call_deepseek),
    ("qwen3",      "Qwen 3.5 397B (NVIDIA)",    MAGENTA, _call_qwen3),
    ("gemini",     "Gemini 2.5 Pro/Flash",      BLUE,    _call_gemini),
    ("openrouter", "OpenRouter free pool",      YELLOW,  _call_openrouter),
    ("groq",       "Groq Llama 3.3 70B",        GREEN,   _call_groq),
]


def _pick_providers(available: dict) -> list:
    """Return up to 2 providers from priority list that passed availability check."""
    live = [(k, label, color, fn) for k, label, color, fn in _PROVIDERS if available.get(k)]
    return live[:2]


# ─── Single coin: AI full analysis ──────────────────────────

def analyze_coin_ai(symbol: str, tf_data: dict, user_prefs: dict = None,
                    btc_context: str = None, market_context: str = None) -> dict:
    """
    Send multi-timeframe indicator data to AI for full trade analysis.
    AI decides: direction, confidence, entry, SL, TP, timeframe, trade type.

    Args:
        symbol: coin symbol
        tf_data: dict of {timeframe: DataFrame} with indicators already added.
                 e.g. {"1m": df_1m, "3m": df_3m, "5m": df_5m, ...}
        btc_context: optional pre-built BTC regime text from build_btc_context(),
                 prepended to the prompt as backdrop for altcoin analysis.
        market_context: optional live microstructure text from
                 fetch_data.get_market_context() (funding rate, order book).
        user_prefs: optional dict with the trader's request for this analysis:
                 {"capital": float, "mode": "SCALP"/"INTRADAY"/"SWING"/"ANY",
                  "leverage": int or None, "profit_target": str or None,
                  "max_loss": str or None}

    Returns:
        dict with: direction, confidence, entry, stop_loss, take_profit,
                   leverage, timeframe, trade_type, hold_time, mode_note,
                   alternative_setups, reasons, risk_score, advice, decided_by
    """

    snapshot = build_indicator_snapshot(tf_data, symbol)

    request_block = ""
    if user_prefs:
        mode = user_prefs.get("mode", "ANY")
        mode_str = "Any — pick the best setup available" if mode == "ANY" else mode
        lev = user_prefs.get("leverage")
        lev_str = f"{lev}x" if lev else "AI decides based on confidence"
        profit_str = user_prefs.get("profit_target") or "AI decides — best R:R available"
        loss_str = user_prefs.get("max_loss") or "Standard risk management"
        capital = user_prefs.get('capital', 0)
        request_block = f"""
TRADER'S REQUEST FOR THIS ANALYSIS:
- Available capital: ${capital:,.2f}
- Preferred trading mode: {mode_str}
- Preferred leverage: {lev_str}
- Profit target: {profit_str}
- Max loss tolerance: {loss_str}

Once you pick entry/SL/TP/leverage, work out the approximate $ risk and $ reward
for ${capital:,.2f} capital at that leverage (position_size = capital * leverage;
risk = position_size * SL distance %; reward = position_size * TP distance %).
Use those dollar amounts — not just percentages — to judge whether the trade_type
(SCALP/INTRADAY/SWING) and hold_time you picked actually make sense (e.g. a SCALP
that nets less than ~$0.20 after friction on this capital is not worth it — prefer
INTRADAY/SWING with a deeper target instead, or NO_TRADE if nothing clears friction).
"""

    confidence_stats_text = format_confidence_stats_text()
    btc_block = f"\n{btc_context}\n" if btc_context else ""
    market_block = f"\n{market_context}\n" if market_context else ""

    prompt = f"""Analyze this coin using multi-timeframe data below. Use top-down analysis:
1. Start from 1D/1H to determine overall bias
2. Use 15m/5m to find entry structure
3. Use 3m/1m for precise entry timing
{btc_block}{market_block}
{snapshot}
{request_block}
{confidence_stats_text}
FRICTION: Total fees + slippage = 0.18% round-trip. Factor this into your R:R calculation.
SL RULES: Place SL beyond structure (support/resistance/EMA). Min distance from entry: 0.30%.
{TP_RULES_TEXT}
TIMEFRAME: Pick the best entry timeframe and trade type (SCALP/INTRADAY/SWING).

If no clear setup exists across ANY timeframe, return direction: "NO_TRADE" with confidence: 0.
Output ONLY valid JSON as specified in your instructions."""


    cached = bool(_availability_cache) and (time.time() - _availability_checked_at < _AVAILABILITY_TTL)
    cache_note = f"{DIM}(cached){RESET}" if cached else f"{DIM}(checking...){RESET}"

    print(f"\n  {CYAN}{BOLD}{'=' * 55}{RESET}")
    print(f"  {CYAN}{BOLD}  AI TRADE ANALYSIS — {symbol}{RESET}")
    print(f"  {CYAN}{BOLD}{'=' * 55}{RESET}")

    available = _check_availability()
    active = _pick_providers(available)

    if not active:
        tried = ", ".join(k for k, _, _, _ in _PROVIDERS if _has_key(k))
        print(f"  {RED}{BOLD}  No AI providers reachable.{RESET} {DIM}(tried: {tried or 'none configured'}){RESET}")
        return {
            "direction": "NO_TRADE", "confidence": 0,
            "entry": None, "stop_loss": None, "take_profit": None, "leverage": 0,
            "reasons": ["All AI providers unreachable — check API keys and connectivity"],
            "risk_score": "HIGH", "advice": "Try again later", "decided_by": "NONE",
        }

    # Show each provider: online / offline / no key
    for key, label, color, _ in _PROVIDERS:
        has_key = _has_key(key)
        if available.get(key):
            role = "primary" if key == active[0][0] else "second opinion"
            print(f"  {color}{BOLD}  ✓ {label}{RESET} {DIM}({role}) {cache_note}{RESET}")
        elif has_key:
            print(f"  {YELLOW}  ✗ {label} — unreachable{RESET}")
        else:
            print(f"  {DIM}  — {label} — no API key{RESET}")

    print(f"  {DIM}Sending indicator data to AIs...{RESET}")

    # Call active providers in parallel
    results = {}
    with ThreadPoolExecutor(max_workers=len(active)) as pool:
        fs = {pool.submit(fn, prompt, ANALYZE_SYSTEM, 6144): key
              for key, _, _, fn in active}
        for future in as_completed(fs):
            key = fs[future]
            try:
                results[key] = future.result()
            except Exception as e:
                results[key] = {"error": str(e), "skipped": True, "source": key}

    # If every active provider failed mid-call (e.g. rate limit, 404),
    # cascade to the next untried provider that has an API key.
    if all(results.get(key, {}).get("skipped", True) for key, _, _, _ in active):
        attempted = {key for key, _, _, _ in active}
        for key, label, color, fn in _PROVIDERS:
            if key in attempted or not _has_key(key):
                continue
            print(f"  {DIM}All active providers failed — falling back to {label}...{RESET}")
            try:
                r = fn(prompt, ANALYZE_SYSTEM, 6144)
            except Exception as e:
                r = {"error": str(e), "skipped": True, "source": key}
            results[key] = r
            active.append((key, label, color, fn))
            if not r.get("skipped", True):
                break

    # Display and parse each result in priority order
    analyses = {}
    for i, (key, label, color, _) in enumerate(active, 1):
        raw = results.get(key, {"skipped": True, "error": "no response"})
        ok = not raw.get("skipped", True)
        if ok:
            parsed = _parse_analysis_response(raw.get("raw_text", ""))
            analyses[key] = parsed
            _print_analysis(parsed, raw, i, len(active), color)
        else:
            print(f"\n  {color}{BOLD}[{i}/{len(active)}] {label}{RESET}")
            print(f"  {YELLOW}  SKIPPED: {raw.get('error', '?')[:100]}{RESET}")

    # Merge: first available is primary, second is second opinion
    primary_key = next((k for k, _, _, _ in active if k in analyses), None)
    secondary_key = next((k for k, _, _, _ in active if k in analyses and k != primary_key), None)

    if not primary_key:
        return {
            "direction": "NO_TRADE", "confidence": 0,
            "entry": None, "stop_loss": None, "take_profit": None, "leverage": 0,
            "reasons": ["All AI calls failed — cannot analyze"],
            "risk_score": "HIGH", "advice": "Try again later", "decided_by": "NONE",
        }

    primary_label = next(label for k, label, _, _ in _PROVIDERS if k == primary_key)
    final = analyses[primary_key]
    final["decided_by"] = primary_label

    # ─── Ensemble agreement ──────────────────────────────────
    # Two providers agreeing on direction is a stronger signal than one;
    # disagreement should temper confidence/leverage rather than just be noted.
    if secondary_key:
        sec = analyses[secondary_key]
        sec_label = next(label for k, label, _, _ in _PROVIDERS if k == secondary_key)

        if sec["direction"] == final["direction"]:
            # Agreement: average numeric SL/TP/entry/leverage from both providers
            # for a less idiosyncratic, blended target.
            if final["direction"] in ("LONG", "SHORT", "WAIT"):
                numeric_fields = ["entry", "stop_loss", "take_profit"]
                if all(final.get(f) is not None and sec.get(f) is not None for f in numeric_fields):
                    for f in numeric_fields:
                        final[f] = (final[f] + sec[f]) / 2
            if final.get("leverage") and sec.get("leverage"):
                final["leverage"] = round((final["leverage"] + sec["leverage"]) / 2)
            final["reasons"].append(
                f"Ensemble: {sec_label} agrees on {sec['direction']} (conf {sec['confidence']}) — "
                f"SL/TP/leverage averaged across both providers"
            )
        else:
            final["reasons"].append(
                f"Note: {sec_label} suggested {sec['direction']} (conf {sec['confidence']})"
            )
            # Disagreement on an actionable trade — temper confidence/leverage
            # rather than acting on the primary's full conviction alone.
            if final["direction"] in ("LONG", "SHORT") and sec["direction"] != "NO_TRADE":
                capped_conf = min(final["confidence"], 6)
                capped_lev = min(final.get("leverage", 5), 5)
                if capped_conf < final["confidence"] or capped_lev < final.get("leverage", 5):
                    final["reasons"].append(
                        f"Providers disagree on direction — confidence capped at {capped_conf}/10 "
                        f"and leverage capped at {capped_lev}x"
                    )
                final["confidence"] = capped_conf
                final["leverage"] = capped_lev

    # Display final verdict
    print(f"\n  {CYAN}{BOLD}{'─' * 55}{RESET}")
    dir_str = final["direction"]
    conf = final["confidence"]
    if dir_str == "NO_TRADE":
        print(f"  {YELLOW}{BOLD}  AI VERDICT: NO TRADE{RESET} {DIM}(conf {conf}/10){RESET}")
    elif dir_str == "WAIT":
        wait_dir = final.get("wait_direction", "?")
        wait_cond = final.get("wait_condition", "")
        wait_color = GREEN if wait_dir == "LONG" else RED
        print(f"  {YELLOW}{BOLD}  AI VERDICT: WAITING FOR ENTRY{RESET} {DIM}(conf {conf}/10){RESET}")
        print(f"  {wait_color}{BOLD}  Bias: {wait_dir}{RESET} {DIM}| Risk: {final.get('risk_score', '?')}{RESET}")
        if wait_cond:
            print(f"  {YELLOW}  Trigger: {wait_cond}{RESET}")
    else:
        dir_color = GREEN if dir_str == "LONG" else RED
        print(f"  {dir_color}{BOLD}  AI VERDICT: {dir_str}{RESET} {DIM}| Confidence: {conf}/10 | Risk: {final.get('risk_score', '?')}{RESET}")
    print(f"  {DIM}  Decided by: {final['decided_by']}{RESET}")
    if final.get("mode_note"):
        print(f"  {YELLOW}  Note: {final['mode_note']}{RESET}")
    if final.get("advice"):
        print(f"  {CYAN}  Advice: {final['advice'][:150]}{RESET}")

    # Other setups the AI considered (other trade types/modes)
    alts = final.get("alternative_setups") or []
    for alt in alts:
        a_dir = str(alt.get("direction", "?")).upper()
        a_type = str(alt.get("trade_type", "?")).upper()
        a_color = GREEN if a_dir == "LONG" else RED
        a_entry, a_sl, a_tp = alt.get("entry"), alt.get("stop_loss"), alt.get("take_profit")
        line = f"  {DIM}  Also viable — {a_color}{a_type} {a_dir}{RESET}{DIM}"
        if a_entry and a_sl and a_tp:
            line += f": entry ${a_entry:,.6f} | SL ${a_sl:,.6f} | TP ${a_tp:,.6f} | {alt.get('leverage', '?')}x"
        if alt.get("reason"):
            line += f" — {alt['reason']}"
        print(line + RESET)

    print(f"  {CYAN}{BOLD}{'=' * 55}{RESET}\n")

    logger.info(f"[AI Analysis] {symbol}: {dir_str} conf={conf} by {final['decided_by']}")

    return final


def _print_analysis(analysis: dict, raw: dict, index: int, total: int, color: str):
    """Print a single AI's analysis result."""
    model = raw.get("model", raw.get("source", "?"))
    elapsed = raw.get("time", "?")
    direction = analysis.get("direction", "NO_TRADE")
    conf = analysis.get("confidence", 0)
    risk = analysis.get("risk_score", "?")

    dir_color = GREEN if direction == "LONG" else (RED if direction == "SHORT" else YELLOW)
    r_color = GREEN if risk == "LOW" else (YELLOW if risk == "MEDIUM" else RED)

    print(f"\n  {color}{BOLD}[{index}/{total}] {model}{RESET} {DIM}({elapsed}s){RESET}")
    print(f"  {dir_color}{BOLD}  {direction}{RESET} | Confidence: {YELLOW}{conf}/10{RESET} | Risk: {r_color}{risk}{RESET}")

    # Show key reasons (max 5)
    reasons = analysis.get("reasons", [])
    for i, r in enumerate(reasons[:5], 1):
        if any(kw in r.upper() for kw in ["WARNING", "REJECT", "RISK", "DANGER"]):
            print(f"    {RED}[{i}] {r}{RESET}")
        else:
            print(f"    {GREEN}[{i}] {r}{RESET}")
    if len(reasons) > 5:
        print(f"    {DIM}... +{len(reasons)-5} more{RESET}")

    # Show SL/TP if provided
    entry = analysis.get("entry")
    sl = analysis.get("stop_loss")
    tp = analysis.get("take_profit")
    if entry and sl and tp:
        sl_dist = abs(entry - sl) / entry * 100
        tp_dist = abs(tp - entry) / entry * 100
        rr = tp_dist / sl_dist if sl_dist > 0 else 0
        print(f"    {DIM}Entry: ${entry:,.6f} | SL: ${sl:,.6f} ({sl_dist:.2f}%) | TP: ${tp:,.6f} ({tp_dist:.2f}%) | R:R: {rr:.2f}:1{RESET}")


# ─── Re-analyze an open position ─────────────────────────────

def reanalyze_position_ai(symbol: str, tf_data: dict, position: dict, quiet: bool = False) -> dict:
    """
    Re-analyze an OPEN position with fresh multi-timeframe data and the
    trade's current state. AI recommends HOLD / MOVE_SL / MOVE_TP /
    MOVE_BOTH / CLOSE_NOW with optional new SL/TP values.

    Args:
        symbol: coin symbol
        tf_data: dict of {timeframe: DataFrame} with indicators already added.
        position: dict with direction, entry_price, current_price, stop_loss,
                  take_profit, leverage, pnl_pct, sl_stage, hold_time

    Returns:
        dict with: action, new_stop_loss, new_take_profit, urgency,
                   reasons, advice, decided_by
    """
    snapshot = build_indicator_snapshot(tf_data, symbol)

    direction = position["direction"]
    entry = position["entry_price"]
    current_price = position["current_price"]
    sl = position["stop_loss"]
    tp = position["take_profit"]
    leverage = position["leverage"]
    pnl_pct = position.get("pnl_pct", 0)
    sl_stage = position.get("sl_stage", "INITIAL")
    hold_time = position.get("hold_time", "?")

    sl_dist = abs(entry - sl) / entry * 100
    tp_dist = abs(tp - entry) / entry * 100

    prompt = f"""Re-analyze this OPEN position using the fresh multi-timeframe data below.

CURRENT TRADE ({symbol}):
  Direction:        {direction}
  Entry price:      ${entry:,.6f}
  Current price:    ${current_price:,.6f}
  Current SL:       ${sl:,.6f} ({sl_dist:.2f}% from entry)
  Current TP:       ${tp:,.6f} ({tp_dist:.2f}% from entry)
  Leverage:         {leverage}x
  Unrealized P&L:   {pnl_pct:+.2f}%
  Trailing stage:   {sl_stage}
  Time in trade:    {hold_time}

{snapshot}

FRICTION: Total fees + slippage = 0.18% round-trip.

Decide whether to HOLD as-is, MOVE_SL (tighten only — never loosen beyond original risk),
MOVE_TP (extend on strong continuation or pull in if momentum is fading), MOVE_BOTH, or
CLOSE_NOW (only if the trend has clearly reversed or a major risk event is visible).
Output ONLY valid JSON as specified in your instructions."""


    cached = bool(_availability_cache) and (time.time() - _availability_checked_at < _AVAILABILITY_TTL)
    cache_note = f"{DIM}(cached){RESET}" if cached else f"{DIM}(checking...){RESET}"

    if not quiet:
        print(f"\n  {CYAN}{BOLD}{'=' * 55}{RESET}")
        print(f"  {CYAN}{BOLD}  AI POSITION RE-ANALYSIS — {symbol}{RESET}")
        print(f"  {CYAN}{BOLD}{'=' * 55}{RESET}")

    available = _check_availability()
    active = _pick_providers(available)

    if not active:
        tried = ", ".join(k for k, _, _, _ in _PROVIDERS if _has_key(k))
        if not quiet:
            print(f"  {RED}{BOLD}  No AI providers reachable.{RESET} {DIM}(tried: {tried or 'none configured'}){RESET}")
        return {
            "action": "HOLD", "new_stop_loss": None, "new_take_profit": None,
            "urgency": "LOW", "reasons": ["All AI providers unreachable — check API keys and connectivity"],
            "advice": "Try again later", "decided_by": "NONE",
        }

    if not quiet:
        for key, label, color, _ in _PROVIDERS:
            has_key = _has_key(key)
            if available.get(key):
                role = "primary" if key == active[0][0] else "second opinion"
                print(f"  {color}{BOLD}  ✓ {label}{RESET} {DIM}({role}) {cache_note}{RESET}")
            elif has_key:
                print(f"  {YELLOW}  ✗ {label} — unreachable{RESET}")
            else:
                print(f"  {DIM}  — {label} — no API key{RESET}")

        print(f"  {DIM}Sending position + indicator data to AI...{RESET}")

    # Re-analysis only needs ONE good answer — try providers sequentially,
    # starting with the active (available) ones, then fall back to any
    # other configured provider.
    order = list(active)
    attempted = {key for key, _, _, _ in order}
    for key, label, color, fn in _PROVIDERS:
        if key not in attempted and _has_key(key):
            order.append((key, label, color, fn))
            attempted.add(key)

    action_colors = {
        "HOLD": GREEN, "MOVE_SL": CYAN, "MOVE_TP": CYAN,
        "MOVE_BOTH": CYAN, "CLOSE_NOW": RED,
    }

    for key, label, color, fn in order:
        try:
            raw = fn(prompt, REANALYZE_SYSTEM, 2048)
        except Exception as e:
            raw = {"error": str(e), "skipped": True, "source": key}

        if raw.get("skipped", True):
            if not quiet:
                print(f"  {YELLOW}  {label}: SKIPPED — {raw.get('error', '?')[:100]}{RESET}")
            continue

        parsed = _parse_reanalysis_response(raw.get("raw_text", ""))
        parsed["decided_by"] = label

        action = parsed["action"]
        urgency = parsed.get("urgency", "LOW")
        a_color = action_colors.get(action, YELLOW)
        u_color = GREEN if urgency == "LOW" else (YELLOW if urgency == "MEDIUM" else RED)

        if not quiet:
            print(f"\n  {color}{BOLD}{label}{RESET} {DIM}({raw.get('time', '?')}s){RESET}")
            print(f"  {a_color}{BOLD}  ACTION: {action}{RESET} {DIM}| Urgency: {u_color}{urgency}{RESET}")

            for i, r in enumerate(parsed.get("reasons", [])[:5], 1):
                print(f"    {DIM}[{i}] {r}{RESET}")

            if parsed.get("new_stop_loss"):
                print(f"    {CYAN}New SL: ${parsed['new_stop_loss']:,.6f}{RESET}")
            if parsed.get("new_take_profit"):
                print(f"    {CYAN}New TP: ${parsed['new_take_profit']:,.6f}{RESET}")
            if parsed.get("advice"):
                print(f"  {CYAN}  Advice: {parsed['advice'][:150]}{RESET}")

            print(f"  {CYAN}{BOLD}{'=' * 55}{RESET}\n")

        logger.info(f"[AI Reanalysis] {symbol}: {action} (urgency {urgency}) by {label}")
        return parsed

    if not quiet:
        print(f"  {RED}{BOLD}  All AI providers failed.{RESET}")
    return {
        "action": "HOLD", "new_stop_loss": None, "new_take_profit": None,
        "urgency": "LOW", "reasons": ["All AI calls failed — cannot re-analyze"],
        "advice": "Try again later", "decided_by": "NONE",
    }


# ─── Batch scan: AI picks best from pre-filtered list ────────

def scan_coins_ai(candidates: list) -> list:
    """
    Send pre-filtered coin data to AI in one batch prompt.
    AI picks the best setups (max 3).

    Args:
        candidates: list of dicts with {symbol, snapshot} (from build_indicator_snapshot)

    Returns:
        list of AI trade signals sorted by confidence
    """
    if not candidates:
        return []

    # Build batch prompt
    coins_text = "\n\n".join(c["snapshot"] for c in candidates)

    prompt = f"""Analyze these {len(candidates)} coins using multi-timeframe data. Pick the BEST setups (max 3).

Use top-down analysis for each coin:
1. 1D/1H → overall bias
2. 15m/5m → entry structure
3. 3m/1m → precise timing

FRICTION: Total fees + slippage = 0.18% round-trip.
SL RULES: Place SL beyond structure. Min distance: 0.30%.
{TP_RULES_TEXT}
LEVERAGE: Max 25x. Scale: conf 7→5x, 8→8x, 9→15x, 10→25x.
For each pick, specify timeframe, trade_type (SCALP/INTRADAY/SWING), and hold_time.

If NO coins have a good setup, return an empty array [].

{coins_text}

Output ONLY a valid JSON array as specified in your instructions."""

    cached = bool(_availability_cache) and (time.time() - _availability_checked_at < _AVAILABILITY_TTL)
    cache_note = f"{DIM}(cached){RESET}" if cached else f"{DIM}(checking...){RESET}"

    print(f"\n  {CYAN}{BOLD}{'=' * 55}{RESET}")
    print(f"  {CYAN}{BOLD}  AI BATCH ANALYSIS — {len(candidates)} coins{RESET}")
    print(f"  {CYAN}{BOLD}{'=' * 55}{RESET}")

    available = _check_availability()
    active = _pick_providers(available)

    if not active:
        print(f"  {RED}{BOLD}  No AI providers reachable.{RESET}")
        return []

    for key, label, color, _ in _PROVIDERS:
        has_key = _has_key(key)
        if available.get(key):
            role = "primary" if key == active[0][0] else "second opinion"
            print(f"  {color}{BOLD}  ✓ {label}{RESET} {DIM}({role}) {cache_note}{RESET}")
        elif has_key:
            print(f"  {YELLOW}  ✗ {label} — unreachable{RESET}")
        else:
            print(f"  {DIM}  — {label} — no API key{RESET}")

    # Call active providers in parallel
    raw_results = {}
    with ThreadPoolExecutor(max_workers=len(active)) as pool:
        fs = {pool.submit(fn, prompt, SCAN_SYSTEM, 6144, 0.15, False): key
              for key, _, _, fn in active}
        for future in as_completed(fs):
            key = fs[future]
            try:
                raw_results[key] = future.result()
            except Exception as e:
                raw_results[key] = {"error": str(e), "skipped": True, "source": key}

    # If every active provider failed mid-call (e.g. rate limit, 404),
    # cascade to the next untried provider that has an API key.
    if all(raw_results.get(key, {}).get("skipped", True) for key, _, _, _ in active):
        attempted = {key for key, _, _, _ in active}
        for key, label, color, fn in _PROVIDERS:
            if key in attempted or not _has_key(key):
                continue
            print(f"  {DIM}All active providers failed — falling back to {label}...{RESET}")
            try:
                r = fn(prompt, SCAN_SYSTEM, 6144, 0.15, False)
            except Exception as e:
                r = {"error": str(e), "skipped": True, "source": key}
            raw_results[key] = r
            active.append((key, label, color, fn))
            if not r.get("skipped", True):
                break

    # Parse and display in priority order
    all_picks = {}
    for i, (key, label, color, _) in enumerate(active, 1):
        raw = raw_results.get(key, {"skipped": True})
        ok = not raw.get("skipped", True)
        picks = _parse_scan_response(raw.get("raw_text", "")) if ok else []
        all_picks[key] = picks
        elapsed = raw.get("time", "?")

        print(f"\n  {color}{BOLD}[{i}/{len(active)}] {label}{RESET} {DIM}({elapsed}s){RESET}")
        if ok and picks:
            for p in picks:
                dc = GREEN if p["direction"] == "LONG" else RED
                print(f"    {dc}{p['direction']}{RESET} {p['symbol']} conf={p['confidence']} | {p.get('reasons', [''])[0][:60]}")
        elif ok:
            print(f"    {YELLOW}No setups found{RESET}")
        else:
            print(f"    {YELLOW}SKIPPED: {raw.get('error', '?')[:80]}{RESET}")

    # Use first provider that returned picks; tag secondary as note
    final_picks = []
    decided_by = "NONE"
    for key, label, color, _ in active:
        picks = all_picks.get(key, [])
        if picks:
            final_picks = picks
            decided_by = label
            break

    if not final_picks:
        # Try merging all picks from all providers
        for key, label, color, _ in active:
            final_picks.extend(all_picks.get(key, []))

    # Tag each with decided_by
    for p in final_picks:
        p["decided_by"] = decided_by

    # Sort by confidence
    final_picks.sort(key=lambda x: x["confidence"], reverse=True)

    n = len(final_picks)
    print(f"\n  {CYAN}{BOLD}{'─' * 55}{RESET}")
    if n > 0:
        print(f"  {GREEN}{BOLD}  AI found {n} setup(s){RESET} {DIM}(decided by {decided_by}){RESET}")
    else:
        print(f"  {YELLOW}{BOLD}  NO SETUPS — AI found nothing worth trading{RESET}")
    print(f"  {CYAN}{BOLD}{'=' * 55}{RESET}\n")

    logger.info(f"[AI Scan] {n} picks from {len(candidates)} candidates (by {decided_by})")

    return final_picks


# ─── CLI test ────────────────────────────────────────────────

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    print("Testing AI analysis mode...")
    print("Use: from multi_ai_verifier import analyze_coin_ai, scan_coins_ai")
