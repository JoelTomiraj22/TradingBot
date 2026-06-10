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
  1. Gemini 2.5 Flash   (Google AI Studio — primary)
  2. Groq Llama 3.3 70B (Groq — fast second opinion)
  3. NVIDIA DeepSeek R1 (NVIDIA NIM — reasoning fallback)
  4. Qwen2.5 72B        (HuggingFace — last resort)

Setup:
  pip install groq openai
  Add to .env: GEMINI_API_KEY, GROQ_API_KEY, NVIDIA_API_KEY, HF_API_KEY
"""

import os
import json
import time
import requests
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from logger_setup import get_logger

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

**Trade Types:**
- SCALP (1m–5m entry): Hold 5–30 min. SL tight. Need very clear pattern + momentum.
- INTRADAY (5m–15m entry): Hold 30 min – 4 hours. More room for SL. Best R:R potential.
- SWING (30m–1h entry): Hold 4–24 hours. Wide SL. Only on very strong setups.

**Strict Rules:**
- NEVER chase: Price must be at a key level (EMA pullback, VWAP, S/R zone).
- Require 3+ confluences across timeframes.
- SL: Place beyond structure on the ENTRY timeframe. Never at round numbers.
- TP: Target next higher-TF S/R level. R:R must be 2.2:1+ after fees (0.18% friction).
- Leverage: Max 12x. Scale: 7→5x, 8→8x, 9→10x, 10→12x.
- Reject if: HTF trend conflicts, RSI extreme, overextended, chasing.

**Direction rules:**
- LONG / SHORT: Clear setup exists RIGHT NOW — price is at the entry level, take it.
- WAIT: Trend/bias is clear BUT price is NOT yet at the entry level (e.g. needs to pull back to EMA, reach a resistance). Use this instead of NO_TRADE when the direction is known and you'd trade it — just not yet.
- NO_TRADE: No clear bias, conflicting signals, or market is not worth trading at all.

**Output ONLY valid JSON (no markdown, no code fences):**
{
  "direction": "LONG" or "SHORT" or "WAIT" or "NO_TRADE",
  "confidence": 0-10,
  "entry": number,
  "stop_loss": number,
  "take_profit": number,
  "leverage": number,
  "timeframe": "1m" or "3m" or "5m" or "15m" or "30m" or "1h",
  "trade_type": "SCALP" or "INTRADAY" or "SWING",
  "hold_time": "estimated hold time e.g. 15min, 2h, etc.",
  "risk_score": "LOW" or "MEDIUM" or "HIGH",
  "wait_condition": "WAIT only: exact condition to watch e.g. 'Price pulls back to 15m EMA21 at $207.05'",
  "wait_direction": "WAIT only: the trade direction once condition is met — LONG or SHORT",
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

**Rules:**
- 3+ confluences across timeframes required
- Never chase — must be at a key level on the entry TF
- R:R must be 2.2:1+ after fees (0.18% total friction)
- Higher TF S/R levels are stronger — use them for TP targets
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
    "trade_type": "SCALP" or "INTRADAY" or "SWING",
    "hold_time": "estimated hold time",
    "risk_score": "LOW" or "MEDIUM" or "HIGH",
    "reasons": ["reason 1 (mention TF)", ...],
    "advice": "One line"
  }
]

Return [] if no good setups exist."""


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

    # S/R levels (important for all TFs)
    support = row.get("nearest_support", "N/A")
    resistance = row.get("nearest_resistance", "N/A")
    if support != "N/A" and not pd.isna(support):
        snapshot += f"\n    Support: {support:.6f} (near: {row.get('near_support', False)}) | Resistance: {resistance:.6f} (near: {row.get('near_resistance', False)})"

    if not full_detail:
        return snapshot

    # Full detail for entry timeframes
    vwap = row.get("vwap", 0)
    if vwap and not pd.isna(vwap):
        vwap_side = "ABOVE" if close > vwap else "BELOW"
        snapshot += f"\n    VWAP: {vwap:.6f} ({vwap_side})"

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

    # POC
    poc = row.get("poc_price", None)
    if poc and not pd.isna(poc):
        snapshot += f"\n    POC: {poc:.6f} (at HVN: {row.get('high_volume_node', False)})"

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


# ─── JSON parsers ────────────────────────────────────────────

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

        data = json.loads(clean)

        return {
            "direction": str(data.get("direction", "NO_TRADE")).upper(),
            "confidence": int(data.get("confidence", 0)),
            "entry": data.get("entry"),
            "stop_loss": data.get("stop_loss"),
            "take_profit": data.get("take_profit"),
            "leverage": int(data.get("leverage", 0)),
            "timeframe": str(data.get("timeframe", "5m")),
            "trade_type": str(data.get("trade_type", "SCALP")).upper(),
            "hold_time": str(data.get("hold_time", "15-30 min")),
            "risk_score": str(data.get("risk_score", "UNKNOWN")).upper(),
            "wait_condition": str(data.get("wait_condition", "")),
            "wait_direction": str(data.get("wait_direction", "")).upper(),
            "reasons": data.get("reasons", []),
            "advice": str(data.get("advice", "")),
        }

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
            "risk_score": "UNKNOWN",
            "wait_condition": "",
            "wait_direction": "",
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

        data = json.loads(clean)

        return {
            "action": str(data.get("action", "HOLD")).upper(),
            "new_stop_loss": data.get("new_stop_loss"),
            "new_take_profit": data.get("new_take_profit"),
            "urgency": str(data.get("urgency", "LOW")).upper(),
            "reasons": data.get("reasons", []),
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

        data = json.loads(clean)
        if not isinstance(data, list):
            data = [data]

        results = []
        for item in data:
            results.append({
                "symbol": str(item.get("symbol", "")),
                "direction": str(item.get("direction", "NO_TRADE")).upper(),
                "confidence": int(item.get("confidence", 0)),
                "entry": item.get("entry"),
                "stop_loss": item.get("stop_loss"),
                "take_profit": item.get("take_profit"),
                "leverage": int(item.get("leverage", 0)),
                "timeframe": str(item.get("timeframe", "5m")),
                "trade_type": str(item.get("trade_type", "SCALP")).upper(),
                "hold_time": str(item.get("hold_time", "15-30 min")),
                "risk_score": str(item.get("risk_score", "UNKNOWN")).upper(),
                "reasons": item.get("reasons", []),
                "advice": str(item.get("advice", "")),
            })
        return results

    except (json.JSONDecodeError, KeyError, IndexError, ValueError, TypeError):
        return []


# ─── AI Callers ──────────────────────────────────────────────

def _call_gemini(prompt: str, system: str = None, max_tokens: int = 2048) -> dict:
    """Call Gemini 2.5 Flash via Google AI Studio REST API."""
    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        return {"error": "No GEMINI_API_KEY in .env", "skipped": True, "source": "gemini"}

    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"

    full_prompt = f"{system or ANALYZE_SYSTEM}\n\n{prompt}"

    payload = {
        "contents": [
            {"parts": [{"text": full_prompt}]}
        ],
        "generationConfig": {
            "temperature": 0.3,
            "maxOutputTokens": max_tokens,
            "responseMimeType": "application/json",
            # Gemini 2.5 Flash reserves part of maxOutputTokens for internal
            # "thinking" by default, which can starve the actual JSON output
            # and cause it to be cut off mid-object. Disable it.
            "thinkingConfig": {"thinkingBudget": 0},
        }
    }

    headers = {
        "Content-Type": "application/json",
        "X-goog-api-key": api_key,
    }

    try:
        start = time.time()
        resp = requests.post(url, json=payload, headers=headers, timeout=45)
        elapsed = time.time() - start

        if resp.status_code != 200:
            error_msg = resp.text[:200]
            return {"error": f"HTTP {resp.status_code}: {error_msg}", "skipped": True, "source": "gemini"}

        data = resp.json()
        candidate = data["candidates"][0]
        if candidate.get("finishReason") == "MAX_TOKENS":
            return {"error": "Response truncated (hit max output tokens)", "skipped": True, "source": "gemini"}
        text = candidate["content"]["parts"][0]["text"]
        return {
            "raw_text": text,
            "source": "gemini",
            "model": "Gemini 2.5 Flash",
            "time": round(elapsed, 1),
            "skipped": False,
        }

    except Exception as e:
        return {"error": str(e), "source": "gemini", "skipped": True}


def _call_groq(prompt: str, system: str = None, max_tokens: int = 2048) -> dict:
    """Call Llama 3.3 70B via Groq."""
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
        chat = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system or ANALYZE_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            max_tokens=max_tokens,
            temperature=0.3,
        )
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


def _call_nvidia(prompt: str, system: str = None, max_tokens: int = 2048) -> dict:
    """
    Call NVIDIA NIM via direct REST (no openai package needed).
    Tries llama-3.1-70b-instruct first (widely available), then deepseek-r1.
    """
    api_key = os.getenv("NVIDIA_API_KEY", "")
    if not api_key:
        return {"error": "No NVIDIA_API_KEY in .env", "skipped": True, "source": "nvidia"}

    url = "https://integrate.api.nvidia.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    # Try models in order of availability; stop on first success
    models_to_try = [
        ("meta/llama-3.1-70b-instruct", "Llama 3.1 70B (NVIDIA)"),
        ("deepseek-ai/deepseek-r1", "DeepSeek R1 (NVIDIA)"),
        ("nvidia/llama-3.1-nemotron-70b-instruct", "Nemotron 70B (NVIDIA)"),
    ]

    messages = [
        {"role": "user", "content": f"{system or ANALYZE_SYSTEM}\n\n{prompt}"},
    ]

    last_error = None
    for model_id, model_label in models_to_try:
        payload = {
            "model": model_id,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.3,
        }
        try:
            start = time.time()
            resp = requests.post(url, json=payload, headers=headers, timeout=90)
            elapsed = time.time() - start

            if resp.status_code == 404:
                last_error = f"Model {model_id} not found (404)"
                continue
            if resp.status_code == 402:
                last_error = f"Model {model_id} requires paid access (402)"
                continue
            if resp.status_code != 200:
                last_error = f"HTTP {resp.status_code}: {resp.text[:150]}"
                continue

            text = resp.json()["choices"][0]["message"]["content"] or ""
            # DeepSeek R1 wraps reasoning in <think>...</think> — strip it
            if "<think>" in text:
                after = text.find("</think>")
                text = text[after + 8:].strip() if after >= 0 else text
            return {
                "raw_text": text,
                "source": "nvidia",
                "model": model_label,
                "time": round(elapsed, 1),
                "skipped": False,
            }
        except Exception as e:
            last_error = str(e)
            continue

    return {"error": last_error or "All NVIDIA models failed", "source": "nvidia", "skipped": True}


def _call_huggingface(prompt: str, system: str = None, max_tokens: int = 2048) -> dict:
    """Call Qwen2.5 72B via HuggingFace Serverless Inference API."""
    api_key = os.getenv("HF_API_KEY", "")
    if not api_key:
        return {"error": "No HF_API_KEY in .env", "skipped": True, "source": "huggingface"}

    model = "Qwen/Qwen2.5-72B-Instruct"
    url = "https://router.huggingface.co/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system or ANALYZE_SYSTEM},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.3,
    }

    try:
        start = time.time()
        resp = requests.post(url, json=payload, headers=headers, timeout=60)
        elapsed = time.time() - start
        if resp.status_code != 200:
            return {"error": f"HTTP {resp.status_code}: {resp.text[:200]}", "skipped": True, "source": "huggingface"}
        text = resp.json()["choices"][0]["message"]["content"]
        return {
            "raw_text": text,
            "source": "huggingface",
            "model": "Qwen2.5 72B (HuggingFace)",
            "time": round(elapsed, 1),
            "skipped": False,
        }
    except Exception as e:
        return {"error": str(e), "source": "huggingface", "skipped": True}


# ─── Provider availability check (cached) ───────────────────

_availability_cache: dict = {}
_availability_checked_at: float = 0.0
_AVAILABILITY_TTL = 300  # re-check every 5 minutes


def _check_availability(timeout: int = 15, force: bool = False) -> dict:
    """
    Ping all configured AI providers in parallel with a tiny request.
    Result is cached for 5 minutes — only runs once per session unless
    a provider fails mid-analysis (force=True re-runs immediately).
    """
    global _availability_cache, _availability_checked_at

    if not force and _availability_cache and (time.time() - _availability_checked_at < _AVAILABILITY_TTL):
        return _availability_cache

    ping = "Reply with the single word: READY"

    def _test_gemini():
        api_key = os.getenv("GEMINI_API_KEY", "")
        if not api_key:
            return False
        url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"
        payload = {"contents": [{"parts": [{"text": ping}]}],
                   "generationConfig": {"maxOutputTokens": 5}}
        try:
            r = requests.post(url, json=payload,
                              headers={"X-goog-api-key": api_key}, timeout=timeout)
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

    def _test_nvidia():
        api_key = os.getenv("NVIDIA_API_KEY", "")
        if not api_key:
            return False
        url = "https://integrate.api.nvidia.com/v1/chat/completions"
        try:
            r = requests.post(
                url,
                json={"model": "meta/llama-3.1-70b-instruct",  # fast ping model
                      "messages": [{"role": "user", "content": ping}],
                      "max_tokens": 5},
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                timeout=timeout,
            )
            return r.status_code == 200
        except Exception:
            return False

    def _test_hf():
        api_key = os.getenv("HF_API_KEY", "")
        if not api_key:
            return False
        url = "https://router.huggingface.co/v1/chat/completions"
        try:
            r = requests.post(
                url,
                json={"model": "Qwen/Qwen2.5-72B-Instruct",
                      "messages": [{"role": "user", "content": ping}],
                      "max_tokens": 5},
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                timeout=timeout,
            )
            return r.status_code == 200
        except Exception:
            return False

    available = {"gemini": False, "groq": False, "nvidia": False, "huggingface": False}
    with ThreadPoolExecutor(max_workers=4) as pool:
        fs = {
            pool.submit(_test_gemini): "gemini",
            pool.submit(_test_groq): "groq",
            pool.submit(_test_nvidia): "nvidia",
            pool.submit(_test_hf): "huggingface",
        }
        try:
            for f in as_completed(fs, timeout=timeout + 2):
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
    ("gemini",      "Gemini 2.5 Flash",      BLUE,    _call_gemini),
    ("groq",        "Groq Llama 3.3 70B",    MAGENTA, _call_groq),
    ("nvidia",      "Llama 3.1 70B (NVIDIA)", CYAN,    _call_nvidia),
    ("huggingface", "Qwen2.5 72B (HF)",      YELLOW,  _call_huggingface),
]


def _pick_providers(available: dict) -> list:
    """Return up to 2 providers from priority list that passed availability check."""
    live = [(k, label, color, fn) for k, label, color, fn in _PROVIDERS if available.get(k)]
    return live[:2]


# ─── Single coin: AI full analysis ──────────────────────────

def analyze_coin_ai(symbol: str, tf_data: dict) -> dict:
    """
    Send multi-timeframe indicator data to AI for full trade analysis.
    AI decides: direction, confidence, entry, SL, TP, timeframe, trade type.

    Args:
        symbol: coin symbol
        tf_data: dict of {timeframe: DataFrame} with indicators already added.
                 e.g. {"1m": df_1m, "3m": df_3m, "5m": df_5m, ...}

    Returns:
        dict with: direction, confidence, entry, stop_loss, take_profit,
                   leverage, timeframe, trade_type, hold_time, reasons,
                   risk_score, advice, decided_by
    """

    snapshot = build_indicator_snapshot(tf_data, symbol)

    prompt = f"""Analyze this coin using multi-timeframe data below. Use top-down analysis:
1. Start from 1D/1H to determine overall bias
2. Use 15m/5m to find entry structure
3. Use 3m/1m for precise entry timing

{snapshot}

FRICTION: Total fees + slippage = 0.18% round-trip. Factor this into your R:R calculation.
SL RULES: Place SL beyond structure (support/resistance/EMA). Min distance from entry: 0.30%.
TP RULES: Must achieve 2.2:1 R:R AFTER friction.
TIMEFRAME: Pick the best entry timeframe and trade type (SCALP/INTRADAY/SWING).

If no clear setup exists across ANY timeframe, return direction: "NO_TRADE" with confidence: 0.
Output ONLY valid JSON as specified in your instructions."""

    _KEY_MAP = {"gemini": "GEMINI_API_KEY", "groq": "GROQ_API_KEY",
                "nvidia": "NVIDIA_API_KEY", "huggingface": "HF_API_KEY"}

    cached = bool(_availability_cache) and (time.time() - _availability_checked_at < _AVAILABILITY_TTL)
    cache_note = f"{DIM}(cached){RESET}" if cached else f"{DIM}(checking...){RESET}"

    print(f"\n  {CYAN}{BOLD}{'=' * 55}{RESET}")
    print(f"  {CYAN}{BOLD}  AI TRADE ANALYSIS — {symbol}{RESET}")
    print(f"  {CYAN}{BOLD}{'=' * 55}{RESET}")

    available = _check_availability()
    active = _pick_providers(available)

    if not active:
        tried = ", ".join(k for k in _KEY_MAP if os.getenv(_KEY_MAP[k], ""))
        print(f"  {RED}{BOLD}  No AI providers reachable.{RESET} {DIM}(tried: {tried or 'none configured'}){RESET}")
        return {
            "direction": "NO_TRADE", "confidence": 0,
            "entry": None, "stop_loss": None, "take_profit": None, "leverage": 0,
            "reasons": ["All AI providers unreachable — check API keys and connectivity"],
            "risk_score": "HIGH", "advice": "Try again later", "decided_by": "NONE",
        }

    # Show each provider: online / offline / no key
    for key, label, color, _ in _PROVIDERS:
        env_key = _KEY_MAP[key]
        has_key = bool(os.getenv(env_key, ""))
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
        fs = {pool.submit(fn, prompt, ANALYZE_SYSTEM, 4096): key
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
            if key in attempted or not os.getenv(_KEY_MAP[key], ""):
                continue
            print(f"  {DIM}All active providers failed — falling back to {label}...{RESET}")
            try:
                r = fn(prompt, ANALYZE_SYSTEM, 4096)
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

    # Note if secondary opinion differs
    if secondary_key:
        sec = analyses[secondary_key]
        sec_label = next(label for k, label, _, _ in _PROVIDERS if k == secondary_key)
        if sec["direction"] != final["direction"]:
            final["reasons"].append(
                f"Note: {sec_label} suggested {sec['direction']} (conf {sec['confidence']})"
            )

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
    if final.get("advice"):
        print(f"  {CYAN}  Advice: {final['advice'][:150]}{RESET}")
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

def reanalyze_position_ai(symbol: str, tf_data: dict, position: dict) -> dict:
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

    _KEY_MAP = {"gemini": "GEMINI_API_KEY", "groq": "GROQ_API_KEY",
                "nvidia": "NVIDIA_API_KEY", "huggingface": "HF_API_KEY"}

    cached = bool(_availability_cache) and (time.time() - _availability_checked_at < _AVAILABILITY_TTL)
    cache_note = f"{DIM}(cached){RESET}" if cached else f"{DIM}(checking...){RESET}"

    print(f"\n  {CYAN}{BOLD}{'=' * 55}{RESET}")
    print(f"  {CYAN}{BOLD}  AI POSITION RE-ANALYSIS — {symbol}{RESET}")
    print(f"  {CYAN}{BOLD}{'=' * 55}{RESET}")

    available = _check_availability()
    active = _pick_providers(available)

    if not active:
        tried = ", ".join(k for k in _KEY_MAP if os.getenv(_KEY_MAP[k], ""))
        print(f"  {RED}{BOLD}  No AI providers reachable.{RESET} {DIM}(tried: {tried or 'none configured'}){RESET}")
        return {
            "action": "HOLD", "new_stop_loss": None, "new_take_profit": None,
            "urgency": "LOW", "reasons": ["All AI providers unreachable — check API keys and connectivity"],
            "advice": "Try again later", "decided_by": "NONE",
        }

    for key, label, color, _ in _PROVIDERS:
        env_key = _KEY_MAP[key]
        has_key = bool(os.getenv(env_key, ""))
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
        if key not in attempted and os.getenv(_KEY_MAP[key], ""):
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
            print(f"  {YELLOW}  {label}: SKIPPED — {raw.get('error', '?')[:100]}{RESET}")
            continue

        parsed = _parse_reanalysis_response(raw.get("raw_text", ""))
        parsed["decided_by"] = label

        action = parsed["action"]
        urgency = parsed.get("urgency", "LOW")
        a_color = action_colors.get(action, YELLOW)
        u_color = GREEN if urgency == "LOW" else (YELLOW if urgency == "MEDIUM" else RED)

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
TP RULES: Must achieve 2.2:1 R:R AFTER friction.
LEVERAGE: Max 12x. Scale: conf 7→5x, 8→8x, 9→10x, 10→12x.
For each pick, specify timeframe, trade_type (SCALP/INTRADAY/SWING), and hold_time.

If NO coins have a good setup, return an empty array [].

{coins_text}

Output ONLY a valid JSON array as specified in your instructions."""

    _KEY_MAP = {"gemini": "GEMINI_API_KEY", "groq": "GROQ_API_KEY",
                "nvidia": "NVIDIA_API_KEY", "huggingface": "HF_API_KEY"}
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
        env_key = _KEY_MAP[key]
        has_key = bool(os.getenv(env_key, ""))
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
        fs = {pool.submit(fn, prompt, SCAN_SYSTEM, 4096): key
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
            if key in attempted or not os.getenv(_KEY_MAP[key], ""):
                continue
            print(f"  {DIM}All active providers failed — falling back to {label}...{RESET}")
            try:
                r = fn(prompt, SCAN_SYSTEM, 4096)
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


# ─── Legacy verify_trade (kept for backward compatibility) ───

def verify_trade(symbol: str, signal: dict, capital: float,
                 leverage: int, balance: float) -> dict:
    """
    Legacy verification function. Now wraps analyze_coin_ai logic
    but with the old return format for backward compatibility.
    """
    # Build a minimal prompt using the old format
    reasons_text = "\n".join(f"  - {r}" for r in signal.get("reasons", []))
    entry = signal.get("entry", 0)
    sl = signal.get("stop_loss", 0)
    tp = signal.get("take_profit", 0)

    prompt = f"""Review this trade signal. Be VERY critical.

TRADE: {signal.get('direction', 'N/A')} {symbol}
  Entry: ${entry:,.6f} | SL: ${sl:,.6f} | TP: ${tp:,.6f}
  Confidence: {signal.get('confidence', 0)}/10
  Capital: ${capital:.2f} | Leverage: {leverage}x | Balance: ${balance:.2f}
  HTF Trend: {signal.get('htf_trend', 'N/A')}

ANALYSIS:
{reasons_text}

Output ONLY valid JSON with: verdict (APPROVED/REJECTED/NO_TRADE), reason, suggested_entry, suggested_sl, suggested_tp, suggested_leverage, risk_score, advice"""

    old_system = """You are an elite crypto futures trader. Review the trade and output ONLY valid JSON:
{
  "verdict": "APPROVED" or "REJECTED" or "NO_TRADE",
  "reason": "explanation",
  "suggested_entry": number or null,
  "suggested_sl": number or null,
  "suggested_tp": number or null,
  "suggested_leverage": number or null,
  "risk_score": "LOW" or "MEDIUM" or "HIGH",
  "advice": "actionable advice"
}"""

    gemini_result = _call_gemini(prompt, old_system)
    groq_result = _call_groq(prompt, old_system)

    # Parse old format
    gem_ok = not gemini_result.get("skipped", True)
    groq_ok = not groq_result.get("skipped", True)

    def _parse_old(text):
        try:
            clean = text.strip()
            if "```" in clean:
                s = clean.find("{"); e = clean.rfind("}") + 1
                clean = clean[s:e]
            data = json.loads(clean)
            return data
        except Exception:
            return {"verdict": "NO_TRADE", "reason": text[:150]}

    gem_data = _parse_old(gemini_result.get("raw_text", "{}")) if gem_ok else {}
    groq_data = _parse_old(groq_result.get("raw_text", "{}")) if groq_ok else {}

    # Pick Gemini, fallback Groq
    if gem_ok:
        verdict_data = gem_data
        decided_by = "Gemini 2.5 Flash"
    elif groq_ok:
        verdict_data = groq_data
        decided_by = "Groq (fallback)"
    else:
        return {"approved": False, "reason": "Both AIs failed", "decided_by": "NONE", "suggestions": {}}

    approved = str(verdict_data.get("verdict", "")).upper() == "APPROVED"
    suggestions = {}
    for k_old, k_new in [("suggested_entry", "entry"), ("suggested_sl", "sl"),
                          ("suggested_tp", "tp"), ("suggested_leverage", "leverage")]:
        if verdict_data.get(k_old):
            suggestions[k_new] = verdict_data[k_old]
    if verdict_data.get("advice"):
        suggestions["advice"] = verdict_data["advice"]

    return {
        "approved": approved,
        "final_verdict": str(verdict_data.get("verdict", "NO_TRADE")).upper(),
        "decided_by": decided_by,
        "reason": str(verdict_data.get("reason", "")),
        "suggestions": suggestions,
    }


# ─── CLI test ────────────────────────────────────────────────

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    print("Testing AI analysis mode...")
    print("Use: from multi_ai_verifier import analyze_coin_ai, scan_coins_ai")
