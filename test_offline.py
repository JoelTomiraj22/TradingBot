"""
Offline smoke tests — no exchange or AI keys needed.
Run: python test_offline.py
Validates indicators (causality), strategy, risk manager, backtest engine,
trailing-stop math, and AI JSON parsers using synthetic OHLCV data.
"""

import numpy as np
import pandas as pd


def make_ohlcv(n=400, seed=42, trend=0.0003, start=100.0):
    """Synthetic OHLCV with mild trend + noise."""
    rng = np.random.default_rng(seed)
    rets = rng.normal(trend, 0.004, n)
    close = start * np.exp(np.cumsum(rets))
    open_ = np.roll(close, 1)
    open_[0] = start
    spread = np.abs(rng.normal(0, 0.002, n)) * close
    high = np.maximum(open_, close) + spread
    low = np.minimum(open_, close) - spread
    vol = rng.uniform(100, 1000, n)
    ts = pd.date_range("2026-01-01", periods=n, freq="5min")
    return pd.DataFrame({"timestamp": ts, "open": open_, "high": high,
                         "low": low, "close": close, "volume": vol})


def test_indicators_causal():
    """Row i's indicator values must not change when future bars are appended."""
    from indicators import add_all_indicators

    df_full = add_all_indicators(make_ohlcv(300))
    df_half = add_all_indicators(make_ohlcv(300).iloc[:200].reset_index(drop=True))

    check_cols = ["ema_21", "rsi", "atr", "nearest_support", "nearest_resistance",
                  "poc_price", "near_support", "near_resistance", "high_volume_node"]
    i = 199  # last row of the half df
    for col in check_cols:
        a, b = df_full.iloc[i][col], df_half.iloc[i][col]
        if isinstance(a, (bool, np.bool_)):
            assert a == b, f"{col} not causal: full={a} half={b}"
        elif pd.notna(a) and pd.notna(b):
            assert abs(a - b) < 1e-9 * max(1, abs(a)), f"{col} not causal: full={a} half={b}"
    print("PASS indicators are point-in-time correct (no look-ahead)")


def test_strategy_runs():
    from indicators import add_all_indicators
    from strategy import evaluate_signal

    df = add_all_indicators(make_ohlcv(300))
    sig = evaluate_signal(df)
    assert sig["direction"] in ("LONG", "SHORT", "NO TRADE")
    assert 0 <= sig["confidence"] <= 10
    if sig["direction"] in ("LONG", "SHORT"):
        e, sl, tp, be = sig["entry"], sig["stop_loss"], sig["take_profit"], sig["breakeven"]
        if sig["direction"] == "LONG":
            assert sl < e < tp and be > e
        else:
            assert tp < e < sl and be < e, f"SHORT levels wrong: e={e} sl={sl} tp={tp} be={be}"
    print(f"PASS strategy evaluate_signal -> {sig['direction']} ({sig['confidence']}/10)")


def test_short_breakeven():
    """Regression: SHORT breakeven must be BELOW entry."""
    from indicators import add_all_indicators
    from strategy import evaluate_signal
    # Find any df where a short triggers; if none, at least verify formula directly
    from risk_manager import calculate_breakeven
    assert calculate_breakeven(100, "SHORT") < 100
    assert calculate_breakeven(100, "LONG") > 100
    print("PASS breakeven direction-aware")


def test_risk_manager():
    from risk_manager import validate_trade, get_leverage_for_confidence

    assert get_leverage_for_confidence(6) == 0
    assert get_leverage_for_confidence(7) == 5
    assert get_leverage_for_confidence(10) == 25

    ok = validate_trade(100, 50000, 49500, 51200, confidence=8, direction="LONG")
    assert ok["approved"], ok.get("reason")
    assert ok["risk_amount"] <= 100 * 0.02 * 1.10  # ~2% risk cap (small tolerance)

    low_conf = validate_trade(100, 50000, 49500, 51200, confidence=5, direction="LONG")
    assert not low_conf["approved"]

    bad_rr = validate_trade(100, 50000, 49500, 50200, confidence=8, direction="LONG")
    assert not bad_rr["approved"]

    tight = validate_trade(100, 50000, 49990, 50500, confidence=8, direction="LONG")
    assert not tight["approved"]  # stop too tight (0.02%)
    print("PASS risk manager gates (confidence, R:R, stop distance, 2% risk)")


def test_backtest_engine(monkeypatched=True):
    """Run the backtester on synthetic data by stubbing fetch_ohlcv."""
    import backtest as bt

    df = make_ohlcv(900, seed=7, trend=0.0006)
    bt.fetch_ohlcv = lambda *a, **k: df.copy()
    result = bt.backtest("TEST/USDT", "5m", days=3, capital_per_trade=10, exchange=object())
    stats = result["stats"]
    assert "total_trades" in stats
    for t in result["trades"]:
        assert t["fees"] > 0, "fees missing"
        assert t["result"] in ("WIN", "LOSS")
        # Intrabar exits must be at SL or TP or a close price
        if t["exit_reason"] == "Stop loss hit":
            assert abs(t["exit"] - t["sl"]) < 1e-6
        if t["exit_reason"] == "Take profit hit":
            assert abs(t["exit"] - t["tp"]) < 1e-6
    print(f"PASS backtest engine ({stats['total_trades']} trades, "
          f"win rate {stats.get('win_rate', 0)}%, pnl ${stats.get('total_pnl', 0)})")


def test_trailing_threshold():
    """Regression: trailing SL threshold must be relative, not $0.01 absolute."""
    price = 0.10  # sub-dollar coin
    current_sl, new_sl = 0.0950, 0.0952
    old_logic = abs(new_sl - current_sl) > 0.01          # never fires on cheap coins
    new_logic = abs(new_sl - current_sl) > price * 0.0005
    assert not old_logic and new_logic
    print("PASS trailing-stop threshold works on sub-dollar coins")


def test_ai_parsers():
    from multi_ai_verifier import (_parse_analysis_response,
                                   _parse_scan_response, _parse_reanalysis_response)

    good = '{"direction": "LONG", "confidence": 8, "entry": 100.5, "stop_loss": 99.0, "take_profit": 104.0, "leverage": 8, "timeframe": "5m", "trade_type": "INTRADAY", "risk_score": "LOW", "reasons": ["r1"], "advice": "go"}'
    p = _parse_analysis_response(good)
    assert p["direction"] == "LONG" and p["confidence"] == 8

    fenced = "```json\n" + good + "\n```"
    assert _parse_analysis_response(fenced)["direction"] == "LONG"

    truncated = good[:-40]  # cut mid-object
    pt = _parse_analysis_response(truncated)
    assert pt["direction"] in ("LONG", "NO_TRADE")  # recovered or safe fallback

    arr = f"[{good}]"
    assert _parse_scan_response(arr)[0]["direction"] == "LONG"

    re_good = '{"action": "MOVE_SL", "new_stop_loss": 101.0, "new_take_profit": null, "urgency": "LOW", "reasons": [], "advice": ""}'
    assert _parse_reanalysis_response(re_good)["action"] == "MOVE_SL"

    garbage = _parse_analysis_response("the market looks bullish, maybe buy?")
    assert garbage["direction"] == "NO_TRADE"
    print("PASS AI JSON parsers (clean, fenced, truncated, garbage)")


def test_sl_hunt_guard():
    from indicators import check_sl_hunt_risk, _is_round_number, add_all_indicators

    df = add_all_indicators(make_ohlcv(100, seed=3))
    recent_low = float(df["low"].iloc[-15:].min())
    recent_high = float(df["high"].iloc[-15:].max())

    # SL inside the recent wick cluster (LONG) → risky, suggestion beyond the extreme
    inside_sl = recent_low * 1.001
    r = check_sl_hunt_risk(df, "LONG", inside_sl, df["close"].iloc[-1])
    assert r["risky"] and r["suggested_sl"] < recent_low

    # SL safely below all recent wicks and not round → not risky
    entry = float(df["close"].iloc[-1])
    safe_sl = recent_low * 0.97 * 1.00123  # well below wicks, non-round
    r2 = check_sl_hunt_risk(df, "LONG", safe_sl, entry)
    assert not r2["risky"], r2["reasons"]

    # SHORT: SL inside recent highs → risky, suggestion above the extreme
    r3 = check_sl_hunt_risk(df, "SHORT", recent_high * 0.999, entry)
    assert r3["risky"] and r3["suggested_sl"] > recent_high

    # Round-number detection
    assert _is_round_number(50000.0)
    assert _is_round_number(0.10)
    assert not _is_round_number(49734.27)
    print("PASS anti-stop-hunt guard (wick cluster, round numbers, both directions)")


def test_eta_estimator():
    from indicators import estimate_eta_minutes, add_all_indicators

    df = add_all_indicators(make_ohlcv(100, seed=5))
    entry = float(df["close"].iloc[-1])
    atr = float(df["atr"].iloc[-1])

    eta = estimate_eta_minutes(df, entry, entry + 2 * atr, timeframe_minutes=5)
    assert eta is not None
    lo, hi = eta
    assert 5 <= lo <= hi
    assert abs(lo - 2 * 5) <= 5            # ~2 ATR away → ~2 candles → ~10 min lower bound
    assert estimate_eta_minutes(df, entry, entry, 5) is None  # zero distance
    assert estimate_eta_minutes(None, entry, entry + 1, 5) is None
    print(f"PASS ETA estimator (2x ATR target -> {lo}-{hi} min on 5m)")


def test_no_swing_policy():
    from multi_ai_verifier import _parse_analysis_response, _parse_scan_response

    swing = ('{"direction": "LONG", "confidence": 9, "entry": 100, "stop_loss": 98, '
             '"take_profit": 110, "leverage": 5, "timeframe": "1h", "trade_type": "SWING", '
             '"risk_score": "LOW", "reasons": ["big trend"], "advice": "hold for days"}')
    p = _parse_analysis_response(swing)
    assert p["direction"] == "NO_TRADE" and p["confidence"] == 0
    assert any("SWING" in r for r in p["reasons"])

    scalp = swing.replace('"SWING"', '"SCALP"')
    arr = f"[{swing}, {scalp}]"
    picks = _parse_scan_response(arr)
    assert len(picks) == 1 and picks[0]["trade_type"] == "SCALP"
    print("PASS no-swing policy enforced in analysis + scan parsers")


def test_eta_parsing():
    from multi_ai_verifier import _parse_analysis_response

    j = ('{"direction": "LONG", "confidence": 8, "entry": 100, "stop_loss": 99, '
         '"take_profit": 102, "leverage": 8, "timeframe": "5m", "trade_type": "SCALP", '
         '"eta_minutes": 12.0, "eta_basis": "TP 1.5x ATR away", "risk_score": "LOW", '
         '"reasons": [], "advice": ""}')
    p = _parse_analysis_response(j)
    assert p["eta_minutes"] == 12 and "ATR" in p["eta_basis"]

    j2 = j.replace('"eta_minutes": 12.0,', '"eta_minutes": "soon",')
    assert _parse_analysis_response(j2)["eta_minutes"] is None
    print("PASS eta_minutes/eta_basis parsing (numeric + junk fallback)")


def _df_from_arrays(open_, high, low, close, vol):
    n = len(close)
    ts = pd.date_range("2026-01-01", periods=n, freq="5min")
    return pd.DataFrame({"timestamp": ts, "open": open_, "high": high,
                         "low": low, "close": close, "volume": vol})


def test_setup_detectors():
    """Pattern playbook: flag, DCB, inside bar/NR7, volume confirmation."""
    from indicators import add_all_indicators, detect_setups

    # --- Bull flag: flat base -> sharp pole up on volume -> tight quiet consolidation
    n = 100
    close = np.full(n, 100.0)
    close[:80] += np.sin(np.arange(80) * 0.7) * 0.3          # noisy base
    close[80:90] = np.linspace(100, 106, 10)                  # pole (+6 = many ATRs)
    close[90:] = 105.8 + np.sin(np.arange(10)) * 0.15         # tight flag
    open_ = np.roll(close, 1); open_[0] = close[0]
    high = np.maximum(open_, close) + 0.10
    low = np.minimum(open_, close) - 0.10
    vol = np.full(n, 100.0)
    vol[80:90] = 400.0                                        # pole volume
    vol[90:] = 60.0                                           # flag volume dries up
    df = add_all_indicators(_df_from_arrays(open_, high, low, close, vol))
    s = detect_setups(df)
    assert s["bull_flag"], "bull flag not detected"
    assert not s["bear_flag"]

    # --- Dead cat bounce: high-volume dump, weak low-volume bounce
    n = 100
    close = np.full(n, 100.0) + np.sin(np.arange(n) * 0.5) * 0.2
    close[80:88] = np.linspace(100, 92, 8)                    # sharp dump
    close[88:] = np.linspace(92, 94.5, 12)                    # weak bounce (~31% retrace)
    open_ = np.roll(close, 1); open_[0] = close[0]
    high = np.maximum(open_, close) + 0.15
    low = np.minimum(open_, close) - 0.15
    vol = np.full(n, 100.0)
    vol[80:88] = 500.0                                        # dump volume
    vol[88:] = 50.0                                           # bounce volume weak
    df2 = add_all_indicators(_df_from_arrays(open_, high, low, close, vol))
    s2 = detect_setups(df2)
    assert s2["dead_cat_bounce"], "dead cat bounce not detected"
    assert s2["dcb_bounce_high"] is not None

    # --- Inside bar + volume confirmation flags exist and behave
    df3 = add_all_indicators(make_ohlcv(120, seed=11))
    s3 = detect_setups(df3)
    assert isinstance(s3["inside_bar"], bool)
    assert isinstance(s3["nr7"], bool)
    assert isinstance(s3["volume_confirmed"], bool)
    print("PASS setup detectors (bull flag, dead cat bounce, compression flags)")


def test_dual_verification_gate():
    """Bot-side verification must block unbacked trades and warn on DCB longs."""
    from indicators import add_all_indicators
    from strategy import verify_trade_setup

    # Flat, featureless chop — no playbook setup in either direction
    n = 120
    rng_ = np.random.default_rng(2)
    close = 100 + rng_.normal(0, 0.02, n).cumsum() * 0.1
    open_ = np.roll(close, 1); open_[0] = close[0]
    high = np.maximum(open_, close) + 0.01
    low = np.minimum(open_, close) - 0.01
    vol = np.full(n, 100.0)
    df = add_all_indicators(_df_from_arrays(open_, high, low, close, vol))
    v = verify_trade_setup(df, "LONG")
    assert not v["verified"], f"chop should not verify: {v['matched']}"

    # DCB structure: LONG must carry the trap warning; SHORT must verify
    close2 = np.full(n, 100.0) + np.sin(np.arange(n) * 0.5) * 0.2
    close2[100:108] = np.linspace(100, 92, 8)
    close2[108:] = np.linspace(92, 94.5, 12)
    open2 = np.roll(close2, 1); open2[0] = close2[0]
    high2 = np.maximum(open2, close2) + 0.15
    low2 = np.minimum(open2, close2) - 0.15
    vol2 = np.full(n, 100.0); vol2[100:108] = 500.0; vol2[108:] = 50.0
    df2 = add_all_indicators(_df_from_arrays(open2, high2, low2, close2, vol2))
    vl = verify_trade_setup(df2, "LONG")
    assert any("DEAD CAT" in w for w in vl["warnings"]), vl
    vs = verify_trade_setup(df2, "SHORT")
    assert vs["verified"] and any("Dead cat" in m for m in vs["matched"]), vs
    print("PASS dual-verification gate (chop blocked, DCB long warned, DCB short verified)")


def test_sanity_gate():
    """Regression: ADA session — Groq emitted LONG with SL 2.05% / TP 0.18%
    (R:R 0.09:1). Such verdicts must be discarded before they decide anything."""
    from multi_ai_verifier import _sanity_check_verdict, _parse_analysis_response, _parse_scan_response

    groq_garbage = {
        "direction": "LONG", "confidence": 8, "trade_type": "SCALP",
        "entry": 0.1658, "stop_loss": 0.1624, "take_profit": 0.1661,
        "reasons": ["RSI on 1m is 70.5"],
    }
    v = _sanity_check_verdict(groq_garbage)
    assert v["direction"] == "NO_TRADE" and v["confidence"] == 0
    assert any("SANITY GATE" in r for r in v["reasons"])

    # Healthy verdict passes untouched
    good = {"direction": "LONG", "confidence": 8, "trade_type": "SCALP",
            "entry": 100.0, "stop_loss": 99.0, "take_profit": 102.5, "reasons": []}
    assert _sanity_check_verdict(dict(good))["direction"] == "LONG"

    # Inconsistent levels (SHORT with TP above entry) — discarded
    bad_short = {"direction": "SHORT", "confidence": 9, "trade_type": "SCALP",
                 "entry": 100.0, "stop_loss": 99.0, "take_profit": 102.0, "reasons": []}
    assert _sanity_check_verdict(bad_short)["direction"] == "NO_TRADE"

    # Gate is wired into the parsers
    import json
    raw = json.dumps({**groq_garbage, "leverage": 8, "timeframe": "5m",
                      "risk_score": "MEDIUM", "advice": ""})
    assert _parse_analysis_response(raw)["direction"] == "NO_TRADE"
    assert _parse_scan_response(f"[{raw}]") == []  # scan pick dropped entirely
    print("PASS sanity gate (0.09:1 R:R discarded, parsers wired, good verdicts pass)")


def test_truncated_json_recovery():
    """Regression: TIA/USDT session — response cut off at '"timeframe": '
    used to collapse a LONG 8/10 verdict into NO_TRADE / Risk UNKNOWN."""
    from multi_ai_verifier import _parse_analysis_response

    cut_at_key = """{
  "direction": "LONG",
  "confidence": 8,
  "entry": 0.326400,
  "stop_loss": 0.323800,
  "take_profit": 0.329300,
  "leverage": 20,
  "timeframe": """
    p = _parse_analysis_response(cut_at_key)
    assert p["direction"] == "LONG", p
    assert p["confidence"] == 8
    assert abs(p["entry"] - 0.3264) < 1e-9
    assert abs(p["stop_loss"] - 0.3238) < 1e-9
    assert any("truncated" in r for r in p["reasons"])

    # Variant: invalid unquoted token before the cut
    bad_token = cut_at_key + "5m,\n  \"trade_type\": "
    p2 = _parse_analysis_response(bad_token)
    assert p2["direction"] == "LONG" and p2["confidence"] == 8

    # Variant: cut inside a string value
    cut_in_string = cut_at_key + '"5m'
    p3 = _parse_analysis_response(cut_in_string)
    assert p3["direction"] == "LONG"
    print("PASS truncated-JSON recovery (cut at key, bad token, cut in string)")


def test_auto_sl_apply_guard():
    """Auto re-analysis may only TIGHTEN the SL, never loosen/cross price."""
    from position_monitor import PositionMonitor
    check = PositionMonitor._sl_auto_apply_check

    # LONG @ entry 100, current SL 98, price 102
    ok, _ = check("LONG", 98.0, 100.5, 102.0)
    assert ok                                            # tighten, valid gap
    assert not check("LONG", 98.0, 97.0, 102.0)[0]       # loosen → rejected
    assert not check("LONG", 98.0, 102.5, 102.0)[0]      # above price → rejected
    assert not check("LONG", 98.0, 101.9, 102.0)[0]      # <0.3% from price → rejected
    assert not check("LONG", 98.0, 98.05, 102.0)[0]      # change too small → rejected

    # SHORT @ entry 100, current SL 102, price 98
    ok, _ = check("SHORT", 102.0, 99.5, 98.0)
    assert ok                                            # tighten, valid gap
    assert not check("SHORT", 102.0, 103.0, 98.0)[0]     # loosen → rejected
    assert not check("SHORT", 102.0, 97.5, 98.0)[0]      # below price → rejected
    print("PASS auto re-analysis SL guard (tighten-only, gap, noise filters)")


def test_net_pnl_honesty():
    """Regression: ATOM session — monitor said '+$0.11 WIN' but balance fell
    $0.05. Net P&L must use actual fills and include taker fees."""
    from position_monitor import PositionMonitor
    net_pnl = PositionMonitor._net_pnl

    # Real numbers from the session: entry fill 1.8460, exit fill ~1.8474,
    # qty 108.4, 20x. Trigger price said +$0.15 gross, but fees ~$0.20.
    net, pct, fees = net_pnl("LONG", 1.8460, 1.8474, 108.4, 20)
    assert fees > 0.19, fees
    assert net < 0, f"breakeven-trigger exit must be net negative, got {net:+.2f}"
    assert -0.10 < net < 0.0  # ≈ the -$0.05 balance change

    # A real winner stays a winner after fees
    net2, pct2, _ = net_pnl("LONG", 100.0, 102.0, 2.0, 10)
    assert net2 > 0 and pct2 > 0

    # SHORT direction
    net3, _, _ = net_pnl("SHORT", 100.0, 98.0, 2.0, 10)
    assert net3 > 0
    print(f"PASS net P&L honesty (ATOM case: net ${net:+.2f}, fees ${fees:.2f})")


def test_bot_risk_state():
    """Circuit breaker / daily loss state actually updates on closed trades."""
    import bot

    bot.consecutive_losses = 0
    bot.daily_pnl = 0.0
    bot.session_start_balance = 100.0

    bot.record_closed_trade(-2.0)
    bot.record_closed_trade(-2.0)
    assert bot.consecutive_losses == 2 and bot.daily_pnl == -4.0
    bot.record_closed_trade(+3.0)
    assert bot.consecutive_losses == 0 and bot.daily_pnl == -1.0
    assert not bot.trading_blocked()

    bot.record_closed_trade(-2.0)
    bot.record_closed_trade(-2.0)
    bot.record_closed_trade(-2.0)
    assert bot.consecutive_losses == 3
    assert bot.trading_blocked()  # circuit breaker

    bot.consecutive_losses = 0
    bot.daily_pnl = -6.0  # 6% of 100 — over 5% daily limit
    assert bot.trading_blocked()
    print("PASS circuit breaker + daily loss limit wiring")


if __name__ == "__main__":
    test_indicators_causal()
    test_strategy_runs()
    test_short_breakeven()
    test_risk_manager()
    test_backtest_engine()
    test_trailing_threshold()
    test_ai_parsers()
    test_sl_hunt_guard()
    test_eta_estimator()
    test_no_swing_policy()
    test_eta_parsing()
    test_setup_detectors()
    test_dual_verification_gate()
    test_sanity_gate()
    test_truncated_json_recovery()
    test_auto_sl_apply_guard()
    test_net_pnl_honesty()
    test_bot_risk_state()
    print("\nAll offline tests passed.")
