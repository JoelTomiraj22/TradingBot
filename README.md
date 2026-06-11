# Crypto Trading Bot — Binance Futures (AI-Assisted)

A CLI trading bot for Binance USDT-M Futures. It combines a multi-confirmation
technical strategy with an ensemble of free LLM analysts (Gemini, Groq, NVIDIA
NIM, HuggingFace), strict risk management, a software trailing stop, and a
point-in-time-correct backtester.

> Trading futures with leverage is high risk. This bot defaults to Binance
> testnet/demo. Nothing here is financial advice — no bot can guarantee
> profitable predictions. Validate on testnet and with backtests first.

## Quick start

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env              # then fill in your keys
python bot.py
```

## Configuration (.env)

| Key | Purpose |
|---|---|
| `BINANCE_TESTNET_API_KEY/SECRET` | Demo keys (https://testnet.binancefuture.com) |
| `BINANCE_API_KEY/SECRET` | Live keys (only when going live) |
| `USE_TESTNET` | `True` (default) = demo, `False` = live |
| `GEMINI_API_KEY` | Primary AI analyst (free at aistudio.google.com) |
| `GROQ_API_KEY` | Second opinion (free at console.groq.com) |
| `NVIDIA_API_KEY`, `HF_API_KEY` | Optional fallbacks |

At least one AI key is required for `analyze`/`allcoins`; the rule-based
`scan` and `backtest` work without any AI keys.

## Commands

| Command | What it does |
|---|---|
| `allcoins` | Scan ~38 majors, pre-filter, AI picks the best ≤3 setups |
| `scan` | Rule-based scan of top gainers (with higher-TF confirmation) |
| `analyze` / `btc`, `eth`, ... | Full multi-timeframe AI analysis of one coin |
| `reanalyse <coin>` | AI re-checks an open trade — may suggest moving SL/TP or closing |
| `backtest` | Backtest the rule strategy (fees, intrabar SL/TP, no look-ahead) |
| `positions` / `monitored` / `balance` | Account state |
| `stats` / `trades` | Trade journal (trades.csv) |
| `close` / `closeall` | Close one / all positions |
| `reset` | Clear circuit breaker + daily loss limit |

## Trading policy

Scalping-first: SCALP is the default mode, INTRADAY only when clearly
superior, and SWING trades are disabled at every layer (prompts, parsers,
and bot).

No chasing — ever. If price has drifted more than `MAX_CHASE_PCT` (0.15%)
from the AI entry, market execution is disabled with no override: the only
option is a limit order at the AI level, registered with the monitor so
SL/TP, the trailing stop, and auto re-analysis all activate the moment it
fills. The AI itself must return WAIT with a precise limit-entry level
unless price is at the level right now.

Liquidity-aware: stops cluster just beyond obvious swing points, equal
highs/lows, and round numbers — price sweeps those pools before reversing.
The AI is instructed to place limit ENTRIES at sweep zones (the sweep fills
you at the best price) while keeping the SL out of them. Independently, the
bot checks every proposed SL against the recent wick cluster and
round-number levels and proposes a safer level beyond the liquidity zone.
Every actionable setup also shows an estimated time-to-TP (AI estimate plus
an independent ATR-based band).

## Risk management

Confidence-gated entries (7+/10 only), confidence-scaled leverage (5x–25x cap),
minimum R:R after fees (1.2:1 scalps / 2.2:1 intraday+), 0.30% minimum stop
distance, max 4h scalp hold, circuit breaker after 3 consecutive losses, 5%
daily loss limit, anti-chase and volatility-spike guards, correlation warning
when stacking same-direction positions.

SL/TP are placed on the exchange when supported; otherwise (e.g. demo) a
3-second software monitor with a staged trailing stop (breakeven → lock profit
→ tight trail) manages the position and survives restarts via
`monitor_state.json`.

While a trade is open, the AI automatically re-analyzes it every 3 minutes
(`AUTO_REANALYZE_MINUTES` in config.py) with fresh multi-timeframe data:
HOLD prints a one-line status, SL suggestions are auto-applied only if they
tighten risk (never loosened, never within 0.3% of price, 10-min cooldown),
TP changes are suggestion-only, and a CLOSE_NOW verdict raises a loud alert
but never closes automatically — you stay in control.

## Architecture

```
bot.py                 CLI loop, trade confirmation flow, risk state
config.py              .env config + exchange factory (demo/live)
fetch_data.py          OHLCV, prices, top gainers, market microstructure
indicators.py          EMA/RSI/MACD/BB/ATR/VWAP, patterns, causal S/R + volume profile
strategy.py            Multi-confirmation scoring engine (rule-based)
multi_ai_verifier.py   Multi-provider LLM analysis (analyze / scan / reanalyze)
scanner.py             Top-gainer scanner with higher-TF confirmation
risk_manager.py        Position sizing, leverage, R:R validation, fees
order_executor.py      Orders, SL/TP (algo + fallbacks), close-all
position_monitor.py    Software SL/TP + trailing stop, limit-order watcher
trade_tracker.py       trades.csv journal + confidence calibration stats
backtest.py            Point-in-time-correct backtester
tradingview_webhook.py Optional Flask webhook validating TradingView alerts
```

See `GO_LIVE_CHECKLIST.md` before switching `USE_TESTNET=False`.

------------------------------------------------
python3 -m venv venv
source venv/bin/activate
which python        # must print .../TradingBot/venv/bin/python
python -m pip install -r requirements.txt
python bot.py