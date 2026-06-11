# Go Live Checklist

Complete all items before switching from testnet to live trading.

## Pre-Live Verification

- [ ] Tested on testnet for at least 1 week
- [ ] Profit factor > 1.3 and win rate > 50% on backtest (run: `python backtest.py`)
- [ ] All API permissions correct (futures enabled, withdrawal DISABLED)
- [ ] Stop loss and take profit orders confirmed working on testnet
      (note: demo mode uses the software monitor — verify exchange SL/TP appear once live)
- [ ] Logging captures everything (check `logs/` folder)
- [ ] Position sizing never exceeds 2% risk (run: `python risk_manager.py`)
- [ ] Emergency close function tested (`closeall` command in bot)
- [ ] Circuit breaker verified: after 3 losses, new entries are blocked until `reset`
- [ ] At least one AI provider key set and reachable (shown at start of `analyze`)
- [ ] `.env` has live keys populated
- [ ] `USE_TESTNET` set to `False` in `.env`
- [ ] Start with minimum capital ($8–$10 per trade)
- [ ] Monitor first 5 trades manually — do not walk away

## How to Go Live

1. Generate live API keys at https://www.binance.com/en/my/settings/api-management
   - Enable Futures permission
   - **Disable** withdrawal permission
   - Restrict to your IP if possible
2. Add keys to `.env`:
   ```
   BINANCE_API_KEY=your_live_key
   BINANCE_API_SECRET=your_live_secret
   USE_TESTNET=False
   ```
3. Run: `python bot.py`
4. Verify balance shows your real account
5. Run a scan, confirm a small trade, verify SL/TP orders appear on Binance

## Post-Live Monitoring

- Check `trades.csv` and `python trade_tracker.py` after each session
- Review logs daily for errors
- If 3 consecutive losses, stop and review strategy parameters
- Never increase leverage beyond what the confidence score recommends
