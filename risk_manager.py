"""
Risk management module.
Position sizing, ATR-based SL/TP, leverage scaling, trailing stops.
Accounts for leverage in risk calculation and includes trading fees.
"""

from config import RISK_PER_TRADE, MIN_RR_RATIO, CAPITAL_PER_TRADE

# ─── Constants ─────────────────────────────────────────────────
TRADING_FEE_PCT = 0.0004        # 0.04% per side (maker), 0.04% round trip = 0.08%
ROUND_TRIP_FEE_PCT = 0.0008     # Total fees for entry + exit
SLIPPAGE_PCT = 0.0010           # 0.10% estimated slippage (entry + exit combined)
TOTAL_COST_PCT = ROUND_TRIP_FEE_PCT + SLIPPAGE_PCT  # ~0.18% total friction
MAX_LEVERAGE = 12               # Hard cap — conservative until strategy is proven profitable
MAX_CONCURRENT_RISK_PCT = 0.06  # Max 6% of account at risk across all positions
MAX_HOLD_HOURS = 4              # Max hold time for scalp trades (hours)
MAX_DAILY_LOSS_PCT = 0.05       # 5% max daily drawdown — stop trading for the day

# Minimum R:R (after fees+slippage) by trade type — scalps have tighter targets
MIN_RR_BY_TYPE = {
    "SCALP": 1.2,
    "INTRADAY": MIN_RR_RATIO,
    "SWING": MIN_RR_RATIO,
}


def calculate_position_size(account_balance: float, entry_price: float,
                            stop_loss: float, leverage: int) -> dict:
    """
    Calculate position size with leverage-aware risk.

    The actual dollar risk on futures = (stop_distance / entry_price) * position_size
    This is then multiplied by leverage implicitly through position_size.
    We cap risk at RISK_PER_TRADE % of account balance.
    """
    stop_distance = abs(entry_price - stop_loss)
    if stop_distance == 0:
        return _reject("Stop distance is zero")

    # Cap leverage
    leverage = min(leverage, MAX_LEVERAGE)

    risk_amount = account_balance * RISK_PER_TRADE
    max_capital = min(CAPITAL_PER_TRADE, account_balance)

    # On futures with leverage:
    # margin_used = position_size / leverage
    # dollar_risk = stop_distance * quantity = stop_distance * (position_size / entry_price)
    # We want: dollar_risk <= risk_amount
    # So: position_size <= (risk_amount * entry_price) / stop_distance
    stop_pct = stop_distance / entry_price
    position_size = risk_amount / stop_pct

    # Cap at max capital * leverage (margin constraint)
    max_position = max_capital * leverage
    if position_size > max_position:
        position_size = max_position

    quantity = position_size / entry_price

    # Verify LEVERAGED risk doesn't exceed limit
    # Actual dollar at risk = stop_distance * quantity
    actual_risk = stop_distance * quantity

    # Add fee + slippage cost to risk
    fee_cost = position_size * TOTAL_COST_PCT
    total_risk = actual_risk + fee_cost

    if total_risk > risk_amount * 1.05:  # 5% tolerance
        # Scale down to fit
        scale = risk_amount / total_risk
        position_size *= scale
        quantity = position_size / entry_price
        actual_risk = stop_distance * quantity
        fee_cost = position_size * TOTAL_COST_PCT
        total_risk = actual_risk + fee_cost

    margin_used = position_size / leverage

    return {
        "approved": True,
        "rejection_reason": None,
        "position_size": round(position_size, 2),
        "quantity": quantity,
        "risk_amount": round(total_risk, 2),
        "risk_from_sl": round(actual_risk, 2),
        "risk_from_fees": round(fee_cost, 2),
        "margin_used": round(margin_used, 2),
        "max_risk_allowed": round(risk_amount, 2),
        "leverage": leverage,
    }


def calculate_stop_loss(entry_price: float, atr: float, direction: str) -> float:
    """Calculate stop loss using 1.5x ATR (wider for safety)."""
    if direction == "LONG":
        return entry_price - (1.5 * atr)
    else:
        return entry_price + (1.5 * atr)


def calculate_take_profit(entry_price: float, stop_loss: float,
                          direction: str, rr_ratio: float = None) -> float:
    """Calculate take profit at minimum 2:1 R:R."""
    rr = max(rr_ratio or MIN_RR_RATIO, MIN_RR_RATIO)
    stop_distance = abs(entry_price - stop_loss)
    target_distance = stop_distance * rr

    if direction == "LONG":
        return entry_price + target_distance
    else:
        return entry_price - target_distance


def calculate_breakeven(entry_price: float, direction: str) -> float:
    """Calculate breakeven including fees + slippage."""
    if direction == "LONG":
        return entry_price * (1 + TOTAL_COST_PCT)
    else:
        return entry_price * (1 - TOTAL_COST_PCT)


def get_leverage_for_confidence(confidence: int) -> int:
    """
    Map confidence score to leverage. CAPPED at 12x until strategy is proven.

    1-6:  Don't trade (0) — raised minimum to 7
    7:    5x  (conservative entry)
    8:    8x  (moderate)
    9:    10x
    10:   12x (hard cap)
    """
    if confidence <= 6:
        return 0
    elif confidence == 7:
        return 5
    elif confidence == 8:
        return 8
    elif confidence == 9:
        return 10
    else:  # 10
        return min(12, MAX_LEVERAGE)


def calculate_expected_pnl(entry_price: float, stop_loss: float,
                           take_profit: float, quantity: float,
                           direction: str, position_size: float = None) -> dict:
    """Calculate expected profit and loss including fees."""
    if direction == "LONG":
        gross_profit = (take_profit - entry_price) * quantity
        gross_loss = (entry_price - stop_loss) * quantity
    else:
        gross_profit = (entry_price - take_profit) * quantity
        gross_loss = (stop_loss - entry_price) * quantity

    # Deduct fees from profit, add to loss
    if position_size is None:
        position_size = entry_price * quantity
    fee = position_size * TOTAL_COST_PCT  # includes slippage

    net_profit = gross_profit - fee
    net_loss = gross_loss + fee

    rr_ratio = net_profit / net_loss if net_loss > 0 else 0

    return {
        "expected_profit": round(net_profit, 2),
        "expected_loss": round(net_loss, 2),
        "fees": round(fee, 2),
        "rr_ratio": round(rr_ratio, 2),
    }


def validate_trade(account_balance: float, entry_price: float,
                   stop_loss: float, take_profit: float,
                   confidence: int, direction: str,
                   trade_type: str = "SCALP") -> dict:
    """Full trade validation with leverage-aware risk and fee calculation."""
    min_rr = MIN_RR_BY_TYPE.get(trade_type, MIN_RR_RATIO)
    # Check confidence
    leverage = get_leverage_for_confidence(confidence)
    if leverage == 0:
        return {
            "approved": False,
            "reason": f"DON'T TRADE — Confidence {confidence}/10 is below threshold (min 6)",
        }

    # Check basic R:R (raw, pre-fees)
    stop_dist = abs(entry_price - stop_loss)
    tp_dist = abs(take_profit - entry_price)
    if stop_dist == 0:
        return {"approved": False, "reason": "Stop distance is zero"}

    rr_raw = tp_dist / stop_dist
    if rr_raw < 1.5:
        return {"approved": False, "reason": f"Raw R:R {rr_raw:.2f} is too low (need 1.5+ raw)"}

    # Check stop is not too tight — friction (0.18%) must be < 60% of SL distance
    stop_pct = (stop_dist / entry_price) * 100
    min_stop_pct = 0.30  # 0.30% min — ensures friction doesn't dominate on 5m scalps
    if stop_pct < min_stop_pct:
        return {"approved": False, "reason": f"Stop too tight ({stop_pct:.2f}%) — min {min_stop_pct:.2f}% for scalps"}

    # Calculate position
    pos = calculate_position_size(account_balance, entry_price, stop_loss, leverage)
    if not pos["approved"]:
        return {"approved": False, "reason": pos["rejection_reason"]}

    # Expected P&L with fees
    pnl = calculate_expected_pnl(entry_price, stop_loss, take_profit,
                                 pos["quantity"], direction, pos["position_size"])

    # Reject if after-fees R:R is below minimum (lower bar for SCALP — tighter targets)
    if pnl["rr_ratio"] < min_rr:
        return {
            "approved": False,
            "reason": f"R:R after fees+slippage is {pnl['rr_ratio']:.2f}:1 — need {min_rr}:1+ for {trade_type}. "
                      f"Raw R:R {rr_raw:.2f}:1 but friction eats too much on this setup."
        }

    # Reject if fees eat too much of the profit
    if pnl["fees"] > pnl["expected_profit"] * 0.3:
        return {"approved": False, "reason": f"Fees+slippage (${pnl['fees']:.2f}) eat >30% of profit — target too small"}

    breakeven = calculate_breakeven(entry_price, direction)

    return {
        "approved": True,
        "reason": "Trade approved",
        "direction": direction,
        "entry": entry_price,
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "breakeven": breakeven,
        "leverage": pos["leverage"],  # May be capped
        "confidence": confidence,
        "position_size": pos["position_size"],
        "quantity": pos["quantity"],
        "margin_used": pos["margin_used"],
        "risk_amount": pos["risk_amount"],
        "fees": pnl["fees"],
        "expected_profit": pnl["expected_profit"],
        "expected_loss": pnl["expected_loss"],
        "rr_ratio": pnl["rr_ratio"],
        "max_hold_hours": MAX_HOLD_HOURS,
    }


def _reject(reason: str) -> dict:
    return {
        "approved": False,
        "rejection_reason": reason,
        "position_size": 0,
        "quantity": 0,
        "risk_amount": 0,
        "max_risk_allowed": 0,
        "leverage": 0,
    }


if __name__ == "__main__":
    result = validate_trade(
        account_balance=100,
        entry_price=50000,
        stop_loss=49500,
        take_profit=51000,
        confidence=7,
        direction="LONG",
    )
    print("Trade Validation:")
    for k, v in result.items():
        print(f"  {k}: {v}")
