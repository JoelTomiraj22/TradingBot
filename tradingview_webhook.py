"""
TradingView Webhook Integration (Optional).
Flask server that receives TradingView alerts and validates them against the strategy.

Setup Instructions:
1. Run this server: python tradingview_webhook.py
2. Expose to internet using ngrok: ngrok http 5000
3. In TradingView, create an alert with:
   - Webhook URL: https://your-ngrok-url.ngrok.io/webhook
   - Message (JSON):
     {
       "coin": "BTC/USDT",
       "direction": "buy",
       "price": {{close}},
       "timeframe": "5m"
     }
4. The server will validate the alert against the strategy before passing to the bot.
"""

from flask import Flask, request, jsonify
from datetime import datetime

from fetch_data import fetch_ohlcv
from indicators import add_all_indicators
from strategy import evaluate_signal
from logger_setup import get_logger

app = Flask(__name__)
logger = get_logger("webhook")

# Store pending alerts for the bot to pick up
pending_alerts = []


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "time": datetime.now().isoformat()})


@app.route("/webhook", methods=["POST"])
def webhook():
    """
    Receive and validate a TradingView alert.

    Expected JSON payload:
    {
        "coin": "BTC/USDT",
        "direction": "buy" or "sell",
        "price": 50000.0,
        "timeframe": "5m"  (optional, default 5m)
    }
    """
    try:
        data = request.get_json(force=True)
    except Exception:
        logger.error("Invalid JSON payload")
        return jsonify({"error": "Invalid JSON"}), 400

    coin = data.get("coin", "").upper()
    direction = data.get("direction", "").lower()
    price = data.get("price")
    timeframe = data.get("timeframe", "5m")

    # Validate required fields
    if not coin or direction not in ("buy", "sell") or price is None:
        logger.error(f"Missing fields: coin={coin}, direction={direction}, price={price}")
        return jsonify({"error": "Missing required fields: coin, direction, price"}), 400

    logger.info(f"Alert received: {direction.upper()} {coin} @ {price} ({timeframe})")

    # Validate against our strategy
    try:
        df = fetch_ohlcv(coin, timeframe, 200)
        df = add_all_indicators(df)
        signal = evaluate_signal(df)

        tv_direction = "LONG" if direction == "buy" else "SHORT"

        alert_result = {
            "timestamp": datetime.now().isoformat(),
            "coin": coin,
            "tv_direction": tv_direction,
            "tv_price": price,
            "strategy_direction": signal["direction"],
            "strategy_confidence": signal["confidence"],
            "confirmed": False,
            "reasons": signal["reasons"],
        }

        # Confirm if strategy agrees with TradingView alert
        if signal["direction"] == tv_direction and signal["confidence"] >= 5:
            alert_result["confirmed"] = True
            alert_result["entry"] = signal["entry"]
            alert_result["stop_loss"] = signal["stop_loss"]
            alert_result["take_profit"] = signal["take_profit"]
            alert_result["leverage"] = signal["leverage"]
            logger.info(f"CONFIRMED: {tv_direction} {coin} — Confidence {signal['confidence']}/10")
            pending_alerts.append(alert_result)
        else:
            logger.info(f"REJECTED: TV says {tv_direction}, strategy says {signal['direction']} "
                        f"(conf: {signal['confidence']}/10)")

        return jsonify(alert_result), 200

    except Exception as e:
        logger.error(f"Error validating alert: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/alerts", methods=["GET"])
def get_alerts():
    """Get pending confirmed alerts."""
    return jsonify({"alerts": pending_alerts, "count": len(pending_alerts)})


@app.route("/alerts/clear", methods=["POST"])
def clear_alerts():
    """Clear all pending alerts."""
    pending_alerts.clear()
    return jsonify({"status": "cleared"})


def get_pending_alerts() -> list:
    """Get pending alerts (called by bot.py)."""
    return list(pending_alerts)


def clear_pending_alerts():
    """Clear pending alerts after processing."""
    pending_alerts.clear()


if __name__ == "__main__":
    print("TradingView Webhook Server")
    print("─" * 40)
    print("Endpoints:")
    print("  GET  /health       — Health check")
    print("  POST /webhook      — Receive TradingView alert")
    print("  GET  /alerts       — View pending alerts")
    print("  POST /alerts/clear — Clear pending alerts")
    print("─" * 40)
    print("To expose to internet: ngrok http 5000")
    print()
    app.run(host="0.0.0.0", port=5000, debug=False)
