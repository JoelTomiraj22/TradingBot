"""
Claude AI Second Opinion — Reviews trade signals before execution.
Sends the full analysis to Claude and only approves if the setup is sound.

Setup:
1. Get an API key from https://console.anthropic.com/
2. Add to your .env file: ANTHROPIC_API_KEY=sk-ant-...
3. pip install anthropic
"""

import os
import json
from logger_setup import get_logger

logger = get_logger("ai_review")

# ANSI colors
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"


def get_claude_analysis(symbol: str, signal: dict, capital: float,
                        leverage: int, balance: float) -> dict:
    """
    Send the trade signal to Claude AI for a second opinion.

    Args:
        symbol: Trading pair (e.g. "BTC/USDT")
        signal: Full signal dict from evaluate_with_mtf()
        capital: User's chosen capital for this trade
        leverage: User's chosen leverage
        balance: Current account balance

    Returns:
        dict with: approved (bool), verdict (str), adjustments (dict or None)
    """
    api_key = os.getenv("ANTHROPIC_API_KEY", "")

    if not api_key:
        print(f"  {YELLOW}[AI Review] No ANTHROPIC_API_KEY in .env — skipping Claude review{RESET}")
        print(f"  {DIM}Add your key to .env to enable AI second opinion{RESET}")
        return {"approved": True, "verdict": "AI review skipped (no API key)", "adjustments": None}

    try:
        import anthropic
    except ImportError:
        print(f"  {YELLOW}[AI Review] anthropic package not installed{RESET}")
        print(f"  {DIM}Run: pip install anthropic{RESET}")
        return {"approved": True, "verdict": "AI review skipped (package not installed)", "adjustments": None}

    # Build the prompt
    prompt = _build_review_prompt(symbol, signal, capital, leverage, balance)

    try:
        print(f"  {CYAN}[AI Review] Asking Claude for second opinion...{RESET}")

        client = anthropic.Anthropic(api_key=api_key)
        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )

        response_text = message.content[0].text
        result = _parse_claude_response(response_text)

        # Display Claude's verdict
        if result["approved"]:
            print(f"  {GREEN}{BOLD}[AI Review] APPROVED{RESET}")
        else:
            print(f"  {RED}{BOLD}[AI Review] REJECTED{RESET}")
        print(f"  {DIM}{result['verdict']}{RESET}")

        if result.get("adjustments"):
            adj = result["adjustments"]
            print(f"  {YELLOW}Suggested adjustments:{RESET}")
            for k, v in adj.items():
                print(f"    {k}: {v}")

        logger.info(f"[AI Review] {symbol}: {'APPROVED' if result['approved'] else 'REJECTED'} — {result['verdict']}")
        return result

    except Exception as e:
        logger.error(f"[AI Review] Error: {e}")
        print(f"  {YELLOW}[AI Review] Error calling Claude: {e}{RESET}")
        print(f"  {DIM}Proceeding without AI review{RESET}")
        return {"approved": True, "verdict": f"AI review failed: {e}", "adjustments": None}


def _build_review_prompt(symbol: str, signal: dict, capital: float,
                         leverage: int, balance: float) -> str:
    """Build the prompt for Claude to review the trade."""

    reasons_text = "\n".join(f"  - {r}" for r in signal.get("reasons", []))

    return f"""You are a professional crypto futures scalp trader reviewing a trade signal from an automated bot. Be critical and conservative. Only approve trades with a clear edge.

TRADE SIGNAL TO REVIEW:
  Symbol: {symbol}
  Direction: {signal.get('direction', 'N/A')}
  Confidence: {signal.get('confidence', 0)}/10
  Entry: ${signal.get('entry', 0):,.4f}
  Stop Loss: ${signal.get('stop_loss', 0):,.4f}
  Take Profit: ${signal.get('take_profit', 0):,.4f}
  Higher TF Trend: {signal.get('htf_trend', 'N/A')}
  Entry Type: {signal.get('entry_type', 'MARKET')}

RISK PARAMETERS:
  Capital: ${capital:.2f}
  Leverage: {leverage}x
  Account Balance: ${balance:.2f}
  Position Size: ${capital * leverage:.2f}

BOT'S ANALYSIS:
{reasons_text}

REVIEW CRITERIA:
1. Is the direction supported by the evidence?
2. Is the stop loss well-placed (not too tight for the volatility)?
3. Is the take profit realistic?
4. Is the leverage appropriate for the confidence level?
5. Are there any red flags the bot might have missed?
6. For scalping: is this a chase or a proper pullback/reversal entry?

RESPOND IN THIS EXACT JSON FORMAT (no markdown, just raw JSON):
{{
  "approved": true or false,
  "verdict": "One sentence summary of your decision",
  "risk_rating": "LOW" or "MEDIUM" or "HIGH",
  "adjustments": {{
    "stop_loss": null or suggested price,
    "take_profit": null or suggested price,
    "leverage": null or suggested leverage
  }}
}}"""


def _parse_claude_response(response_text: str) -> dict:
    """Parse Claude's JSON response."""
    try:
        # Try to extract JSON from the response
        text = response_text.strip()

        # Handle if wrapped in markdown code blocks
        if "```" in text:
            start = text.find("{")
            end = text.rfind("}") + 1
            text = text[start:end]

        result = json.loads(text)

        approved = result.get("approved", False)
        verdict = result.get("verdict", "No verdict provided")
        risk = result.get("risk_rating", "UNKNOWN")
        adjustments = result.get("adjustments")

        # Clean up adjustments (remove nulls)
        if adjustments:
            adjustments = {k: v for k, v in adjustments.items() if v is not None}
            if not adjustments:
                adjustments = None

        verdict_with_risk = f"[{risk} RISK] {verdict}"

        return {
            "approved": approved,
            "verdict": verdict_with_risk,
            "adjustments": adjustments,
        }

    except (json.JSONDecodeError, KeyError, IndexError):
        # If we can't parse JSON, look for keywords
        lower = response_text.lower()
        approved = "approved" in lower or "approve" in lower
        rejected = "reject" in lower or "don't trade" in lower or "do not trade" in lower

        if rejected:
            approved = False

        return {
            "approved": approved,
            "verdict": response_text[:200],
            "adjustments": None,
        }


if __name__ == "__main__":
    # Test with a dummy signal
    test_signal = {
        "direction": "SHORT",
        "confidence": 8,
        "entry": 60000.0,
        "stop_loss": 60500.0,
        "take_profit": 59000.0,
        "htf_trend": "BEARISH",
        "entry_type": "MARKET",
        "reasons": [
            "EMA 9 below EMA 21 (trend aligned)",
            "Price below VWAP — sellers in control",
            "Bearish Engulfing candle",
            "Higher TF trend BEARISH (aligned)",
        ],
    }
    result = get_claude_analysis("BTC/USDT", test_signal, 10.0, 15, 5000.0)
    print(f"\nResult: {result}")
