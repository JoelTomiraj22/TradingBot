"""
DEPRECATED — All AI review is now handled by multi_ai_verifier.py

This file exists only for backward compatibility.
Use: from multi_ai_verifier import verify_trade
"""

from multi_ai_verifier import verify_trade


def get_claude_analysis(symbol, signal, capital, leverage, balance):
    """Legacy wrapper — redirects to multi-AI verifier."""
    result = verify_trade(symbol, signal, capital, leverage, balance)
    return {
        "approved": result["approved"],
        "verdict": result["reason"],
        "adjustments": result.get("suggestions"),
    }


def get_dual_ai_review(symbol, signal, capital, leverage, balance):
    """Legacy wrapper — redirects to multi-AI verifier."""
    return verify_trade(symbol, signal, capital, leverage, balance)
