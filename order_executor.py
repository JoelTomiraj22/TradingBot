"""
Order execution module for Binance Futures.
Market/limit orders, SL/TP, position management.
"""

import math
from config import get_exchange


class OrderExecutor:
    def __init__(self, exchange=None):
        self.exchange = exchange or get_exchange()
        self._market_info_cache = {}

    # ─── Market Info ───────────────────────────────────────────

    def _get_market_info(self, symbol: str) -> dict:
        """Get market precision and limits for a symbol."""
        if symbol not in self._market_info_cache:
            self.exchange.load_markets()
            market = self.exchange.market(symbol)
            self._market_info_cache[symbol] = market
        return self._market_info_cache[symbol]

    def _precision_to_step(self, precision_val) -> tuple:
        """Convert precision (int or float) to (step_size, decimal_places)."""
        if isinstance(precision_val, float) and precision_val < 1:
            # Precision is already a step size (e.g., 0.001)
            step = precision_val
            decimals = max(0, len(str(precision_val).rstrip('0').split('.')[-1]))
        else:
            # Precision is number of decimal places (e.g., 3)
            decimals = int(precision_val)
            step = 10 ** (-decimals)
        return step, decimals

    def _round_quantity(self, symbol: str, quantity: float) -> float:
        """Round quantity to Binance's required precision."""
        try:
            market = self._get_market_info(symbol)
            precision_val = market.get("precision", {}).get("amount", 3)
            step, decimals = self._precision_to_step(precision_val)
            rounded = math.floor(quantity / step) * step
            # Ensure we meet minimum order quantity
            min_qty = market.get("limits", {}).get("amount", {}).get("min", 0)
            if min_qty and rounded < min_qty:
                rounded = min_qty
            return round(rounded, decimals)
        except Exception as e:
            print(f"[Round] Quantity rounding error for {symbol}: {e}")
            return round(math.floor(quantity * 1000) / 1000, 3)

    def _round_price(self, symbol: str, price: float) -> float:
        """Round price to Binance's required precision."""
        try:
            market = self._get_market_info(symbol)
            precision_val = market.get("precision", {}).get("price", 2)
            step, decimals = self._precision_to_step(precision_val)
            return round(round(price / step) * step, decimals)
        except Exception as e:
            print(f"[Round] Price rounding error for {symbol}: {e}")
            return round(price, 2)

    # ─── Leverage ──────────────────────────────────────────────

    def set_leverage(self, symbol: str, leverage: int) -> bool:
        """Set leverage for a trading pair."""
        try:
            self.exchange.set_leverage(leverage, symbol)
            print(f"[Leverage] Set {symbol} to {leverage}x")
            return True
        except Exception as e:
            print(f"[Leverage] Error setting {symbol} to {leverage}x: {e}")
            return False

    # ─── Account ───────────────────────────────────────────────

    def get_balance(self) -> float:
        """Get USDT available balance."""
        try:
            balance = self.exchange.fetch_balance()
            return float(balance.get("USDT", {}).get("free", 0))
        except Exception as e:
            print(f"[Balance] Error: {e}")
            return 0.0

    def get_total_balance(self) -> float:
        """Get USDT total balance (available + in positions)."""
        try:
            balance = self.exchange.fetch_balance()
            return float(balance.get("USDT", {}).get("total", 0))
        except Exception as e:
            print(f"[Balance] Error: {e}")
            return 0.0

    def get_open_positions(self) -> list:
        """Get all open positions."""
        try:
            positions = self.exchange.fetch_positions()
            return [p for p in positions if float(p.get("contracts", 0)) > 0]
        except Exception as e:
            print(f"[Positions] Error: {e}")
            return []

    # ─── Orders ────────────────────────────────────────────────

    def place_market_order(self, symbol: str, side: str, quantity: float) -> dict:
        """
        Place a market order.
        side: "buy" or "sell"
        """
        quantity = self._round_quantity(symbol, quantity)
        if quantity <= 0:
            return {"error": "Quantity rounds to zero"}

        # Check balance
        balance = self.get_balance()
        if balance <= 0:
            return {"error": f"Insufficient balance: ${balance:.2f}"}

        try:
            order = self.exchange.create_order(
                symbol=symbol,
                type="market",
                side=side,
                amount=quantity,
            )
            print(f"[Order] Market {side.upper()} {quantity} {symbol} — ID: {order['id']}")
            return order
        except Exception as e:
            print(f"[Order] Market order error: {e}")
            return {"error": str(e)}

    def place_limit_order(self, symbol: str, side: str, quantity: float, price: float) -> dict:
        """Place a limit order."""
        quantity = self._round_quantity(symbol, quantity)
        price = self._round_price(symbol, price)

        if quantity <= 0:
            return {"error": "Quantity rounds to zero"}

        try:
            order = self.exchange.create_order(
                symbol=symbol,
                type="limit",
                side=side,
                amount=quantity,
                price=price,
            )
            print(f"[Order] Limit {side.upper()} {quantity} {symbol} @ {price} — ID: {order['id']}")
            return order
        except Exception as e:
            print(f"[Order] Limit order error: {e}")
            return {"error": str(e)}

    def place_stop_loss(self, symbol: str, side: str, quantity: float, stop_price: float) -> dict:
        """
        Place a stop loss order.
        side: "sell" for long positions, "buy" for short positions
        """
        quantity = self._round_quantity(symbol, quantity)
        stop_price = self._round_price(symbol, stop_price)

        # Try different order types in order of preference
        attempts = [
            ("stop_market", None, {"stopPrice": stop_price, "reduceOnly": True}),
            ("STOP", stop_price, {"stopPrice": stop_price, "reduceOnly": True}),
            ("stop", stop_price, {"stopPrice": stop_price, "reduceOnly": True}),
        ]

        last_error = None
        for order_type, price, params in attempts:
            try:
                order = self.exchange.create_order(
                    symbol=symbol,
                    type=order_type,
                    side=side,
                    amount=quantity,
                    price=price,
                    params=params,
                )
                print(f"[Order] Stop Loss {side.upper()} {quantity} {symbol} @ {stop_price} — ID: {order['id']}")
                return order
            except Exception as e:
                last_error = str(e)
                if "-4120" in last_error or "requires a price" in last_error:
                    continue
                print(f"[Order] Stop loss error: {e}")
                return {"error": last_error}

        print(f"[Order] Stop loss error: {last_error}")
        return {"error": last_error or "Stop loss order type not supported"}

    def place_take_profit(self, symbol: str, side: str, quantity: float, stop_price: float) -> dict:
        """
        Place a take profit order.
        side: "sell" for long positions, "buy" for short positions
        """
        quantity = self._round_quantity(symbol, quantity)
        stop_price = self._round_price(symbol, stop_price)

        attempts = [
            ("take_profit_market", None, {"stopPrice": stop_price, "reduceOnly": True}),
            ("TAKE_PROFIT", stop_price, {"stopPrice": stop_price, "reduceOnly": True}),
            ("take_profit", stop_price, {"stopPrice": stop_price, "reduceOnly": True}),
        ]

        last_error = None
        for order_type, price, params in attempts:
            try:
                order = self.exchange.create_order(
                    symbol=symbol,
                    type=order_type,
                    side=side,
                    amount=quantity,
                    price=price,
                    params=params,
                )
                print(f"[Order] Take Profit {side.upper()} {quantity} {symbol} @ {stop_price} — ID: {order['id']}")
                return order
            except Exception as e:
                last_error = str(e)
                if "-4120" in last_error or "requires a price" in last_error:
                    continue
                print(f"[Order] Take profit error: {e}")
                return {"error": last_error}

        print(f"[Order] Take profit error: {last_error}")
        return {"error": last_error or "Take profit order type not supported"}

    # ─── Position Management ───────────────────────────────────

    def cancel_all_orders(self, symbol: str) -> bool:
        """Cancel all open orders for a symbol."""
        try:
            self.exchange.cancel_all_orders(symbol)
            print(f"[Orders] Cancelled all orders for {symbol}")
            return True
        except Exception as e:
            print(f"[Orders] Cancel error for {symbol}: {e}")
            return False

    def close_position(self, symbol: str) -> dict:
        """Emergency close — market close any open position. Returns clean result with P&L."""
        try:
            positions = self.exchange.fetch_positions([symbol])
            for pos in positions:
                contracts = float(pos.get("contracts", 0))
                if contracts > 0:
                    pos_side = pos["side"]  # "long" or "short"
                    entry_price = float(pos.get("entryPrice", 0))
                    unrealized_pnl = float(pos.get("unrealizedPnl", 0))
                    side = "sell" if pos_side == "long" else "buy"
                    # Cancel existing orders first
                    self.cancel_all_orders(symbol)
                    order = self.place_market_order(symbol, side, contracts)
                    if "error" in order:
                        return order
                    close_price = float(order.get("average") or order.get("price", 0))
                    return {
                        "closed": True,
                        "symbol": symbol,
                        "side": pos_side,
                        "entry_price": entry_price,
                        "close_price": close_price,
                        "quantity": contracts,
                        "pnl": unrealized_pnl,
                        "order_id": order.get("id"),
                    }
            return {"info": "No open position found"}
        except Exception as e:
            print(f"[Close] Error closing {symbol}: {e}")
            return {"error": str(e)}

    def close_all_positions(self) -> list:
        """Emergency close ALL positions."""
        results = []
        positions = self.get_open_positions()
        for pos in positions:
            symbol = pos["symbol"]
            result = self.close_position(symbol)
            results.append({"symbol": symbol, "result": result})
        if not positions:
            print("[Close] No open positions to close.")
        return results

    # ─── Full Trade Execution ──────────────────────────────────

    def execute_trade(self, symbol: str, direction: str, quantity: float,
                      leverage: int, stop_loss: float, take_profit: float) -> dict:
        """
        Execute a full trade: set leverage, place entry, then try SL/TP.
        If exchange SL/TP fails (demo), returns sl_tp_mode="monitor" so the
        bot can use software-based monitoring instead.

        Args:
            direction: "LONG" or "SHORT"
        """
        # Set leverage
        if not self.set_leverage(symbol, leverage):
            return {"error": "Failed to set leverage"}

        # Entry order
        entry_side = "buy" if direction == "LONG" else "sell"
        entry_order = self.place_market_order(symbol, entry_side, quantity)
        if "error" in entry_order:
            return entry_order

        # Try exchange SL/TP orders
        exit_side = "sell" if direction == "LONG" else "buy"
        sl_order = self.place_stop_loss(symbol, exit_side, quantity, stop_loss)
        tp_order = self.place_take_profit(symbol, exit_side, quantity, take_profit)

        sl_ok = "error" not in sl_order
        tp_ok = "error" not in tp_order

        return {
            "entry": entry_order,
            "stop_loss": sl_order,
            "take_profit": tp_order,
            "symbol": symbol,
            "direction": direction,
            "quantity": quantity,
            "leverage": leverage,
            "sl_tp_mode": "exchange" if (sl_ok and tp_ok) else "monitor",
        }


if __name__ == "__main__":
    executor = OrderExecutor()
    print(f"USDT Balance: ${executor.get_balance():.2f}")
    print(f"Open Positions: {len(executor.get_open_positions())}")
