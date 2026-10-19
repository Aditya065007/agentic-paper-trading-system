# servers/tools/live_orders.py
# M6 - LIVE submission of guardian-approved limit orders to the Alpaca PAPER account.
#
# This is deliberately NOT an MCP tool and is never imported by the MCP server, so no agent can
# call it. Nothing is sent unless EVERY gate holds:
#   1. ALPACA_ORDERS_ENABLED=true   master switch in the environment (default OFF; a UI cannot set it)
#   2. BROKER_MODE=alpaca_paper     the guardian sized against the real paper account
#   3. the Alpaca URL is the PAPER host (broker._alpaca_base_url refuses anything else) and keys exist
#   4. each order is a whole-share DAY limit order with a sane symbol and price
#   5. pre-flight with Alpaca: enough buying power (buys) / enough shares held (sells)
#   6. no order with the same client_order_id exists, and no open order for that symbol and side
# One live order per symbol, side and UTC day: client_order_id = pt-YYYYMMDD-SYMBOL-side.
#
# Stops (G8): the guardian's stop_price is advisory by default. A protective stop is attached to a BUY only
# when a SEPARATE switch is on: ALPACA_STOPS_ENABLED=true in the real environment (default OFF; a UI cannot
# set it). The stop goes in the same request as the entry as a One-Triggers-Other order (order_class "oto",
# stop_loss.stop_price), so Alpaca activates it only after the entry fills. A BUY without a valid stop below its
# limit price is REJECTED when the switch is on (never sent unprotected). [VERIFY live: Alpaca's OTO fields
# and how the stop leg's time in force behaves for a DAY entry.]
# Nothing in this module replaces or cancels orders; cancel in the Alpaca dashboard.

import math
import os
import re
from datetime import datetime, timezone

import requests

from trading_system.servers.tools import broker

MAX_ORDERS_PER_SUBMISSION = 10
_TIMEOUT_SECONDS = 10
_SYMBOL_PATTERN = re.compile(r"[A-Z0-9][A-Z0-9.\-]{0,9}")


class OrderSubmissionBlocked(RuntimeError):
    """A safety gate is closed. Nothing was sent."""


class _Unreachable(Exception):
    """Alpaca could not be asked (network, HTTP error or odd reply)."""


def orders_enabled() -> bool:
    return os.environ.get("ALPACA_ORDERS_ENABLED", "").strip().lower() == "true"


def stops_enabled() -> bool:
    return os.environ.get("ALPACA_STOPS_ENABLED", "").strip().lower() == "true"


def _stop_for(order, clean):
    """The protective stop price for a BUY: a positive number at least 0.01 below the limit price, else None."""
    if clean["side"] != "buy" or not isinstance(order, dict):
        return None
    raw = order.get("stop_price")
    if isinstance(raw, bool) or not isinstance(raw, (int, float)) or not math.isfinite(raw):
        return None
    stop = round(float(raw), 2)
    return stop if 0 < stop <= round(clean["limit_price"] - 0.01, 2) else None


def check_gates() -> None:
    """Raise OrderSubmissionBlocked unless gates 1-3 all hold."""
    if not orders_enabled():
        raise OrderSubmissionBlocked(
            "Live paper orders are switched off: ALPACA_ORDERS_ENABLED is not 'true'.")
    try:
        if broker._mode() != "alpaca_paper":
            raise OrderSubmissionBlocked(
                "BROKER_MODE must be 'alpaca_paper': the guardian has to size against the real "
                "paper account, not sample data.")
        broker._alpaca_base_url()   # refuses anything but the paper host
        broker._alpaca_headers()    # refuses missing keys
    except broker.BrokerError as exc:
        raise OrderSubmissionBlocked(str(exc)) from None


def make_client_order_id(symbol: str, side: str, run_date=None) -> str:
    day = (run_date or datetime.now(timezone.utc).date()).strftime("%Y%m%d")
    return f"pt-{day}-{symbol}-{side}"


def _redact(text) -> str:
    message = " ".join(str(text).split())
    for name in ("ALPACA_API_KEY", "ALPACA_SECRET_KEY"):
        secret = os.environ.get(name, "").strip()
        if secret:
            message = message.replace(secret, "[REDACTED]")
    return message[:200]


def _request(method: str, path: str, params: dict = None, body: dict = None):
    url = f"{broker._alpaca_base_url()}{path}"
    headers = broker._alpaca_headers()
    try:
        if method == "GET":
            return requests.get(url, headers=headers, params=params, timeout=_TIMEOUT_SECONDS)
        return requests.post(url, headers=dict(headers, **{"Content-Type": "application/json"}),
                             json=body, timeout=_TIMEOUT_SECONDS)
    except Exception as exc:  # never echo the exception text: it may carry request details
        raise _Unreachable(type(exc).__name__) from None


def _json(response):
    try:
        return response.json()
    except ValueError:
        return None


def _reason(response) -> str:
    data = _json(response)
    message = str(data.get("message", "")) if isinstance(data, dict) else ""
    return _redact(f"HTTP {response.status_code}" + (f": {message}" if message else ""))


def _find_by_client_id(client_id: str):
    """Return the existing order or None. Anything but a clear 200/404 means we cannot tell."""
    response = _request("GET", "/orders:by_client_order_id", params={"client_order_id": client_id})
    if response.status_code == 404:
        return None
    data = _json(response)
    if response.status_code != 200 or not isinstance(data, dict):
        raise _Unreachable(f"HTTP {response.status_code}")
    return data


def _open_orders(symbol: str) -> list:
    response = _request("GET", "/orders", params={"status": "open", "symbols": symbol, "limit": 100})
    data = _json(response)
    if response.status_code != 200 or not isinstance(data, list):
        raise _Unreachable(f"HTTP {response.status_code}")
    return [o for o in data if isinstance(o, dict)]


def _clean(order) -> dict:
    """Validate one approved order; raise ValueError(reason) if it is not a sane limit order."""
    if not isinstance(order, dict):
        raise ValueError("order must be a dict")
    raw = order.get("symbol")
    symbol = raw.strip().upper() if isinstance(raw, str) else ""
    if not _SYMBOL_PATTERN.fullmatch(symbol):
        raise ValueError("symbol is not a valid ticker")
    side = order.get("side")
    if side not in ("buy", "sell"):
        raise ValueError("side must be 'buy' or 'sell'")
    qty = order.get("qty")
    if (isinstance(qty, bool) or not isinstance(qty, (int, float)) or not math.isfinite(qty)
            or qty <= 0 or qty != int(qty)):
        raise ValueError("qty must be a positive whole number")
    price = order.get("limit_price")
    if (isinstance(price, bool) or not isinstance(price, (int, float)) or not math.isfinite(price)
            or price <= 0):
        raise ValueError("limit_price must be a positive number")
    if order.get("order_type", "limit") != "limit":
        raise ValueError("only limit orders are allowed")
    if order.get("time_in_force", "day") != "day":
        raise ValueError("only day orders are allowed")
    limit = round(float(price), 2)
    if limit <= 0:
        raise ValueError("limit_price rounds to zero")
    return {"symbol": symbol, "side": side, "qty": int(qty), "limit_price": limit}


def _details(clean: dict, client_id: str, alpaca: dict = None) -> dict:
    alpaca = alpaca or {}
    return {"symbol": clean["symbol"], "side": clean["side"], "qty": clean["qty"],
            "order_type": "limit", "limit_price": clean["limit_price"], "time_in_force": "day",
            "client_order_id": alpaca.get("client_order_id") or client_id,
            "submitted_at": alpaca.get("submitted_at") or alpaca.get("created_at")}


def _result(clean, client_id, status, submitted, reason=None, alpaca=None, **extra) -> dict:
    out = {"order_id": (alpaca or {}).get("id"), "status": status, "submitted": submitted,
           "live": True, "details": _details(clean, client_id, alpaca)}
    if reason:
        out["reason"] = reason
    out.update(extra)
    return out


def submit_order(order: dict, run_date=None) -> dict:
    """Send ONE approved order to the Alpaca paper account. Never raises for per-order problems.

    Returns {order_id, status, submitted, live: True, details, reason?}.
    status is Alpaca's own status when the order is on the books, else one of:
    "rejected" (nothing sent), "skipped" (already handled), "unconfirmed" (sent but not confirmed)."""
    try:
        clean = _clean(order)
    except ValueError as exc:
        return {"order_id": None, "status": "rejected", "submitted": False, "live": True,
                "reason": str(exc), "details": {"symbol": str(order.get("symbol")) if isinstance(order, dict) else None}}
    client_id = make_client_order_id(clean["symbol"], clean["side"], run_date)
    stop = _stop_for(order, clean) if stops_enabled() else None
    if stops_enabled() and clean["side"] == "buy" and stop is None:
        return _result(clean, client_id, "rejected", False,
                       "protective stops are on but the order has no valid stop price below its limit "
                       "price; nothing was sent")

    try:
        # --- pre-flight against the real paper account
        if clean["side"] == "buy":
            needed = clean["qty"] * clean["limit_price"]
            buying_power = broker.get_account()["buying_power"]
            if needed > buying_power:
                return _result(clean, client_id, "rejected", False,
                               f"not enough buying power: needs ${needed:,.2f}, has ${buying_power:,.2f}")
        else:
            held = next((p["qty"] for p in broker.get_portfolio()["positions"]
                         if p["symbol"] == clean["symbol"]), 0)
            if held < clean["qty"]:
                return _result(clean, client_id, "rejected", False,
                               f"cannot sell {clean['qty']}: only {held:g} held at Alpaca")

        # --- duplicates: same client id, or any open order for this symbol and side
        existing = _find_by_client_id(client_id)
        if existing is not None:
            return _result(clean, client_id, str(existing.get("status", "unknown")), False,
                           "already submitted today", alpaca=existing, reason_code="duplicate")
        for open_order in _open_orders(clean["symbol"]):
            if open_order.get("side") == clean["side"]:
                return _result(clean, client_id, "skipped", False,
                               f"an open {clean['side']} order for {clean['symbol']} already exists",
                               alpaca=open_order, reason_code="open_order_exists")
    except (broker.BrokerError, _Unreachable) as exc:
        return _result(clean, client_id, "rejected", False,
                       _redact(f"could not verify the account before sending ({exc}); nothing was sent"))

    body = {"symbol": clean["symbol"], "qty": str(clean["qty"]), "side": clean["side"],
            "type": "limit", "time_in_force": "day",
            "limit_price": f"{clean['limit_price']:.2f}", "client_order_id": client_id}
    if stop is not None:
        body["order_class"] = "oto"
        body["stop_loss"] = {"stop_price": f"{stop:.2f}"}
    try:
        response = _request("POST", "/orders", body=body)
    except _Unreachable as exc:
        return _reconcile(clean, client_id, f"request failed ({exc})")

    data = _json(response)
    if response.status_code in (200, 201) and isinstance(data, dict) and data.get("id"):
        extra = {"stop_attached": True, "stop_price": stop} if stop is not None else {}
        return _result(clean, client_id, str(data.get("status", "accepted")), True, alpaca=data, **extra)
    return _result(clean, client_id, "rejected", False, _reason(response))


def _reconcile(clean, client_id, why) -> dict:
    """The send failed in flight, so we do not know whether Alpaca got it. Ask, never resend blindly."""
    try:
        existing = _find_by_client_id(client_id)
    except (broker.BrokerError, _Unreachable):
        existing = None
        known = False
    else:
        known = True
    if existing is not None:
        return _result(clean, client_id, str(existing.get("status", "accepted")), True,
                       f"{why}; found at Alpaca afterwards", alpaca=existing, reconciled=True)
    if known:
        return _result(clean, client_id, "rejected", False, f"{why}; Alpaca has no such order")
    return _result(clean, client_id, "unconfirmed", False,
                   f"{why}; could not confirm. Check the Alpaca dashboard before trying again.")


def submit_orders(orders: list, only_symbols=None, run_date=None) -> list:
    """Send the chosen guardian-approved orders. Raises OrderSubmissionBlocked if a gate is closed."""
    check_gates()
    chosen = {s.strip().upper() for s in only_symbols} if only_symbols is not None else None
    selected = [o for o in (orders or [])
                if chosen is None or (isinstance(o, dict) and str(o.get("symbol", "")).strip().upper() in chosen)]
    if len(selected) > MAX_ORDERS_PER_SUBMISSION:
        raise OrderSubmissionBlocked(
            f"{len(selected)} orders is more than the limit of {MAX_ORDERS_PER_SUBMISSION} per submission.")
    return [submit_order(order, run_date) for order in selected]
