# servers/tools/broker.py
# M6 - Account, portfolio & orders.
#
# BROKER_MODE=offline       (default) fixtures + simulated orders. No network.
# BROKER_MODE=alpaca_paper  READ-ONLY calls to the Alpaca PAPER account (account, positions,
#                           today's orders). Orders are still NEVER sent: dry-run only.
# Keys come from the environment only (ALPACA_API_KEY, ALPACA_SECRET_KEY), never from code.

import json
import math
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import requests

_FIXTURES = Path(__file__).resolve().parents[2] / "fixtures"
_PAPER_HOST = "paper-api.alpaca.markets"
_TIMEOUT_SECONDS = 10

# Orders placed during this run live here (a simulated order book).
_orders: list = []


class BrokerError(RuntimeError):
    """The broker could not be read safely. The caller must stop and NOT trade."""


def _mode() -> str:
    mode = os.environ.get("BROKER_MODE", "offline").strip().lower() or "offline"
    if mode not in ("offline", "alpaca_paper"):
        raise BrokerError("BROKER_MODE must be 'offline' or 'alpaca_paper'.")
    return mode


def _read_fixture(name: str) -> dict:
    with open(_FIXTURES / name, "r", encoding="utf-8") as stream:
        return json.load(stream)


def _number(value, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        raise BrokerError(f"Alpaca field '{field}' is not a number.") from None
    if not math.isfinite(result):
        raise BrokerError(f"Alpaca field '{field}' is not a finite number.")
    return result


# ------------------------------------------------------------ Alpaca (read-only)

def _alpaca_base_url() -> str:
    raw = (os.environ.get("ALPACA_TRADING_URL") or f"https://{_PAPER_HOST}").strip().rstrip("/")
    if raw.endswith("/v2"):
        raw = raw[:-3]
    parsed = urlparse(raw)
    if parsed.scheme != "https" or parsed.hostname != _PAPER_HOST:
        raise BrokerError("Only the Alpaca PAPER trading URL is allowed in this project.")
    return f"{raw}/v2"


def _alpaca_headers() -> dict:
    key = os.environ.get("ALPACA_API_KEY", "").strip()
    secret = os.environ.get("ALPACA_SECRET_KEY", "").strip()
    if not key or not secret:
        raise BrokerError("ALPACA_API_KEY and ALPACA_SECRET_KEY must be set in the environment.")
    return {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}


def _alpaca_get(path: str, params: dict = None):
    url = f"{_alpaca_base_url()}{path}"
    headers = _alpaca_headers()
    try:
        response = requests.get(url, headers=headers, params=params, timeout=_TIMEOUT_SECONDS)
    except Exception as exc:  # never echo the exception text: it may contain request details
        raise BrokerError(f"Alpaca request failed ({type(exc).__name__}).") from None
    if response.status_code != 200:
        raise BrokerError(f"Alpaca returned HTTP {response.status_code} for {path}.")
    try:
        return response.json()
    except ValueError:
        raise BrokerError("Alpaca returned invalid JSON.") from None


# ---------------------------------------------------------------- the five tools

def get_account() -> dict:
    """Return {equity, cash, buying_power} (plus last_equity when known)."""
    if _mode() == "offline":
        data = _read_fixture("account.json")
        result = {"equity": data["equity"], "cash": data["cash"],
                  "buying_power": data["buying_power"]}
        if data.get("last_equity") is not None:
            result["last_equity"] = data["last_equity"]
        return result
    data = _alpaca_get("/account")
    if not isinstance(data, dict):
        raise BrokerError("Alpaca account response was not an object.")
    result = {"equity": _number(data.get("equity"), "equity"),
              "cash": _number(data.get("cash"), "cash"),
              "buying_power": _number(data.get("buying_power"), "buying_power")}
    if data.get("last_equity") is not None:
        result["last_equity"] = _number(data["last_equity"], "last_equity")
    return result


def get_portfolio() -> dict:
    """Return {positions: [{symbol, qty, avg_entry, market_value, pl_pct, ...}]}."""
    if _mode() == "offline":
        return {"positions": _read_fixture("portfolio.json")["positions"]}
    data = _alpaca_get("/positions")
    if not isinstance(data, list):
        raise BrokerError("Alpaca positions response was not a list.")
    positions = []
    for item in data:
        if not isinstance(item, dict) or not str(item.get("symbol", "")).strip():
            raise BrokerError("Alpaca returned a position without a symbol.")
        qty = _number(item.get("qty"), "qty")
        if str(item.get("side", "long")).lower() == "short":
            qty = -abs(qty)
        positions.append({
            "symbol": str(item["symbol"]).strip().upper(),
            "qty": qty,
            "avg_entry": _number(item.get("avg_entry_price"), "avg_entry_price"),
            "market_value": _number(item.get("market_value"), "market_value"),
            "pl_pct": round(_number(item.get("unrealized_plpc"), "unrealized_plpc") * 100, 4),
            "side": str(item.get("side", "long")).lower(),
        })
    return {"positions": positions}


def _count_buys_today() -> int:
    today = datetime.now(timezone.utc).date().isoformat()
    count = sum(1 for o in _orders
                if o["status"] == "accepted_dry_run" and o["details"]["side"] == "buy"
                and o["details"]["placed_at"][:10] == today)
    if _mode() == "alpaca_paper":
        after = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0,
                                                   microsecond=0).isoformat()
        data = _alpaca_get("/orders", {"status": "all", "after": after, "limit": 500})
        if not isinstance(data, list):
            raise BrokerError("Alpaca orders response was not a list.")
        count += sum(1 for o in data if isinstance(o, dict) and o.get("side") == "buy"
                     and o.get("status") not in ("canceled", "expired", "rejected"))
    return count


def get_portfolio_snapshot(sector_map: dict = None) -> dict:
    """Account + positions in the exact shape the guardian (risk.check_rules) expects."""
    account = get_account()
    portfolio = get_portfolio()
    sectors = {str(k).strip().upper(): v for k, v in (sector_map or {}).items()}
    positions = {}
    for item in portfolio["positions"]:
        symbol = str(item["symbol"]).strip().upper()
        entry = {"qty": item["qty"], "market_value": item["market_value"],
                 "avg_entry": item.get("avg_entry")}
        sector = sectors.get(symbol) or item.get("sector")
        if sector:
            entry["sector"] = sector
        positions[symbol] = entry
    snapshot = {"equity": account["equity"], "cash": account["cash"],
                "buying_power": account["buying_power"], "positions": positions,
                "new_buys_today": _count_buys_today()}
    if account.get("last_equity"):
        snapshot["day_start_equity"] = account["last_equity"]
    return snapshot


def _whole_positive(qty):
    if isinstance(qty, bool) or not isinstance(qty, (int, float)) or not math.isfinite(qty):
        return None
    if qty <= 0 or qty != int(qty):
        return None
    return int(qty)


def place_order(symbol: str, side: str, qty: int,
                order_type: str = "limit", limit_price: float = None,
                time_in_force: str = "day", stop_price: float = None,
                dry_run: bool = True, client_order_id: str = None) -> dict:
    """Place ONE order. DRY-RUN only: simulated, never sent to a broker.
    Assumes the order already passed the guardian. Returns {order_id, status, details}."""
    def reject(reason):
        return {"order_id": None, "status": "rejected", "details": {"reason": reason}}

    if not dry_run:
        return reject("Live mode not configured; use dry_run=True.")
    if not isinstance(symbol, str) or not symbol.strip():
        return reject("symbol must be a non-empty string")
    whole = _whole_positive(qty)
    if whole is None:
        return reject("qty must be a positive whole number")
    if side not in ("buy", "sell"):
        return reject("side must be 'buy' or 'sell'")
    if order_type == "limit" and (
            isinstance(limit_price, bool) or not isinstance(limit_price, (int, float))
            or not math.isfinite(limit_price) or limit_price <= 0):
        return reject("limit order needs a positive limit_price")
    client_id = client_order_id or str(uuid.uuid4())
    if any(o["details"]["client_order_id"] == client_id for o in _orders):
        return reject("duplicate client_order_id")

    order = {
        "order_id": str(uuid.uuid4())[:8],
        "status": "accepted_dry_run",
        "details": {
            "symbol": symbol.strip().upper(), "side": side, "qty": whole,
            "order_type": order_type, "limit_price": limit_price,
            "time_in_force": time_in_force, "stop_price": stop_price,
            "client_order_id": client_id,
            "placed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        },
    }
    _orders.append(order)
    return order


def get_open_orders() -> dict:
    """Return orders placed this run that are still open. {open_orders: [...]}."""
    return {"open_orders": [o for o in _orders if o["status"] == "accepted_dry_run"]}


def cancel_order(order_id: str) -> dict:
    """Cancel one open order by its id. Returns {order_id, status}."""
    for o in _orders:
        if o["order_id"] == order_id:
            o["status"] = "cancelled"
            return {"order_id": order_id, "status": "cancelled"}
    return {"order_id": order_id, "status": "not_found"}


def register_broker_tools(mcp) -> None:
    """Attach the M6 tools to an existing MCP server instance (the server file is not edited here)."""
    for tool in (get_account, get_portfolio, place_order, get_open_orders, cancel_order):
        mcp.tool()(tool)


if __name__ == "__main__":
    print(get_account())