# servers/tools/risk.py
# M5 - Risk, sizing & guardian. Limits come from config/rules.yaml (one source of truth).
#
# Pure functions: numbers in, numbers out. Nothing here calls a broker or an LLM.
# Bad TRADING input never raises: it means "no trade" / "blocked" (fail closed).
# A bad rules.yaml DOES raise RulesConfigError: that is a setup mistake, not a market event.

import math
from datetime import date, datetime, timezone
from pathlib import Path

import yaml

_RULES_PATH = Path(__file__).resolve().parents[2] / "config" / "rules.yaml"


class RulesConfigError(ValueError):
    """rules.yaml is missing, unreadable, or has an invalid value."""


# section -> key -> (lowest allowed, highest allowed or None, must be a whole number)
_SCHEMA = {
    "position": {
        "max_single_position_pct": (0.0, 1.0, False),
        "cash_reserve_pct": (0.0, 1.0, False),
        "max_sector_exposure_pct": (0.0, 1.0, False),
    },
    "sizing": {
        "risk_per_trade_pct": (0.0, 1.0, False),
        "stop_loss_atr_mult": (0.0, None, False),
        "trailing_exit_atr_mult": (0.0, None, False),
    },
    "limits": {
        "daily_loss_limit_pct": (0.0, 1.0, False),
        "max_new_orders_per_day": (0, None, True),
        "earnings_blackout_days": (0, None, True),
    },
    "orders": {
        "limit_buffer_pct": (0.0, 0.05, False),
    },
}
_MUST_BE_POSITIVE = {
    "max_single_position_pct", "max_sector_exposure_pct", "risk_per_trade_pct",
    "stop_loss_atr_mult", "trailing_exit_atr_mult",
}


def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _validate_rules(rules) -> dict:
    if not isinstance(rules, dict):
        raise RulesConfigError("rules must be a mapping with position/sizing/limits/orders sections.")
    for section, keys in _SCHEMA.items():
        block = rules.get(section)
        if not isinstance(block, dict):
            raise RulesConfigError(f"rules are missing the '{section}' section.")
        for key, (low, high, whole) in keys.items():
            value = block.get(key)
            if not _is_number(value):
                raise RulesConfigError(f"{section}.{key} must be a finite number.")
            if whole and value != int(value):
                raise RulesConfigError(f"{section}.{key} must be a whole number.")
            if value < low or (high is not None and value > high):
                raise RulesConfigError(f"{section}.{key} is outside its allowed range.")
            if key in _MUST_BE_POSITIVE and value <= 0:
                raise RulesConfigError(f"{section}.{key} must be greater than zero.")
    if rules["orders"].get("order_type") != "limit":
        raise RulesConfigError("orders.order_type must be 'limit' (limit orders only).")
    if rules["orders"].get("whole_shares") is not True:
        raise RulesConfigError("orders.whole_shares must be true (no fractional shares).")
    cooldown = rules["limits"].get("cooldown_days")
    if cooldown is not None and (not _is_number(cooldown) or cooldown < 0 or cooldown != int(cooldown)):
        raise RulesConfigError("limits.cooldown_days must be a whole number of days (or absent).")
    return rules


def load_rules(path=None) -> dict:
    """Read and validate rules.yaml (or the file at `path`)."""
    source = Path(path) if path is not None else _RULES_PATH
    try:
        with open(source, "r", encoding="utf-8") as stream:
            rules = yaml.safe_load(stream)
    except OSError as exc:
        raise RulesConfigError(f"rules file could not be read: {source.name}") from exc
    except yaml.YAMLError as exc:
        raise RulesConfigError("rules file contains invalid YAML.") from exc
    return _validate_rules(rules)


def _no_trade(symbol, reason, *warnings) -> dict:
    return {"symbol": symbol, "qty": 0, "notional": 0.0, "stop_price": 0.0,
            "reason": f"No trade: {reason}", "warnings": list(warnings)}


def size_position(
    symbol: str,
    price: float,
    atr: float,
    equity: float,
    risk_pct: float = None,
    stop_atr_mult: float = None,
    max_position_pct: float = None,
    existing_value: float = 0.0,
    rules: dict = None,
) -> dict:
    """How many whole shares to buy, and where the stop goes.

    Any limit left as None comes from the rules. `existing_value` is the dollar value
    already held in this stock, so the position cap only leaves the remaining room."""
    if rules is None:
        rules = load_rules()
    else:
        _validate_rules(rules)
    if risk_pct is None:
        risk_pct = rules["sizing"]["risk_per_trade_pct"]
    if stop_atr_mult is None:
        stop_atr_mult = rules["sizing"]["stop_loss_atr_mult"]
    if max_position_pct is None:
        max_position_pct = rules["position"]["max_single_position_pct"]

    numbers_ok = all(_is_number(v) and v > 0 for v in (price, atr, equity))
    if not numbers_ok or not _is_number(existing_value) or existing_value < 0:
        return _no_trade(symbol, "price, ATR or equity was zero or negative.", "bad input")

    risk_per_share = stop_atr_mult * atr
    stop_price = round(price - risk_per_share, 2)
    if stop_price <= 0:
        return _no_trade(symbol, "the stop price would be at or below zero.", "bad stop")

    dollars_at_risk = equity * risk_pct
    qty = int(dollars_at_risk / risk_per_share)

    warnings = []
    room = max(0.0, equity * max_position_pct - existing_value)
    if qty * price > room:
        qty = int(room / price)
        warnings.append("Trimmed to the position cap.")
    if qty <= 0:
        return _no_trade(symbol, "the risk budget or the position cap allows less than one share.",
                         *warnings)

    reason = (f"Risk {risk_pct:.0%} of ${equity:,.0f} = ${dollars_at_risk:,.0f}; "
              f"stop at ${stop_price}; so {qty} share(s).")
    return {"symbol": symbol, "qty": qty, "notional": round(qty * price, 2),
            "stop_price": stop_price, "reason": reason, "warnings": warnings}


def make_limit_price(side: str, last_price: float, rules: dict = None) -> float:
    """Limit price with the small cushion from the rules: buys a bit above, sells a bit below."""
    if rules is None:
        rules = load_rules()
    else:
        _validate_rules(rules)
    if side not in ("buy", "sell"):
        raise ValueError("side must be 'buy' or 'sell'.")
    if not _is_number(last_price) or last_price <= 0:
        raise ValueError("last_price must be a positive number.")
    buffer = rules["orders"]["limit_buffer_pct"]
    factor = 1 + buffer if side == "buy" else 1 - buffer
    return round(last_price * factor, 2)


# ---------------------------------------------------------------- the guardian

def _parse_day(value, name: str) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value.strip()[:10])
        except ValueError:
            pass
    raise ValueError(f"{name} is not a valid date.")


def _day_map(raw, name: str) -> dict:
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError(f"{name} must be a mapping of symbol to date.")
    return {str(sym).strip().upper(): _parse_day(day, name) for sym, day in raw.items()}


def _normalize_snapshot(snapshot) -> dict:
    """Accept the portfolio in either shape (dict by symbol or list) and validate it.
    Raises ValueError for anything unusable; the guardian then blocks everything."""
    if not isinstance(snapshot, dict):
        raise ValueError("portfolio snapshot must be a dict.")
    equity, cash = snapshot.get("equity"), snapshot.get("cash")
    if not (_is_number(equity) and equity > 0):
        raise ValueError("equity is missing or not positive.")
    if not (_is_number(cash) and cash >= 0):
        raise ValueError("cash is missing or negative.")

    raw = snapshot.get("positions", {})
    if isinstance(raw, dict):
        items = list(raw.items())
    elif isinstance(raw, (list, tuple)):
        items = [(info.get("symbol") if isinstance(info, dict) else None, info) for info in raw]
    else:
        raise ValueError("positions must be a dict or a list.")
    positions = {}
    for symbol, info in items:
        if not isinstance(symbol, str) or not symbol.strip() or not isinstance(info, dict):
            raise ValueError("a position entry is malformed.")
        qty, value = info.get("qty", 0), info.get("market_value", 0.0)
        if not (_is_number(qty) and qty >= 0 and _is_number(value) and value >= 0):
            raise ValueError("a position has a negative or non-numeric qty/market_value.")
        positions[symbol.strip().upper()] = {
            "qty": qty, "market_value": float(value), "sector": info.get("sector") or None}

    new_buys = snapshot.get("new_buys_today", 0)
    if isinstance(new_buys, bool) or not isinstance(new_buys, int) or new_buys < 0:
        raise ValueError("new_buys_today must be a whole number of at least 0.")
    day_start = snapshot.get("day_start_equity")
    if day_start is not None and not (_is_number(day_start) and day_start > 0):
        raise ValueError("day_start_equity must be a positive number.")
    sectors = snapshot.get("sectors") or {}
    if not isinstance(sectors, dict):
        raise ValueError("sectors must be a mapping of symbol to sector.")
    today = snapshot.get("today")
    return {
        "equity": float(equity), "cash": float(cash), "positions": positions,
        "new_buys_today": new_buys, "day_start_equity": day_start,
        "sectors": {str(k).strip().upper(): v for k, v in sectors.items() if v},
        "today": _parse_day(today, "today") if today is not None
        else datetime.now(timezone.utc).date(),
        "earnings_dates": _day_map(snapshot.get("earnings_dates"), "earnings_dates"),
        "recent_exits": _day_map(snapshot.get("recent_exits"), "recent_exits"),
    }


def _whole_positive(qty):
    if isinstance(qty, bool) or not _is_number(qty) or qty <= 0 or qty != int(qty):
        return None
    return int(qty)


def check_rules(proposed_orders: list, portfolio_snapshot: dict, rules: dict = None) -> dict:
    """The guardian. Approve or block each order against the hard limits.

    proposed_orders    : [{symbol, side 'buy'/'sell', qty, limit_price, sector (optional)}]
    portfolio_snapshot : {equity, cash, positions (dict by symbol OR list of dicts), and the
                          optional new_buys_today, day_start_equity, today, earnings_dates,
                          recent_exits, sectors}
    Returns {approved, blocked: [{order, rule, detail}], warnings, new_buys_approved}.
    The FIRST failing rule is the one reported. Never raises on bad trading input."""
    rules = load_rules() if rules is None else _validate_rules(rules)
    orders = list(proposed_orders) if isinstance(proposed_orders, (list, tuple)) else []

    try:
        snap = _normalize_snapshot(portfolio_snapshot)
    except ValueError as exc:
        return {"approved": [],
                "blocked": [{"order": order, "rule": "bad_snapshot", "detail": str(exc)}
                            for order in orders],
                "warnings": [f"Portfolio snapshot rejected: {exc}"], "new_buys_approved": 0}

    equity, cash, positions = snap["equity"], snap["cash"], snap["positions"]
    cap_dollars = equity * rules["position"]["max_single_position_pct"]
    sector_cap = equity * rules["position"]["max_sector_exposure_pct"]
    spendable = cash - equity * rules["position"]["cash_reserve_pct"]
    max_orders = rules["limits"]["max_new_orders_per_day"]
    blackout_days = rules["limits"]["earnings_blackout_days"]
    cooldown_days = rules["limits"].get("cooldown_days")
    loss_limit = rules["limits"]["daily_loss_limit_pct"]

    down_too_far = (snap["day_start_equity"] is not None
                    and (equity - snap["day_start_equity"]) / snap["day_start_equity"] <= -loss_limit)

    approved, blocked, warnings = [], [], []
    buy_count = snap["new_buys_today"]
    approved_buys = 0
    running_spend = 0.0
    planned_symbol, planned_sector = {}, {}
    warned_loss, warned_earnings, warned_sector = False, False, set()

    def block(order, rule, detail):
        blocked.append({"order": order, "rule": rule, "detail": detail})

    for order in orders:
        if not isinstance(order, dict):
            block(order, "bad_order", "order must be a dict"); continue
        symbol = order.get("symbol")
        side = order.get("side")
        qty = _whole_positive(order.get("qty"))
        price = order.get("limit_price")
        if (not isinstance(symbol, str) or not symbol.strip() or side not in ("buy", "sell")
                or qty is None or not _is_number(price) or price <= 0):
            block(order, "bad_order",
                  "needs a symbol, side buy/sell, a positive whole qty and a positive limit_price")
            continue
        sym = symbol.strip().upper()
        notional = qty * price

        if side == "sell":
            held = positions.get(sym)
            if held is None or held["qty"] <= 0:
                block(order, "long_only", f"{sym} not held; short selling not allowed"); continue
            if qty > held["qty"]:
                block(order, "oversell", f"selling {qty} but only {held['qty']} held"); continue
            approved.append(order)
            continue

        # ---- buys
        if down_too_far:
            block(order, "daily_loss_limit", "down too far today; no new buys"); continue
        if snap["day_start_equity"] is None and not warned_loss:
            warnings.append("daily_loss unchecked: day_start_equity was not supplied.")
            warned_loss = True
        if not snap["earnings_dates"] and not warned_earnings:
            warnings.append("earnings_blackout unchecked: earnings_dates was not supplied.")
            warned_earnings = True
        days_to_earnings = (snap["earnings_dates"][sym] - snap["today"]).days \
            if sym in snap["earnings_dates"] else None
        if days_to_earnings is not None and 0 <= days_to_earnings <= blackout_days:
            block(order, "earnings_blackout",
                  f"earnings in {days_to_earnings} day(s); no new buys within {blackout_days}"); continue
        if cooldown_days is not None and sym in snap["recent_exits"]:
            since_exit = (snap["today"] - snap["recent_exits"][sym]).days
            if 0 <= since_exit < cooldown_days:
                block(order, "cooldown",
                      f"exited {since_exit} day(s) ago; cooldown is {cooldown_days}"); continue
        if buy_count >= max_orders:
            block(order, "max_new_orders", f"already at {max_orders} new buys today"); continue
        existing = positions.get(sym, {}).get("market_value", 0.0) + planned_symbol.get(sym, 0.0)
        if existing + notional > cap_dollars:
            block(order, "position_cap",
                  f"{sym} would exceed the "
                  f"{rules['position']['max_single_position_pct']:.0%} cap"); continue
        sector = (order.get("sector") or positions.get(sym, {}).get("sector")
                  or snap["sectors"].get(sym))
        bucket = sector if isinstance(sector, str) and sector.strip() else "UNKNOWN"
        if bucket == "UNKNOWN" and sym not in warned_sector:
            warnings.append(f"sector unknown for {sym}; counted in an UNKNOWN bucket.")
            warned_sector.add(sym)
        in_sector = sum(p["market_value"] for p in positions.values()
                        if (p["sector"] or "UNKNOWN") == bucket) + planned_sector.get(bucket, 0.0)
        if in_sector + notional > sector_cap:
            block(order, "sector_cap",
                  f"{bucket} would exceed the "
                  f"{rules['position']['max_sector_exposure_pct']:.0%} sector cap"); continue
        if running_spend + notional > spendable:
            block(order, "cash_reserve", "would dip into the cash reserve"); continue

        running_spend += notional
        planned_symbol[sym] = planned_symbol.get(sym, 0.0) + notional
        planned_sector[bucket] = planned_sector.get(bucket, 0.0) + notional
        buy_count += 1
        approved_buys += 1
        approved.append(order)

    return {"approved": approved, "blocked": blocked, "warnings": warnings,
            "new_buys_approved": approved_buys}


def register_risk_tools(mcp) -> None:
    """Attach the M5 tools to an existing MCP server instance (the server file is not edited here)."""
    mcp.tool()(size_position)
    mcp.tool()(check_rules)


if __name__ == "__main__":
    print(size_position("AAPL", 190.0, 3.20, 100_000))