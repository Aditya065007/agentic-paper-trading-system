# crew/guarded_plan.py
# Stage B: turns the deterministic BUY/SELL/HOLD decisions into guardian-checked order proposals.
#
# decisions (M2/M8, after the optional M7 debate) --size_position / make_limit_price--> proposals
#   --check_rules (M5 guardian)--> approved / blocked
#
# Code decides, the LLM only advises (D3): this module never reads the advisory text, only the final
# decision strings. It places NO order. Placing (dry-run only) is a separate, explicit step:
# see place_approved() and crew/run_cycle.py.

import math
import re
from datetime import datetime, timezone
from pathlib import Path

from trading_system.servers.tools import broker, risk

_PACKAGE_DIR = Path(__file__).resolve().parents[1]


def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _money(value) -> float:
    """A dollar amount from the snapshot; anything that is not a finite number counts as 0."""
    return float(value) if _is_number(value) else 0.0


_SYMBOL_PATTERN = re.compile(r"[A-Z0-9][A-Z0-9.\-]{0,9}")


def load_sector_map(path=None) -> dict:
    """Optional config/universe.yaml `sectors: {AAPL: Technology, ...}`. Missing or bad file -> {}."""
    try:
        import yaml
        with open(Path(path) if path is not None else _PACKAGE_DIR / "config" / "universe.yaml",
                  "r", encoding="utf-8") as stream:
            data = yaml.safe_load(stream) or {}
        sectors = data.get("sectors") if isinstance(data, dict) else None
        if isinstance(sectors, dict):
            return {str(k).strip().upper(): str(v) for k, v in sectors.items() if v}
    except Exception:
        pass
    return {}


_SLEEVES = ("core", "high_beta")


def load_sleeve_map(path=None) -> dict:
    """Optional config/universe.yaml `sleeves: {NVDA: high_beta, ...}`.

    Only "core" and "high_beta" are accepted; anything else is dropped, so a typo leaves
    that ticker on the default "core" sleeve. Missing or bad file -> {}."""
    try:
        import yaml
        with open(Path(path) if path is not None else _PACKAGE_DIR / "config" / "universe.yaml",
                  "r", encoding="utf-8") as stream:
            data = yaml.safe_load(stream) or {}
        sleeves = data.get("sleeves") if isinstance(data, dict) else None
        if isinstance(sleeves, dict):
            return {str(k).strip().upper(): str(v).strip().lower() for k, v in sleeves.items()
                    if str(v).strip().lower() in _SLEEVES}
    except Exception:
        pass
    return {}


def load_earnings_dates(path=None) -> dict:
    """Optional fixtures/earnings.json {"AAPL": "2026-10-29"}. Missing or bad file -> {} (guardian warns)."""
    import json
    try:
        with open(Path(path) if path is not None else _PACKAGE_DIR / "fixtures" / "earnings.json",
                  "r", encoding="utf-8") as stream:
            raw = json.load(stream)
        if isinstance(raw, dict):
            return {str(k).strip().upper(): str(v) for k, v in raw.items() if v}
    except Exception:
        pass
    return {}


def build_guarded_plan(decisions: list, snapshot: dict, rules: dict = None,
                       earnings_dates: dict = None, today: str = None) -> dict:
    """Size every BUY/SELL decision, then run all proposals through the guardian.

    Returns {
      proposals : orders sized by code (before the guardian),
      approved  : orders the guardian passed (each has side, qty, limit_price, stop_price, symbol, reason),
      blocked   : [{order, rule, detail}] from the guardian,
      skipped   : [{symbol, decision, reason}] decisions that could not be turned into an order,
      warnings  : guardian + sizing warnings,
      plan      : one row per decision, in the starter-plan shape the report and M9 already use,
      snapshot_summary : {equity, cash, positions}
    }
    Never raises on bad trading input."""
    rules = risk.load_rules() if rules is None else rules
    positions = snapshot.get("positions") if isinstance(snapshot, dict) else None
    positions = positions if isinstance(positions, dict) else {}
    equity = snapshot.get("equity") if isinstance(snapshot, dict) else None

    proposals, skipped, warnings = [], [], []
    proposal_rows = []
    plan = []
    for row in decisions if isinstance(decisions, list) else []:
        if not isinstance(row, dict):
            continue
        raw_symbol = row.get("symbol")
        symbol = raw_symbol.strip().upper() if isinstance(raw_symbol, str) else ""
        decision = row.get("decision", "HOLD")
        price = row.get("latest_close")
        atr = row.get("atr")
        plan_row = {"symbol": symbol, "decision": decision, "latest_close": price,
                    "planned_value": 0.0, "planned_quantity": 0,
                    "max_position_value": round(float(equity) * rules["position"]["max_single_position_pct"], 2)
                    if _is_number(equity) else None,
                    "cash_reserve": round(float(equity) * rules["position"]["cash_reserve_pct"], 2)
                    if _is_number(equity) else None}
        plan.append(plan_row)
        if decision not in ("BUY", "SELL") or not symbol:
            continue

        def skip(why, _symbol=symbol, _decision=decision):
            skipped.append({"symbol": _symbol, "decision": _decision, "reason": why})

        if not _SYMBOL_PATTERN.fullmatch(symbol):
            skip("the symbol is not a valid ticker.")
            continue
        if not _is_number(price) or price <= 0:
            skip("no valid latest_close, so no limit price.")
            continue

        held = positions.get(symbol)
        held = held if isinstance(held, dict) else {}
        if decision == "BUY":
            if not _is_number(atr) or atr <= 0:
                skip("ATR is missing, so the stop distance and size cannot be computed.")
                continue
            sized = risk.size_position(symbol, float(price), float(atr), _money(equity),
                                       existing_value=_money(held.get("market_value")), rules=rules)
            warnings.extend(f"{symbol}: {w}" for w in sized.get("warnings", []) if w != "bad input")
            if sized["qty"] <= 0:
                skip(sized["reason"])
                continue
            limit = risk.make_limit_price("buy", float(price), rules)
            # The guardian checks the LIMIT price, which is a little above the last price, so make
            # sure the order still fits the position cap at the limit price.
            room = float(equity) * rules["position"]["max_single_position_pct"] \
                - _money(held.get("market_value"))
            fit = int(max(0.0, room) / limit)
            if fit < sized["qty"]:
                warnings.append(f"{symbol}: reduced to {fit} share(s) so the limit price fits the position cap.")
                sized = dict(sized, qty=fit, reason=(
                    f"{sized['reason']} Fitted to {fit} share(s) so the order fits the position "
                    "cap at the limit price."))
                if fit <= 0:
                    skip("no room under the position cap at the limit price.")
                    continue
            order = {"symbol": symbol, "side": "buy", "qty": sized["qty"], "order_type": "limit",
                     "limit_price": limit, "stop_price": sized["stop_price"],
                     "time_in_force": "day", "reason": sized["reason"],
                     "composite": row.get("composite")}
        else:
            qty = held.get("qty")
            if not _is_number(qty) or qty <= 0 or qty != int(qty):
                skip("SELL decision but no whole-share position in the broker snapshot.")
                continue
            limit = risk.make_limit_price("sell", float(price), rules)
            order = {"symbol": symbol, "side": "sell", "qty": int(qty), "order_type": "limit",
                     "limit_price": limit, "stop_price": None, "time_in_force": "day",
                     "reason": "M2 deterministic SELL: exit the full position.",
                     "composite": row.get("composite")}
        if held.get("sector"):
            order["sector"] = held["sector"]
        proposals.append(order)
        proposal_rows.append((order, plan_row))
        plan_row["planned_quantity"] = order["qty"]
        plan_row["planned_value"] = round(order["qty"] * order["limit_price"], 2)

    guard_snapshot = dict(snapshot) if isinstance(snapshot, dict) else {}
    guard_snapshot["today"] = today or datetime.now(timezone.utc).date().isoformat()
    if earnings_dates:
        guard_snapshot["earnings_dates"] = dict(earnings_dates)
    verdict = risk.check_rules(proposals, guard_snapshot, rules)
    warnings.extend(verdict.get("warnings", []))

    row_of = {id(order): plan_row for order, plan_row in proposal_rows}
    for item in verdict.get("blocked", []):  # a blocked order is not a planned order
        plan_row = row_of.get(id(item.get("order")))
        if plan_row is not None:
            plan_row["planned_quantity"], plan_row["planned_value"] = 0, 0.0
            plan_row["blocked_by"] = item["rule"]

    return {"proposals": proposals, "approved": list(verdict.get("approved", [])),
            "blocked": verdict.get("blocked", []), "skipped": skipped, "warnings": warnings,
            "plan": plan,
            "snapshot_summary": {"equity": guard_snapshot.get("equity"), "cash": guard_snapshot.get("cash"),
                                 "positions": sorted(positions)}}


def place_approved(approved_orders: list, only_symbols=None) -> list:
    """Send guardian-approved orders to the broker as DRY-RUN only. Returns one result per order.

    only_symbols: None places nothing-extra (all of `approved_orders`); a list limits placement to those
    symbols (the human approval step). A client_order_id makes a repeat of the same order a duplicate."""
    chosen = {s.strip().upper() for s in only_symbols} if only_symbols is not None else None
    results = []
    for order in approved_orders:
        if chosen is not None and order["symbol"] not in chosen:
            continue
        client_id = f"{order['symbol']}-{order['side']}-{order['qty']}-{order['limit_price']}-" \
                    f"{datetime.now(timezone.utc).date().isoformat()}"
        results.append(broker.place_order(
            order["symbol"], order["side"], order["qty"], order_type="limit",
            limit_price=order["limit_price"], time_in_force=order.get("time_in_force", "day"),
            stop_price=order.get("stop_price"), dry_run=True, client_order_id=client_id))
    return results
