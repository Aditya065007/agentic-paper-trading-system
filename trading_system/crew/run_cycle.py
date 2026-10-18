# crew/run_cycle.py
# B3 - one entry point for a full cycle: run_cycle(config) -> summary dict. A frontend (or the CLI at the
# bottom) calls only this. By default no order leaves the machine: placement is DRY-RUN (M6 broker).
# Live PAPER orders need submit="alpaca_paper" AND an explicit symbol list AND the environment switch
# ALPACA_ORDERS_ENABLED=true (see servers/tools/live_orders.py for every gate).
#
# Flow: fetch -> M2 scores -> decisions -> (optional M7 debate) -> M5 sizing + guardian -> PROPOSALS
#       -> human approval boundary -> dry-run (or live paper) placement of only the approved symbols.
#
# config keys (all optional except symbols):
#   symbols          list[str]            watch-list
#   lookback_days    int                  default: orchestrator default
#   timeframe        "1Day" | "1Hour"     default: orchestrator default
#   advisory         bool                 run the M7 debate (needs GROQ_API_KEY)
#   debate_band      "spec" | "borderline"
#   approval         None | "all" | list[str]   None = stop and wait (default). "all" or a symbol list =
#                    the human said yes; only guardian-approved orders for those symbols are placed.
#   submit           None | "alpaca_paper"   None = dry-run (default). "alpaca_paper" = send the approved
#                    orders to the Alpaca PAPER account; needs approval to be an explicit list of symbols
#                    ("all" is refused) and ALPACA_ORDERS_ENABLED=true in the real environment.
#   env              {NAME: value}        temporary environment overrides for this cycle only
#                    (e.g. keys typed into a UI). Restored afterwards, never logged or returned.

import asyncio
import contextlib
import os
from datetime import datetime, timezone

from trading_system.crew import guarded_plan as gp
from trading_system.crew import trading_crew_multiStocks as crew
from trading_system.servers.tools import live_orders

ALLOWED_ENV = ("ALPACA_API_KEY", "ALPACA_SECRET_KEY", "GROQ_API_KEY", "BROKER_MODE")
VALID_TIMEFRAMES = ("1Day", "1Hour")
VALID_BANDS = ("spec", "borderline")
VALID_SUBMIT = (None, "alpaca_paper")


class CycleConfigError(ValueError):
    """The cycle config is invalid; nothing was run."""


@contextlib.contextmanager
def _temporary_env(overrides):
    saved = {}
    try:
        for name, value in (overrides or {}).items():
            if name not in ALLOWED_ENV:
                raise CycleConfigError(f"env override '{name}' is not allowed.")
            if not isinstance(value, str) or not value.strip():
                raise CycleConfigError(f"env override '{name}' must be a non-empty string.")
            saved[name] = os.environ.get(name)
            os.environ[name] = value.strip()
        yield
    finally:
        for name, old in saved.items():
            if old is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = old


def _validate(config) -> dict:
    if not isinstance(config, dict):
        raise CycleConfigError("config must be a dict.")
    symbols = config.get("symbols")
    if not isinstance(symbols, (list, tuple)) or not symbols or \
            not all(isinstance(s, str) and s.strip() for s in symbols):
        raise CycleConfigError("symbols must be a non-empty list of strings.")
    timeframe = config.get("timeframe", crew.DEFAULT_TIMEFRAME)
    if timeframe not in VALID_TIMEFRAMES:
        raise CycleConfigError(f"timeframe must be one of {VALID_TIMEFRAMES}.")
    lookback = config.get("lookback_days", crew.DEFAULT_LOOKBACK_DAYS)
    if isinstance(lookback, bool) or not isinstance(lookback, int) or lookback <= 0:
        raise CycleConfigError("lookback_days must be a positive integer.")
    band = config.get("debate_band", "spec")
    if band not in VALID_BANDS:
        raise CycleConfigError(f"debate_band must be one of {VALID_BANDS}.")
    approval = config.get("approval")
    if approval is not None and approval != "all" and not (
            isinstance(approval, (list, tuple)) and all(isinstance(s, str) for s in approval)):
        raise CycleConfigError("approval must be None, 'all' or a list of symbols.")
    submit = config.get("submit")
    if submit not in VALID_SUBMIT:
        raise CycleConfigError("submit must be None or 'alpaca_paper'.")
    if submit == "alpaca_paper" and not (
            isinstance(approval, (list, tuple)) and approval and all(s.strip() for s in approval)):
        raise CycleConfigError(
            "live paper orders need approval to be an explicit, non-empty list of symbols "
            "('all' and None are refused).")
    return {"symbols": [s.strip().upper() for s in symbols], "timeframe": timeframe,
            "lookback_days": lookback, "advisory": bool(config.get("advisory")),
            "debate_band": band, "approval": approval, "submit": submit, "env": config.get("env")}


def _journal_live_orders(result: dict, placed: list) -> list:
    """One M9 journal entry per live order (audit trail). Never raises and never contains secrets."""
    paths = []
    try:
        from trading_system.crew.memory import record_memory
    except Exception:
        return paths
    stamp = datetime.now(timezone.utc).isoformat()
    for item in placed:
        details = item.get("details") or {}
        try:
            paths.append(str(record_memory({
                "run_id": result.get("run_id"), "timestamp": stamp,
                "agent": f"M6 Order {details.get('symbol')} {details.get('side')}",
                "status": "ok" if item.get("submitted") else "failed",
                "symbols": [details.get("symbol")], "timeframe": result.get("timeframe"),
                "result": {"live": True, "account": "alpaca_paper", "order_id": item.get("order_id"),
                           "status": item.get("status"), "submitted": item.get("submitted"),
                           "reason": item.get("reason"), "order": details},
                "warnings": [item["reason"]] if item.get("reason") else []})))
        except Exception:
            continue
    return paths


async def run_cycle(config: dict) -> dict:
    """Run one full cycle. Returns a plain, JSON-friendly summary (never contains secrets).

    status: "ok"                    cycle completed, approval given, orders placed (dry-run or live paper)
            "awaiting_approval"     proposals exist and no approval was given; nothing placed
            "submission_blocked"    live paper submission was asked for but a safety gate is closed; nothing sent
            "no_orders"             nothing to propose
    Raises CycleConfigError for a bad config; a failing run raises like run_strategy does."""
    cfg = _validate(config)
    with _temporary_env(cfg["env"]):
        result = await crew.run_strategy(
            symbols=cfg["symbols"], lookback_days=cfg["lookback_days"], timeframe=cfg["timeframe"],
            **({"advisory_band": cfg["debate_band"]} if cfg["advisory"] else {}), guarded=True)

        plan = result.get("guarded_plan") or {}
        approved_orders = plan.get("approved", [])

        placed, approval_state = [], "not_needed"
        submission, submission_error, extra_warnings = "dry_run", None, []
        if approved_orders:
            if cfg["approval"] is None:
                approval_state = "awaiting_approval"
            else:
                only = None if cfg["approval"] == "all" else list(cfg["approval"])
                if cfg["submit"] == "alpaca_paper":
                    try:
                        placed = live_orders.submit_orders(approved_orders, only_symbols=only)
                        submission = "alpaca_paper"
                        if any(p.get("submitted") and p["details"]["side"] == "buy" for p in placed):
                            extra_warnings.append(
                                "No protective stop is placed at the broker: stop_price is advisory only.")
                    except live_orders.OrderSubmissionBlocked as exc:
                        submission, submission_error = "blocked", str(exc)
                else:
                    placed = gp.place_approved(approved_orders, only_symbols=only)
                approval_state = "approved"

    if not approved_orders:
        status = "no_orders"
    elif approval_state == "awaiting_approval":
        status = "awaiting_approval"
    elif submission == "blocked":
        status = "submission_blocked"
    else:
        status = "ok"

    journal_paths = _journal_live_orders(result, placed) if submission == "alpaca_paper" else []
    if placed:
        try:
            from trading_system.crew import journal
            journal.append_order_results(result.get("run_id"), result.get("timeframe"), placed,
                                         live=submission == "alpaca_paper")
        except Exception:
            pass  # the audit trail must never change the cycle

    return {
        "status": status,
        "run_id": result.get("run_id"),
        "degraded": result.get("degraded"),
        "timeframe": result.get("timeframe"),
        "symbols": result.get("symbols"),
        "decisions": result.get("decisions"),
        "regime": result.get("regime"),
        "plan": result.get("plan"),
        "proposals": plan.get("proposals", []),
        "approved": approved_orders,
        "blocked": plan.get("blocked", []),
        "skipped": plan.get("skipped", []),
        "warnings": list(plan.get("warnings", [])) + extra_warnings,
        "approval": approval_state,
        "broker_mode": plan.get("broker_mode"),
        "broker_error": plan.get("broker_error"),
        "placed": placed,
        "submission": submission,
        "submission_error": submission_error,
        "dry_run": submission != "alpaca_paper",
        "order_journal_paths": journal_paths,
        "report_path": result.get("report_path"),
        "journal_path": result.get("journal_path"),
    }


def run_cycle_sync(config: dict) -> dict:
    return asyncio.run(run_cycle(config))


if __name__ == "__main__":
    import json
    import sys
    print(json.dumps(run_cycle_sync({"symbols": sys.argv[1:] or ["AAPL"]}), indent=2, default=str))
