"""G7: Reflection agent. A short post-mortem on each CLOSED trade; the lesson feeds the next run's Judge.

Design rules (architecture sections 7 step 13 and 11; decisions D3/D9):
- Code decides, the LLM only advises. Nothing here changes a decision, a score, a size or an order.
- Opt-in and cheap: no closed trade -> no LLM call at all. At most MAX_REFLECTIONS_PER_RUN per run.
- Compact input only (spec 10.3): the entry score row, the exit decision and the outcome numbers.
- Output is JSON {"lesson": <one line, <= 300 chars>, "assessment": good|bad|neutral}, validated;
  one retry; any failure -> no lesson (an "outcome" line with status "fallback" records the attempt).
- A closed trade is a SELL matched (FIFO) against earlier BUY orders for the same symbol and account,
  read from decisions.jsonl (G6). A trade counts only if it really happened:
    * live (alpaca_paper) orders: only when fill_lookup confirms status "filled" (a live order is just
      "accepted" when submitted, and may never fill or be cancelled). No fill_lookup -> they are skipped.
    * dry-run orders: only when include_dry_run is true (they have no fills; limit prices are used).
- Reflected trades are remembered by "outcome" lines in decisions.jsonl, so a trade is reflected once
  (a failed attempt may be retried once more).
Everything is offline-testable; reflect() never raises."""

import json
import math
import re
import sys
from collections.abc import Mapping
from datetime import datetime, timezone

from . import agent_files, journal
from .agents import ROLE_GUIDANCE, create_agents, materialize_advisory_agent
from .llm import load_llm_config
from .memory import redact_text

MAX_REFLECTIONS_PER_RUN = 5
MAX_ATTEMPTS = 2                       # a trade whose reflection failed twice is not retried again
ASSESSMENTS = ("good", "bad", "neutral")
EXPECTED_REFLECTION = ('JSON only: {"lesson": "<one short line>", '
                       '"assessment": "good" or "bad" or "neutral"}')
GROUNDING = ("Use ONLY the facts in the compact context JSON. Do not cite earnings, news, prices or any "
             "other outside facts. Do not recommend a trade, a size or an order.")
_ORDER_ID = re.compile(r"[0-9A-Za-z-]{8,64}")
_NOT_EXECUTED = ("rejected", "duplicate", "cancelled", "canceled", "not_found", "expired")


# ------------------------------------------------------------------ helpers
def _num(value):
    """A finite positive float, else None."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) and value > 0 else None


def _when(value):
    try:
        moment = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)


def _account(record) -> str:
    return "alpaca_paper" if record.get("live") is True else "offline"


def _exit_key(account, symbol, order_id, timestamp) -> str:
    return f"{account}|{symbol}|{order_id or timestamp}"


def _reflected(decisions):
    """(keys already reflected OK, failed-attempt count per key) from the outcome lines."""
    done, attempts = set(), {}
    for record in decisions:
        if record.get("type") != "outcome" or not record.get("exit_key"):
            continue
        if record.get("status") == "ok":
            done.add(record["exit_key"])
        else:
            attempts[record["exit_key"]] = attempts.get(record["exit_key"], 0) + 1
    return done, attempts


# ------------------------------------------------------------------ fills
def alpaca_fill_lookup(order_id) -> dict:
    """Ask the Alpaca PAPER account whether one order filled. Read-only GET /orders/{id}.

    Returns {"status", "filled_qty", "filled_avg_price"}. Raises on any problem (the caller treats that
    as "unconfirmed"). [VERIFY live: field names follow Alpaca's documented order object.]"""
    from trading_system.servers.tools import live_orders
    if not isinstance(order_id, str) or not _ORDER_ID.fullmatch(order_id):
        raise ValueError("invalid order id")
    response = live_orders._request("GET", f"/orders/{order_id}")
    data = live_orders._json(response)
    if response.status_code != 200 or not isinstance(data, Mapping):
        raise RuntimeError(f"HTTP {response.status_code}")
    return {"status": data.get("status"), "filled_qty": data.get("filled_qty"),
            "filled_avg_price": data.get("filled_avg_price")}


# ------------------------------------------------------------------ closed trades
def find_closed_trades(decisions, include_dry_run=False, fill_lookup=None, warnings=None) -> list:
    """Closed trades not yet reflected, oldest first. Pure apart from fill_lookup calls.

    warnings: optional list that receives human-readable notes (skipped orders, failed lookups)."""
    notes = warnings if warnings is not None else []
    records = [r for r in decisions or [] if isinstance(r, Mapping) and r.get("type") == "order"
               and r.get("symbol") and r.get("side") in ("buy", "sell")]
    done, attempts = _reflected(decisions or [])

    candidates, skipped_live = [], 0
    for rec in records:
        if rec.get("live") is True:
            if rec.get("submitted") is not True or str(rec.get("status")) in _NOT_EXECUTED:
                continue
            if fill_lookup is None:
                skipped_live += 1
                continue
        else:
            if not include_dry_run or rec.get("status") != "accepted_dry_run":
                continue
        candidates.append(rec)
    if skipped_live:
        notes.append(f"{skipped_live} live order(s) skipped: no fill confirmation available "
                     "(set BROKER_MODE=alpaca_paper with Alpaca keys).")

    sells = [r for r in candidates if r["side"] == "sell"]
    if not sells:
        return []

    cache, lookup_failed = {}, False

    def resolve(rec):
        """(qty, price, basis) or None when the order did not (provably) execute."""
        nonlocal lookup_failed
        qty, price = _num(rec.get("qty")), _num(rec.get("limit_price"))
        if rec.get("live") is not True:
            return (qty, price, "dry_run_limit") if qty and price else None
        order_id = rec.get("order_id")
        if order_id not in cache:
            try:
                cache[order_id] = fill_lookup(order_id)
            except Exception as exc:
                lookup_failed = True
                notes.append(f"fill lookup failed for an order ({type(exc).__name__}); live trades "
                             "wait for the next run.")
                cache[order_id] = None
        fill = cache[order_id]
        if not isinstance(fill, Mapping) or str(fill.get("status")) != "filled":
            return None
        try:
            filled_qty, filled_price = _num(float(fill.get("filled_qty"))), _num(float(fill.get("filled_avg_price")))
        except (TypeError, ValueError):
            return None
        return (filled_qty, filled_price, "fill") if filled_qty and filled_price else None

    lots, trades = {}, []
    for rec in candidates:
        got = resolve(rec)
        if got is None:
            continue
        qty, price, basis = got
        account, symbol = _account(rec), str(rec["symbol"]).upper()
        book = lots.setdefault((account, symbol), [])
        if rec["side"] == "buy":
            book.append({"qty": qty, "price": price, "ts": rec.get("timestamp"),
                         "run_id": rec.get("run_id"), "order_id": rec.get("order_id"), "basis": basis})
            continue
        remaining, taken = qty, []
        while remaining > 1e-9 and book:
            lot = book[0]
            part = min(lot["qty"], remaining)
            taken.append((part, lot))
            lot["qty"] -= part
            remaining -= part
            if lot["qty"] <= 1e-9:
                book.pop(0)
        matched = qty - remaining
        if matched <= 1e-9:
            notes.append(f"{symbol}: a sell has no logged buy before it; skipped.")
            continue
        if remaining > 1e-9:
            notes.append(f"{symbol}: only {matched:g} of {qty:g} sold shares have a logged buy.")
        entry_price = sum(part * lot["price"] for part, lot in taken) / matched
        entry_ts = min((lot["ts"] for _, lot in taken if lot["ts"]), default=None)
        key = _exit_key(account, symbol, rec.get("order_id"), rec.get("timestamp"))
        if key in done or attempts.get(key, 0) >= MAX_ATTEMPTS:
            continue
        start, end = _when(entry_ts), _when(rec.get("timestamp"))
        trades.append({
            "symbol": symbol, "account": account, "qty": matched,
            "entry_price": round(entry_price, 4), "exit_price": round(price, 4),
            "pnl": round((price - entry_price) * matched, 2),
            "pnl_pct": round((price / entry_price - 1) * 100, 2),
            "days_held": round((end - start).total_seconds() / 86400, 2) if start and end else None,
            "price_basis": basis, "entry_ts": entry_ts, "exit_ts": rec.get("timestamp"),
            "entry_run_id": taken[0][1]["run_id"], "exit_run_id": rec.get("run_id"),
            "entry_order_ids": [lot["order_id"] for _, lot in taken], "exit_order_id": rec.get("order_id"),
            "exit_key": key,
        })
    if lookup_failed:
        trades = [t for t in trades if t["account"] != "alpaca_paper"]
    return trades


# ------------------------------------------------------------------ prompt and validation
def _last(decisions, run_id, symbol):
    found = None
    for record in decisions or []:
        if (isinstance(record, Mapping) and record.get("type") == "decision"
                and record.get("run_id") == run_id and record.get("symbol") == symbol):
            found = record
    return found


def _scalar(value):
    if isinstance(value, str):
        return value[:160]
    if isinstance(value, float):
        return round(value, 4) if math.isfinite(value) else None
    return value if isinstance(value, (int, bool)) or value is None else None


def build_reflection_context(trade, entry_decision=None, exit_decision=None) -> dict:
    """Compact evidence for one closed trade. Never raw bars, never full reports."""
    entry, leave = entry_decision or {}, exit_decision or {}
    sub = entry.get("sub_scores")
    advisory = entry.get("advisory") if isinstance(entry.get("advisory"), Mapping) else {}
    return {
        "symbol": trade["symbol"],
        "outcome": {key: _scalar(trade.get(key)) for key in (
            "qty", "entry_price", "exit_price", "pnl", "pnl_pct", "days_held", "price_basis")},
        "entry": {
            **{key: _scalar(entry.get(key)) for key in (
                "decision", "triage", "composite", "regime", "news_score", "market_risk_mode",
                "stop_price")},
            "sub_scores": {str(k): _scalar(v) for k, v in sub.items()} if isinstance(sub, Mapping) else {},
            "debate_adjustment": _scalar(advisory.get("adjustment")),
        },
        "exit": {key: _scalar(leave.get(key)) for key in ("decision", "triage", "composite", "reason")},
    }


def format_reflection_prompt(context, recent_lessons=()) -> str:
    lessons = [str(item)[:journal.MAX_LESSON_CHARS] for item in list(recent_lessons)[-3:]]
    return (f"Role: Reflection\n{ROLE_GUIDANCE['reflection']}\n"
            "Return JSON only: {\"lesson\": \"one short line, at most 300 characters\", "
            "\"assessment\": \"good\" or \"bad\" or \"neutral\"}.\n"
            "The lesson says what the entry evidence and the outcome teach about the scoring, "
            "not what to buy or sell next.\n"
            f"Compact context JSON:\n{json.dumps(context, ensure_ascii=True, separators=(',', ':'))}\n"
            f"Recent lessons (do not repeat): {json.dumps(lessons, ensure_ascii=True)}\n{GROUNDING}")


def validate_reflection_output(response) -> dict:
    """Exactly {lesson, assessment}; lesson one non-empty line <= 300 chars; assessment in ASSESSMENTS."""
    if isinstance(response, Mapping):
        value = dict(response)
    else:
        text = response if isinstance(response, str) else getattr(response, "raw", None)
        if not isinstance(text, str):
            raise ValueError("Reflection output must be JSON text or an object.")
        stripped = text.strip()
        if stripped.startswith("```") and stripped.endswith("```"):
            text = stripped[3:-3].strip()
            if text.startswith("json"):
                text = text[4:].strip()
        try:
            value = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError("Reflection output is not valid JSON.") from exc
    if not isinstance(value, dict) or set(value) != {"lesson", "assessment"}:
        raise ValueError("Reflection output must contain exactly lesson and assessment.")
    lesson, assessment = value["lesson"], value["assessment"]
    if (not isinstance(lesson, str) or not lesson.strip() or "\n" in lesson or "\r" in lesson
            or len(lesson.strip()) > journal.MAX_LESSON_CHARS):
        raise ValueError("Reflection lesson must be one short line.")
    if assessment not in ASSESSMENTS:
        raise ValueError("Reflection assessment must be good, bad or neutral.")
    return {"lesson": redact_text(lesson.strip()), "assessment": assessment}


# ------------------------------------------------------------------ the LLM step
def run_reflection(trades, decisions, recent_lessons=(), config=None, client_factory=None,
                   agent_factory=None, crew_runner=None, printer=print) -> list:
    """One Reflection call per closed trade. Returns one outcome dict per trade; never changes anything.

    Raises only for setup problems (unverified config, missing key, no CrewAI); reflect() catches that."""
    from . import advisory   # lazy: advisory pulls in the scoring tools
    loaded = load_llm_config(advisory.ADVISORY_CONFIG_PATH if config is None else config)
    printer()
    printer("-" * 90)
    printer(f"M7 REFLECTION ({len(trades)} closed trade(s); advisory only, never changes a decision)")
    printer("-" * 90)
    if not trades:
        return []
    advisory._disable_tracing_by_default()
    if client_factory is None:
        advisory.install_groq_message_compat()
    definitions = create_agents(config=loaded, client_factory=client_factory, enable_advisory_llm=True)
    agent = materialize_advisory_agent(definitions["reflection"], agent_factory)
    model = advisory._model_of(definitions["reflection"])
    runner = crew_runner or advisory.default_crew_runner

    outcomes = []
    for trade in trades:
        context = build_reflection_context(
            trade, _last(decisions, trade["entry_run_id"], trade["symbol"]),
            _last(decisions, trade["exit_run_id"], trade["symbol"]))
        prompt = format_reflection_prompt(context, recent_lessons)
        result, warnings = None, []
        for attempt in range(2):
            try:
                reply = runner(agent, prompt if attempt == 0 else
                               prompt + "\nRetry once: return only valid JSON.", EXPECTED_REFLECTION)
                result = validate_reflection_output(advisory._text(reply))
                break
            except Exception as exc:
                warnings.append(f"{trade['symbol']} reflection attempt {attempt + 1} failed: "
                                f"{advisory._redact(exc)}")
        outcome = {key: trade.get(key) for key in (
            "symbol", "account", "qty", "entry_price", "exit_price", "pnl", "pnl_pct", "days_held",
            "price_basis", "entry_run_id", "exit_run_id", "exit_key")}
        outcome.update({"status": "ok" if result else "fallback",
                        "lesson": result["lesson"] if result else None,
                        "assessment": result["assessment"] if result else None,
                        "model": model, "warnings": warnings})
        outcomes.append(outcome)
        printer(f"  {trade['symbol']} | pnl {trade['pnl']:+.2f} ({trade['pnl_pct']:+.2f}%) | "
                f"{outcome['assessment'] or 'no lesson'} | {outcome['lesson'] or '(unavailable)'}")
        for warning in warnings:
            printer(f"    ! {warning}")
    return outcomes


# ------------------------------------------------------------------ entry point
def reflect(run_id, timestamp, symbols=None, timeframe=None, include_dry_run=False, fill_lookup=None,
            memory_dir=None, reports_dir=None, max_reflections=MAX_REFLECTIONS_PER_RUN, config=None,
            client_factory=None, agent_factory=None, crew_runner=None, printer=print) -> dict:
    """Reflect on closed trades not yet reflected; persist outcome lines, lessons and the agent file.

    Returns {"status": "no_closed_trades" | "ok" | "degraded" | "unavailable", "closed_trades": n,
    "reflections": [...], "warnings": [...], "agent_file": path or None}. Never raises."""
    warnings = []
    try:
        decisions = journal.read_decisions(memory_dir)
        trades = find_closed_trades(decisions, include_dry_run, fill_lookup, warnings)
        if not trades:
            return {"status": "no_closed_trades", "closed_trades": 0, "reflections": [],
                    "warnings": warnings, "agent_file": None}
        pending = trades[:max(0, int(max_reflections))]
        if len(trades) > len(pending):
            warnings.append(f"{len(trades) - len(pending)} more closed trade(s) wait for the next run.")
        try:
            outcomes = run_reflection(pending, decisions, journal.load_recent_lessons(3, memory_dir),
                                      config, client_factory, agent_factory, crew_runner, printer)
        except Exception as exc:   # setup problem: nothing is written, the trades stay pending
            warnings.append(f"Reflection unavailable ({type(exc).__name__}): {redact_text(str(exc))[:200]}")
            return {"status": "unavailable", "closed_trades": len(trades), "reflections": [],
                    "warnings": warnings, "agent_file": None}
        for outcome in outcomes:
            journal.append_outcome({**outcome, "run_id": run_id, "timestamp": timestamp,
                                    "kind": "closed_trade"}, memory_dir)
            if outcome["status"] == "ok":
                journal.add_lesson(outcome["lesson"], outcome["symbol"], run_id, timestamp, memory_dir)
            warnings.extend(outcome["warnings"])
        agent_file = agent_files.write_agent_file(
            "Reflection", {"symbols": symbols, "timeframe": timeframe, "reflections": outcomes},
            run_id, timestamp, reports_dir)
        degraded = any(o["status"] != "ok" for o in outcomes)
        return {"status": "degraded" if degraded else "ok", "closed_trades": len(trades),
                "reflections": outcomes, "warnings": warnings, "agent_file": agent_file}
    except Exception as exc:
        print(f"M7 reflection warning: {type(exc).__name__}: {exc}", file=sys.stderr)
        warnings.append(f"Reflection failed ({type(exc).__name__}).")
        return {"status": "unavailable", "closed_trades": 0, "reflections": [], "warnings": warnings,
                "agent_file": None}
