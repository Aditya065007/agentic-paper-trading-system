"""M9 journal: decisions.jsonl, trade_journal.md and lessons.md (architecture 7.5 and section 12).

Offline and tolerant, like memory.py: every record is sanitized (secrets redacted), every writer
returns an empty result instead of raising, and nothing here makes a decision or places an order.
Files live in memory.MEMORY_DIR (looked up at call time so tests can point it elsewhere):

  decisions.jsonl   append-only, one JSON object per line. type "decision" (one per symbol per run),
                    type "order" (one per order result). Later outcomes and reflection notes (G7)
                    are appended as further lines with type "outcome".
  trade_journal.md  the same story for humans: one section per run, appended.
  lessons.md        reflection notes; load_recent_lessons() feeds the last few into the Judge context.
"""

import json
import sys
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path

from . import memory
from .memory import redact_text, safe_inline, sanitize

DECISIONS_FILE = "decisions.jsonl"
JOURNAL_FILE = "trade_journal.md"
LESSONS_FILE = "lessons.md"
MAX_LESSON_CHARS = 300
_LESSONS_HEADER = ("# Lessons\n\nOne line per lesson: timestamp | run id | symbol | note.\n"
                   "The most recent few are given to the Judge as context.\n\n")


def _directory(memory_dir=None) -> Path:
    return Path(memory_dir) if memory_dir is not None else Path(memory.MEMORY_DIR)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _warn(message: str) -> None:
    print(f"M9 journal warning: {message}", file=sys.stderr)


def _append(path: Path, text: str, header: str = "") -> None:
    """Append text in one write (a header first if the file is new)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists() or path.stat().st_size == 0
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write((header if new else "") + text)


def _cell(value) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4f}"
    return safe_inline(value).replace("|", "/")


def _by_symbol(items, key="symbol") -> dict:
    return {str(item.get(key)).upper(): item for item in items or []
            if isinstance(item, Mapping) and item.get(key)}


# ------------------------------------------------------------------ decisions
def decision_records(result: Mapping, run_id, timestamp) -> list:
    """One record per decision row of a run, joined with news, debate and risk data. Pure function."""
    result = result if isinstance(result, Mapping) else {}
    news_scores = result.get("news_scores") if isinstance(result.get("news_scores"), Mapping) else {}
    plan_rows = _by_symbol(result.get("plan"))
    guarded = result.get("guarded_plan") if isinstance(result.get("guarded_plan"), Mapping) else {}
    proposals = _by_symbol([p for p in guarded.get("proposals") or [] if isinstance(p, Mapping)])
    blocked = {}
    for item in guarded.get("blocked") or []:
        order = item.get("order") if isinstance(item, Mapping) else None
        if isinstance(order, Mapping) and order.get("symbol"):
            blocked.setdefault(str(order["symbol"]).upper(), []).append(item.get("rule"))
    advisory = _by_symbol(result.get("advisory"))
    regime = result.get("regime") if isinstance(result.get("regime"), Mapping) else {}
    records = []
    for row in result.get("decisions") or []:
        if not isinstance(row, Mapping) or not row.get("symbol"):
            continue
        symbol = str(row["symbol"]).upper()
        debate = advisory.get(symbol)
        plan = plan_rows.get(symbol, {})
        order = proposals.get(symbol, {})
        models = (debate or {}).get("models") or {}
        records.append({
            "type": "decision", "timestamp": timestamp, "run_id": run_id,
            "timeframe": result.get("timeframe"), "symbol": symbol,
            "decision": row.get("decision"), "triage": row.get("triage"),
            "composite": row.get("composite"), "sub_scores": row.get("sub_scores"),
            "news_score": news_scores.get(symbol, news_scores.get(row["symbol"])),
            "regime": row.get("regime"), "adx": row.get("adx"), "atr": row.get("atr"),
            "latest_close": row.get("latest_close"), "held": row.get("held"),
            "missing_indicators": row.get("missing_indicators"), "reason": row.get("reason"),
            "market_risk_mode": regime.get("risk_mode"),
            "llm_used": debate is not None,
            "model_name": models.get("judge") if debate else None,
            "advisory": ({key: debate.get(key) for key in (
                "adjustment", "adjusted_composite", "decision_before", "decision_after",
                "status", "reason", "models")} if debate else None),
            "planned_quantity": plan.get("planned_quantity"),
            "stop_price": order.get("stop_price"),
            "rule_flags": blocked.get(symbol) or ([plan["blocked_by"]] if plan.get("blocked_by") else []),
        })
    return sanitize(records)


def _order_records(run_id, timeframe, placed, live: bool, timestamp) -> list:
    records = []
    for item in placed or []:
        if not isinstance(item, Mapping):
            continue
        details = item.get("details") if isinstance(item.get("details"), Mapping) else item
        records.append({
            "type": "order", "timestamp": timestamp, "run_id": run_id, "timeframe": timeframe,
            "symbol": details.get("symbol"), "side": details.get("side"), "qty": details.get("qty"),
            "limit_price": details.get("limit_price"), "stop_price": details.get("stop_price"),
            "order_id": item.get("order_id"), "status": item.get("status"),
            "submitted": item.get("submitted"), "reason": item.get("reason"),
            "live": bool(live), "account": "alpaca_paper" if live else "offline",
        })
    return sanitize(records)


def _jsonl(records) -> str:
    return "".join(json.dumps(record, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
                   for record in records)


def record_run(result: Mapping, run_id, timestamp, memory_dir=None) -> dict:
    """Append this run's decisions to decisions.jsonl and a section to trade_journal.md.

    Returns {"decisions_log": path, "trade_journal": path}; {} if nothing was written. Never raises."""
    try:
        records = decision_records(result, run_id, timestamp)
        if not records:
            return {}
        directory = _directory(memory_dir)
        log_path, journal_path = directory / DECISIONS_FILE, directory / JOURNAL_FILE
        _append(log_path, _jsonl(records))
        lines = [f"## {safe_inline(timestamp)} | run {safe_inline(run_id)} | timeframe "
                 f"{safe_inline(result.get('timeframe'))}", ""]
        regime = result.get("regime") if isinstance(result.get("regime"), Mapping) else None
        if regime:
            lines += [f"Market regime: {_cell(regime.get('risk_mode'))} (SPY trend "
                      f"{_cell(regime.get('spy_trend'))}, volatility {_cell(regime.get('spy_vol'))}, "
                      f"breadth {_cell(regime.get('breadth_pct'))}%)", ""]
        lines += ["| Symbol | Decision | Triage | Composite | News | Regime | LLM | Qty | Rule flags |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for r in records:
            lines.append("| " + " | ".join(_cell(v) for v in (
                r["symbol"], r["decision"], r["triage"], r["composite"], r["news_score"], r["regime"],
                "yes" if r["llm_used"] else "no", r["planned_quantity"],
                ", ".join(r["rule_flags"]) or None)) + " |")
        for r in records:
            if r["advisory"]:
                a = r["advisory"]
                lines += ["", f"- {r['symbol']} debate: adjustment {_cell(a.get('adjustment'))}, "
                              f"{_cell(a.get('decision_before'))} -> {_cell(a.get('decision_after'))}, "
                              f"reason: {_cell(a.get('reason'))}"]
        _append(journal_path, "\n".join(lines) + "\n\n", "# Trade journal\n\n")
        return {"decisions_log": str(log_path), "trade_journal": str(journal_path)}
    except Exception as exc:
        _warn(f"{type(exc).__name__}: {exc}")
        return {}


def append_order_results(run_id, timeframe, placed, live=False, timestamp=None, memory_dir=None) -> dict:
    """Append one "order" line per order result (and a short block in trade_journal.md). Never raises.

    live=True means real paper-account submission; False means the offline/dry-run broker."""
    try:
        stamp = timestamp or _now()
        records = _order_records(run_id, timeframe, placed, live, stamp)
        if not records:
            return {}
        directory = _directory(memory_dir)
        _append(directory / DECISIONS_FILE, _jsonl(records))
        lines = [f"### Orders | {safe_inline(stamp)} | run {safe_inline(run_id)} | "
                 f"{'alpaca_paper (live)' if live else 'offline (dry run)'}", "",
                 "| Symbol | Side | Qty | Limit | Status | Order id | Note |", "|---|---|---|---|---|---|---|"]
        for r in records:
            lines.append("| " + " | ".join(_cell(v) for v in (
                r["symbol"], r["side"], r["qty"], r["limit_price"], r["status"], r["order_id"],
                r["reason"])) + " |")
        _append(directory / JOURNAL_FILE, "\n".join(lines) + "\n\n", "# Trade journal\n\n")
        return {"decisions_log": str(directory / DECISIONS_FILE),
                "trade_journal": str(directory / JOURNAL_FILE)}
    except Exception as exc:
        _warn(f"{type(exc).__name__}: {exc}")
        return {}


def append_outcome(record: Mapping, memory_dir=None) -> bool:
    """Append one "outcome" line (a closed trade and its reflection) and a short block in trade_journal.md.

    Returns False if nothing was written. Never raises."""
    try:
        if not isinstance(record, Mapping):
            return False
        entry = sanitize({**record, "type": "outcome"})
        directory = _directory(memory_dir)
        _append(directory / DECISIONS_FILE, _jsonl([entry]))
        pnl, pct = entry.get("pnl"), entry.get("pnl_pct")
        lines = [f"### Outcome | {safe_inline(entry.get('timestamp') or _now())} | run "
                 f"{safe_inline(entry.get('run_id'))} | {safe_inline(entry.get('symbol'))} "
                 f"({safe_inline(entry.get('account'))})", "",
                 f"- P/L: {_cell(pnl)} ({_cell(pct)}%), qty {_cell(entry.get('qty'))}, "
                 f"entry {_cell(entry.get('entry_price'))}, exit {_cell(entry.get('exit_price'))}, "
                 f"days held {_cell(entry.get('days_held'))}, prices: {_cell(entry.get('price_basis'))}",
                 f"- Reflection ({_cell(entry.get('status'))}, {_cell(entry.get('model'))}): "
                 f"{_cell(entry.get('assessment'))} | {_cell(entry.get('lesson'))}"]
        _append(directory / JOURNAL_FILE, "\n".join(lines) + "\n\n", "# Trade journal\n\n")
        return True
    except Exception as exc:
        _warn(f"{type(exc).__name__}: {exc}")
        return False


def read_decisions(memory_dir=None) -> list:
    """All parseable lines of decisions.jsonl (bad lines are skipped). Never raises."""
    try:
        text = (_directory(memory_dir) / DECISIONS_FILE).read_text(encoding="utf-8")
    except Exception:
        return []
    out = []
    for line in text.splitlines():
        try:
            item = json.loads(line)
        except ValueError:
            continue
        if isinstance(item, dict):
            out.append(item)
    return out


# ------------------------------------------------------------------ lessons
def add_lesson(text, symbol=None, run_id=None, timestamp=None, memory_dir=None) -> bool:
    """Append one lesson line. Returns False (writing nothing) for empty text. Never raises."""
    try:
        note = " ".join(redact_text(str(text or "")).split())[:MAX_LESSON_CHARS]
        if not note:
            return False
        who = " ".join(str(symbol).split()).upper()[:12] if symbol else "-"
        rid = " ".join(redact_text(str(run_id)).split())[:64] if run_id else "-"
        stamp = " ".join(redact_text(str(timestamp or _now())).split())
        _append(_directory(memory_dir) / LESSONS_FILE, f"- {stamp} | {rid} | {who} | {note}\n",
                _LESSONS_HEADER)
        return True
    except Exception as exc:
        _warn(f"{type(exc).__name__}: {exc}")
        return False


def load_recent_lessons(count=3, memory_dir=None) -> list:
    """The note text of the last `count` lessons, oldest first. [] if none, or on any problem."""
    try:
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            return []
        text = (_directory(memory_dir) / LESSONS_FILE).read_text(encoding="utf-8")
        notes = []
        for line in text.splitlines():
            if not line.startswith("- "):
                continue
            parts = line[2:].split(" | ", 3)
            note = parts[3] if len(parts) == 4 else line[2:]
            if note.strip():
                notes.append(redact_text(note.strip())[:MAX_LESSON_CHARS])
        return notes[-count:]
    except Exception:
        return []
