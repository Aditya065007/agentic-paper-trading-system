"""M9: one timestamped Markdown file per agent per run (course brief; architecture section 12).

  run_reports/YYYYMMDD/agent<Name>_DDMMYYYY-HH-MM_<TF>.md      TF = 1D or 1H

(HH-MM, not HH:MM: a colon is not allowed in Windows file names. The folder date and the file time are
New York time, the market's clock; the header also gives UTC.)

Every file starts with the same header: run id, timestamp, watch-list, timeframe, LLM provider and model
id, config hash. The agents are the roster of spec section 11: Data, Scoring, News, Regime, Bull, Bear,
Judge, Risk, Reflection. A file is written only for an agent that produced something in the run (the
debate agents only when --advisory ran, Risk only with --guarded, Reflection only when a closed trade was
reflected; crew/reflection.py writes that file after the cycle).

Offline and tolerant: all text goes through memory.sanitize (secrets redacted) and write_agent_files
never raises."""

import hashlib
import json
import sys
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path

from . import memory, reporting
from .memory import redact_text, safe_inline, sanitize

CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"
AGENTS = ("Data", "Scoring", "News", "Regime", "Bull", "Bear", "Judge", "Risk", "Reflection")
TIMEFRAME_CODES = {"1Day": "1D", "1Hour": "1H"}
CONFIG_FILES = ("data.yaml", "llm_verified.yaml", "profiles.yaml", "regime.yaml", "rules.yaml",
                "universe.yaml", "weights.yaml")
MAX_ITEMS, MAX_TEXT = 40, 600


def config_hash(config_dir=None) -> str:
    """Short SHA-256 of the settings files (names and bytes). A changed weight, threshold or model
    changes the hash, so two runs can be compared. Missing files count as empty; never raises."""
    digest = hashlib.sha256()
    folder = Path(config_dir) if config_dir is not None else CONFIG_DIR
    for name in CONFIG_FILES:
        digest.update(name.encode("utf-8") + b"\0")
        try:
            digest.update((folder / name).read_bytes().replace(b"\r\n", b"\n"))
        except OSError:
            digest.update(b"<missing>")
        digest.update(b"\0")
    return digest.hexdigest()[:12]


def _llm_provider() -> str:
    try:
        import yaml
        with open(CONFIG_DIR / "llm_verified.yaml", "r", encoding="utf-8") as stream:
            return str(yaml.safe_load(stream).get("provider") or "unknown")
    except Exception:
        return "unknown"


def _compact(value, depth=0):
    """Bound the size of a nested value: lists capped, long text cut, deep levels summarised."""
    if isinstance(value, Mapping):
        if depth >= 4:
            return f"<{len(value)} fields>"
        return {str(k): _compact(v, depth + 1) for k, v in list(value.items())[:MAX_ITEMS]}
    if isinstance(value, (list, tuple)):
        if depth >= 4:
            return f"<{len(value)} items>"
        items = [_compact(v, depth + 1) for v in list(value)[:MAX_ITEMS]]
        if len(value) > MAX_ITEMS:
            items.append(f"... {len(value) - MAX_ITEMS} more")
        return items
    if isinstance(value, str) and len(value) > MAX_TEXT:
        return value[:MAX_TEXT] + "..."
    return value


def _block(value) -> list:
    clean = sanitize(_compact(value))
    return ["```json", json.dumps(clean, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False),
            "```"]


def _table(headers, rows) -> list:
    def cell(value):
        if value is None:
            return "-"
        if isinstance(value, float):
            return f"{value:.4f}"
        return safe_inline(value).replace("|", "/")
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join(cell(v) for v in row) + " |" for row in rows]
    return lines


def _ny_time(timestamp):
    """(datetime in New York, zone label) from an ISO string; now() if unparseable."""
    try:
        moment = datetime.fromisoformat(str(timestamp).replace("Z", "+00:00"))
    except ValueError:
        moment = datetime.now(timezone.utc)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    try:
        from zoneinfo import ZoneInfo
        return moment.astimezone(ZoneInfo("America/New_York")), "America/New_York"
    except Exception:
        return moment.astimezone(timezone.utc), "UTC"


def file_name(agent, local_time, timeframe) -> str:
    code = TIMEFRAME_CODES.get(timeframe, "NA")
    return f"agent{agent}_{local_time:%d%m%Y-%H-%M}_{code}.md"


# ------------------------------------------------------------------ bodies (None = nothing to write)
def _body_data(result):
    report = result.get("fetch_report")
    if not isinstance(report, Mapping):
        return None
    return ["M1 market data (one batched Alpaca call for the whole watch-list).", ""] + _block(report)


def _body_scoring(result):
    rows = [d for d in result.get("decisions") or [] if isinstance(d, Mapping)]
    if not rows:
        return None
    lines = ["M2 indicators and deterministic scoring. Code only; no LLM.", ""]
    lines += _table(["Symbol", "Decision", "Triage", "Composite", "Regime", "ADX", "ATR", "Close"],
                    [(d.get("symbol"), d.get("decision"), d.get("triage"), d.get("composite"),
                      d.get("regime"), d.get("adx"), d.get("atr"), d.get("latest_close")) for d in rows])
    lines += ["", "## Sub-scores and reasons", ""]
    lines += _block({str(d.get("symbol")): {"sub_scores": d.get("sub_scores"),
                                            "missing_indicators": d.get("missing_indicators"),
                                            "reason": d.get("reason")} for d in rows})
    return lines


def _body_news(result):
    scores = result.get("news_scores")
    warnings = result.get("news_warnings")
    if not scores and not warnings:
        return None
    lines = ["M4 news sentiment (FinBERT over NewsAPI headlines). Code only; the LLM News agent "
             "is used inside the debate.", ""]
    if isinstance(scores, Mapping) and scores:
        lines += _table(["Symbol", "News score"], sorted(scores.items())) + [""]
    if warnings:
        lines += ["## Warnings", ""] + [f"- {safe_inline(w)}" for w in warnings]
    return lines


def _body_regime(result):
    regime = result.get("regime")
    warnings = result.get("regime_warnings")
    if not isinstance(regime, Mapping) and not warnings:
        return None
    lines = ["M3 market regime from SPY and the watch-list. Code only.", ""]
    lines += _block(regime) if isinstance(regime, Mapping) else ["Market regime was not available for this run."]
    if warnings:
        lines += ["", "## Warnings", ""] + [f"- {safe_inline(w)}" for w in warnings]
    return lines


def _debate(result):
    return [o for o in result.get("advisory") or [] if isinstance(o, Mapping)]


def _body_debate_side(result, side):
    outcomes = _debate(result)
    if not outcomes:
        return None
    lines = [f"M7 {side.capitalize()} agent: case {'for' if side == 'bull' else 'against'} each "
             "borderline stock. Advisory only; the Judge's adjustment is capped and rechecked in code.", ""]
    for o in outcomes:
        model = (o.get("models") or {}).get(side)
        lines += [f"## {safe_inline(o.get('symbol'))} (model: {safe_inline(model)})", ""]
        bullets = o.get(side) or []
        lines += [f"- {safe_inline(b)}" for b in bullets] or ["- (unavailable)"]
        lines.append("")
    return lines


def _body_judge(result):
    outcomes = _debate(result)
    if not outcomes:
        return None
    lines = ["M7 Judge agent: weighs Bull and Bear and returns one bounded adjustment.", ""]
    lines += _table(["Symbol", "Composite", "Adjustment", "Adjusted", "Before", "After", "Status", "Model"],
                    [(o.get("symbol"), o.get("composite"), o.get("adjustment"), o.get("adjusted_composite"),
                      o.get("decision_before"), o.get("decision_after"), o.get("status"),
                      (o.get("models") or {}).get("judge")) for o in outcomes])
    lines += ["", "## Reasons and warnings", ""]
    for o in outcomes:
        lines.append(f"- {safe_inline(o.get('symbol'))}: {safe_inline(o.get('reason'))}")
        lines += [f"  - warning: {safe_inline(w)}" for w in o.get("warnings") or []]
    return lines


def _body_risk(result):
    plan = result.get("guarded_plan")
    if not isinstance(plan, Mapping):
        return None
    lines = ["M5 sizing and rules guardian over the M6 broker snapshot. Code only; proposes, "
             "never places.", ""]
    lines += [f"- Broker mode: {safe_inline(plan.get('broker_mode'))}"]
    if plan.get("broker_error"):
        lines.append(f"- Broker error: {safe_inline(plan.get('broker_error'))}")
    orders = [o for o in plan.get("approved") or [] if isinstance(o, Mapping)]
    lines += ["", "## Approved", ""]
    lines += _table(["Symbol", "Side", "Qty", "Limit", "Stop"],
                    [(o.get("symbol"), o.get("side"), o.get("qty"), o.get("limit_price"),
                      o.get("stop_price")) for o in orders]) if orders else ["(none)"]
    blocked = [b for b in plan.get("blocked") or [] if isinstance(b, Mapping)]
    lines += ["", "## Blocked", ""]
    lines += _table(["Symbol", "Rule", "Detail"],
                    [((b.get("order") or {}).get("symbol"), b.get("rule"), b.get("detail"))
                     for b in blocked]) if blocked else ["(none)"]
    skipped = plan.get("skipped") or []
    lines += ["", "## Skipped (no proposal)", ""] + ([f"- {safe_inline(s)}" for s in skipped] or ["(none)"])
    if plan.get("warnings"):
        lines += ["", "## Warnings", ""] + [f"- {safe_inline(w)}" for w in plan["warnings"]]
    return lines


def _body_reflection(result):
    items = [r for r in result.get("reflections") or [] if isinstance(r, Mapping)]
    if not items:
        return None
    lines = ["M7 Reflection agent: a short post-mortem on each closed trade. Advisory only; it never "
             "changes a decision, a size or an order.", ""]
    lines += _table(["Symbol", "Account", "Qty", "Entry", "Exit", "P/L", "P/L %", "Days", "Prices",
                     "Assessment", "Status"],
                    [(r.get("symbol"), r.get("account"), r.get("qty"), r.get("entry_price"),
                      r.get("exit_price"), r.get("pnl"), r.get("pnl_pct"), r.get("days_held"),
                      r.get("price_basis"), r.get("assessment"), r.get("status")) for r in items])
    lines += ["", "## Lessons", ""]
    for r in items:
        lines.append(f"- {safe_inline(r.get('symbol'))}: {safe_inline(r.get('lesson') or '(none)')}")
        lines += [f"  - warning: {safe_inline(w)}" for w in r.get("warnings") or []]
    return lines


_BODIES = {"Data": _body_data, "Scoring": _body_scoring, "News": _body_news, "Regime": _body_regime,
           "Bull": lambda r: _body_debate_side(r, "bull"), "Bear": lambda r: _body_debate_side(r, "bear"),
           "Judge": _body_judge, "Risk": _body_risk, "Reflection": _body_reflection}


def _llm_model(agent, result):
    if agent == "Reflection":
        for item in result.get("reflections") or []:
            if isinstance(item, Mapping) and item.get("model"):
                return str(item["model"])
        return None
    if agent not in ("Bull", "Bear", "Judge"):
        return None
    for outcome in _debate(result):
        model = (outcome.get("models") or {}).get(agent.lower())
        if model:
            return str(model)
    return None


def render_agent_file(agent, result, run_id, timestamp, config_digest=None, provider=None):
    """The Markdown text of one agent file, or None if the agent had nothing to report."""
    body = _BODIES[agent](result)
    if body is None:
        return None
    local, zone = _ny_time(timestamp)
    model = _llm_model(agent, result)
    llm = f"{provider or _llm_provider()} / {model}" if model else "none (deterministic code)"
    symbols = result.get("symbols")
    watch = ", ".join(safe_inline(s) for s in symbols) if isinstance(symbols, (list, tuple)) else safe_inline(symbols)
    header = [f"# Agent {agent}", "",
              f"- Run ID: {safe_inline(run_id)}",
              f"- Timestamp ({zone}): {local:%d-%m-%Y %H:%M}",
              f"- Timestamp (UTC): {safe_inline(timestamp)}",
              f"- Watch-list: {watch}",
              f"- Timeframe: {safe_inline(result.get('timeframe'))}",
              f"- LLM provider / model: {safe_inline(llm)}",
              f"- Config hash: {config_digest or config_hash()}", "", "---", ""]
    return redact_text("\n".join(header + list(body))).rstrip() + "\n"


def _write_one(agent, result, run_id, timestamp, folder, local, digest, provider):
    """Render and write one agent file; returns its path, or None if the agent had nothing to report."""
    text = render_agent_file(agent, result, run_id, timestamp, digest, provider)
    if text is None:
        return None
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / file_name(agent, local, result.get("timeframe"))
    suffix = 1
    while True:
        candidate = target if suffix == 1 else target.with_name(f"{target.stem}_{suffix}{target.suffix}")
        try:
            with candidate.open("x", encoding="utf-8", newline="\n") as stream:
                stream.write(text)
            return str(candidate)
        except FileExistsError:
            suffix += 1


def _folder_and_time(timestamp, reports_dir):
    local, _ = _ny_time(timestamp)
    base = Path(reports_dir) if reports_dir is not None else Path(reporting.REPORTS_DIR)
    return base / f"{local:%Y%m%d}", local


def write_agent_file(agent, result, run_id, timestamp, reports_dir=None):
    """Write the file of ONE agent (used by the Reflection step, which runs after the cycle).
    Returns the path, or None if there was nothing to report or the write failed. Never raises."""
    try:
        result = result if isinstance(result, Mapping) else {}
        folder, local = _folder_and_time(timestamp, reports_dir)
        return _write_one(agent, result, run_id, timestamp, folder, local, config_hash(), _llm_provider())
    except Exception as exc:
        print(f"M9 agent file warning ({agent}): {type(exc).__name__}: {exc}", file=sys.stderr)
        return None


def write_agent_files(result, run_id, timestamp, reports_dir=None) -> list:
    """Write one file per agent that has output. Returns the paths; never raises."""
    paths = []
    try:
        result = result if isinstance(result, Mapping) else {}
        folder, local = _folder_and_time(timestamp, reports_dir)
        digest, provider = config_hash(), _llm_provider()
        for agent in AGENTS:
            try:
                path = _write_one(agent, result, run_id, timestamp, folder, local, digest, provider)
                if path:
                    paths.append(path)
            except Exception as exc:
                print(f"M9 agent file warning ({agent}): {type(exc).__name__}: {exc}", file=sys.stderr)
    except Exception as exc:
        print(f"M9 agent file warning: {type(exc).__name__}: {exc}", file=sys.stderr)
    return paths
