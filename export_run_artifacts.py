#!/usr/bin/env python3
"""
export_run_artifacts.py  -  collect EVERYTHING a run of the paper-trading system leaves behind.

Put this file in the repository root (next to the trading_system folder) and run it AFTER a dashboard run:

    python export_run_artifacts.py                 # latest completed dashboard run
    python export_run_artifacts.py --run-id <id>   # a specific run (id from the dashboard or runs.json)
    python export_run_artifacts.py --all           # every run and every file
    python export_run_artifacts.py --no-cache      # skip the raw price/news cache (smaller zip)
    python export_run_artifacts.py --fresh AAPL MSFT NVDA JPM [--advisory] [--timeframe 1Day]
                                                   # runs ONE new analysis (no orders) and also dumps
                                                   # every indicator value (RSI, MACD, ...), which no
                                                   # file keeps otherwise

Output: exports/export_<time>/  and a zip next to it. Open 00_WHERE_IS_WHAT.md first.
Secrets: .env is never copied; API-key values are replaced by [REDACTED] in every text file written.
"""
import argparse
import asyncio
import csv
import hashlib
import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

TEXT_SUFFIXES = {".md", ".json", ".jsonl", ".yaml", ".yml", ".txt", ".csv", ".log"}
SECRET_NAMES = ("ALPACA_API_KEY", "ALPACA_SECRET_KEY", "GROQ_API_KEY", "NEWSAPI_KEY")
DROP_KEYS = {"key_fingerprint", "secret_env", "raw_exception", "api_key", "secret_key", "password", "token"}
SUBSCORES = ("trend", "momentum", "volume", "mean_reversion", "breakout_rs", "news")


# ----------------------------------------------------------------------------- helpers
def load_env(root):
    try:
        from dotenv import load_dotenv
        load_dotenv(root / ".env", override=False)
    except Exception:
        pass
    return [os.environ[n] for n in SECRET_NAMES if len(os.environ.get(n, "")) >= 6]


class Redactor:
    def __init__(self, secrets):
        self.secrets = secrets

    def text(self, value):
        for secret in self.secrets:
            value = value.replace(secret, "[REDACTED]")
        return value

    def data(self, value):
        if isinstance(value, dict):
            return {str(k): self.data(v) for k, v in value.items() if str(k).lower() not in DROP_KEYS}
        if isinstance(value, (list, tuple)):
            return [self.data(v) for v in value]
        if isinstance(value, str):
            return self.text(value)
        return value


def write_text(path, text, redactor):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(redactor.text(text), encoding="utf-8")


def write_json(path, value, redactor):
    write_text(path, json.dumps(redactor.data(value), indent=2, ensure_ascii=False, default=str), redactor)


def copy_file(src, dst, redactor):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.suffix.lower() in TEXT_SUFFIXES:
        dst.write_text(redactor.text(src.read_text(encoding="utf-8", errors="replace")), encoding="utf-8")
    else:
        shutil.copy2(src, dst)


def write_csv(path, header, rows, redactor):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        for row in rows:
            writer.writerow([redactor.text(v) if isinstance(v, str) else v for v in row])


def cell(value):
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, default=str)
    return value


def read_runs(state_file):
    try:
        value = json.loads(state_file.read_text(encoding="utf-8"))
        return value if isinstance(value, list) else []
    except (OSError, json.JSONDecodeError):
        return []


def pick_record(runs, run_id):
    if run_id:
        for rec in runs:
            if rec.get("id") == run_id or (rec.get("result") or {}).get("run_id") == run_id:
                return rec
        return None
    done = [r for r in runs if r.get("status") == "completed" and r.get("result")]
    return done[-1] if done else None


def folder_size(path):
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) if path.exists() else 0


# ----------------------------------------------------------------------------- readable exports
def normalise(result):
    """Accept the dashboard's stored result OR the full in-memory result of run_strategy."""
    result = result or {}
    guarded = result.get("guarded_plan") if isinstance(result.get("guarded_plan"), dict) else {}
    return {
        "run_id": result.get("run_id"),
        "timeframe": result.get("timeframe"),
        "symbols": result.get("symbols"),
        "decisions": result.get("decisions") or [],
        "regime": result.get("regime"),
        "proposals": result.get("proposals") or guarded.get("proposals") or [],
        "approved": result.get("approved") or guarded.get("approved") or [],
        "blocked": result.get("blocked") or guarded.get("blocked") or [],
        "skipped": result.get("skipped") or guarded.get("skipped") or [],
        "warnings": result.get("warnings") or guarded.get("warnings") or [],
        "placed": result.get("placed") or [],
        "news_scores": result.get("news_scores"),
        "broker_mode": result.get("broker_mode") or guarded.get("broker_mode"),
    }


def export_readable(res, out, redactor):
    made = []

    rows = []
    for d in res["decisions"]:
        if not isinstance(d, dict):
            continue
        sub = d.get("sub_scores") or {}
        debate = d.get("debate") or {}
        rows.append([d.get("symbol"), d.get("decision"), d.get("triage"), d.get("composite"), d.get("adx"),
                     d.get("atr"), d.get("latest_close"), d.get("held"), d.get("regime"),
                     *[sub.get(name) for name in SUBSCORES], cell(d.get("missing_indicators")),
                     d.get("reason"), debate.get("status"), debate.get("adjustment"),
                     debate.get("adjusted_composite"), debate.get("decision_before"), debate.get("decision_after")])
    write_csv(out / "01_decisions_and_scores.csv",
              ["symbol", "decision", "triage", "composite", "adx", "atr", "latest_close", "held", "adx_regime",
               *[f"sub_{n}" for n in SUBSCORES], "missing_indicators", "reason", "debate_status",
               "debate_adjustment", "debate_adjusted_composite", "decision_before_debate", "decision_after_debate"],
              rows, redactor)
    made.append("01_decisions_and_scores.csv")

    write_json(out / "02_market_regime.json", res["regime"] or {}, redactor)
    made.append("02_market_regime.json")

    lines = ["# Bull / Bear / Judge debate (M7)", ""]
    debated = [d for d in res["decisions"] if isinstance(d, dict) and d.get("debate")]
    if not debated:
        lines.append("No stock was debated in this run (advisory was off, or no stock was within 0.10 of a threshold).")
    for d in debated:
        db = d["debate"]
        lines += [f"## {d.get('symbol')}", "",
                  f"- Composite before: {db.get('composite')}   Judge adjustment: {db.get('adjustment')}   "
                  f"Adjusted: {db.get('adjusted_composite')}",
                  f"- Decision: {db.get('decision_before')} -> {db.get('decision_after')} "
                  f"(proposed by thresholds: {db.get('decision_proposed')}, mode: {db.get('debate_mode')}, "
                  f"status: {db.get('status')})"]
        if db.get("held_back"):
            lines.append(f"- Held back: {db.get('held_back')}")
        lines += ["", "**Bull case**", ""] + [f"- {b}" for b in (db.get("bull") or ["(unavailable)"])]
        lines += ["", "**Bear case**", ""] + [f"- {b}" for b in (db.get("bear") or ["(unavailable)"])]
        lines += ["", f"**Judge reason:** {db.get('reason')}", f"**Models:** {cell(db.get('models'))}"]
        if db.get("warnings"):
            lines += ["", "Warnings: " + "; ".join(str(w) for w in db["warnings"])]
        lines.append("")
    write_text(out / "03_debate_cases.md", "\n".join(lines), redactor)
    made.append("03_debate_cases.md")

    order_rows = []
    for o in res["approved"]:
        if isinstance(o, dict):
            order_rows.append(["APPROVED", o.get("symbol"), o.get("side"), o.get("qty"), o.get("limit_price"),
                               o.get("stop_price"), "", cell({k: v for k, v in o.items() if k not in
                                                              ("symbol", "side", "qty", "limit_price", "stop_price")})])
    for b in res["blocked"]:
        if isinstance(b, dict):
            order = b.get("order") or {}
            order_rows.append(["BLOCKED", order.get("symbol"), order.get("side"), order.get("qty"),
                               order.get("limit_price"), order.get("stop_price"), b.get("rule"), b.get("detail")])
    for s in res["skipped"]:
        order_rows.append(["SKIPPED", "", "", "", "", "", "", cell(s)])
    write_csv(out / "04_guardian_orders.csv",
              ["outcome", "symbol", "side", "qty", "limit_price", "stop_price", "rule", "detail_or_extra"],
              order_rows, redactor)
    made.append("04_guardian_orders.csv")

    write_json(out / "05_orders_placed.json", res["placed"], redactor)
    made.append("05_orders_placed.json")
    write_text(out / "06_warnings.txt", "\n".join(f"- {w}" for w in res["warnings"]) or "(none)", redactor)
    made.append("06_warnings.txt")
    if res["news_scores"] is not None:
        write_json(out / "07_news_scores.json", res["news_scores"], redactor)
        made.append("07_news_scores.json")
    return made


def export_indicators(m2_report, out, redactor):
    """Every indicator reading for every stock (only available from a fresh in-memory run)."""
    rows, scores = [], {}
    for item in (m2_report or {}).get("results", []):
        symbol = item.get("symbol")
        ind = item.get("indicators") or {}
        readings = ind.get("indicators") if isinstance(ind.get("indicators"), dict) else {}
        for name, reading in readings.items():
            if isinstance(reading, dict):
                rows.append([symbol, name, cell(reading.get("raw")), reading.get("normalized"), reading.get("available")])
        for name, reading in (ind.get("context") or {}).items():
            if isinstance(reading, dict):
                rows.append([symbol, f"context:{name}", cell(reading.get("raw")), reading.get("normalized"),
                             reading.get("available")])
        scores[symbol] = item.get("score")
    write_csv(out / "08_indicators_full.csv", ["symbol", "indicator", "raw_value", "normalized_-1_to_+1", "available"],
              rows, redactor)
    write_json(out / "09_score_rows_full.json", scores, redactor)
    write_json(out / "10_m2_report_full.json", m2_report or {}, redactor)
    return ["08_indicators_full.csv", "09_score_rows_full.json", "10_m2_report_full.json"]


# ----------------------------------------------------------------------------- file collection
def collect_files(pkg, root, out, run_ids, everything, no_cache, redactor):
    copied = {}
    reports = pkg / "run_reports"
    picked = []
    if reports.exists():
        for md in sorted(reports.rglob("*.md")):
            if everything:
                picked.append(md)
                continue
            try:
                text = md.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            if any(rid and rid in text for rid in run_ids):
                picked.append(md)
    for md in picked:
        copy_file(md, out / "run_reports" / md.relative_to(reports), redactor)
    copied["run_reports/ (agent files and run report: Data, Scoring, News, Regime, Bull, Bear, Judge, Risk)"] = len(picked)

    memory = pkg / "memory"
    count = 0
    if memory.exists():
        for f in sorted(memory.rglob("*")):
            if f.is_file() and f.name != ".gitkeep":
                copy_file(f, out / "memory" / f.relative_to(memory), redactor)
                count += 1
        decisions = memory / "decisions.jsonl"
        if decisions.exists() and not everything:
            mine = [ln for ln in decisions.read_text(encoding="utf-8", errors="replace").splitlines()
                    if any(rid and rid in ln for rid in run_ids)]
            write_text(out / "memory" / "decisions_THIS_RUN.jsonl", "\n".join(mine) + ("\n" if mine else ""), redactor)
    copied["memory/ (decisions.jsonl, trade_journal.md, lessons.md)"] = count

    cfg_count = 0
    for sub, name in (("config", "settings_used"), ("fixtures", "fixtures_used")):
        folder = pkg / sub
        if folder.exists():
            for f in sorted(folder.glob("*")):
                if f.is_file():
                    copy_file(f, out / name / f.name, redactor)
                    cfg_count += 1
    copied["settings_used/ + fixtures_used/ (the exact settings and sample account)"] = cfg_count

    docs = pkg / "docs"
    docs_count = 0
    if docs.exists():
        for f in sorted(docs.glob("*.md")):
            copy_file(f, out / "project_docs" / f.name, redactor)
            docs_count += 1
    copied["project_docs/ (design decisions and experiments log)"] = docs_count

    if not no_cache:
        cache = pkg / "data_cache"
        n = 0
        if cache.exists():
            for f in sorted(cache.rglob("*")):
                if f.is_file():
                    copy_file(f, out / "raw_data_cache" / f.relative_to(cache), redactor)
                    n += 1
        copied["raw_data_cache/ (raw Alpaca price rows and NewsAPI headlines)"] = n
    return copied


def config_hash(root, pkg):
    sys.path.insert(0, str(root))
    try:
        from trading_system.crew.agent_files import config_hash as fn
        return fn()
    except Exception:
        names = ("data.yaml", "llm_verified.yaml", "profiles.yaml", "regime.yaml", "rules.yaml",
                 "universe.yaml", "weights.yaml")
        digest = hashlib.sha256()
        for name in names:
            digest.update(name.encode() + b"\0")
            try:
                digest.update((pkg / "config" / name).read_bytes().replace(b"\r\n", b"\n"))
            except OSError:
                digest.update(b"<missing>")
            digest.update(b"\0")
        return digest.hexdigest()[:12]


# ----------------------------------------------------------------------------- the "where is what" index
QUESTIONS = [
    ("Where are the agent .md files?", "run_reports/YYYYMMDD/agent<Name>_DDMMYYYY-HH-MM_1D.md",
     "run_reports/ in this export. Written by crew/agent_files.py : write_agent_files"),
    ("Where is the scoring?", "01_decisions_and_scores.csv (+ run_reports/.../agentScoring*.md)",
     "M2: servers/tools/scoring.py : score_indicators. Sub-scores = trend, momentum, volume, mean reversion, breakout, news"),
    ("Where are the Bull and Bear cases?", "03_debate_cases.md (+ run_reports/.../agentBull*.md, agentBear*.md)",
     "M7: crew/advisory.py : run_advisory_debate. Only borderline stocks are debated"),
    ("Where is the Judge adjustment?", "03_debate_cases.md, 01_decisions_and_scores.csv, memory/decisions.jsonl",
     "M7: crew/agents.py : validate_judge_output (cap +/-0.15)"),
    ("Where is the market regime?", "02_market_regime.json (+ run_reports/.../agentRegime*.md)",
     "M3: servers/tools/regime.py : compute_market_regime"),
    ("Where are the news scores?", "07_news_scores.json (fresh runs), run_reports/.../agentNews*.md, raw_data_cache/news*",
     "M4: servers/tools/news.py : get_news_by_symbol, score_headlines (FinBERT)"),
    ("Where are the raw prices?", "raw_data_cache/ (raw Alpaca rows)",
     "M1: servers/technical_indicators_mcp_server_multiStocks.py : fetch_ohlcv_batch; cleaned by servers/tools/data.py"),
    ("Where are the indicator values (RSI, MACD ...)?", "08_indicators_full.csv (ONLY from --fresh)",
     "M2: servers/tools/indicators.py : compute_indicators. Not stored by any normal run file"),
    ("Where are the risk checks and order proposals?", "04_guardian_orders.csv (+ run_reports/.../agentRisk*.md)",
     "M5: servers/tools/risk.py : check_rules; crew/guarded_plan.py : build_guarded_plan"),
    ("Where is the placed order?", "05_orders_placed.json, memory/trade_journal.md, memory/decisions.jsonl",
     "M6: servers/tools/live_orders.py (live) or crew/guarded_plan.py : place_approved (dry run)"),
    ("Where is the journal / history?", "memory/trade_journal.md, memory/decisions.jsonl, memory/lessons.md",
     "M9: crew/journal.py"),
    ("Which settings were used?", "settings_used/ and config_hash.txt",
     "config/*.yaml; the same hash appears in every agent file header"),
    ("What did the dashboard store?", "full_run_record.json (dashboard runs only)",
     "web/store.py -> web/.local_state/runs.json"),
]


def write_index(out, record, res, copied, made, chash, mode):
    lines = ["# WHERE IS WHAT: export of a paper-trading run", "",
             f"- Exported: {datetime.now().isoformat(timespec='seconds')} ({mode})",
             f"- Run id: {res.get('run_id')}   Timeframe: {res.get('timeframe')}   Symbols: {res.get('symbols')}",
             f"- Settings hash (same as in every agent file): {chash}",
             f"- Broker mode: {res.get('broker_mode')}",
             f"- Dashboard status: {record.get('status') if record else 'n/a'}   "
             f"Live mode: {record.get('live_mode') if record else 'n/a'}", "",
             "## If the professor asks...", "",
             "| Question | Open this file | Which module and code |", "|---|---|---|"]
    lines += [f"| {q} | {f} | {m} |" for q, f, m in QUESTIONS]
    lines += ["", "## Readable files made from this run", ""] + [f"- {name}" for name in made]
    lines += ["", "## Folders copied", ""] + [f"- {k}: {v} files" for k, v in copied.items()]
    lines += ["", "## Honest notes", "",
              "- Individual indicator values (RSI, MACD, ...) are not saved by a normal run. Only `--fresh` captures them.",
              "- The raw price cache holds the unprocessed Alpaca rows; cleaning happens in servers/tools/data.py.",
              "- API keys are never copied: .env is skipped and key values are replaced by [REDACTED].",
              "- If advisory (the AI debate) was off, there are no Bull/Bear files and 03_debate_cases.md says so."]
    write_text(out / "00_WHERE_IS_WHAT.md", "\n".join(lines) + "\n", Redactor([]))


# ----------------------------------------------------------------------------- main
def run_fresh(root, args):
    sys.path.insert(0, str(root))
    from trading_system.crew import trading_crew_multiStocks as crew
    print("Running ONE new analysis (no orders will be placed)...")
    return asyncio.run(crew.run_strategy(
        symbols=[s.upper() for s in args.fresh], lookback_days=args.lookback, timeframe=args.timeframe,
        advisory_band="spec" if args.advisory else None, guarded=True))


def main():
    ap = argparse.ArgumentParser(description="Export everything a paper-trading run produced.")
    ap.add_argument("--root", default=str(Path(__file__).resolve().parent), help="repository root")
    ap.add_argument("--run-id")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--fresh", nargs="+", metavar="SYMBOL")
    ap.add_argument("--advisory", action="store_true")
    ap.add_argument("--timeframe", default="1Day", choices=["1Day", "1Hour"])
    ap.add_argument("--lookback", type=int, default=180)
    args = ap.parse_args()

    root = Path(args.root).resolve()
    pkg = root / "trading_system"
    if not pkg.exists():
        sys.exit(f"Cannot find {pkg}. Put this file in the repository root or use --root.")
    redactor = Redactor(load_env(root))

    record, mode, extra_made = None, "latest dashboard run", []
    out = root / "exports" / f"export_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out.mkdir(parents=True, exist_ok=True)

    if args.fresh:
        full = run_fresh(root, args)
        mode = "fresh analysis (no orders)"
        res = normalise(full)
        extra_made += export_indicators(full.get("m2_report"), out, redactor)
        write_json(out / "11_fetch_report.json", full.get("fetch_report") or {}, redactor)
        extra_made.append("11_fetch_report.json")
        run_ids = [res["run_id"]]
    else:
        runs = read_runs(pkg / "web" / ".local_state" / "runs.json")
        record = pick_record(runs, args.run_id)
        if record is None and not args.all:
            print("No completed dashboard run found in web/.local_state/runs.json. Run one first, or use --all.")
        result = (record or {}).get("result") or {}
        res = normalise(result)
        run_ids = [res["run_id"], result.get("hourly_run_id"), (record or {}).get("id")]
        if record:
            write_json(out / "full_run_record.json", record, redactor)
            extra_made.append("full_run_record.json")

    made = export_readable(res, out, redactor) + extra_made
    copied = collect_files(pkg, root, out, [r for r in run_ids if r], args.all, args.no_cache, redactor)
    chash = config_hash(root, pkg)
    write_text(out / "config_hash.txt", chash + "\n", redactor)
    write_index(out, record, res, copied, made, chash, mode)

    archive = shutil.make_archive(str(out), "zip", root_dir=str(out))
    print(f"\nExport folder: {out}\nZip file:      {archive}\n")
    print("Open 00_WHERE_IS_WHAT.md first. Contents:")
    for f in sorted(p for p in out.rglob("*") if p.is_file()):
        if f.parent == out:
            print(f"  {f.name}")
    for k, v in copied.items():
        print(f"  {k.split(' ')[0]}  ({v} files)")


if __name__ == "__main__":
    main()
