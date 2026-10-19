"""G6: per-agent md files, decisions.jsonl, trade_journal.md, lessons.md (offline)."""

import asyncio
import json
import sys
from pathlib import Path

import pytest

from trading_system.crew import advisory, agent_files, journal, memory, reporting
from trading_system.crew import trading_crew_multiStocks as crew

SECRET = "sk-test-SECRET-1234567890"
STAMP = "2026-10-12T14:15:00+00:00"          # 10:15 in New York (EDT)


def full_result(**overrides):
    result = {
        "status": "ok", "symbols": ["AAPL", "MSFT"], "timeframe": "1Day",
        "fetch_report": {"status": "ok", "fetched": [{"symbol": "AAPL", "bars_fetched": 124}], "missing": []},
        "news_scores": {"AAPL": 0.12, "MSFT": -0.2}, "news_warnings": [],
        "regime": {"risk_mode": "neutral", "spy_trend": "down", "spy_vol": 0.1, "breadth_pct": 75.0},
        "regime_warnings": [],
        "decisions": [
            {"symbol": "AAPL", "decision": "BUY", "triage": "clear_buy", "composite": 0.45,
             "regime": "trending", "adx": 30.0, "atr": 3.1, "latest_close": 200.0, "held": False,
             "sub_scores": {"trend": 0.4}, "missing_indicators": [], "reason": "deterministic"},
            {"symbol": "MSFT", "decision": "HOLD", "triage": "borderline", "composite": 0.31,
             "regime": "ranging", "adx": 18.0, "atr": 4.0, "latest_close": 400.0, "held": False,
             "sub_scores": {}, "missing_indicators": [], "reason": "deterministic"}],
        "plan": [{"symbol": "AAPL", "decision": "BUY", "planned_quantity": 20},
                 {"symbol": "MSFT", "decision": "HOLD"}],
        "guarded_plan": {"broker_mode": "offline", "broker_error": None,
                         "proposals": [{"symbol": "AAPL", "side": "buy", "qty": 20, "limit_price": 200.0,
                                        "stop_price": 193.8}],
                         "approved": [{"symbol": "AAPL", "side": "buy", "qty": 20, "limit_price": 200.0,
                                       "stop_price": 193.8}],
                         "blocked": [], "skipped": ["MSFT"], "warnings": []},
        "advisory": [{"symbol": "MSFT", "band_mode": "spec", "composite": 0.31, "adjustment": 0.05,
                      "adjusted_composite": 0.36, "decision_before": "HOLD", "decision_after": "HOLD",
                      "bull": ["momentum is strong"], "bear": ["stretched valuation"],
                      "reason": "bull slightly stronger", "status": "ok",
                      "models": {"bull": "openai/gpt-oss-20b", "bear": "openai/gpt-oss-20b",
                                 "judge": "openai/gpt-oss-120b"}, "warnings": []}],
    }
    result.update(overrides)
    return result


# ----------------------------------------------------------------- agent files
def test_one_file_per_agent_with_the_required_name_and_header(tmp_path):
    paths = agent_files.write_agent_files(full_result(), "run-1", STAMP, tmp_path)
    names = sorted(Path(p).name for p in paths)
    assert names == sorted(f"agent{a}_12102026-10-15_1D.md" for a in
                           ("Data", "Scoring", "News", "Regime", "Bull", "Bear", "Judge", "Risk"))
    assert all(Path(p).parent.name == "20261012" for p in paths)
    assert all(":" not in Path(p).name for p in paths)           # Windows-safe
    text = Path([p for p in paths if "agentJudge" in p][0]).read_text(encoding="utf-8")
    for needle in ("Run ID: run-1", "Watch-list: AAPL, MSFT", "Timeframe: 1Day", "openai/gpt-oss-120b",
                   "Config hash:", "Timestamp (UTC): " + STAMP, "12-10-2026 10:15"):
        assert needle in text


def test_hourly_runs_use_the_1h_code(tmp_path):
    paths = agent_files.write_agent_files(full_result(timeframe="1Hour"), "r", STAMP, tmp_path)
    assert paths and all(Path(p).name.endswith("_1H.md") for p in paths)


def test_an_agent_without_output_gets_no_file(tmp_path):
    result = full_result()
    for key in ("advisory", "guarded_plan", "regime", "news_scores", "news_warnings", "fetch_report",
                "regime_warnings"):
        result.pop(key, None)
    names = sorted(Path(p).name.split("_")[0] for p in agent_files.write_agent_files(result, "r", STAMP, tmp_path))
    assert names == ["agentScoring"]


def test_an_unavailable_regime_is_recorded_not_silently_dropped(tmp_path):
    result = full_result(regime=None, regime_warnings=["Market regime tool unavailable: get_market_regime"])
    (path,) = [p for p in agent_files.write_agent_files(result, "r", STAMP, tmp_path) if "agentRegime" in p]
    text = Path(path).read_text(encoding="utf-8")
    assert "was not available" in text and "tool unavailable" in text


def test_deterministic_agents_say_no_llm_and_debate_agents_name_their_model(tmp_path):
    paths = agent_files.write_agent_files(full_result(), "r", STAMP, tmp_path)
    scoring = Path([p for p in paths if "agentScoring" in p][0]).read_text(encoding="utf-8")
    bull = Path([p for p in paths if "agentBull" in p][0]).read_text(encoding="utf-8")
    assert "none (deterministic code)" in scoring
    assert "openai/gpt-oss-20b" in bull and "momentum is strong" in bull


def test_two_runs_in_the_same_minute_do_not_overwrite(tmp_path):
    first = agent_files.write_agent_files(full_result(), "r1", STAMP, tmp_path)
    second = agent_files.write_agent_files(full_result(), "r2", STAMP, tmp_path)
    assert not set(first) & set(second) and all(Path(p).is_file() for p in first + second)


def test_secrets_never_reach_agent_files(tmp_path, monkeypatch):
    monkeypatch.setenv("NEWSAPI_KEY", SECRET)
    result = full_result(news_warnings=[f"upstream said key={SECRET}"])
    for path in agent_files.write_agent_files(result, "r", STAMP, tmp_path):
        assert SECRET not in Path(path).read_text(encoding="utf-8")


def test_config_hash_changes_when_a_setting_changes(tmp_path):
    (tmp_path / "weights.yaml").write_text("a: 1\n", encoding="utf-8")
    before = agent_files.config_hash(tmp_path)
    assert before == agent_files.config_hash(tmp_path)
    (tmp_path / "weights.yaml").write_text("a: 2\n", encoding="utf-8")
    assert agent_files.config_hash(tmp_path) != before


def test_config_hash_ignores_line_endings(tmp_path):
    (tmp_path / "weights.yaml").write_bytes(b"a: 1\nb: 2\n")
    lf = agent_files.config_hash(tmp_path)
    (tmp_path / "weights.yaml").write_bytes(b"a: 1\r\nb: 2\r\n")
    assert agent_files.config_hash(tmp_path) == lf


def test_agent_file_writer_never_raises(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(agent_files, "render_agent_file",
                        lambda *a, **k: (_ for _ in ()).throw(OSError("disk full")))
    assert agent_files.write_agent_files(full_result(), "r", STAMP, tmp_path) == []
    assert "M9 agent file warning" in capsys.readouterr().err
    assert agent_files.write_agent_files("not a mapping", "r", "garbage", tmp_path) == []


# ----------------------------------------------------------------- decisions.jsonl
def test_decision_records_join_news_debate_and_risk():
    records = {r["symbol"]: r for r in journal.decision_records(full_result(), "run-1", STAMP)}
    aapl, msft = records["AAPL"], records["MSFT"]
    assert aapl["news_score"] == 0.12 and aapl["planned_quantity"] == 20 and aapl["stop_price"] == 193.8
    assert aapl["llm_used"] is False and aapl["model_name"] is None
    assert msft["llm_used"] is True and msft["model_name"] == "openai/gpt-oss-120b"
    assert msft["advisory"]["adjustment"] == 0.05 and msft["market_risk_mode"] == "neutral"
    assert all(r["type"] == "decision" and r["run_id"] == "run-1" for r in records.values())


def test_blocked_orders_become_rule_flags():
    result = full_result()
    result["guarded_plan"]["blocked"] = [{"order": {"symbol": "AAPL"}, "rule": "max_position"}]
    (aapl,) = [r for r in journal.decision_records(result, "r", STAMP) if r["symbol"] == "AAPL"]
    assert aapl["rule_flags"] == ["max_position"]


def test_record_run_appends_jsonl_and_a_journal_section(tmp_path):
    journal.record_run(full_result(), "run-1", STAMP, tmp_path)
    paths = journal.record_run(full_result(), "run-2", STAMP, tmp_path)
    assert set(paths) == {"decisions_log", "trade_journal"}
    lines = (tmp_path / "decisions.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 4 and [json.loads(x)["run_id"] for x in lines] == ["run-1"] * 2 + ["run-2"] * 2
    text = (tmp_path / "trade_journal.md").read_text(encoding="utf-8")
    assert text.startswith("# Trade journal") and text.count("run run-") == 2
    assert "MSFT debate: adjustment 0.0500" in text and "Market regime: neutral" in text


def test_record_run_with_nothing_to_record_writes_nothing(tmp_path):
    assert journal.record_run({"decisions": []}, "r", STAMP, tmp_path) == {}
    assert journal.record_run("junk", "r", STAMP, tmp_path) == {}
    assert not list(tmp_path.iterdir())


def test_secrets_are_redacted_in_the_journal(tmp_path, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", SECRET)
    result = full_result()
    result["decisions"][0]["reason"] = f"leak {SECRET}"
    journal.record_run(result, "r", STAMP, tmp_path)
    for name in ("decisions.jsonl", "trade_journal.md"):
        assert SECRET not in (tmp_path / name).read_text(encoding="utf-8")


def test_order_results_are_logged_dry_run_and_live(tmp_path):
    placed = [{"submitted": True, "order_id": "abc", "status": "accepted",
               "details": {"symbol": "AAPL", "side": "buy", "qty": 20, "limit_price": 200.0}}]
    journal.append_order_results("run-1", "1Day", placed, live=True, timestamp=STAMP, memory_dir=tmp_path)
    journal.append_order_results("run-1", "1Day", placed, live=False, timestamp=STAMP, memory_dir=tmp_path)
    orders = [r for r in journal.read_decisions(tmp_path) if r["type"] == "order"]
    assert [o["account"] for o in orders] == ["alpaca_paper", "offline"]
    assert orders[0]["live"] is True and orders[0]["order_id"] == "abc" and orders[0]["symbol"] == "AAPL"
    assert "alpaca_paper (live)" in (tmp_path / "trade_journal.md").read_text(encoding="utf-8")
    assert journal.append_order_results("r", "1Day", [], memory_dir=tmp_path) == {}


def test_read_decisions_skips_bad_lines(tmp_path):
    (tmp_path / "decisions.jsonl").write_text('{"a": 1}\nnot json\n[1]\n{"b": 2}\n', encoding="utf-8")
    assert journal.read_decisions(tmp_path) == [{"a": 1}, {"b": 2}]
    assert journal.read_decisions(tmp_path / "missing") == []


def test_journal_writers_never_raise(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(journal, "_append", lambda *a, **k: (_ for _ in ()).throw(OSError("read-only")))
    assert journal.record_run(full_result(), "r", STAMP, tmp_path) == {}
    assert journal.append_order_results("r", "1Day", [{"details": {"symbol": "A"}}], memory_dir=tmp_path) == {}
    assert journal.add_lesson("x", memory_dir=tmp_path) is False
    assert capsys.readouterr().err.count("M9 journal warning") == 3


# ----------------------------------------------------------------- lessons.md
def test_lessons_round_trip_oldest_first_and_limited(tmp_path):
    for i in range(5):
        assert journal.add_lesson(f"lesson {i}", symbol="aapl", run_id=f"r{i}", memory_dir=tmp_path)
    assert journal.load_recent_lessons(3, tmp_path) == ["lesson 2", "lesson 3", "lesson 4"]
    assert "AAPL" in (tmp_path / "lessons.md").read_text(encoding="utf-8")


def test_lessons_edge_cases(tmp_path):
    assert journal.add_lesson("   ", memory_dir=tmp_path) is False
    assert journal.load_recent_lessons(3, tmp_path) == []
    journal.add_lesson("x" * 1000 + " | pipe", memory_dir=tmp_path)
    (note,) = journal.load_recent_lessons(3, tmp_path)
    assert len(note) <= journal.MAX_LESSON_CHARS
    for bad in (0, -1, True, "3", None):
        assert journal.load_recent_lessons(bad, tmp_path) == []


def test_lessons_are_redacted_and_single_line(tmp_path, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", SECRET)
    journal.add_lesson(f"line one\nline two {SECRET}", memory_dir=tmp_path)
    text = (tmp_path / "lessons.md").read_text(encoding="utf-8")
    assert SECRET not in text and "line one line two" in text
    assert len([l for l in text.splitlines() if l.startswith("- ")]) == 1


def test_recent_lessons_reach_the_judge_context(tmp_path, monkeypatch):
    journal.add_lesson("do not chase gaps", memory_dir=tmp_path)
    monkeypatch.setattr(memory, "MEMORY_DIR", tmp_path)
    from trading_system.tests.test_advisory import make_runner, row, run_debate
    calls = []
    run_debate([row("AAPL", 0.38, "borderline", "HOLD")], runner=make_runner(calls=calls))
    assert any("do not chase gaps" in prompt for role, prompt in calls)


# ----------------------------------------------------------------- orchestrator integration
def test_a_run_writes_agent_files_and_the_decision_log(monkeypatch, tmp_path):
    from trading_system.tests.test_m9_orchestrator_hooks import install, responses
    install(monkeypatch, responses(calculate_m2_batch={
        "status": "ok", "results": [], "missing": [{"symbol": "AAPL", "error": "no bars"}]}))
    result = asyncio.run(crew.run_strategy(symbols=["AAPL"], lookback_days=180, timeframe="1Day"))
    assert result["status"] == "ok"      # the fail-safe HOLD row still gets a Scoring file and a log line
    names = {Path(p).name.split("_")[0] for p in result["agent_report_paths"]}
    assert {"agentData", "agentScoring"} <= names and all(Path(p).is_file() for p in result["agent_report_paths"])
    assert Path(result["journal_files"]["decisions_log"]).is_file()


def test_a_failing_g6_writer_never_changes_the_run(monkeypatch, capsys):
    from trading_system.tests.test_m9_orchestrator_hooks import install, responses
    install(monkeypatch, responses())
    monkeypatch.setattr(agent_files, "write_agent_files",
                        lambda *a, **k: (_ for _ in ()).throw(OSError("disk full")))
    result = asyncio.run(crew.run_strategy(symbols=["AAPL"], lookback_days=180, timeframe="1Day"))
    assert result["status"] == "ok" and result["agent_report_paths"] == []
    assert "M9 journal warning: OSError: disk full" in capsys.readouterr().err


def test_run_cycle_journals_dry_run_orders(monkeypatch):
    from trading_system.crew import run_cycle as rc
    result = full_result()
    result.update(run_id="run-9", degraded=False)

    async def fake_run_strategy(**kwargs):
        return result

    monkeypatch.setattr(rc.crew, "run_strategy", fake_run_strategy)
    monkeypatch.setattr(rc.gp, "place_approved",
                        lambda orders, only_symbols=None: [{"submitted": True, "order_id": "dry-1",
                                                            "status": "simulated", "details": orders[0]}])
    out = rc.run_cycle_sync({"symbols": ["AAPL"], "approval": "all"})
    assert out["status"] == "ok"
    (order,) = [r for r in journal.read_decisions() if r["type"] == "order"]
    assert order["run_id"] == "run-9" and order["live"] is False and order["symbol"] == "AAPL"
