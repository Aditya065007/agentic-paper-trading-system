"""G8: BOTH timeframes (hourly veto), approve/submit CLI flags, gated broker stop, debate consensus,
SPY exclusion and the thin CrewAI crew (all offline)."""

import asyncio
import copy
import io
import json
import types
from contextlib import redirect_stdout
from pathlib import Path

import pytest

from trading_system.crew import advisory, agent_files, both_timeframes, guarded_plan, journal, trading_crew
from trading_system.crew import run_cycle as rc
from trading_system.crew import trading_crew_multiStocks as crew
from trading_system.servers.tools import live_orders
from trading_system.tests.test_advisory import (BEAR, BULL, JUDGE_DOWN, JUDGE_UP, make_runner, row,
                                                stub_agent_factory, stub_client_factory)
from trading_system.tests.test_g6_journal import STAMP, full_result
from trading_system.tests.test_live_orders import KEY, FakeAlpaca, alpaca, buy, env, sell  # noqa: F401


def run(coro):
    return asyncio.run(coro)


# ----------------------------------------------------------------- the veto rule
def test_only_clear_sell_hourly_rows_veto():
    rows = [{"symbol": "aapl", "triage": "clear_sell", "composite": -0.5123},
            {"symbol": "MSFT", "triage": "borderline", "composite": -0.4},
            {"symbol": "NVDA", "triage": "clear_buy", "composite": 0.9},
            {"symbol": "JPM", "triage": "hold", "composite": 0.0},
            {"symbol": "XOM", "triage": "clear_sell", "composite": None}, "junk", {"triage": "clear_sell"}]
    assert both_timeframes.hourly_vetoes(rows) == {"AAPL": "hourly composite -0.5123 is a clear SELL",
                                                  "XOM": "hourly composite n/a is a clear SELL"}
    assert both_timeframes.hourly_vetoes(None) == {}


def test_the_veto_only_turns_a_buy_into_a_hold():
    rows = [{"symbol": "AAPL", "decision": "BUY", "reason": "r"}, {"symbol": "MSFT", "decision": "HOLD", "reason": "r"},
            {"symbol": "NVDA", "decision": "SELL", "reason": "r"}, {"symbol": "JPM", "decision": "BUY", "reason": "r"}]
    veto = {"AAPL": "x", "MSFT": "x", "NVDA": "x"}
    assert crew.apply_hourly_veto(rows, veto) == ["AAPL"]
    assert [r["decision"] for r in rows] == ["HOLD", "HOLD", "SELL", "BUY"]
    assert rows[0]["hourly_veto"] is True and "Hourly veto" in rows[0]["reason"]
    assert "hourly_veto" not in rows[1] and rows[1]["reason"] == "r"


def test_no_veto_changes_nothing():
    rows = [{"symbol": "AAPL", "decision": "BUY", "reason": "r"}]
    before = copy.deepcopy(rows)
    assert crew.apply_hourly_veto(rows, None) == [] and crew.apply_hourly_veto(rows, {}) == [] and rows == before


# ----------------------------------------------------------------- run_both
class FakeStrategy:
    def __init__(self, hourly=None, hourly_error=None, daily_error=None):
        self.calls, self.hourly, self.hourly_error, self.daily_error = [], hourly, hourly_error, daily_error

    async def __call__(self, **kw):
        self.calls.append(kw)
        if kw["timeframe"] == "1Hour":
            if self.hourly_error:
                raise self.hourly_error
            return self.hourly or {"run_id": "run-h", "decisions": [], "degraded": False}
        if self.daily_error:
            raise self.daily_error
        return {"run_id": "run-d", "decisions": [], "timeframe": "1Day"}


def test_hourly_runs_first_decisions_only_then_daily_gets_the_veto():
    fake = FakeStrategy(hourly={"run_id": "run-h", "degraded": False,
                                "decisions": [{"symbol": "AAPL", "triage": "clear_sell", "composite": -0.6}]})
    result = run(both_timeframes.run_both(fake, ["AAPL", "MSFT"], 180, advisory_band="spec"))
    hourly, daily = fake.calls
    assert hourly == {"symbols": ["AAPL", "MSFT"], "lookback_days": 30, "timeframe": "1Hour"}
    assert daily["timeframe"] == "1Day" and daily["lookback_days"] == 180 and daily["guarded"] is True
    assert daily["hourly_veto"] == {"AAPL": "hourly composite -0.6000 is a clear SELL"}
    assert daily["timeframe_mode"] == "BOTH" and daily["advisory_band"] == "spec" and daily["hourly_warnings"] == []
    assert result["run_id"] == "run-d" and result["hourly_run_id"] == "run-h"


def test_the_debate_is_never_requested_for_the_hourly_run():
    fake = FakeStrategy()
    run(both_timeframes.run_both(fake, ["AAPL"], 180, advisory_band="spec"))
    assert "advisory_band" not in fake.calls[0] and "guarded" not in fake.calls[0]
    fake = FakeStrategy()
    run(both_timeframes.run_both(fake, ["AAPL"], 180))
    assert "advisory_band" not in fake.calls[1]


def test_a_failed_hourly_run_does_not_stop_the_daily_run(capsys):
    fake = FakeStrategy(hourly_error=RuntimeError("alpaca down"))
    result = run(both_timeframes.run_both(fake, ["AAPL"], 180))
    daily = fake.calls[1]
    assert daily["hourly_veto"] == {} and any("Hourly run failed" in w for w in daily["hourly_warnings"])
    assert result["run_id"] == "run-d" and result["hourly_run_id"] is None
    assert "BOTH timeframes warning" in capsys.readouterr().err


def test_a_degraded_hourly_run_is_noted():
    fake = FakeStrategy(hourly={"run_id": "h", "decisions": [], "degraded": True})
    run(both_timeframes.run_both(fake, ["AAPL"], 180))
    assert any("degraded" in w for w in fake.calls[1]["hourly_warnings"])


def test_a_failed_daily_run_still_raises():
    with pytest.raises(RuntimeError, match="daily down"):
        run(both_timeframes.run_both(FakeStrategy(daily_error=RuntimeError("daily down")), ["AAPL"], 180))


# ----------------------------------------------------------------- orchestrator plumbing
def test_run_strategy_forwards_the_both_fields_only_when_asked(monkeypatch):
    seen = []

    async def impl(**kw):
        seen.append(kw)
        return {"status": "ok", "decisions": [], "symbols": kw["symbols"], "timeframe": kw["timeframe"]}

    monkeypatch.setattr(crew, "_run_strategy_impl", impl)
    run(crew.run_strategy(symbols=["AAPL"], lookback_days=180, timeframe="1Day"))
    run(crew.run_strategy(symbols=["AAPL"], lookback_days=180, timeframe="1Day", hourly_veto={"AAPL": "x"},
                          timeframe_mode="BOTH", hourly_warnings=["w"]))
    assert set(seen[0]) == {"symbols", "lookback_days", "timeframe"}      # the daily call is exactly as before
    assert seen[1]["hourly_veto"] == {"AAPL": "x"} and seen[1]["timeframe_mode"] == "BOTH"
    assert seen[1]["hourly_warnings"] == ["w"]


def test_hourly_warnings_reach_the_report_and_mark_the_run_degraded():
    result = full_result(hourly_warnings=["Hourly run failed (RuntimeError); the daily run proceeds with no veto."])
    result.update(fetch_report={"missing": []}, m2_report={"missing": []})
    assert crew._is_degraded(result) is True
    payload = crew._build_m9_payload(result, "r", STAMP, True)
    assert any("Hourly run failed" in w for w in payload["warnings"])


def test_the_journal_and_file_names_know_about_both(tmp_path):
    result = full_result(timeframe_mode="BOTH")
    result["decisions"][0]["hourly_veto"] = True
    records = {r["symbol"]: r for r in journal.decision_records(result, "r", STAMP)}
    assert records["AAPL"]["hourly_veto"] is True and records["AAPL"]["timeframe_mode"] == "BOTH"
    assert records["MSFT"]["hourly_veto"] is False
    names = {Path(p).name for p in agent_files.write_agent_files(result, "r", STAMP, tmp_path)}
    assert names and all(n.endswith("_BOTH.md") for n in names)
    plain = {Path(p).name for p in agent_files.write_agent_files(full_result(), "r", STAMP, tmp_path / "x")}
    assert all(n.endswith("_1D.md") for n in plain)


# ----------------------------------------------------------------- run_cycle: BOTH and CLI
def fake_cycle(monkeypatch):
    calls = []
    result = full_result()
    result.update(run_id="run-d", degraded=False)

    async def fake_run_strategy(**kw):
        calls.append(kw)
        if kw["timeframe"] == "1Hour":
            return {"run_id": "run-h", "decisions": [{"symbol": "AAPL", "triage": "clear_sell", "composite": -0.7}]}
        return dict(result, timeframe_mode=kw.get("timeframe_mode"), hourly_vetoed=["AAPL"],
                    hourly_warnings=kw.get("hourly_warnings"))

    monkeypatch.setattr(rc.crew, "run_strategy", fake_run_strategy)
    monkeypatch.setattr(rc.gp, "place_approved", lambda orders, only_symbols=None: [])
    return calls


def test_run_cycle_both_runs_hourly_then_daily_and_reports_the_veto(monkeypatch):
    calls = fake_cycle(monkeypatch)
    out = rc.run_cycle_sync({"symbols": ["AAPL"], "timeframe": "BOTH"})
    assert [c["timeframe"] for c in calls] == ["1Hour", "1Day"]
    assert out["timeframe_mode"] == "BOTH" and out["hourly_vetoed"] == ["AAPL"] and out["hourly_run_id"] == "run-h"
    assert out["timeframe"] == "1Day" and out["run_id"] == "run-d"


def test_other_timeframes_use_a_single_run(monkeypatch):
    calls = fake_cycle(monkeypatch)
    rc.run_cycle_sync({"symbols": ["AAPL"], "timeframe": "1Day"})
    assert [c["timeframe"] for c in calls] == ["1Day"] and "hourly_veto" not in calls[0]


def test_an_unknown_timeframe_is_still_refused():
    with pytest.raises(rc.CycleConfigError):
        rc.run_cycle_sync({"symbols": ["AAPL"], "timeframe": "5Min"})


def test_cli_timeframe_approve_and_submit_flags():
    assert rc._parse_cli(["AAPL", "--timeframe", "BOTH"])["timeframe"] == "BOTH"
    assert "timeframe" not in rc._parse_cli(["AAPL"])
    assert rc._parse_cli(["AAPL", "--approve", "all"])["approval"] == "all"
    assert rc._parse_cli(["AAPL", "MSFT", "--approve", "aapl, msft"])["approval"] == ["AAPL", "MSFT"]
    cfg = rc._parse_cli(["AAPL", "--approve", "AAPL", "--submit"])
    assert cfg["submit"] == "alpaca_paper" and cfg["approval"] == ["AAPL"]
    assert "approval" not in rc._parse_cli(["AAPL"]) and "submit" not in rc._parse_cli(["AAPL"])
    with pytest.raises(SystemExit):
        rc._parse_cli(["AAPL", "--timeframe", "5Min"])


def test_submit_from_the_cli_still_needs_an_explicit_symbol_list():
    for cfg in (rc._parse_cli(["AAPL", "--submit"]), rc._parse_cli(["AAPL", "--approve", "all", "--submit"])):
        with pytest.raises(rc.CycleConfigError):
            rc.run_cycle_sync(cfg)


def test_approve_alone_places_dry_run_orders_only(monkeypatch):
    fake_cycle(monkeypatch)
    placed = []
    monkeypatch.setattr(rc.gp, "place_approved", lambda orders, only_symbols=None: placed.append(only_symbols) or
                        [{"status": "accepted_dry_run", "order_id": "d1", "details": orders[0]}])
    monkeypatch.setattr(rc.live_orders, "submit_orders", lambda *a, **k: pytest.fail("live must not be called"))
    out = rc.run_cycle_sync(rc._parse_cli(["AAPL", "--approve", "AAPL"]))
    assert out["status"] == "ok" and out["dry_run"] is True and placed == [["AAPL"]] and out["submission"] == "dry_run"


# ----------------------------------------------------------------- the gated broker stop
def last_post(alpaca):
    return alpaca.posts[-1]


def test_stops_are_off_by_default_and_the_order_body_is_unchanged(alpaca):
    (result,) = live_orders.submit_orders([buy()])
    body = last_post(alpaca)
    assert "order_class" not in body and "stop_loss" not in body and result["submitted"] is True
    assert "stop_attached" not in result


def test_with_the_switch_on_the_buy_carries_an_oto_stop(env, alpaca):
    env.setenv("ALPACA_STOPS_ENABLED", "true")
    (result,) = live_orders.submit_orders([buy(price=530.05)])
    body = last_post(alpaca)
    assert body["order_class"] == "oto" and body["stop_loss"] == {"stop_price": "505.21"}
    assert (body["type"], body["time_in_force"], body["limit_price"], body["qty"]) == ("limit", "day", "530.05", "22")
    assert result["submitted"] is True and result["stop_attached"] is True and result["stop_price"] == 505.21
    assert len(alpaca.posts) == 1


@pytest.mark.parametrize("value", [None, "", "false", "1", "yes", "on", "truee"])
def test_the_stop_switch_must_be_exactly_true(env, alpaca, value):
    if value is not None:
        env.setenv("ALPACA_STOPS_ENABLED", value)
    live_orders.submit_orders([buy()])
    assert "stop_loss" not in last_post(alpaca)


def test_a_sell_never_gets_a_stop(env, alpaca):
    env.setenv("ALPACA_STOPS_ENABLED", "true")
    alpaca.positions = [{"symbol": "TSLA", "qty": "10", "side": "long", "avg_entry_price": "190",
                         "market_value": "1900", "unrealized_plpc": "0.01"}]
    live_orders.submit_orders([sell()])
    assert alpaca.posts and "order_class" not in last_post(alpaca) and "stop_loss" not in last_post(alpaca)


@pytest.mark.parametrize("stop", [None, 0, -1, 530.05, 600.0, float("nan"), float("inf"), True, "505.21"])
def test_with_stops_on_a_buy_without_a_valid_stop_is_rejected_and_nothing_is_sent(env, alpaca, stop):
    env.setenv("ALPACA_STOPS_ENABLED", "true")
    (result,) = live_orders.submit_orders([buy(price=530.05, stop_price=stop)])
    assert result["submitted"] is False and result["status"] == "rejected" and "stop" in result["reason"]
    assert alpaca.posts == []


def test_the_stop_switch_cannot_open_the_order_gates(env, alpaca):
    env.setenv("ALPACA_STOPS_ENABLED", "true")
    env.setenv("ALPACA_ORDERS_ENABLED", "false")
    with pytest.raises(live_orders.OrderSubmissionBlocked):
        live_orders.submit_orders([buy()])
    assert alpaca.posts == []


def test_the_ui_cannot_set_the_stop_switch():
    assert "ALPACA_STOPS_ENABLED" not in rc.ALLOWED_ENV
    with pytest.raises(rc.CycleConfigError):
        with rc._temporary_env({"ALPACA_STOPS_ENABLED": "true"}):
            pass


def live_cycle(monkeypatch, placed):
    fake_cycle(monkeypatch)
    monkeypatch.setattr(rc.live_orders, "submit_orders", lambda orders, only_symbols=None: placed)
    return rc.run_cycle_sync({"symbols": ["AAPL"], "approval": ["AAPL"], "submit": "alpaca_paper"})


def test_the_no_stop_warning_only_appears_when_a_buy_has_no_stop(monkeypatch):
    base = {"submitted": True, "order_id": "id", "status": "accepted", "details": {"symbol": "AAPL", "side": "buy"}}
    out = live_cycle(monkeypatch, [dict(base)])
    assert any("No protective stop" in w and "ALPACA_STOPS_ENABLED" in w for w in out["warnings"])
    out = live_cycle(monkeypatch, [dict(base, stop_attached=True, stop_price=193.8)])
    assert not any("No protective stop" in w for w in out["warnings"])


def test_the_journal_records_whether_a_stop_was_attached(tmp_path):
    placed = [{"submitted": True, "order_id": "id", "status": "accepted", "stop_attached": True,
               "details": {"symbol": "AAPL", "side": "buy", "qty": 5, "limit_price": 200.0, "stop_price": 193.8}}]
    journal.append_order_results("r", "1Day", placed, live=True, timestamp=STAMP, memory_dir=tmp_path)
    (line,) = journal.read_decisions(tmp_path)
    assert line["stop_attached"] is True and line["stop_price"] == 193.8


# ----------------------------------------------------------------- debate mode: consensus
@pytest.mark.parametrize("mode, before, proposed, adj, bull, bear, status, expected", [
    ("as_built", "HOLD", "BUY", 0.05, [], [], "ok", "BUY"),            # as built: any crossing counts
    ("report", "HOLD", "BUY", 0.05, ["b"], ["x"], "ok", "HOLD"),       # report-only: never changes
    ("consensus", "HOLD", "BUY", 0.05, ["b"], ["x"], "ok", "BUY"),     # agreeing, complete, ok
    ("consensus", "BUY", "HOLD", -0.05, ["b"], ["x"], "ok", "HOLD"),
    ("consensus", "HOLD", "BUY", -0.05, ["b"], ["x"], "ok", "HOLD"),   # wrong way round: held back
    ("consensus", "HOLD", "BUY", 0.05, [], ["x"], "ok", "HOLD"),       # Bull missing
    ("consensus", "HOLD", "BUY", 0.05, ["b"], [], "ok", "HOLD"),       # Bear missing
    ("consensus", "HOLD", "BUY", 0.05, ["b"], ["x"], "fallback", "HOLD"),
    ("consensus", "HOLD", "HOLD", 0.05, [], [], "fallback", "HOLD"),   # no crossing: nothing to hold back
])
def test_apply_debate_mode_matrix(mode, before, proposed, adj, bull, bear, status, expected):
    assert advisory.apply_debate_mode(mode, before, proposed, adj, bull, bear, status)[0] == expected


def test_an_unknown_debate_mode_is_refused():
    with pytest.raises(ValueError):
        advisory.apply_debate_mode("yolo", "HOLD", "BUY", 0.1, ["b"], ["x"], "ok")
    with pytest.raises(ValueError):
        advisory.run_advisory_debate([row("AAPL", 0.36, "borderline", "HOLD")], "1Day", debate_mode="yolo")


def debate(rows, mode=None, runner=None):
    kw = {} if mode is None else {"debate_mode": mode}
    with redirect_stdout(io.StringIO()):
        return advisory.run_advisory_debate(rows, "1Day", "spec", client_factory=stub_client_factory,
                                            agent_factory=stub_agent_factory,
                                            crew_runner=runner or make_runner(), **kw)


def test_consensus_is_the_default_and_a_complete_agreeing_debate_still_flips_the_decision():
    rows = [row("AAPL", 0.36, "borderline", "HOLD")]
    (outcome,) = debate(rows)                      # no mode given -> the default
    assert advisory.DEFAULT_DEBATE_MODE == "consensus" and outcome["debate_mode"] == "consensus"
    assert rows[0]["decision"] == "BUY" and outcome["decision_after"] == "BUY" and outcome["held_back"] == ""


def test_report_mode_logs_the_adjustment_but_the_decision_stands():
    rows = [row("AAPL", 0.36, "borderline", "HOLD")]
    (outcome,) = debate(rows, "report")
    assert rows[0]["decision"] == "HOLD" and outcome["decision_proposed"] == "BUY"
    assert outcome["decision_after"] == "HOLD" and "report-only" in outcome["held_back"]
    assert rows[0]["advisory_adjustment"] == pytest.approx(0.05) and "stands" in rows[0]["reason"]


def test_a_one_sided_debate_still_changes_nothing_in_any_mode():
    for mode in advisory.DEBATE_MODES:
        rows = [row("AAPL", 0.36, "borderline", "HOLD")]
        (outcome,) = debate(rows, mode, make_runner(bear=RuntimeError("down")))
        assert rows[0]["decision"] == "HOLD" and outcome["adjustment"] == 0.0


def test_a_downward_consensus_still_works():
    rows = [row("AAPL", 0.44, "borderline", "BUY")]
    (outcome,) = debate(rows, "consensus", make_runner(judge=JUDGE_DOWN))
    assert rows[0]["decision"] == "HOLD" and outcome["decision_proposed"] == "HOLD"


def test_as_literally_defined_consensus_equals_as_built_when_the_debate_is_complete_and_ok():
    """The direction test is implied by arithmetic (composite + adjustment), so consensus adds no filter
    beyond Judge status and a complete Bull/Bear case. Recorded here so nobody assumes more."""
    for rows_args, runner in (((0.36, "borderline", "HOLD"), make_runner()),
                              ((0.44, "borderline", "BUY"), make_runner(judge=JUDGE_DOWN))):
        a, b = [row("AAPL", *rows_args)], [row("AAPL", *rows_args)]
        debate(a, "consensus", runner)
        debate(b, "as_built", runner)
        assert a[0]["decision"] == b[0]["decision"]


# ----------------------------------------------------------------- SPY is never an order candidate
def test_the_benchmark_is_skipped_for_buys_and_sells():
    rows = [{"symbol": "SPY", "decision": "BUY", "latest_close": 500.0, "atr": 5.0, "held": False},
            {"symbol": "SPY", "decision": "SELL", "latest_close": 500.0, "atr": 5.0, "held": True}]
    snapshot = {"equity": 100000.0, "cash": 100000.0, "positions": {"SPY": {"qty": 5, "market_value": 2500.0}},
                "sectors": {}}
    plan = guarded_plan.build_guarded_plan(rows, snapshot)
    assert plan["proposals"] == [] and plan["approved"] == []
    assert [s["reason"] for s in plan["skipped"]] == ["SPY is the benchmark, not a trading candidate."] * 2


# ----------------------------------------------------------------- the thin CrewAI crew
SNAPSHOT = {"equity": 100000.0, "cash": 100000.0, "positions": {}, "sectors": {"AAPL": "Technology"}}


def decisions():
    return [{"symbol": "AAPL", "decision": "BUY", "triage": "clear_buy", "composite": 0.45, "regime": "trending",
             "held": False, "latest_close": 200.0, "atr": 3.0, "reason": "deterministic"},
            {"symbol": "MSFT", "decision": "HOLD", "triage": "hold", "composite": 0.1, "held": False,
             "latest_close": 400.0, "atr": 4.0, "reason": "deterministic"}]


def test_the_tool_functions_read_the_deterministic_results():
    tools = trading_crew.make_tool_functions(decisions(), SNAPSHOT)
    seen = json.loads(tools["scoring"]("aapl"))
    assert seen["decision"] == "BUY" and seen["composite"] == 0.45 and "atr" not in seen
    verdict = json.loads(tools["risk"]("AAPL"))
    assert verdict["approved"] and verdict["approved"][0]["side"] == "buy" and verdict["blocked"] == []
    assert json.loads(tools["risk"]("MSFT"))["approved"] == []
    assert "error" in json.loads(tools["scoring"]("NOPE")) and "error" in json.loads(tools["risk"]("NOPE"))


def test_one_scoring_and_one_risk_task_per_stock():
    specs = trading_crew.build_stock_tasks(["aapl", "msft"])
    assert [s["symbol"] for s in specs] == ["AAPL", "MSFT"]
    assert "scoring_lookup" in specs[0]["scoring"] and "risk_check" in specs[0]["risk"]
    assert trading_crew.build_stock_tasks(None) == []


class FakeCrew:
    def __init__(self, **kw):
        self.agents, self.tasks = kw["agents"], kw["tasks"]

    def kickoff(self):
        return types.SimpleNamespace(raw="AAPL: BUY, approved 59 shares")


def crew_factories():
    return dict(agent_factory=lambda **kw: types.SimpleNamespace(**kw),
                task_factory=lambda **kw: types.SimpleNamespace(**kw),
                crew_factory=lambda **kw: FakeCrew(**kw),
                tool_factory=lambda name: (lambda fn: types.SimpleNamespace(name=name, fn=fn)))


def test_the_crew_has_a_scoring_and_a_risk_agent_each_with_its_own_tool():
    built = trading_crew.build_crew(decisions(), SNAPSHOT, llm=object(), **crew_factories())
    roles = {a.role: a for a in built.agents}
    assert set(roles) == {"Scoring", "Risk"} and roles["Scoring"].tools[0].name == "scoring_lookup"
    assert roles["Risk"].tools[0].name == "risk_check"
    assert all(a.allow_delegation is False for a in built.agents)
    assert len(built.tasks) == 4 and [t.agent.role for t in built.tasks] == ["Scoring", "Risk"] * 2


def test_the_crew_needs_an_llm_and_never_changes_a_decision():
    with pytest.raises(ValueError):
        trading_crew.build_crew(decisions(), SNAPSHOT, llm=None, **crew_factories())
    rows = decisions()
    before = copy.deepcopy(rows)
    out = trading_crew.run_thin_crew(rows, SNAPSHOT, llm=object(), **crew_factories())
    assert out == {"status": "ok", "report": "AAPL: BUY, approved 59 shares", "tasks": 4} and rows == before


def test_a_crew_that_mutates_a_decision_is_caught():
    class Meddling(FakeCrew):
        def kickoff(self):
            rows[0]["decision"] = "SELL"
            return "x"

    rows = decisions()
    factories = dict(crew_factories(), crew_factory=lambda **kw: Meddling(**kw))
    with pytest.raises(RuntimeError, match="never change a decision"):
        trading_crew.run_thin_crew(rows, SNAPSHOT, llm=object(), **factories)


def test_the_real_crewai_tool_decorator_accepts_the_wrappers():
    from crewai.tools import tool
    fns = trading_crew.make_tool_functions(decisions(), SNAPSHOT)
    assert json.loads(tool("scoring_lookup")(fns["scoring"]).run(symbol="AAPL"))["decision"] == "BUY"
