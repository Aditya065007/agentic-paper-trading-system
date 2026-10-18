"""Follow-up fixes for the M5/M6 orchestrator wiring. Fully offline, no pipeline harness needed.

Found by an independent review of the wiring:
  * the sector map reached the guardian only for stocks that were ALREADY held, so a new buy in a
    crowded sector was counted in an UNKNOWN bucket and passed the 30% sector cap;
  * the default broker mode is offline (sample fixtures) and nothing said so;
  * a non-numeric market_value or a garbage symbol could crash or slip into the plan;
  * the order reason said "56 share(s)" while the order had been fitted down to 55."""

import asyncio
import os
from unittest import mock

import pytest

from trading_system.crew import guarded_plan, run_cycle as cycle
from trading_system.crew import trading_crew_multiStocks as crew
from trading_system.servers.tools import broker, risk

SECTORS = {"AAPL": "Technology", "NVDA": "Technology", "XOM": "Energy"}
HELD_TECH = {"equity": 100_000.0, "cash": 80_000.0, "buying_power": 100_000.0,
             "positions": {"AAPL": {"qty": 70, "market_value": 25_000.0, "sector": "Technology"}},
             "new_buys_today": 0, "day_start_equity": 100_000.0}


def buy(symbol, price=214.125, atr=7.5):
    return {"symbol": symbol, "decision": "BUY", "latest_close": price, "atr": atr, "composite": 0.6}


@pytest.fixture()
def step(monkeypatch):
    """Runs the orchestrator's guarded step with the broker snapshot supplied by the test."""
    def run(snapshot, mode=None):
        if mode is None:
            monkeypatch.delenv("BROKER_MODE", raising=False)
        else:
            monkeypatch.setenv("BROKER_MODE", mode)
        monkeypatch.setattr(guarded_plan, "load_sector_map", lambda path=None: dict(SECTORS))
        monkeypatch.setattr(guarded_plan, "load_earnings_dates", lambda path=None: {})
        monkeypatch.setattr(broker, "get_portfolio_snapshot", lambda sector_map=None: dict(snapshot))
        return crew._run_guarded_step([buy("NVDA")])
    return run


# --------------------------------------------------------------- the sector cap
def test_a_new_buy_in_a_crowded_sector_is_blocked_by_the_sector_cap(step):
    plan = step(HELD_TECH)
    assert plan["approved"] == []
    assert [b["rule"] for b in plan["blocked"]] == ["sector_cap"]
    assert "Technology" in plan["blocked"][0]["detail"]
    assert not any("sector unknown" in w for w in plan["warnings"])


def test_a_new_buy_in_an_empty_sector_still_goes_through(step):
    snapshot = dict(HELD_TECH, positions={"XOM": {"qty": 10, "market_value": 1_000.0, "sector": "Energy"}})
    plan = step(snapshot)
    assert [o["symbol"] for o in plan["approved"]] == ["NVDA"]


def test_the_review_scenario_failed_before_the_fix():
    """Documents the old behaviour: without snapshot['sectors'] the same buy is approved."""
    out = guarded_plan.build_guarded_plan([buy("NVDA")], dict(HELD_TECH), today="2026-10-18")
    assert [o["symbol"] for o in out["approved"]] == ["NVDA"]          # the guardian had no sector for NVDA
    assert any("sector unknown for NVDA" in w for w in out["warnings"])  # ...and only warned
    fixed = guarded_plan.build_guarded_plan([buy("NVDA")], dict(HELD_TECH, sectors=SECTORS), today="2026-10-18")
    assert fixed["approved"] == [] and fixed["blocked"][0]["rule"] == "sector_cap"


# ---------------------------------------------------- which account was used
def test_offline_mode_is_announced_loudly_and_recorded(step):
    plan = step(HELD_TECH, mode=None)
    assert plan["broker_mode"] == "offline"
    assert plan["warnings"][0].startswith("OFFLINE FIXTURES")
    assert "not your Alpaca account" in plan["warnings"][0]


def test_paper_mode_has_no_offline_warning(step):
    plan = step(HELD_TECH, mode="alpaca_paper")
    assert plan["broker_mode"] == "alpaca_paper"
    assert not any("OFFLINE" in w for w in plan["warnings"])


def test_a_broker_failure_still_fails_closed_and_names_the_mode(monkeypatch):
    monkeypatch.setenv("BROKER_MODE", "alpaca_paper")
    monkeypatch.setattr(guarded_plan, "load_sector_map", lambda path=None: {})

    def boom(sector_map=None):
        raise broker.BrokerError("Alpaca unreachable")

    monkeypatch.setattr(broker, "get_portfolio_snapshot", boom)
    plan = crew._run_guarded_step([buy("NVDA")])
    assert plan["approved"] == [] and plan["proposals"] == []
    assert plan["broker_error"] == "BrokerError" and plan["broker_mode"] == "alpaca_paper"
    assert any("Broker snapshot unavailable" in w for w in plan["warnings"])


def test_the_console_summary_prints_the_data_source(capsys):
    crew._print_guarded_summary({"approved": [], "blocked": [], "skipped": [], "warnings": [],
                                 "broker_mode": "offline"})
    assert "Account data source: offline" in capsys.readouterr().out


def test_run_cycle_returns_the_mode_and_any_broker_error(monkeypatch):
    async def fake_run_strategy(**kwargs):
        return {"run_id": "r1", "degraded": False, "timeframe": "1Day", "symbols": ["NVDA"], "decisions": [],
                "plan": [], "guarded_plan": {"approved": [], "broker_mode": "offline", "broker_error": "BrokerError"},
                "report_path": None, "journal_path": None}

    monkeypatch.setattr(crew, "run_strategy", fake_run_strategy)
    result = asyncio.run(cycle.run_cycle({"symbols": ["NVDA"]}))
    assert result["broker_mode"] == "offline" and result["broker_error"] == "BrokerError"
    assert result["status"] == "no_orders" and result["placed"] == []


# -------------------------------------------------------------- robustness
def test_non_numeric_market_values_and_garbage_positions_do_not_crash():
    snapshot = dict(HELD_TECH, positions={"AAPL": {"qty": 5, "market_value": "abc", "sector": "Technology"},
                                          "XOM": "garbage"})
    out = guarded_plan.build_guarded_plan([buy("AAPL", 100.0, 3.2), buy("XOM", 100.0, 3.2)], snapshot,
                                          today="2026-10-18")
    assert isinstance(out["approved"], list)          # no exception; the guardian decides what passes


def test_a_non_numeric_equity_proposes_nothing_instead_of_crashing():
    for bad_equity in ("abc", None, float("nan"), True, -5):
        out = guarded_plan.build_guarded_plan([buy("NVDA")], dict(HELD_TECH, equity=bad_equity), today="2026-10-18")
        assert out["approved"] == [], bad_equity


def test_garbage_symbols_never_become_orders():
    decisions = [dict(buy("AAPL"), symbol=None), dict(buy("AAPL"), symbol=True), dict(buy("AAPL"), symbol={}),
                 dict(buy("AAPL"), symbol="   "), dict(buy("AAPL"), symbol="NOT A TICKER"),
                 dict(buy("AAPL"), symbol="[]")]
    out = guarded_plan.build_guarded_plan(decisions, dict(HELD_TECH), today="2026-10-18")
    assert out["proposals"] == [] and out["approved"] == []


def test_valid_class_share_tickers_are_accepted():
    snapshot = dict(HELD_TECH, positions={})
    out = guarded_plan.build_guarded_plan([buy("BRK.B", 400.0, 6.0)], snapshot, today="2026-10-18")
    assert [o["symbol"] for o in out["approved"]] == ["BRK.B"]


def test_the_reason_matches_the_fitted_quantity():
    out = guarded_plan.build_guarded_plan([buy("NVDA")], dict(HELD_TECH, positions={}), today="2026-10-18")
    (order,) = out["approved"]
    assert order["qty"] == 55
    assert "Fitted to 55 share(s)" in order["reason"]


def test_inputs_are_never_mutated():
    import copy
    decisions, snapshot = [buy("NVDA")], copy.deepcopy(HELD_TECH)
    before = copy.deepcopy((decisions, snapshot))
    guarded_plan.build_guarded_plan(decisions, snapshot, today="2026-10-18")
    assert (decisions, snapshot) == before
