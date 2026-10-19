"""Stage B: M5/M6 wired into the orchestrator (opt-in) and run_cycle(config). Fully offline.

The pipeline harness (fake bars/indicators, no network) comes from test_pipeline_offline.py.
Real code under test: guarded_plan, risk (guardian), broker (offline mode), run_cycle, run_strategy.
Nothing is ever sent to Alpaca: placement is dry-run only and the tests prove it."""

import asyncio
import os
import sys
from pathlib import Path

import pytest

REPOSITORY_DIR = Path(__file__).resolve().parents[2]
if str(REPOSITORY_DIR) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_DIR))

from trading_system.crew import guarded_plan, run_cycle as cycle
from trading_system.crew import trading_crew_multiStocks as crew
from trading_system.servers.tools import broker, risk
from trading_system.tests.test_pipeline_offline import (  # noqa: F401  (fixtures + harness)
    OfflineMCP, SYMBOLS, isolated_artifact_dirs, pipeline)

SNAPSHOT = {"equity": 100_000.0, "cash": 60_000.0, "buying_power": 100_000.0,
            "positions": {"TSLA": {"qty": 10, "market_value": 2_000.0, "avg_entry": 190.0,
                                   "sector": "Consumer"}},
            "new_buys_today": 0, "day_start_equity": 100_000.0}


def decision(symbol, action, price=100.0, atr=3.2, held=False):
    return {"symbol": symbol, "decision": action, "latest_close": price, "atr": atr,
            "held": held, "composite": 0.8}


# ------------------------------------------------------------------ pure plan

def test_buy_is_sized_by_risk_then_trimmed_to_the_position_cap():
    out = guarded_plan.build_guarded_plan([decision("AAPL", "BUY")], SNAPSHOT)
    (order,) = out["approved"]
    # risk 1% of 100k = 1000; stop distance 2*3.2 = 6.4 -> 156 shares; cap 12% = 12000 -> 119 shares at 100
    assert order["side"] == "buy" and order["qty"] == 119
    assert order["limit_price"] == 100.2 and order["stop_price"] == 93.6
    assert order["order_type"] == "limit"
    assert out["blocked"] == [] and out["skipped"] == []
    assert out["plan"][0]["planned_quantity"] == 119


def test_sell_exits_the_whole_position_with_a_limit_below_the_last_price():
    out = guarded_plan.build_guarded_plan([decision("TSLA", "SELL", held=True)], SNAPSHOT)
    (order,) = out["approved"]
    assert order["side"] == "sell" and order["qty"] == 10 and order["limit_price"] == 99.8


def test_hold_makes_no_order():
    out = guarded_plan.build_guarded_plan([decision("KO", "HOLD")], SNAPSHOT)
    assert out["proposals"] == [] and out["plan"][0]["planned_quantity"] == 0


def test_missing_atr_or_price_is_skipped_with_a_reason_not_guessed():
    out = guarded_plan.build_guarded_plan(
        [decision("AAPL", "BUY", atr=None), decision("MSFT", "BUY", price=None)], SNAPSHOT)
    assert out["proposals"] == []
    assert {s["symbol"] for s in out["skipped"]} == {"AAPL", "MSFT"}


def test_sell_of_something_the_broker_does_not_hold_is_skipped():
    out = guarded_plan.build_guarded_plan([decision("NVDA", "SELL", held=True)], SNAPSHOT)
    assert out["approved"] == [] and "no whole-share position" in out["skipped"][0]["reason"]


def test_the_guardian_blocks_and_the_plan_row_shows_zero():
    out = guarded_plan.build_guarded_plan(
        [decision("AAPL", "BUY")], SNAPSHOT, earnings_dates={"AAPL": "2026-10-19"}, today="2026-10-18")
    assert out["approved"] == []
    assert out["blocked"][0]["rule"] == "earnings_blackout"
    assert out["plan"][0]["planned_quantity"] == 0 and out["plan"][0]["blocked_by"] == "earnings_blackout"


def test_daily_loss_limit_blocks_new_buys_but_not_sells():
    down = dict(SNAPSHOT, day_start_equity=110_000.0)    # equity 100k is -9% on the day
    out = guarded_plan.build_guarded_plan(
        [decision("AAPL", "BUY"), decision("TSLA", "SELL", held=True)], down)
    assert [o["symbol"] for o in out["approved"]] == ["TSLA"]
    assert out["blocked"][0]["rule"] == "daily_loss_limit"


def test_place_approved_is_dry_run_and_only_for_the_chosen_symbols(monkeypatch):
    monkeypatch.setattr(broker, "_orders", [])
    out = guarded_plan.build_guarded_plan(
        [decision("AAPL", "BUY"), decision("TSLA", "SELL", held=True)], SNAPSHOT)
    placed = guarded_plan.place_approved(out["approved"], only_symbols=["tsla"])
    assert [p["status"] for p in placed] == ["accepted_dry_run"]
    assert [o["details"]["symbol"] for o in broker.get_open_orders()["open_orders"]] == ["TSLA"]
    repeat = guarded_plan.place_approved(out["approved"], only_symbols=["TSLA"])
    assert repeat[0]["status"] == "rejected"          # same order twice is a duplicate


# ------------------------------------------------------------------ orchestrator + run_cycle

@pytest.fixture()
def cycle_env(monkeypatch, pipeline):
    monkeypatch.delenv("BROKER_MODE", raising=False)
    monkeypatch.setattr(broker, "_orders", [])
    mcp = OfflineMCP(pipeline)

    class Client:
        def __init__(self, path):
            pass

        async def __aenter__(self):
            return mcp

        async def __aexit__(self, *exc):
            return False

    monkeypatch.setattr(crew, "DirectMCPClient", Client)
    monkeypatch.setattr(crew, "get_holdings", lambda: ["TSLA"])
    monkeypatch.setattr(broker, "get_portfolio_snapshot", lambda sector_map=None: dict(SNAPSHOT))
    return mcp


def run(config):
    return asyncio.run(cycle.run_cycle(dict({"symbols": SYMBOLS}, **config)))


def test_default_run_strategy_is_unchanged_and_has_no_guarded_plan(cycle_env):
    result = asyncio.run(crew.run_strategy(symbols=SYMBOLS, lookback_days=180, timeframe="1Day"))
    assert "guarded_plan" not in result


def test_decision_rows_now_carry_atr(cycle_env):
    result = run({})
    assert {d["symbol"]: d["atr"] for d in result["decisions"] if d["symbol"] == "AAPL"} == {"AAPL": 3.2}


def test_without_approval_the_cycle_proposes_but_places_nothing(cycle_env):
    result = run({})
    assert result["status"] == "awaiting_approval" and result["dry_run"] is True
    assert {o["symbol"]: o["side"] for o in result["approved"]} == {"AAPL": "buy", "TSLA": "sell"}
    assert result["placed"] == [] and broker.get_open_orders()["open_orders"] == []


def test_approval_of_one_symbol_places_only_that_dry_run_order(cycle_env):
    result = run({"approval": ["AAPL"]})
    assert result["status"] == "ok" and result["approval"] == "approved"
    assert [p["details"]["symbol"] for p in result["placed"]] == ["AAPL"]
    assert all(p["status"] == "accepted_dry_run" for p in result["placed"])


def test_approve_all_places_every_guardian_approved_order(cycle_env):
    result = run({"approval": "all"})
    assert sorted(p["details"]["symbol"] for p in result["placed"]) == ["AAPL", "TSLA"]


def test_m9_report_records_the_risk_results(cycle_env):
    result = run({})
    text = Path(result["report_path"]).read_text(encoding="utf-8")
    assert "## Risk / Guardian context" in text and '"risk_results"' in text
    assert '"limit_price"' in text and "gsk_" not in text


def test_a_broker_failure_fails_closed(cycle_env, monkeypatch):
    def boom(sector_map=None):
        raise broker.BrokerError("Alpaca unreachable")
    monkeypatch.setattr(broker, "get_portfolio_snapshot", boom)
    result = run({"approval": "all"})
    assert result["status"] == "no_orders" and result["placed"] == []
    assert any("Broker snapshot unavailable" in w for w in result["warnings"])


def test_the_real_offline_broker_feeds_the_guardian_without_any_network(monkeypatch):
    import requests
    monkeypatch.delenv("BROKER_MODE", raising=False)
    monkeypatch.setattr(broker, "_orders", [])
    monkeypatch.setattr(requests, "get", lambda *a, **k: pytest.fail("network used"))
    monkeypatch.setattr(requests, "post", lambda *a, **k: pytest.fail("network used"))
    snapshot = broker.get_portfolio_snapshot()
    out = guarded_plan.build_guarded_plan([decision("NVDA", "BUY", price=120.0)], snapshot)
    assert out["approved"] and out["approved"][0]["symbol"] == "NVDA"
    assert guarded_plan.place_approved(out["approved"])[0]["status"] == "accepted_dry_run"


def test_env_overrides_are_temporary_and_only_allowed_names(cycle_env, monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    seen = {}
    real = crew.run_strategy

    async def spy(**kwargs):
        seen["key"] = os.environ.get("GROQ_API_KEY")
        return await real(**kwargs)

    monkeypatch.setattr(crew, "run_strategy", spy)
    result = run({"env": {"GROQ_API_KEY": "gsk_test_value"}})
    assert seen["key"] == "gsk_test_value"
    assert "GROQ_API_KEY" not in os.environ                       # restored
    assert "gsk_test_value" not in repr(result)                   # never returned
    with pytest.raises(cycle.CycleConfigError):
        run({"env": {"PATH": "/tmp"}})


@pytest.mark.parametrize("bad", [{"symbols": []}, {"symbols": "AAPL"}, {"timeframe": "5Min"},
                                 {"lookback_days": 0}, {"approval": 5}, {"debate_band": "x"}])
def test_bad_config_raises_before_anything_runs(cycle_env, bad):
    with pytest.raises(cycle.CycleConfigError):
        asyncio.run(cycle.run_cycle(dict({"symbols": ["AAPL"]}, **bad)))
    assert cycle_env.calls == []


def test_the_guardian_still_reads_the_architecture_rules():
    assert risk.load_rules()["position"]["max_single_position_pct"] == 0.12


def test_the_shipped_sector_map_loads():
    sectors = guarded_plan.load_sector_map()
    assert sectors["AAPL"] == "Technology" and sectors["XOM"] == "Energy"
