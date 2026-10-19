"""G3 / M3: market regime (SPY trend, volatility, breadth) and the earnings calendar. Fully offline."""

import asyncio
import json
import math
import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml

REPOSITORY_DIR = Path(__file__).resolve().parents[2]
if str(REPOSITORY_DIR) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_DIR))

from trading_system.crew import guarded_plan  # noqa: E402
from trading_system.crew import trading_crew_multiStocks as crew  # noqa: E402
from trading_system.servers import technical_indicators_mcp_server_multiStocks as server  # noqa: E402
from trading_system.servers.tools import regime  # noqa: E402

YAML_PATH = Path(__file__).resolve().parents[1] / "config" / "regime.yaml"


def rising(n=80, step=0.002, start=100.0):
    return [start * (1 + step) ** i for i in range(n)]


def falling(n=80, step=0.002, start=100.0):
    return [start * (1 - step) ** i for i in range(n)]


def noisy_falling(n=80, start=100.0):
    """Falls, with +/-3% swings so annualised volatility is far above 30%."""
    out, price = [], start
    for i in range(n):
        price *= 0.985 if i % 2 == 0 else 1.0
        price *= 1.03 if i % 3 == 0 else 0.99
        out.append(price)
    return out


# ------------------------------------------------------------------ compute_market_regime
def test_calm_uptrend_with_good_breadth_is_risk_on():
    result = regime.compute_market_regime(rising(), {"AAPL": rising(40), "MSFT": rising(40)})
    assert result["spy_trend"] == "up"
    assert result["breadth_pct"] == 100.0
    assert result["spy_vol"] < 0.30
    assert result["risk_mode"] == "risk_on"
    assert result["warnings"] == []


def test_downtrend_with_stressed_volatility_is_risk_off():
    spy = noisy_falling()
    result = regime.compute_market_regime(spy, {"AAPL": falling(40)})
    assert result["spy_trend"] == "down"
    assert result["spy_vol"] >= 0.30
    assert result["risk_mode"] == "risk_off"


def test_downtrend_calm_but_weak_breadth_is_risk_off_and_good_breadth_is_neutral():
    weak = regime.compute_market_regime(falling(), {"A": falling(40), "B": falling(40), "C": falling(40)})
    assert weak["spy_trend"] == "down" and weak["spy_vol"] < 0.30
    assert weak["breadth_pct"] == 0.0 and weak["risk_mode"] == "risk_off"
    strong = regime.compute_market_regime(falling(), {"A": rising(40), "B": rising(40)})
    assert strong["breadth_pct"] == 100.0 and strong["risk_mode"] == "neutral"


def test_uptrend_with_weak_breadth_is_neutral_not_risk_on():
    result = regime.compute_market_regime(rising(), {"A": falling(40), "B": falling(40)})
    assert result["spy_trend"] == "up" and result["risk_mode"] == "neutral"


def test_breadth_counts_the_share_above_their_ema():
    watch = {"A": rising(40), "B": rising(40), "C": falling(40), "D": falling(40)}
    assert regime.compute_market_regime(rising(), watch)["breadth_pct"] == 50.0


@pytest.mark.parametrize("spy", [None, [], "abc", [100.0] * 10, [100.0] * 80 + [float("nan")],
                                 [100.0] * 80 + [-5.0], [100.0] * 80 + [True], [0.0] * 80])
def test_missing_or_bad_spy_is_neutral_with_a_warning_never_risk_on(spy):
    result = regime.compute_market_regime(spy, {"AAPL": rising(40)})
    assert result["risk_mode"] == "neutral"
    assert result["warnings"]


def test_missing_breadth_keeps_risk_on_out_of_reach():
    result = regime.compute_market_regime(rising(), None)
    assert result["breadth_pct"] is None and result["risk_mode"] == "neutral"
    assert any("Breadth unavailable" in w for w in result["warnings"])
    skipped = regime.compute_market_regime(rising(), {"AAPL": [1.0, 2.0], "BAD": "x"})
    assert skipped["breadth_pct"] is None
    assert any("Breadth skipped AAPL" in w for w in skipped["warnings"])


def test_hourly_volatility_is_annualised_with_more_bars_per_year():
    series = noisy_falling()
    daily = regime.compute_market_regime(series, {}, timeframe="1Day")["spy_vol"]
    hourly = regime.compute_market_regime(series, {}, timeframe="1Hour")["spy_vol"]
    assert hourly == pytest.approx(daily * math.sqrt(7), rel=1e-3)
    bad = regime.compute_market_regime(series, {}, timeframe="5Min")
    assert bad["risk_mode"] == "neutral" and bad["warnings"]


def test_inputs_are_not_mutated():
    spy, watch = rising(), {"A": rising(40)}
    before = (list(spy), {k: list(v) for k, v in watch.items()})
    regime.compute_market_regime(spy, watch)
    assert (spy, watch) == before


def test_thresholds_come_from_settings():
    stricter = dict(regime.DEFAULT_SETTINGS, breadth_risk_on_pct=101.0)
    result = regime.compute_market_regime(rising(), {"A": rising(40)}, stricter)
    assert result["risk_mode"] == "neutral"


# ------------------------------------------------------------------ settings file
def test_shipped_regime_yaml_loads_cleanly_and_equals_builtin_defaults():
    settings, warning = regime.load_regime_settings(YAML_PATH)
    assert warning is None
    assert settings == regime.DEFAULT_SETTINGS
    assert settings["block_new_buys_in_risk_off"] is False  # report-only until the team decides


def write(tmp_path, data):
    path = tmp_path / "r.yaml"
    path.write_text(data if isinstance(data, str) else yaml.safe_dump(data))
    return path


def test_bad_settings_fall_back_to_defaults_with_a_warning(tmp_path):
    good = yaml.safe_load(YAML_PATH.read_text())
    for patch in ({"spy_ema_period": 0}, {"spy_ema_period": 12.5}, {"risk_off_vol": -1},
                  {"risk_off_vol": "high"}, {"vol_window": True},
                  {"block_new_buys_in_risk_off": "yes"},
                  {"breadth_risk_off_pct": 60.0, "breadth_risk_on_pct": 50.0}):
        settings, warning = regime.load_regime_settings(write(tmp_path, dict(good, **patch)))
        assert settings == regime.DEFAULT_SETTINGS and warning and "built-in default" in warning
    for text in ("", "- a list", "not: [valid"):
        settings, warning = regime.load_regime_settings(write(tmp_path, text))
        assert settings == regime.DEFAULT_SETTINGS and warning
    assert regime.load_regime_settings(tmp_path / "absent.yaml")[1]


def test_a_valid_edit_is_used(tmp_path):
    good = yaml.safe_load(YAML_PATH.read_text())
    settings, warning = regime.load_regime_settings(write(tmp_path, dict(good, spy_ema_period=20,
                                                                         block_new_buys_in_risk_off=True)))
    assert warning is None
    assert settings["spy_ema_period"] == 20 and settings["block_new_buys_in_risk_off"] is True


# ------------------------------------------------------------------ earnings calendar
def test_earnings_calendar_returns_future_dates_only(tmp_path):
    path = tmp_path / "earnings.json"
    path.write_text(json.dumps({"aapl": "2026-10-29", "MSFT": "2026-10-10", "NVDA": "2026-10-18",
                                "JPM": "garbage"}))
    result = regime.get_earnings_calendar([" AAPL ", "MSFT", "NVDA", "JPM", "XOM", "", 5],
                                          manual_path=path, today="2026-10-18")
    assert result == {"AAPL": "2026-10-29", "MSFT": None, "NVDA": "2026-10-18", "JPM": None, "XOM": None}


@pytest.mark.parametrize("content", [None, "not json", "[1, 2]"])
def test_earnings_calendar_with_missing_or_bad_file_is_all_unknown(tmp_path, content):
    path = tmp_path / "earnings.json"
    if content is not None:
        path.write_text(content)
    assert regime.get_earnings_calendar(["AAPL"], manual_path=path, today="2026-10-18") == {"AAPL": None}
    assert regime.get_earnings_calendar("AAPL") == {}


def test_earnings_calendar_and_the_guardian_loader_read_the_same_file(tmp_path):
    path = tmp_path / "earnings.json"
    path.write_text(json.dumps({"AAPL": "2026-10-29", "MSFT": "2026-11-02"}))
    loaded = guarded_plan.load_earnings_dates(path)
    calendar = regime.get_earnings_calendar(["AAPL", "MSFT"], manual_path=path, today="2026-10-18")
    assert loaded == calendar


# ------------------------------------------------------------------ MCP server tools
@pytest.fixture()
def cache(monkeypatch):
    store = {}
    monkeypatch.setattr(server, "_cache", store)

    def put(symbol, closes, timeframe="1Day"):
        store[(symbol, timeframe)] = {"timeframe": timeframe, "df": pd.DataFrame({"close": closes})}
    return put


def test_server_exposes_both_m3_tools():
    names = {tool.name for tool in asyncio.run(server.mcp.list_tools())}
    assert {"get_market_regime", "get_earnings_calendar"} <= names


def test_get_market_regime_reads_the_cached_bars(cache):
    cache("SPY", rising())
    cache("AAPL", rising(40))
    cache("MSFT", rising(40))
    result = server.get_market_regime(["AAPL", "MSFT", "SPY", "NOPE"], "1Day")
    assert result["risk_mode"] == "risk_on" and result["breadth_pct"] == 100.0
    assert result["block_new_buys_in_risk_off"] is False
    assert result["timeframe"] == "1Day"


def test_get_market_regime_without_spy_is_neutral(cache):
    cache("AAPL", rising(40))
    result = server.get_market_regime(["AAPL"], "1Day")
    assert result["risk_mode"] == "neutral"
    assert any("SPY" in w for w in result["warnings"])


def test_get_market_regime_uses_the_requested_timeframe(cache):
    cache("SPY", rising(), "1Day")
    assert server.get_market_regime(["AAPL"], "1Hour")["spy_trend"] is None
    with pytest.raises(ValueError):
        server.get_market_regime(["AAPL"], "5Min")


def test_get_market_regime_surfaces_a_bad_settings_file(cache, monkeypatch):
    cache("SPY", rising())
    monkeypatch.setattr(server, "load_regime_settings", lambda: (dict(regime.DEFAULT_SETTINGS), "regime.yaml unusable (test)"))
    assert "regime.yaml unusable (test)" in server.get_market_regime([], "1Day")["warnings"]


# ------------------------------------------------------------------ orchestrator helper
class FakeMCP:
    def __init__(self, reply=None, error=None):
        self.reply, self.error, self.calls = reply, error, []

    async def call_tool(self, name, arguments):
        self.calls.append((name, arguments))
        if self.error:
            raise self.error
        return self.reply


def run(coro):
    return asyncio.run(coro)


def test_fetch_market_regime_success_passes_symbols_and_timeframe():
    reply = {"risk_mode": "risk_on", "spy_trend": "up", "warnings": ["Breadth skipped X: not enough valid closes."]}
    mcp = FakeMCP(reply)
    got, warnings = run(crew.fetch_market_regime(mcp, ["AAPL"], "1Day", ["get_market_regime"]))
    assert got == reply
    assert warnings == ["M3 Breadth skipped X: not enough valid closes."]
    assert mcp.calls == [("get_market_regime", {"symbols": ["AAPL"], "timeframe": "1Day"})]


def test_fetch_market_regime_is_optional_and_never_raises():
    got, warnings = run(crew.fetch_market_regime(FakeMCP(), ["AAPL"], "1Day", ["other"]))
    assert got is None and "unavailable" in warnings[0]
    got, warnings = run(crew.fetch_market_regime(FakeMCP(error=RuntimeError("boom")), ["AAPL"], "1Day",
                                                 ["get_market_regime"]))
    assert got is None and "RuntimeError" in warnings[0] and "boom" not in warnings[0]
    for reply in (None, "x", {"no": "mode"}):
        got, warnings = run(crew.fetch_market_regime(FakeMCP(reply), ["AAPL"], "1Day", ["get_market_regime"]))
        assert got is None and warnings


# ------------------------------------------------------------------ guarded plan: optional pause
SNAPSHOT = {"equity": 100_000.0, "cash": 100_000.0, "buying_power": 100_000.0,
            "positions": {"AAPL": {"qty": 50, "market_value": 15_000.0, "sector": "Technology"}},
            "new_buys_today": 0, "day_start_equity": 100_000.0,
            "sectors": {"AAPL": "Technology", "XOM": "Energy"}}
DECISIONS = [
    {"symbol": "XOM", "decision": "BUY", "latest_close": 110.0, "atr": 2.5, "composite": 0.6},
    {"symbol": "AAPL", "decision": "SELL", "latest_close": 300.0, "atr": 5.0, "composite": -0.6},
]
RISK_OFF_PAUSING = {"risk_mode": "risk_off", "block_new_buys_in_risk_off": True}


def plan(regime_value):
    return guarded_plan.build_guarded_plan(DECISIONS, SNAPSHOT, earnings_dates={}, today="2026-10-19",
                                           regime=regime_value)


def test_without_a_regime_nothing_changes():
    base = plan(None)
    assert [o["symbol"] for o in base["approved"]] == ["XOM", "AAPL"]
    assert plan({"risk_mode": "neutral", "block_new_buys_in_risk_off": True})["approved"] == base["approved"]
    assert plan({"risk_mode": "risk_on", "block_new_buys_in_risk_off": True})["approved"] == base["approved"]


def test_risk_off_with_the_switch_on_pauses_buys_but_never_sells():
    paused = plan(RISK_OFF_PAUSING)
    assert [o["symbol"] for o in paused["approved"]] == ["AAPL"]
    assert paused["approved"][0]["side"] == "sell"
    assert [(b["order"]["symbol"], b["rule"]) for b in paused["blocked"]] == [("XOM", "regime_risk_off")]
    assert "risk_off" in paused["blocked"][0]["detail"]
    xom = next(row for row in paused["plan"] if row["symbol"] == "XOM")
    assert xom["planned_quantity"] == 0 and xom["blocked_by"] == "regime_risk_off"
    assert any(o["symbol"] == "XOM" for o in paused["proposals"])  # still shown as sized by code


def test_risk_off_with_the_switch_off_only_warns():
    result = plan({"risk_mode": "risk_off", "block_new_buys_in_risk_off": False})
    assert [o["symbol"] for o in result["approved"]] == ["XOM", "AAPL"]
    assert any("new buys are not paused" in w for w in result["warnings"])


@pytest.mark.parametrize("bad", ["risk_off", 5, [], {"risk_mode": "risk_off"},
                                 {"risk_mode": "risk_off", "block_new_buys_in_risk_off": "true"}])
def test_garbage_regime_never_pauses_anything(bad):
    assert [o["symbol"] for o in plan(bad)["approved"]] == ["XOM", "AAPL"]


def test_regime_block_reason_needs_both_conditions():
    assert guarded_plan.regime_block_reason(RISK_OFF_PAUSING)
    assert guarded_plan.regime_block_reason({"risk_mode": "risk_off"}) is None
    assert guarded_plan.regime_block_reason(None) is None


# ------------------------------------------------------------------ whole pipeline (harness of test_pipeline_offline)
from trading_system.tests.test_pipeline_offline import (  # noqa: E402,F401  (fixtures + harness)
    isolated_artifact_dirs, pipeline, run_pipeline)


def test_pipeline_without_the_regime_tool_runs_and_says_so(monkeypatch, pipeline):
    result, _ = run_pipeline(monkeypatch, pipeline)
    assert result["regime"] is None
    assert result["regime_warnings"] == ["Market regime tool unavailable: get_market_regime"]
    assert result["status"] == "ok"


def test_pipeline_carries_the_regime_into_the_result_and_the_run_report(monkeypatch, pipeline):
    seen = {}

    async def fake_regime(mcp, symbols, timeframe, available_tools):
        seen.update(symbols=list(symbols), timeframe=timeframe)
        return ({"risk_mode": "risk_off", "spy_trend": "down", "spy_vol": 0.41, "breadth_pct": 25.0,
                 "timeframe": timeframe, "block_new_buys_in_risk_off": False, "warnings": []},
                ["M3 sample warning"])

    monkeypatch.setattr(crew, "fetch_market_regime", fake_regime)
    result, _ = run_pipeline(monkeypatch, pipeline)
    assert result["regime"]["risk_mode"] == "risk_off"
    assert seen["timeframe"] == "1Day" and "AAPL" in seen["symbols"]
    assert "M3 sample warning" in Path(result["report_path"]).read_text(encoding="utf-8")
    assert "risk_off" in Path(result["report_path"]).read_text(encoding="utf-8")
