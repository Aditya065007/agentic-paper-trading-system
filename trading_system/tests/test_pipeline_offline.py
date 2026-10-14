"""T5: offline end-to-end run (M1 -> M4 -> M2 -> decisions -> plan -> M9), no network.

Real code under test: the MCP server's calculate_m2_batch, scoring.py, make_decisions,
make_plan, run_strategy and the M9 writers. Faked: Alpaca bars, the indicator readings
(one uniform normalized value per symbol, so the expected composite is known), the news
tools, the holdings call and the MCP transport (JSON round trip, like the real one).
make_plan sizing is placeholder logic owned by M5/M6 and is deliberately NOT asserted.
"""

import asyncio
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

import pandas as pd
import pytest

PROJECT_DIR = Path(__file__).resolve().parents[1]
REPOSITORY_DIR = PROJECT_DIR.parent
if str(REPOSITORY_DIR) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_DIR))

from trading_system.crew import memory, reporting
from trading_system.crew import trading_crew_multiStocks as crew

INDICATOR_NAMES = (
    "ema_spread", "adx_direction", "supertrend", "rsi14", "macd_histogram", "stochastic_rsi",
    "obv_slope", "cmf20", "relative_volume", "bollinger_percent_b", "zscore20",
    "donchian20", "gap_percent", "relative_strength_spy",
)
# Uniform normalized reading per symbol -> the composite equals this value when news is missing.
PROFILES = {"AAPL": 0.8, "TSLA": -0.8, "JPM": -0.8, "XOM": 0.35, "KO": 0.0, "SPY": 0.0}
HOLDINGS = ["TSLA"]
SYMBOLS = ["AAPL", "TSLA", "JPM", "XOM", "KO", "BAD"]   # BAD has no data at all


@pytest.fixture(autouse=True)
def isolated_artifact_dirs(tmp_path, monkeypatch):
    monkeypatch.setattr(reporting, "REPORTS_DIR", tmp_path / "run_reports")
    monkeypatch.setattr(memory, "MEMORY_DIR", tmp_path / "memory")


def load_server():
    class FakeFastMCP:
        def __init__(self, name):
            self.name, self.tools = name, {}

        def tool(self):
            return lambda function: self.tools.setdefault(function.__name__, function)

        def run(self, *args, **kwargs):
            return None

    packages = {}
    for name in ("mcp", "mcp.server"):
        module = ModuleType(name)
        module.__path__ = []
        packages[name] = module
    fastmcp = ModuleType("mcp.server.fastmcp")
    fastmcp.FastMCP = FakeFastMCP
    packages["mcp.server.fastmcp"] = fastmcp
    path = PROJECT_DIR / "servers" / "technical_indicators_mcp_server_multiStocks.py"
    spec = importlib.util.spec_from_file_location("offline_pipeline_server", path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, packages):
        spec.loader.exec_module(module)
    return module


class FakeIndicators:
    def __init__(self, symbol, timeframe):
        self.symbol, self.timeframe = symbol, timeframe

    def to_dict(self):
        value = PROFILES[self.symbol]
        reading = lambda name: {"raw": {"adx": 27.0} if name == "adx_direction" else 1.0,
                                "normalized": value, "available": True}
        return {
            "symbol": self.symbol, "timeframe": self.timeframe, "as_of": "2026-10-09T20:00:00+00:00",
            "last_close": 100.0 + len(self.symbol),
            "indicators": {name: reading(name) for name in INDICATOR_NAMES},
            "context": {"atr": {"raw": 3.2, "available": True}, "natr": {"raw": 1.7, "available": True},
                        "bb_squeeze": {"raw": False, "available": True}},
            "missing_indicators": [], "warnings": [],
        }


@pytest.fixture()
def pipeline(monkeypatch):
    server = load_server()
    frame = pd.DataFrame({"close": [1.0] * 40}, index=pd.date_range("2026-08-01", periods=40))

    def get_df(symbol):
        if symbol.upper() not in PROFILES:
            raise ValueError(f"No cached data for '{symbol}'.")
        return frame

    monkeypatch.setattr(server, "_get_df", get_df)
    monkeypatch.setattr(server, "compute_indicators",
                        lambda frame, symbol, timeframe, spy_frame=None: FakeIndicators(symbol, timeframe))
    return server


class OfflineMCP:
    def __init__(self, server, news=None):
        self.server, self.news, self.calls = server, news, []

    async def list_tools(self):
        tools = ["fetch_ohlcv_batch", "calculate_m2_batch"]
        return tools + (["get_news", "score_headlines"] if self.news is not None else [])

    async def call_tool(self, name, arguments):
        self.calls.append(name)
        if name == "fetch_ohlcv_batch":
            wanted = [s for s in arguments["symbols"] if s in PROFILES]
            return {"status": "ok",
                    "fetched": [{"symbol": s, "bars_fetched": 126, "end": "2026-10-09",
                                 "last_close": 100.0 + len(s)} for s in wanted],
                    "missing": [{"symbol": s} for s in arguments["symbols"] if s not in PROFILES]}
        if name == "calculate_m2_batch":
            return json.loads(json.dumps(self.server.calculate_m2_batch(**arguments), default=str))
        if name == "get_news":
            return {"headlines": [], "warnings": [], "errors": []}
        if name == "score_headlines":
            return {"by_symbol": {s: {"news_score": v} for s, v in self.news.items()}}
        raise KeyError(name)


def run_pipeline(monkeypatch, server, news=None, holdings=HOLDINGS):
    mcp = OfflineMCP(server, news)

    class Client:
        def __init__(self, path):
            pass

        async def __aenter__(self):
            return mcp

        async def __aexit__(self, *exc):
            return False

    monkeypatch.setattr(crew, "DirectMCPClient", Client)
    monkeypatch.setattr(crew, "get_holdings", lambda: list(holdings))
    return asyncio.run(crew.run_strategy(symbols=SYMBOLS, lookback_days=180, timeframe="1Day")), mcp


def decisions_of(result):
    return {item["symbol"]: item for item in result["decisions"]}


def test_every_decision_class_through_the_real_pipeline(monkeypatch, pipeline):
    result, _ = run_pipeline(monkeypatch, pipeline)
    decisions = decisions_of(result)
    assert {s: d["decision"] for s, d in decisions.items()} == {
        "AAPL": "BUY", "TSLA": "SELL", "JPM": "HOLD", "XOM": "HOLD", "KO": "HOLD", "BAD": "HOLD"}
    assert decisions["AAPL"]["triage"] == "clear_buy"
    assert decisions["TSLA"]["held"] is True and decisions["TSLA"]["triage"] == "clear_sell"
    assert "long-only" in decisions["JPM"]["reason"].lower()
    assert decisions["XOM"]["reason"] == "borderline, awaiting M7 debate"
    assert decisions["BAD"]["composite"] is None and "fail-safe HOLD" in decisions["BAD"]["reason"]
    assert result["status"] == "ok" and result["degraded"] is True    # BAD has no data


def test_missing_news_does_not_blank_the_decisions(monkeypatch, pipeline):
    result, mcp = run_pipeline(monkeypatch, pipeline, news=None)   # news tools not even offered
    assert result["news_warnings"] == ["News tools unavailable: get_news, score_headlines"]
    assert decisions_of(result)["AAPL"]["composite"] == pytest.approx(0.8)   # renormalized, not zero-filled
    assert decisions_of(result)["AAPL"]["decision"] == "BUY"


def test_positive_news_can_lift_a_borderline_symbol_to_buy(monkeypatch, pipeline):
    result, mcp = run_pipeline(monkeypatch, pipeline, news={"XOM": 0.9})
    assert "get_news" in mcp.calls and "score_headlines" in mcp.calls
    xom = decisions_of(result)["XOM"]
    assert xom["composite"] == pytest.approx(0.85 * 0.35 + 0.15 * 0.9)   # 0.4325
    assert xom["triage"] == "clear_buy" and xom["decision"] == "BUY"


def test_news_scores_are_per_symbol(monkeypatch, pipeline):
    result, _ = run_pipeline(monkeypatch, pipeline, news={"AAPL": -0.9, "XOM": 0.9})
    assert decisions_of(result)["AAPL"]["composite"] == pytest.approx(0.85 * 0.8 + 0.15 * -0.9)
    assert decisions_of(result)["KO"]["composite"] == pytest.approx(0.0)    # no news for KO: untouched


def test_the_run_is_deterministic(monkeypatch, pipeline):
    first, _ = run_pipeline(monkeypatch, pipeline, news={"XOM": 0.9})
    second, _ = run_pipeline(monkeypatch, pipeline, news={"XOM": 0.9})
    assert first["decisions"] == second["decisions"] and first["plan"] == second["plan"]


def test_nothing_touches_the_network_and_no_order_is_placed(monkeypatch, pipeline, capsys):
    import requests

    def forbidden(*args, **kwargs):
        raise AssertionError("network access during an offline test")

    for name in ("get", "post", "put", "delete", "request"):
        monkeypatch.setattr(requests, name, forbidden)
    result, _ = run_pipeline(monkeypatch, pipeline)
    output = capsys.readouterr().out.lower()
    assert "order placed" not in output and "order submitted" not in output
    assert len(result["plan"]) == len(result["decisions"])
    assert not any("order_id" in row for row in result["plan"])


def test_artifacts_are_written_and_the_exit_code_is_zero(monkeypatch, pipeline):
    run_pipeline(monkeypatch, pipeline)
    monkeypatch.setattr(sys, "argv", ["crew"] + SYMBOLS)
    assert asyncio.run(crew.async_main()) == 0
    assert list(reporting.REPORTS_DIR.glob("*.md")) and list(memory.MEMORY_DIR.glob("*.md"))
