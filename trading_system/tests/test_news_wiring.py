"""T3: M4 news scores flow into M2 through the orchestrator (offline, mocked)."""

import asyncio
import importlib.util
import math
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


@pytest.fixture(autouse=True)
def isolated_artifact_dirs(tmp_path, monkeypatch):
    """Never write reports or journal entries into the real repository folders."""
    monkeypatch.setattr(reporting, "REPORTS_DIR", tmp_path / "run_reports")
    monkeypatch.setattr(memory, "MEMORY_DIR", tmp_path / "memory")


def load_server():
    class FakeFastMCP:
        def __init__(self, name):
            self.name = name
            self.tools = {}

        def tool(self):
            return lambda function: self.tools.setdefault(function.__name__, function)

        def run(self, *args, **kwargs):
            return None

    mcp_package = ModuleType("mcp")
    mcp_package.__path__ = []
    server_package = ModuleType("mcp.server")
    server_package.__path__ = []
    fastmcp_module = ModuleType("mcp.server.fastmcp")
    fastmcp_module.FastMCP = FakeFastMCP
    path = PROJECT_DIR / "servers" / "technical_indicators_mcp_server_multiStocks.py"
    spec = importlib.util.spec_from_file_location("m2_news_wiring_server", path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {"mcp": mcp_package, "mcp.server": server_package,
                                  "mcp.server.fastmcp": fastmcp_module}):
        spec.loader.exec_module(module)
    return module


class FakeIndicatorResult:
    def __init__(self, symbol, timeframe):
        self.symbol, self.timeframe = symbol, timeframe

    def to_dict(self):
        reading = lambda name: {"raw": {"adx": 27.0} if name == "adx_direction" else 1.0,
                                "normalized": 0.5, "available": True}
        return {
            "symbol": self.symbol, "timeframe": self.timeframe,
            "as_of": "2026-10-09T20:00:00+00:00", "last_close": 190.0,
            "indicators": {name: reading(name) for name in INDICATOR_NAMES},
            "context": {"atr": {"raw": 3.2, "available": True},
                        "natr": {"raw": 1.7, "available": True},
                        "bb_squeeze": {"raw": False, "available": True}},
            "missing_indicators": [], "warnings": [],
        }


@pytest.fixture()
def server(monkeypatch):
    module = load_server()
    frame = pd.DataFrame({"close": [1.0] * 40}, index=pd.date_range("2026-08-01", periods=40))
    monkeypatch.setattr(module, "_get_df", lambda symbol: frame)
    monkeypatch.setattr(
        module, "compute_indicators",
        lambda frame, symbol, timeframe, spy_frame=None: FakeIndicatorResult(symbol, timeframe))
    return module


def rows_by_symbol(report):
    return {row["symbol"]: row["score"] for row in report["results"]}


# --------------------------- server: calculate_m2_batch ---------------------------

def test_news_scores_reach_the_scorer(server):
    scores = rows_by_symbol(server.calculate_m2_batch(
        ["AAPL", "MSFT"], news_scores={"AAPL": 0.6, "MSFT": None}))
    assert scores["AAPL"]["sub_scores"]["news"] == 0.6
    assert scores["AAPL"]["composite"] == pytest.approx(0.85 * 0.5 + 0.15 * 0.6)
    assert scores["MSFT"]["sub_scores"]["news"] is None
    assert scores["MSFT"]["composite"] == pytest.approx(0.5)  # renormalized, not zero-filled
    assert "news" in scores["MSFT"]["missing_indicators"]


def test_omitting_news_scores_keeps_the_old_behaviour(server):
    score = rows_by_symbol(server.calculate_m2_batch(["AAPL"]))["AAPL"]
    assert score["sub_scores"]["news"] is None
    assert score["composite"] == pytest.approx(0.5)


def test_symbol_keys_in_the_map_are_normalized(server):
    score = rows_by_symbol(server.calculate_m2_batch(["AAPL"], news_scores={" aapl ": 0.6}))["AAPL"]
    assert score["sub_scores"]["news"] == 0.6


def test_a_non_dict_map_is_ignored_not_fatal(server):
    report = server.calculate_m2_batch(["AAPL"], news_scores=["AAPL"])
    assert report["missing"] == []
    assert rows_by_symbol(report)["AAPL"]["sub_scores"]["news"] is None


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), 7, "x", True])
def test_invalid_news_values_are_excluded_by_the_scorer(server, bad):
    report = server.calculate_m2_batch(["AAPL"], news_scores={"AAPL": bad})
    assert report["missing"] == []
    score = rows_by_symbol(report)["AAPL"]
    assert score["sub_scores"]["news"] is None
    assert score["composite"] is not None and math.isfinite(score["composite"])


# --------------------------- crew: fetch_news_scores ---------------------------

class RecordingMCP:
    def __init__(self, responses, tools=("get_news", "score_headlines", "calculate_m2_batch",
                                         "fetch_ohlcv_batch")):
        self.responses, self.tools, self.calls = responses, list(tools), []

    async def list_tools(self):
        return self.tools

    async def call_tool(self, name, arguments):
        self.calls.append((name, arguments))
        response = self.responses[name]
        if isinstance(response, Exception):
            raise response
        return response


def run(coroutine):
    return asyncio.run(coroutine)


def test_helper_success_path_and_argument_shape():
    news = {"headlines": [{"title": "x"}], "warnings": ["slow feed"], "errors": []}
    mcp = RecordingMCP({"get_news": news, "score_headlines": {
        "by_symbol": {"AAPL": {"news_score": 0.4}, "MSFT": {"news_score": None}}}})
    scores, warnings = run(crew.fetch_news_scores(mcp, ["AAPL", "MSFT", "NVDA"], mcp.tools))
    assert scores == {"AAPL": 0.4, "MSFT": None, "NVDA": None}
    assert warnings == ["slow feed"]
    assert mcp.calls[0] == ("get_news", {"symbols": ["AAPL", "MSFT", "NVDA"]})
    assert mcp.calls[1] == ("score_headlines", {"headlines": news})


def test_helper_get_news_failure_is_non_fatal():
    mcp = RecordingMCP({"get_news": RuntimeError("boom"), "score_headlines": {}})
    scores, warnings = run(crew.fetch_news_scores(mcp, ["AAPL"], mcp.tools))
    assert scores == {}
    assert warnings == ["News unavailable: RuntimeError: boom"]


def test_helper_score_headlines_failure_is_non_fatal():
    mcp = RecordingMCP({"get_news": {"headlines": []}, "score_headlines": ValueError("bad")})
    scores, warnings = run(crew.fetch_news_scores(mcp, ["AAPL"], mcp.tools))
    assert scores == {} and warnings == ["News unavailable: ValueError: bad"]


def test_helper_without_by_symbol_data_returns_empty_map():
    mcp = RecordingMCP({"get_news": {"warnings": ["no key"]}, "score_headlines": {"available": False}})
    scores, warnings = run(crew.fetch_news_scores(mcp, ["AAPL"], mcp.tools))
    assert scores == {}
    assert warnings[0] == "no key" and "no by_symbol" in warnings[1]


def test_helper_with_missing_tools_makes_no_calls():
    mcp = RecordingMCP({}, tools=("fetch_ohlcv_batch", "calculate_m2_batch"))
    scores, warnings = run(crew.fetch_news_scores(mcp, ["AAPL"], mcp.tools))
    assert scores == {}
    assert warnings == ["News tools unavailable: get_news, score_headlines"]
    assert mcp.calls == []


# --------------------------- crew: run_strategy wiring ---------------------------

def patch_run_strategy_environment(monkeypatch, mcp):
    class Client:
        def __init__(self, path):
            pass

        async def __aenter__(self):
            return mcp

        async def __aexit__(self, *exc):
            return False

    monkeypatch.setattr(crew, "DirectMCPClient", Client)
    monkeypatch.setattr(crew, "get_holdings", lambda: [])


BASE_RESPONSES = {
    "fetch_ohlcv_batch": {"status": "ok", "missing": [], "fetched": [
        {"symbol": "AAPL", "bars_fetched": 126, "end": "2026-10-09", "last_close": 190.0}]},
    "calculate_m2_batch": {"status": "ok", "results": [], "missing": []},
    "get_news": {"headlines": [], "warnings": [], "errors": []},
    "score_headlines": {"by_symbol": {"AAPL": {"news_score": 0.6}}},
}


def m2_arguments(mcp):
    return next(arguments for name, arguments in mcp.calls if name == "calculate_m2_batch")


def test_run_strategy_passes_news_scores_to_m2(monkeypatch):
    mcp = RecordingMCP(dict(BASE_RESPONSES))
    patch_run_strategy_environment(monkeypatch, mcp)
    result = run(crew.run_strategy(symbols=["AAPL"], lookback_days=180, timeframe="1Day"))
    assert m2_arguments(mcp)["news_scores"] == {"AAPL": 0.6}
    assert result["status"] == "ok"
    assert result["news_scores"] == {"AAPL": 0.6} and result["news_warnings"] == []


def test_run_strategy_continues_when_news_tools_are_missing(monkeypatch):
    mcp = RecordingMCP(dict(BASE_RESPONSES), tools=("fetch_ohlcv_batch", "calculate_m2_batch"))
    patch_run_strategy_environment(monkeypatch, mcp)
    result = run(crew.run_strategy(symbols=["AAPL"], lookback_days=180, timeframe="1Day"))
    assert m2_arguments(mcp)["news_scores"] == {}
    assert result["status"] == "ok"
    assert result["news_warnings"] == ["News tools unavailable: get_news, score_headlines"]
