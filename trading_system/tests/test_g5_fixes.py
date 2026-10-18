"""G5: profiles.yaml wiring, missing_indicators fix, news confidence guard, per-symbol news (cache,
budget), report headings, requirements. Fully offline: NewsAPI is a fake, each test gets its own cache."""

import asyncio
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
import requests
import yaml

REPOSITORY_DIR = Path(__file__).resolve().parents[2]
if str(REPOSITORY_DIR) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_DIR))

from trading_system.crew import advisory  # noqa: E402
from trading_system.crew import trading_crew_multiStocks as crew  # noqa: E402
from trading_system.servers import technical_indicators_mcp_server_multiStocks as server  # noqa: E402
from trading_system.servers.tools import indicators, news, scoring  # noqa: E402
from trading_system.tests.test_guarded_cycle import SYMBOLS, cycle_env, run  # noqa: E402,F401
from trading_system.tests.test_pipeline_offline import isolated_artifact_dirs, pipeline  # noqa: E402,F401
from trading_system.tests.test_weights_config import m2a  # noqa: E402

PACKAGE = Path(__file__).resolve().parents[1]
PROFILES_YAML = PACKAGE / "config" / "profiles.yaml"


# =================================================================== profiles.yaml
def write(tmp_path, data):
    path = tmp_path / "profiles.yaml"
    path.write_text(data if isinstance(data, str) else yaml.safe_dump(data), encoding="utf-8")
    return path


def shipped():
    return yaml.safe_load(PROFILES_YAML.read_text(encoding="utf-8"))


def test_shipped_profiles_load_cleanly_and_equal_the_builtin_defaults():
    thresholds, buffer, warning = scoring.load_profiles(PROFILES_YAML)
    assert warning is None
    assert thresholds == scoring.DEFAULT_TRIAGE_THRESHOLDS
    assert buffer == scoring.DEFAULT_BORDERLINE_BUFFER
    assert scoring.PROFILES_WARNING is None


def test_spec_5_5_values():
    data = shipped()
    assert (data["1Day"]["buy_threshold"], data["1Day"]["sell_threshold"]) == (0.40, -0.40)
    assert (data["1Hour"]["buy_threshold"], data["1Hour"]["sell_threshold"]) == (0.45, -0.45)
    assert data["1Day"]["cooldown_bars"] == 3 and data["1Hour"]["cooldown_bars"] == 4
    assert data["1Day"]["history_days"] == 250 and data["1Hour"]["history_days"] == 30


def test_advisory_uses_the_very_same_thresholds_as_scoring():
    assert advisory.TRIAGE_THRESHOLDS is scoring.TRIAGE_THRESHOLDS
    assert advisory.BAND_BUFFER == scoring.BORDERLINE_BUFFER


def test_a_valid_edit_is_used(tmp_path):
    data = shipped()
    data["1Day"]["buy_threshold"] = 0.5
    data["borderline_buffer"] = 0.05
    thresholds, buffer, warning = scoring.load_profiles(write(tmp_path, data))
    assert warning is None and thresholds["1Day"]["buy"] == 0.5 and buffer == 0.05


@pytest.mark.parametrize("edit", [
    lambda d: d.pop("borderline_buffer"),
    lambda d: d.update(borderline_buffer=0),
    lambda d: d.update(borderline_buffer=0.5),
    lambda d: d.update(borderline_buffer=True),
    lambda d: d.update(borderline_buffer="wide"),
    lambda d: d.pop("1Hour"),
    lambda d: d["1Day"].update(buy_threshold="high"),
    lambda d: d["1Day"].update(buy_threshold=-0.4),
    lambda d: d["1Day"].update(sell_threshold=0.4),
    lambda d: d["1Day"].update(buy_threshold=1.5),
    lambda d: d["1Hour"].update(buy_threshold=0.05),
    lambda d: d["1Hour"].pop("sell_threshold"),
    lambda d: d["1Hour"].update(sell_threshold=float("nan")),
])
def test_bad_profiles_fall_back_with_a_warning(tmp_path, edit):
    data = shipped()
    edit(data)
    thresholds, buffer, warning = scoring.load_profiles(write(tmp_path, data))
    assert thresholds == scoring.DEFAULT_TRIAGE_THRESHOLDS and buffer == 0.10
    assert warning and "built-in default" in warning


@pytest.mark.parametrize("text", ["", "- a list", "not: [valid", "3"])
def test_unreadable_profiles_fall_back_with_a_warning(tmp_path, text):
    assert scoring.load_profiles(write(tmp_path, text))[2]
    assert scoring.load_profiles(tmp_path / "absent.yaml")[2]


def test_fallback_defaults_are_copies_not_the_shared_dict(tmp_path):
    thresholds, _, _ = scoring.load_profiles(tmp_path / "absent.yaml")
    thresholds["1Day"]["buy"] = 9.0
    assert scoring.DEFAULT_TRIAGE_THRESHOLDS["1Day"]["buy"] == 0.40


def test_the_profile_warning_reaches_every_score_row(monkeypatch):
    monkeypatch.setattr(scoring, "PROFILES_WARNING", "profiles.yaml unusable (test)")
    row = scoring.score_indicators("AAPL", "1Day", "core", "Technology", m2a(27.0), news_score=0.2)
    assert "profiles.yaml unusable (test)" in row["warnings"]
    monkeypatch.setattr(scoring, "PROFILES_WARNING", None)
    clean = scoring.score_indicators("AAPL", "1Day", "core", "Technology", m2a(27.0), news_score=0.2)
    assert not any("profiles.yaml" in w for w in clean["warnings"])


def test_triage_follows_the_loaded_thresholds(monkeypatch):
    assert scoring.classify_triage(0.41, "1Day") == "clear_buy"
    monkeypatch.setattr(scoring, "TRIAGE_THRESHOLDS", {"1Day": {"buy": 0.6, "sell": -0.6},
                                                       "1Hour": {"buy": 0.65, "sell": -0.65}})
    assert scoring.classify_triage(0.41, "1Day") == "hold"
    assert scoring.classify_triage(0.61, "1Day") == "clear_buy"


# =================================================================== missing_indicators
def frame(n, seed=1):
    rng = np.random.default_rng(seed)
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    index = pd.date_range("2026-01-02", periods=n, freq="B", tz="UTC")
    return pd.DataFrame({"open": close - 0.2, "high": close + 1, "low": close - 1, "close": close,
                         "volume": rng.integers(1_000_000, 2_000_000, n)}, index=index)


def missing(n, timeframe="1Day"):
    data = frame(n)
    return indicators.compute_indicators(frame=data, symbol="AAPL", timeframe=timeframe,
                                         spy_frame=data.copy()).to_dict()


def test_with_plenty_of_data_nothing_is_reported_missing():
    result = missing(200)
    assert result["missing_indicators"] == []
    assert not any("unavailable" in w for w in result["warnings"])


def test_with_too_little_data_the_gaps_are_reported_but_not_everything():
    names = set(missing(30)["missing_indicators"])
    # 30 bars: the 50-bar EMA and the squeeze window are not computable, but RSI (14 bars) is.
    assert {"ema_spread", "bb_squeeze"} <= names and "rsi14" not in names
    assert len(names) < len(indicators.INDICATOR_NAMES) + len(indicators.CONTEXT_NAMES)


def test_timeframe_specific_indicators_are_not_reported_missing_on_the_other_timeframe():
    assert "vwap" not in missing(200, "1Day")["missing_indicators"]
    assert "gap_percent" not in missing(200, "1Hour")["missing_indicators"]


def test_a_180_day_daily_fetch_has_no_squeeze_context_by_design():
    # 124 daily bars: the 120-bar squeeze quantile on top of a 20-bar band width needs about 140.
    assert missing(124)["missing_indicators"] == ["bb_squeeze"]


def test_an_empty_input_still_reports_everything_missing():
    result = indicators.compute_indicators(frame=None, symbol="AAPL", timeframe="1Day").to_dict()
    assert set(result["missing_indicators"]) == set(indicators.INDICATOR_NAMES) | set(indicators.CONTEXT_NAMES)


# =================================================================== news confidence guard
def articles(*values):
    return [{"symbol": "JPM", "sentiment_score": v} for v in values]


@pytest.mark.parametrize("values, raw, confidence", [
    ([-0.93], -0.93, 1 / 3), ([0.9, 0.7], 0.8, 2 / 3), ([0.6, 0.6, 0.6], 0.6, 1.0),
    ([0.6] * 7, 0.6, 1.0), ([0.2, -0.2, 0.6, -0.6], 0.0, 1.0),
])
def test_the_news_score_is_shrunk_by_the_headline_count(values, raw, confidence):
    out = news._aggregate("JPM", articles(*values))
    assert out["raw_news_score"] == pytest.approx(raw, abs=1e-6)
    assert out["confidence"] == pytest.approx(confidence, abs=1e-4)
    assert out["news_score"] == pytest.approx(raw * confidence, abs=1e-5)


def test_the_jpm_case_one_cautionary_headline_no_longer_swings_the_score():
    out = news._aggregate("JPM", articles(-0.93))
    assert -0.35 < out["news_score"] < -0.30
    assert out["sentiment"] == "negative" and out["scored_article_count"] == 1


def test_unscored_articles_do_not_count_toward_confidence():
    out = news._aggregate("JPM", articles(0.9, None, None))
    assert out["scored_article_count"] == 1 and out["confidence"] == pytest.approx(1 / 3, abs=1e-4)
    none = news._aggregate("JPM", articles(None, None))
    assert none["news_score"] is None and none["sentiment_available"] is False


def test_the_guard_never_flips_a_sign_or_leaves_the_range():
    for values in ([1.0], [-1.0], [1.0, 1.0], [0.0], [0.5, -0.5, 0.1]):
        out = news._aggregate("X", articles(*values))
        assert -1.0 <= out["news_score"] <= 1.0
        assert out["news_score"] == 0 or (out["news_score"] > 0) == (out["raw_news_score"] > 0)


# =================================================================== per-symbol news
class FakeResponse:
    def __init__(self, payload=None, status=200):
        self.payload, self.status_code = payload, status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")

    def json(self):
        if isinstance(self.payload, Exception):
            raise self.payload
        return self.payload


def story(symbol, n=1):
    return {"title": f"{symbol} announces item {n}", "source": {"name": "Example"},
            "publishedAt": "2026-10-17T12:00:00Z", "url": f"https://example.test/{symbol}/{n}",
            "description": ""}


def ok(*items):
    return FakeResponse({"status": "ok", "articles": list(items)})


@pytest.fixture()
def feed(tmp_path, monkeypatch):
    """Private cache folder, a key, a generous budget, and a recorder for NewsAPI calls."""
    monkeypatch.setenv("TRADING_DATA_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("NEWSAPI_KEY", "offline-test-key")
    monkeypatch.setenv("NEWSAPI_DAILY_LIMIT", "50")
    state = MagicMock()
    state.replies = {}

    def fake_get(url, params=None, headers=None, timeout=None):
        state.calls.append((params["q"], dict(params), dict(headers)))
        reply = state.replies.get(params["q"].strip('"'))
        if reply is None:
            symbol = params["q"].strip('"')
            return ok(story(symbol))
        if isinstance(reply, Exception):
            raise reply
        return reply

    state.calls = []
    monkeypatch.setattr(news.requests, "get", fake_get)
    state.cache = tmp_path / "cache"
    return state


def budget_used(state):
    path = state.cache / "news_budget.json"
    return json.loads(path.read_text())["count"] if path.exists() else 0


def test_one_request_per_symbol_each_tagged_with_its_own_symbol(feed):
    result = news.get_news_by_symbol(["jpm", "AAPL", "MSFT"])
    assert [call[0] for call in feed.calls] == ['"AAPL"', '"JPM"', '"MSFT"']
    assert sorted(h["symbol"] for h in result["headlines"]) == ["AAPL", "JPM", "MSFT"]
    assert result["article_count"] == 3 and result["available"] is True
    assert result["budget"] == {"limit": 50, "used_today": 3, "requests_made": 3, "cache_hits": 0}
    assert budget_used(feed) == 3


def test_the_result_feeds_score_headlines_unchanged(feed):
    result = news.get_news_by_symbol(["AAPL"])
    scored = news.score_headlines(result)
    assert set(scored["by_symbol"]) <= {"AAPL"} and scored["article_count"] == 1


def test_a_busy_symbol_cannot_crowd_out_a_quiet_one(feed):
    feed.replies["JPM"] = ok(story("JPM"))
    feed.replies["AAPL"] = ok(*[story("AAPL", i) for i in range(40)])
    result = news.get_news_by_symbol(["AAPL", "JPM"])
    counts = {s: sum(h["symbol"] == s for h in result["headlines"]) for s in ("AAPL", "JPM")}
    assert counts == {"AAPL": 40, "JPM": 1}


def test_a_repeat_call_is_served_from_the_disk_cache_with_no_request(feed):
    first = news.get_news_by_symbol(["AAPL", "JPM"])
    feed.calls.clear()
    second = news.get_news_by_symbol(["JPM", "AAPL"])
    assert feed.calls == []
    assert second["headlines"] == first["headlines"]
    assert second["budget"]["requests_made"] == 0 and second["budget"]["cache_hits"] == 2
    assert budget_used(feed) == 2  # cache hits cost nothing


def test_a_stale_cache_entry_is_refetched(feed, monkeypatch):
    news.get_news_by_symbol(["AAPL"])
    monkeypatch.setattr(news, "NEWS_CACHE_TTL_MINUTES", 0)
    feed.calls.clear()
    news.get_news_by_symbol(["AAPL"])
    assert len(feed.calls) == 1


def test_a_new_day_window_is_a_new_cache_entry(feed, monkeypatch):
    news.get_news_by_symbol(["AAPL"])
    monkeypatch.setattr(news, "_window", lambda days: ("2026-10-18", "2026-10-19"))
    feed.calls.clear()
    news.get_news_by_symbol(["AAPL"])
    assert len(feed.calls) == 1


def test_the_daily_budget_stops_new_requests_and_names_the_skipped_symbols(feed, monkeypatch):
    monkeypatch.setenv("NEWSAPI_DAILY_LIMIT", "2")
    result = news.get_news_by_symbol(["AAPL", "JPM", "MSFT", "NVDA"])
    assert len(feed.calls) == 2 and budget_used(feed) == 2
    assert sorted({h["symbol"] for h in result["headlines"]}) == ["AAPL", "JPM"]
    assert any("News skipped for MSFT, NVDA" in w and "daily request budget of 2" in w
               for w in result["warnings"])
    assert result["degraded"] is True


def test_cached_symbols_are_still_served_after_the_budget_is_spent(feed, monkeypatch):
    news.get_news_by_symbol(["AAPL"])
    monkeypatch.setenv("NEWSAPI_DAILY_LIMIT", "1")
    feed.calls.clear()
    result = news.get_news_by_symbol(["AAPL", "JPM"])
    assert feed.calls == []
    assert [h["symbol"] for h in result["headlines"]] == ["AAPL"]
    assert any("News skipped for JPM" in w for w in result["warnings"])


def test_yesterdays_tally_does_not_count_today(feed):
    feed.cache.mkdir(parents=True)
    yesterday = (datetime.now(timezone.utc) - timedelta(days=1)).date().isoformat()
    (feed.cache / "news_budget.json").write_text(json.dumps({"date": yesterday, "count": 999}))
    news.get_news_by_symbol(["AAPL"])
    assert budget_used(feed) == 1 and len(feed.calls) == 1


@pytest.mark.parametrize("raw", ["abc", "-5", ""])
def test_a_bad_limit_setting_falls_back_to_the_default(feed, monkeypatch, raw):
    monkeypatch.setenv("NEWSAPI_DAILY_LIMIT", raw)
    assert news._daily_limit() == news.DEFAULT_DAILY_REQUEST_LIMIT


def test_a_limit_of_zero_makes_no_request(feed, monkeypatch):
    monkeypatch.setenv("NEWSAPI_DAILY_LIMIT", "0")
    result = news.get_news_by_symbol(["AAPL"])
    assert feed.calls == [] and result["headlines"] == [] and result["degraded"]


def test_http_429_stops_further_requests_and_spends_the_budget(feed):
    feed.replies["AAPL"] = FakeResponse(status=429)
    result = news.get_news_by_symbol(["AAPL", "JPM", "MSFT"])
    assert len(feed.calls) == 1
    assert any("HTTP 429" in w and "AAPL, JPM, MSFT" in w for w in result["warnings"])
    assert budget_used(feed) == 50


def test_one_failing_symbol_does_not_stop_the_others_and_is_not_cached(feed):
    feed.replies["JPM"] = requests.Timeout("slow")
    feed.replies["MSFT"] = requests.ConnectionError("down")
    result = news.get_news_by_symbol(["AAPL", "JPM", "MSFT"])
    assert [h["symbol"] for h in result["headlines"]] == ["AAPL"]
    assert "NewsAPI timeout for JPM." in result["errors"]
    assert "NewsAPI request failure for MSFT." in result["errors"]
    feed.replies.clear()
    feed.calls.clear()
    again = news.get_news_by_symbol(["AAPL", "JPM", "MSFT"])
    assert [call[0] for call in feed.calls] == ['"JPM"', '"MSFT"']
    assert again["budget"]["cache_hits"] == 1


def test_a_malformed_or_unsuccessful_response_is_reported_not_cached(feed):
    feed.replies["AAPL"] = FakeResponse(ValueError("bad json"))
    feed.replies["JPM"] = FakeResponse({"status": "error"})
    result = news.get_news_by_symbol(["AAPL", "JPM"])
    assert result["headlines"] == [] and len(result["errors"]) == 2
    assert not list(feed.cache.glob("news_*.json")) or all(
        "news_budget" in p.name for p in feed.cache.glob("news_*.json"))


def test_an_article_that_does_not_mention_the_symbol_is_dropped(feed):
    feed.replies["JPM"] = ok({"title": "Weather turns cold", "url": "https://example.test/w",
                              "source": {"name": "X"}, "description": ""})
    result = news.get_news_by_symbol(["JPM"])
    assert result["headlines"] == [] and "No matching headlines for JPM." in result["warnings"]


def test_empty_but_successful_answers_are_cached_to_save_requests(feed):
    feed.replies["JPM"] = ok()
    news.get_news_by_symbol(["JPM"])
    feed.calls.clear()
    news.get_news_by_symbol(["JPM"])
    assert feed.calls == []


def test_missing_key_bad_input_make_no_request(feed, monkeypatch):
    monkeypatch.delenv("NEWSAPI_KEY")
    assert news.get_news_by_symbol(["AAPL"])["errors"] == ["NewsAPI credentials are unavailable."]
    monkeypatch.setenv("NEWSAPI_KEY", "offline-test-key")
    assert news.get_news_by_symbol([])["errors"]
    assert news.get_news_by_symbol(["AAPL"], lookback_days=0)["errors"]
    assert news.get_news_by_symbol(["AAPL"], lookback_days=True)["errors"] == ["Invalid lookback window."]
    assert feed.calls == []


def test_the_api_key_is_only_ever_sent_as_a_header(feed):
    result = news.get_news_by_symbol(["AAPL"])
    query, params, headers = feed.calls[0]
    assert headers["X-Api-Key"] == "offline-test-key"
    assert "offline-test-key" not in str(params) and "offline-test-key" not in str(result)
    for path in feed.cache.glob("*"):
        assert "offline-test-key" not in path.read_text()


def test_an_unwritable_cache_folder_never_breaks_news(feed, monkeypatch, tmp_path):
    blocker = tmp_path / "file-not-folder"
    blocker.write_text("x")
    monkeypatch.setenv("TRADING_DATA_CACHE_DIR", str(blocker / "inside"))
    result = news.get_news_by_symbol(["AAPL"])
    assert [h["symbol"] for h in result["headlines"]] == ["AAPL"]


def test_bars_cache_files_and_news_cache_files_do_not_collide(feed):
    from trading_system.servers.tools import data
    key = data.cache_key(["AAPL"], "news", 3, "newsapi", "x")
    assert data._cache_file(feed.cache, key).name.startswith("bars_")
    assert data._cache_file(feed.cache, key, "news").name.startswith("news_")


# =================================================================== wiring
def test_the_real_server_exposes_the_new_tool_next_to_get_news():
    names = {tool.name for tool in asyncio.run(server.mcp.list_tools())}
    assert {"get_news", "get_news_by_symbol", "score_headlines"} <= names


class RecordingMCP:
    def __init__(self):
        self.calls = []

    async def call_tool(self, name, arguments):
        self.calls.append(name)
        if name == "score_headlines":
            return {"by_symbol": {"AAPL": {"news_score": 0.1}}}
        return {"headlines": [], "warnings": [], "errors": []}


def test_the_orchestrator_prefers_the_per_symbol_tool_when_it_exists():
    mcp = RecordingMCP()
    scores, _ = asyncio.run(crew.fetch_news_scores(
        mcp, ["AAPL"], ["get_news", "get_news_by_symbol", "score_headlines"]))
    assert mcp.calls == ["get_news_by_symbol", "score_headlines"] and scores == {"AAPL": 0.1}


def test_the_orchestrator_falls_back_to_get_news_on_an_older_server():
    mcp = RecordingMCP()
    asyncio.run(crew.fetch_news_scores(mcp, ["AAPL"], ["get_news", "score_headlines"]))
    assert mcp.calls == ["get_news", "score_headlines"]
    _, warnings = asyncio.run(crew.fetch_news_scores(mcp, ["AAPL"], ["score_headlines"]))
    assert warnings == ["News tools unavailable: get_news"]


# =================================================================== report headings and requirements
def test_plain_report_says_starter_and_what_the_llm_and_orders_do(cycle_env, capsys):
    asyncio.run(crew.run_strategy(symbols=SYMBOLS, lookback_days=180, timeframe="1Day"))
    out = capsys.readouterr().out
    assert "STARTER ACTION PLAN (simple sizing, no risk check; use --guarded)" in out
    assert "LLM decision-making: NONE (code decides; the LLM only advises with --advisory)" in out
    assert "Order submission: NONE from this report (orders need run_cycle approval)" in out
    assert "DISABLED" not in out


def test_guarded_report_does_not_call_the_plan_a_starter_plan(cycle_env, capsys):
    run({})
    out = capsys.readouterr().out
    assert "ACTION PLAN (sized and risk-checked by the guardian; nothing is sent)" in out
    assert "STARTER ACTION PLAN" not in out


def test_requirements_list_the_finbert_packages():
    lines = [line.strip() for line in (PACKAGE / "requirements.txt").read_text().splitlines()]
    assert "transformers" in lines and "torch" in lines
