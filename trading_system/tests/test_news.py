import os
import importlib
import importlib.util
import sys
from types import ModuleType
import unittest
from pathlib import Path
from unittest.mock import patch

import requests

PROJECT_DIR = Path(__file__).resolve().parents[1]
REPOSITORY_DIR = PROJECT_DIR.parent
for import_path in (str(PROJECT_DIR), str(REPOSITORY_DIR)):
    if import_path not in sys.path:
        sys.path.insert(0, import_path)

from servers.tools import news
from servers.tools.scoring import WEIGHTS, score_indicators


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


def prediction_pipeline(probabilities):
    return lambda *args, **kwargs: [
        {"label": label, "score": value} for label, value in probabilities.items()]


def m2a_result():
    def reading(raw, normalized):
        return {"raw": raw, "normalized": normalized, "available": True}

    return {
        "symbol": "AAPL",
        "timeframe": "1Day",
        "as_of": "2026-10-08T12:00:00+00:00",
        "last_close": 190.1,
        "indicators": {
            "ema_spread": reading({"ema_short": 105, "ema_long": 100}, 0.5),
            "adx_direction": reading({"adx": 27.1, "plus_di": 30, "minus_di": 10}, 0.4),
            "supertrend": reading({"line": 100, "direction": 1}, 1.0),
            "rsi14": reading(60.0, 0.2),
            "macd_histogram": reading(0.3, 0.3),
            "stochastic_rsi": reading({"k": 70, "d": 65}, 0.4),
            "obv_slope": reading(3.0, 0.3),
            "cmf20": reading(0.2, 0.2),
            "relative_volume": reading(1.5, 0.4),
            "bollinger_percent_b": reading(0.6, 0.2),
            "zscore20": reading(0.3, 0.1),
            "donchian20": reading({"upper": 110, "lower": 90}, 0.5),
            "gap_percent": reading(0.5, 0.2),
            "relative_strength_spy": reading({"asset_return": 0.1, "spy_return": 0.05}, 0.3),
        },
        "context": {
            "atr": {"raw": 3.2, "normalized": None, "available": True},
            "natr": {"raw": 1.7, "normalized": None, "available": True},
            "bb_squeeze": {"raw": False, "normalized": None, "available": True},
        },
        "missing_indicators": [],
        "warnings": [],
    }


class NewsToolTests(unittest.TestCase):
    def setUp(self):
        news._news_cache.clear()

    @patch.dict(os.environ, {"NEWSAPI_KEY": "offline-test-key"})
    @patch("servers.tools.news.requests.get")
    def test_parsing_deduplicates_and_caches_batched_query(self, get):
        article = {
            "title": "AAPL and MSFT report quarterly results",
            "source": {"name": "Example News"},
            "publishedAt": "2026-10-06T12:00:00Z",
            "url": "https://example.test/story",
            "description": "",
        }
        get.return_value = FakeResponse({"status": "ok", "articles": [article, article]})
        result = news.get_news(["AAPL", "MSFT"])
        repeated = news.get_news(["MSFT", "AAPL"])
        self.assertEqual(len(result["headlines"]), 2)
        self.assertEqual(result["headlines"][0]["title"], article["title"])
        self.assertEqual(result["headlines"][0]["source"], "Example News")
        self.assertEqual(result["headlines"][0]["published_at"], article["publishedAt"])
        self.assertEqual(result["headlines"][0]["url"], article["url"])
        self.assertEqual(result["warnings"], [])
        self.assertEqual(repeated, result)
        get.assert_called_once()
        self.assertNotIn("offline-test-key", str(result))
        self.assertNotIn("offline-test-key", str(get.call_args.kwargs["params"]))
        self.assertEqual(get.call_args.kwargs["headers"]["X-Api-Key"], "offline-test-key")

    @patch.dict(os.environ, {}, clear=True)
    @patch("servers.tools.news.requests.get")
    def test_missing_newsapi_key_warns_without_request(self, get):
        result = news.get_news(["AAPL"])
        self.assertEqual(result["headlines"], [])
        self.assertTrue(result["warnings"])
        get.assert_not_called()

    @patch.dict(os.environ, {"NEWSAPI_KEY": "offline-test-key"})
    @patch("servers.tools.news.requests.get", side_effect=requests.Timeout)
    def test_timeout_has_degraded_state(self, get):
        result = news.get_news(["AAPL"])
        self.assertTrue(result["degraded"])
        self.assertIn("NewsAPI timeout.", result["errors"])

    @patch.dict(os.environ, {"NEWSAPI_KEY": "offline-test-key"})
    @patch("servers.tools.news.requests.get", side_effect=requests.ConnectionError)
    def test_api_failure_is_structured_and_hides_key(self, get):
        result = news.get_news(["AAPL"])
        self.assertFalse(result["available"])
        self.assertTrue(result["errors"])
        self.assertNotIn("offline-test-key", str(result))

    @patch.dict(os.environ, {"NEWSAPI_KEY": "offline-test-key"})
    @patch("servers.tools.news.requests.get")
    def test_empty_results_are_unavailable(self, get):
        get.return_value = FakeResponse({"status": "ok", "articles": []})
        result = news.get_news(["AAPL"])
        self.assertFalse(result["available"])
        self.assertEqual(result["article_count"], 0)
        self.assertTrue(result["warnings"])

    @patch.dict(os.environ, {"NEWSAPI_KEY": "offline-test-key"})
    @patch("servers.tools.news.requests.get")
    def test_malformed_response_is_normalized(self, get):
        get.return_value = FakeResponse({"status": "ok", "articles": None})
        result = news.get_news(["AAPL"])
        self.assertTrue(result["degraded"])
        self.assertIn("NewsAPI response did not contain a valid articles list.", result["warnings"])

    @patch.dict(os.environ, {"NEWSAPI_KEY": "offline-test-key"})
    @patch("trading_system.servers.tools.news.requests.get")
    def test_offline_fixture_uses_mocked_response(self, get):
        get.return_value = FakeResponse({"status": "ok", "articles": [{
            "title": "AAPL announces product news",
            "source": {"name": "Fixture"},
            "publishedAt": "2026-10-07T10:00:00Z",
            "url": "https://fixture.test/aapl",
        }]})
        result = news.get_news(["AAPL"])
        self.assertEqual(result["headlines"][0]["symbol"], "AAPL")
        get.assert_called_once()

    @patch("servers.tools.news._load_finbert", side_effect=RuntimeError("offline"))
    def test_finbert_unavailable_is_not_neutral(self, load_finbert):
        result = news.score_headlines([{"symbol": "AAPL", "title": "AAPL update"}])
        self.assertIsNone(result["news_score"])
        self.assertFalse(result["sentiment_available"])
        self.assertEqual(result["scores"][0]["sentiment"], "unavailable")
        self.assertTrue(result["errors"])

    @patch("servers.tools.news._load_finbert")
    def test_finbert_inference_failure_keeps_signal_unavailable(self, load_finbert):
        load_finbert.return_value = lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError())
        result = news.score_headlines(["AAPL update"])
        self.assertIsNone(result["scores"][0]["sentiment_score"])
        self.assertIsNone(result["news_score"])
        self.assertFalse(result["sentiment_available"])
        self.assertTrue(result["warnings"])

    @patch("servers.tools.news._load_finbert")
    def test_positive_sentiment_is_aggregated(self, load_finbert):
        load_finbert.return_value = prediction_pipeline(
            {"positive": 0.7, "negative": 0.1, "neutral": 0.2})
        result = news.score_headlines([{"symbol": "AAPL", "title": "AAPL rises"}])
        self.assertEqual(result["news_score"], 0.6)
        self.assertEqual(result["sentiment"], "positive")

    @patch("servers.tools.news._load_finbert")
    def test_negative_sentiment_is_aggregated(self, load_finbert):
        load_finbert.return_value = prediction_pipeline(
            {"positive": 0.05, "negative": 0.85, "neutral": 0.10})
        result = news.score_headlines([{"symbol": "AAPL", "title": "AAPL plunges"}])
        self.assertAlmostEqual(result["news_score"], -0.8)
        self.assertEqual(result["sentiment"], "negative")

    @patch("servers.tools.news._load_finbert")
    def test_genuine_neutral_is_zero_and_available(self, load_finbert):
        load_finbert.return_value = prediction_pipeline(
            {"positive": 0.1, "negative": 0.1, "neutral": 0.8})
        result = news.score_headlines([{"symbol": "AAPL", "title": "AAPL unchanged"}])
        self.assertEqual(result["news_score"], 0.0)
        self.assertTrue(result["sentiment_available"])
        self.assertEqual(result["sentiment"], "neutral")

    def test_no_news_is_unavailable_not_neutral(self):
        result = news.score_headlines([])
        self.assertIsNone(result["news_score"])
        self.assertFalse(result["sentiment_available"])
        self.assertEqual(result["sentiment"], "unavailable")

    @patch("servers.tools.news._load_finbert")
    def test_duplicates_and_malformed_headlines_are_filtered(self, load_finbert):
        load_finbert.return_value = prediction_pipeline(
            {"positive": 0.6, "negative": 0.2, "neutral": 0.2})
        article = {"symbol": "AAPL", "title": "AAPL gains", "url": "https://fixture.test/story"}
        result = news.score_headlines([article, article, None, {"symbol": "AAPL"}])
        self.assertEqual(result["article_count"], 1)
        self.assertEqual(result["news_score"], 0.4)
        self.assertTrue(any("Duplicate" in warning for warning in result["warnings"]))
        self.assertTrue(any("malformed" in warning or "without valid" in warning
                            for warning in result["warnings"]))

    @patch("servers.tools.news._load_finbert")
    def test_m4_signal_integrates_with_existing_m2b_news_score_contract(self, load_finbert):
        load_finbert.return_value = prediction_pipeline(
            {"positive": 0.7, "negative": 0.1, "neutral": 0.2})
        sentiment = news.score_headlines([{"symbol": "AAPL", "title": "AAPL rises"}])
        with_news = score_indicators(
            "AAPL", "1Day", "core", "Technology", m2a_result(),
            news_score=sentiment["news_score"])
        no_news = score_indicators(
            "AAPL", "1Day", "core", "Technology", m2a_result(), news_score=None)
        without_news_component = score_indicators(
            "AAPL", "1Day", "core", "Technology", m2a_result(), news_score=0.0)
        self.assertEqual(with_news["sub_scores"]["news"], sentiment["news_score"])
        self.assertAlmostEqual(
            with_news["composite"] - without_news_component["composite"],
            WEIGHTS["core"]["news"] * sentiment["news_score"])
        self.assertIsNone(no_news["sub_scores"]["news"])
        self.assertIsNone(no_news["composite"])

    @patch("servers.tools.news._load_finbert", side_effect=RuntimeError("offline"))
    def test_unavailable_news_remains_missing_in_m2b(self, load_finbert):
        sentiment = news.score_headlines([{"symbol": "AAPL", "title": "AAPL update"}])
        result = score_indicators(
            "AAPL", "1Day", "core", "Technology", m2a_result(),
            news_score=sentiment["news_score"])
        self.assertIsNone(sentiment["news_score"])
        self.assertIsNone(result["sub_scores"]["news"])
        self.assertIsNone(result["composite"])

    def test_news_registration_uses_the_supplied_mcp_instance(self):
        registered = []

        class FakeMCP:
            def tool(self):
                return lambda function: registered.append(function.__name__) or function

        news.register_news_tools(FakeMCP())
        self.assertEqual(registered, ["get_news", "score_headlines"])

    def test_main_server_imports_with_m4_registered_on_existing_instance(self):
        created = []

        class FakeFastMCP:
            def __init__(self, name):
                self.name = name
                self.tools = {}
                created.append(self)

            def tool(self):
                return lambda function: self.tools.setdefault(function.__name__, function)

        mcp_package = ModuleType("mcp")
        mcp_package.__path__ = []
        server_package = ModuleType("mcp.server")
        server_package.__path__ = []
        fastmcp_module = ModuleType("mcp.server.fastmcp")
        fastmcp_module.FastMCP = FakeFastMCP

        server_path = PROJECT_DIR / "servers" / "technical_indicators_mcp_server_multiStocks.py"
        spec = importlib.util.spec_from_file_location("mcp_server_registration_test", server_path)
        server = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {
            "mcp": mcp_package,
            "mcp.server": server_package,
            "mcp.server.fastmcp": fastmcp_module,
        }):
            spec.loader.exec_module(server)
        self.assertEqual(server.mcp.name, "TechnicalIndicators")
        self.assertEqual(created, [server.mcp])
        self.assertIn("get_news", server.mcp.tools)
        self.assertIn("score_headlines", server.mcp.tools)


if __name__ == "__main__":
    unittest.main()