"""Edge-case tests for the M2-B scoring missing-data policy."""

import unittest

from servers.tools.scoring import WEIGHTS, score_indicators


def m2a_result(missing=()):
    def reading(value):
        return {"raw": value, "normalized": value, "available": True}

    values = {
        "ema_spread": 0.2,
        "adx_direction": 0.4,
        "supertrend": 0.6,
        "rsi14": 0.1,
        "macd_histogram": 0.3,
        "stochastic_rsi": 0.5,
        "obv_slope": 0.2,
        "cmf20": 0.4,
        "relative_volume": 0.6,
        "bollinger_percent_b": -0.2,
        "zscore20": 0.0,
        "donchian20": 0.4,
        "gap_percent": 0.2,
        "relative_strength_spy": 0.6,
    }
    indicators = {
        name: reading(value) for name, value in values.items()
        if name not in set(missing)
    }
    indicators["adx_direction"] = {
        "raw": {"adx": 27.0}, "normalized": 0.4, "available": True
    } if "adx_direction" not in set(missing) else {
        "raw": None, "normalized": None, "available": False
    }
    return {
        "symbol": "AAPL",
        "timeframe": "1Day",
        "as_of": "2026-10-08T12:00:00+00:00",
        "last_close": 190.1,
        "indicators": indicators,
        "context": {
            "atr": {"raw": 3.2, "available": True},
            "natr": {"raw": 1.7, "available": True},
            "bb_squeeze": {"raw": False, "available": True},
        },
        "missing_indicators": [],
        "warnings": [],
    }


class MissingDataPolicyTests(unittest.TestCase):
    def score(self, *, sleeve="core", missing=(), news_score=0.2):
        return score_indicators(
            "AAPL", "1Day", sleeve, "Technology",
            m2a_result(missing), news_score=news_score,
        )

    def test_all_present_matches_configured_weighted_sum(self):
        result = self.score()
        expected = sum(
            WEIGHTS["core"][group] * result["sub_scores"][group]
            for group in WEIGHTS["core"]
        )
        self.assertAlmostEqual(result["composite"], expected)

    def test_missing_news_is_renormalized_not_zero_filled(self):
        missing = self.score(news_score=None)
        neutral = self.score(news_score=0.0)
        self.assertIsNone(missing["sub_scores"]["news"])
        self.assertAlmostEqual(
            missing["composite"],
            neutral["composite"] / (1 - WEIGHTS["core"]["news"]),
        )
        self.assertIn("news", missing["missing_indicators"])

    def test_more_than_30_percent_missing_forces_hold(self):
        result = self.score(missing=("ema_spread", "adx_direction", "supertrend",
                                     "rsi14", "macd_histogram", "stochastic_rsi"))
        self.assertIsNone(result["composite"])
        self.assertEqual(result["triage"], "hold")
        self.assertTrue(any("exceeds the 30% limit" in w for w in result["warnings"]))

    def test_exactly_30_percent_missing_still_computes(self):
        result = self.score(missing=("ema_spread", "adx_direction", "supertrend"))
        self.assertAlmostEqual(
            sum(WEIGHTS["core"][g] for g in ("trend",)), 0.30
        )
        self.assertIsNotNone(result["composite"])

    def test_everything_missing_forces_hold(self):
        all_names = (
            "ema_spread", "adx_direction", "supertrend",
            "rsi14", "macd_histogram", "stochastic_rsi",
            "obv_slope", "cmf20", "relative_volume",
            "bollinger_percent_b", "zscore20",
            "donchian20", "gap_percent", "relative_strength_spy",
        )
        result = self.score(missing=all_names, news_score=None)
        self.assertIsNone(result["composite"])
        self.assertEqual(result["triage"], "hold")

    def test_high_beta_uses_its_own_weights_and_renormalizes(self):
        result = self.score(sleeve="high_beta", news_score=None)
        expected = sum(
            WEIGHTS["high_beta"][group] * result["sub_scores"][group]
            for group in WEIGHTS["high_beta"] if group != "news"
        ) / (1 - WEIGHTS["high_beta"]["news"])
        self.assertAlmostEqual(result["composite"], expected)

    def test_one_missing_indicator_does_not_discard_its_group(self):
        result = self.score(missing=("ema_spread",))
        self.assertAlmostEqual(
            result["sub_scores"]["trend"], (0.4 + 0.6) / 2
        )
        self.assertIn("ema_spread", result["missing_indicators"])
        self.assertIsNotNone(result["composite"])


if __name__ == "__main__":
    unittest.main()
