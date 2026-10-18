"""G2: scoring weights come from config/weights.yaml and blend by ADX regime. Fully offline."""

import sys
from pathlib import Path

import pytest
import yaml

REPOSITORY_DIR = Path(__file__).resolve().parents[2]
if str(REPOSITORY_DIR) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_DIR))

from trading_system.servers.tools import scoring  # noqa: E402

YAML_PATH = Path(__file__).resolve().parents[1] / "config" / "weights.yaml"
GROUPS = scoring.GROUPS


def m2a(adx, value=0.4):
    """Same-shape M2-A result for every group, with a chosen ADX; mean_reversion differs."""
    def reading(v):
        return {"raw": v, "normalized": v, "available": True}
    names = ["ema_spread", "supertrend", "rsi14", "macd_histogram", "stochastic_rsi", "obv_slope",
             "cmf20", "relative_volume", "donchian20", "gap_percent", "relative_strength_spy"]
    indicators = {n: reading(value) for n in names}
    indicators["bollinger_percent_b"] = reading(-0.6)
    indicators["zscore20"] = reading(-0.6)
    indicators["adx_direction"] = ({"raw": {"adx": adx}, "normalized": value, "available": True}
                                   if adx is not None else
                                   {"raw": None, "normalized": None, "available": False})
    return {"symbol": "AAPL", "timeframe": "1Day", "as_of": "2026-10-08T12:00:00+00:00",
            "last_close": 190.1, "indicators": indicators,
            "context": {"atr": {"raw": 3.2, "available": True},
                        "natr": {"raw": 1.7, "available": True},
                        "bb_squeeze": {"raw": False, "available": True}},
            "missing_indicators": [], "warnings": []}


def score(adx, sleeve="core", news=0.2):
    return scoring.score_indicators("AAPL", "1Day", sleeve, "Technology", m2a(adx), news_score=news)


def test_shipped_yaml_loads_cleanly_and_matches_builtin_defaults():
    config, warning = scoring.load_weight_config(YAML_PATH)
    assert warning is None
    assert config == scoring.DEFAULT_WEIGHT_CONFIG
    assert scoring.WEIGHT_CONFIG_WARNING is None


def test_every_column_sums_to_one_and_matches_spec_8_3():
    config, _ = scoring.load_weight_config(YAML_PATH)
    for sleeve in ("core", "high_beta"):
        for column in ("trending", "ranging"):
            assert sum(config[sleeve][column].values()) == pytest.approx(1.0)
    assert config["core"]["ranging"]["mean_reversion"] == 0.30
    assert config["high_beta"]["ranging"]["mean_reversion"] == 0.35
    assert config["high_beta"]["ranging"]["trend"] == 0.10


def test_legacy_weights_constant_is_the_trending_column():
    assert scoring.WEIGHTS["core"]["trend"] == 0.30
    assert scoring.WEIGHTS["high_beta"]["news"] == 0.10


@pytest.mark.parametrize("sleeve", ["core", "high_beta"])
def test_blend_endpoints_and_midpoint(sleeve):
    cfg = scoring.WEIGHT_CONFIG[sleeve]
    assert scoring.blended_weights(sleeve, 27.0) == pytest.approx(cfg["trending"])
    assert scoring.blended_weights(sleeve, 25.0) == pytest.approx(cfg["trending"])
    assert scoring.blended_weights(sleeve, 20.0) == pytest.approx(cfg["ranging"])
    assert scoring.blended_weights(sleeve, 10.0) == pytest.approx(cfg["ranging"])
    mid = scoring.blended_weights(sleeve, 22.5)
    for group in GROUPS:
        assert mid[group] == pytest.approx((cfg["trending"][group] + cfg["ranging"][group]) / 2)
    assert sum(mid.values()) == pytest.approx(1.0)


@pytest.mark.parametrize("adx", [None, float("nan"), -3.0, "x"])
def test_unknown_adx_uses_trending_column(adx):
    assert scoring.blended_weights("core", adx) == scoring.WEIGHTS["core"]


def test_composite_follows_the_regime():
    ranging, trending = score(15.0), score(27.0)
    expect_r = sum(scoring.WEIGHT_CONFIG["core"]["ranging"][g] * ranging["sub_scores"][g] for g in GROUPS)
    expect_t = sum(scoring.WEIGHT_CONFIG["core"]["trending"][g] * trending["sub_scores"][g] for g in GROUPS)
    assert ranging["composite"] == pytest.approx(expect_r)
    assert trending["composite"] == pytest.approx(expect_t)
    assert ranging["composite"] < trending["composite"]  # mean reversion is negative here
    assert ranging["regime"] == "ranging" and trending["regime"] == "trending"


def test_blend_regime_row_uses_mixed_weights():
    row = score(22.5)
    mixed = scoring.blended_weights("core", 22.5)
    assert row["regime"] == "blend"
    assert row["composite"] == pytest.approx(sum(mixed[g] * row["sub_scores"][g] for g in GROUPS))


def test_missing_news_still_renormalizes_under_blend():
    with_none = score(15.0, news=None)
    w = scoring.blended_weights("core", 15.0)
    expect = sum(w[g] * with_none["sub_scores"][g] for g in GROUPS if g != "news") / (1 - w["news"])
    assert with_none["composite"] == pytest.approx(expect)


def test_regime_blend_false_always_trending(tmp_path):
    raw = yaml.safe_load(YAML_PATH.read_text())
    raw["regime_blend"] = False
    path = tmp_path / "w.yaml"
    path.write_text(yaml.safe_dump(raw))
    config, warning = scoring.load_weight_config(path)
    assert warning is None
    assert scoring.blended_weights("core", 10.0, config) == config["core"]["trending"]


def write_and_load(tmp_path, text):
    path = tmp_path / "w.yaml"
    path.write_text(text)
    return scoring.load_weight_config(path)


def test_bad_files_fall_back_with_a_warning(tmp_path):
    good = yaml.safe_load(YAML_PATH.read_text())
    bad_sum = yaml.safe_load(YAML_PATH.read_text())
    bad_sum["core"]["ranging"]["trend"] = 0.5
    missing_group = yaml.safe_load(YAML_PATH.read_text())
    del missing_group["high_beta"]["trending"]["news"]
    negative = yaml.safe_load(YAML_PATH.read_text())
    negative["core"]["trending"]["trend"] = -0.1
    negative["core"]["trending"]["news"] = 0.55
    for text in (yaml.safe_dump(bad_sum), yaml.safe_dump(missing_group), yaml.safe_dump(negative),
                 "regime_blend: maybe\ncore: 1\n", "not: [valid", "", "- a list"):
        config, warning = write_and_load(tmp_path, text)
        assert config == scoring.DEFAULT_WEIGHT_CONFIG
        assert warning and "built-in default" in warning
    assert scoring.load_weight_config(tmp_path / "absent.yaml")[1]
    assert scoring.load_weight_config(YAML_PATH)[0] == yaml.safe_load(yaml.safe_dump(good))


def test_fallback_warning_reaches_every_score_row(monkeypatch):
    monkeypatch.setattr(scoring, "WEIGHT_CONFIG_WARNING", "weights.yaml unusable (test); built-in default weights were used.")
    assert any("weights.yaml unusable" in w for w in score(27.0)["warnings"])


def test_clean_config_adds_no_weights_warning():
    assert not any("weights.yaml" in w for w in score(27.0)["warnings"])
