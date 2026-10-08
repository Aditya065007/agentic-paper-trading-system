"""Deterministic aggregation of M2-A indicators into architecture score rows."""

import math
from collections.abc import Mapping
from numbers import Real


WEIGHTS = {
    "core": {
        "trend": 0.30,
        "momentum": 0.20,
        "volume": 0.15,
        "mean_reversion": 0.05,
        "breakout_rs": 0.15,
        "news": 0.15,
    },
    "high_beta": {
        "trend": 0.25,
        "momentum": 0.25,
        "volume": 0.20,
        "mean_reversion": 0.05,
        "breakout_rs": 0.15,
        "news": 0.10,
    },
}

GROUP_INDICATORS = {
    "trend": ("ema_spread", "adx_direction", "supertrend"),
    "momentum": ("rsi14", "macd_histogram", "stochastic_rsi"),
    "volume": ("obv_slope", "cmf20", "relative_volume"),
    "mean_reversion": ("bollinger_percent_b", "zscore20"),
}

TRIAGE_THRESHOLDS = {
    "1Day": {"buy": 0.40, "sell": -0.40},
    "1Hour": {"buy": 0.45, "sell": -0.45},
}
BORDERLINE_BUFFER = 0.10
VERIFY_WARNINGS = (
    "[VERIFY] Equal-mean aggregation within scoring groups has no approved intra-group weights.",
    "[VERIFY] Missing weighted groups block the composite; confirm this strict missing-data policy.",
    "[VERIFY] Borderline covers the 0.10 interval immediately inside each clear threshold; clear thresholds take precedence.",
)


def _finite(value) -> bool:
    return (isinstance(value, Real) and not isinstance(value, bool)
            and math.isfinite(value))


def _as_mapping(value):
    if isinstance(value, Mapping):
        return value
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        converted = to_dict()
        return converted if isinstance(converted, Mapping) else None
    return None


def _reading(indicators: Mapping, name: str) -> tuple[float | None, object, bool]:
    item = indicators.get(name)
    item = _as_mapping(item)
    if item is None:
        return None, None, False
    raw = item.get("raw")
    normalized = item.get("normalized")
    available = item.get("available", raw is not None)
    if available is not True or not _finite(normalized) or not -1.0 <= normalized <= 1.0:
        return None, raw, False
    return float(normalized), raw, True


def _raw_adx(indicators: Mapping) -> float | None:
    item = _as_mapping(indicators.get("adx_direction"))
    if item is None or item.get("available", item.get("raw") is not None) is not True:
        return None
    raw = item.get("raw")
    if not isinstance(raw, Mapping):
        return None
    adx = raw.get("adx")
    return float(adx) if _finite(adx) and adx >= 0 else None


def classify_regime(adx) -> tuple[str | None, float | None]:
    """Return ADX-based regime and t=clip((ADX-20)/5, 0, 1)."""
    if not _finite(adx) or adx < 0:
        return None, None
    strength = max(0.0, min(1.0, (float(adx) - 20.0) / 5.0))
    if adx <= 20:
        return "ranging", strength
    if adx >= 25:
        return "trending", strength
    return "blend", strength


def classify_triage(composite, timeframe: str) -> str:
    if not _finite(composite) or timeframe not in TRIAGE_THRESHOLDS:
        return "hold"
    thresholds = TRIAGE_THRESHOLDS[timeframe]
    if composite >= thresholds["buy"]:
        return "clear_buy"
    if composite <= thresholds["sell"]:
        return "clear_sell"
    near_buy = (composite + BORDERLINE_BUFFER >= thresholds["buy"]
                or math.isclose(composite + BORDERLINE_BUFFER, thresholds["buy"], abs_tol=1e-12))
    near_sell = (composite - BORDERLINE_BUFFER <= thresholds["sell"]
                 or math.isclose(composite - BORDERLINE_BUFFER, thresholds["sell"], abs_tol=1e-12))
    if near_buy or near_sell:
        return "borderline"
    return "hold"


def _lean(value) -> str:
    if not _finite(value):
        return "NEUTRAL"
    if value > 0:
        return "BULLISH"
    if value < 0:
        return "BEARISH"
    return "NEUTRAL"


def _indicator_groups(timeframe: str) -> dict[str, tuple[str, ...]]:
    breakout = ["donchian20"]
    breakout.append("gap_percent" if timeframe == "1Day" else "vwap")
    breakout.append("relative_strength_spy")
    return {**GROUP_INDICATORS, "breakout_rs": tuple(breakout)}


def _group_score(names, indicators: Mapping, supplied_news_score=None) -> tuple[float | None, list[str]]:
    if names == ("news",):
        if _finite(supplied_news_score) and -1.0 <= supplied_news_score <= 1.0:
            return float(supplied_news_score), []
        return None, ["news"]

    values = []
    missing = []
    for name in names:
        value, _, available = _reading(indicators, name)
        if available:
            values.append(value)
        else:
            missing.append(name)
    if missing:
        return None, missing
    return sum(values) / len(values), []


def _normalize_result(indicator_result):
    result = _as_mapping(indicator_result)
    if result is None:
        return None, None
    indicators = _as_mapping(result.get("indicators"))
    context = _as_mapping(result.get("context"))
    return result, (indicators or {}, context or {})


def _context_reading(context: Mapping, name: str):
    item = _as_mapping(context.get(name))
    if item is None or item.get("available", item.get("raw") is not None) is not True:
        return None
    raw = item.get("raw")
    if isinstance(raw, bool):
        return raw
    return float(raw) if _finite(raw) else None


def _iso_timestamp(value):
    if isinstance(value, str):
        return value
    isoformat = getattr(value, "isoformat", None)
    return isoformat() if callable(isoformat) else None


def _base_row(symbol, timeframe, sleeve, sector, as_of):
    return {
        "symbol": str(symbol).strip().upper() if symbol is not None else "",
        "timeframe": timeframe,
        "as_of": _iso_timestamp(as_of),
        "sleeve": sleeve,
        "sector": sector,
        "regime": None,
        "regime_strength": None,
        "adx": None,
        "sub_scores": {
            "trend": None,
            "momentum": None,
            "volume": None,
            "mean_reversion": None,
            "breakout_rs": None,
            "news": None,
        },
        "composite": None,
        "triage": "hold",
        "leans": {"rsi": "NEUTRAL", "macd": "NEUTRAL", "trend": "NEUTRAL"},
        "atr": None,
        "natr": None,
        "bb_squeeze": None,
        "last_close": None,
        "missing_indicators": [],
        "warnings": [],
    }


def score_indicators(symbol: str, timeframe: str, sleeve: str, sector: str,
                     indicator_result, config=None, news_score=None) -> dict:
    """Convert one M2-A result to deterministic group scores and triage.

    Missing group policy is strict: if any weighted group is unavailable, the
    composite is unavailable rather than zero-filled or renormalized.
    """
    normalized_input, unpacked = _normalize_result(indicator_result)
    result = _base_row(symbol, timeframe, sleeve, sector,
                       normalized_input.get("as_of") if normalized_input else None)
    result["warnings"].extend(VERIFY_WARNINGS)

    if not isinstance(sleeve, str) or sleeve not in WEIGHTS:
        result["warnings"].append("sleeve must be 'core' or 'high_beta'; scores are unavailable.")
        return result
    if timeframe not in TRIAGE_THRESHOLDS:
        result["warnings"].append("timeframe must be '1Day' or '1Hour'; scores are unavailable.")
        return result
    if normalized_input is None or unpacked is None:
        result["warnings"].append("M2-A indicator result is missing or invalid.")
        result["missing_indicators"] = list(dict.fromkeys(
            name for names in _indicator_groups(timeframe).values() for name in names
        )) + ["news"]
        return result

    expected_symbol = str(symbol).strip().upper() if symbol is not None else ""
    result_symbol = normalized_input.get("symbol")
    result_timeframe = normalized_input.get("timeframe")
    if isinstance(result_symbol, str) and result_symbol.strip().upper() != expected_symbol:
        result["warnings"].append("M2-A symbol does not match the requested symbol; scoring was blocked.")
        result["missing_indicators"] = ["m2a_symbol_match"]
        return result
    if isinstance(result_timeframe, str) and result_timeframe != timeframe:
        result["warnings"].append("M2-A timeframe does not match the requested timeframe; scoring was blocked.")
        result["missing_indicators"] = ["m2a_timeframe_match"]
        return result

    indicators, context = unpacked
    if news_score is None and isinstance(config, Mapping):
        news_score = config.get("news_score")
    if not indicators:
        result["warnings"].append("M2-A indicator result has no indicator readings.")
    groups = _indicator_groups(timeframe)
    source_missing = normalized_input.get("missing_indicators", [])
    missing = {name for name in source_missing if isinstance(name, str)} \
        if isinstance(source_missing, (list, tuple, set)) else set()
    for group_name, indicator_names in groups.items():
        sub_score, unavailable = _group_score(indicator_names, indicators)
        result["sub_scores"][group_name] = sub_score
        missing.update(unavailable)
        if sub_score is None:
            result["warnings"].append(f"Scoring group '{group_name}' is unavailable.")

    news_value, news_missing = _group_score(("news",), indicators, news_score)
    result["sub_scores"]["news"] = news_value
    missing.update(news_missing)
    if news_value is None:
        result["warnings"].append("News score is unavailable; no news score was fabricated or fetched.")

    active_indicators = {name for names in groups.values() for name in names}
    active_indicators.add("news")
    result["missing_indicators"] = sorted(name for name in missing if name in active_indicators)

    weights = WEIGHTS[sleeve]
    if all(_finite(result["sub_scores"][group]) for group in weights):
        result["composite"] = sum(weights[group] * result["sub_scores"][group]
                                   for group in weights)
        result["triage"] = classify_triage(result["composite"], timeframe)
    else:
        result["warnings"].append(
            "Composite unavailable because one or more weighted scoring groups are missing; no renormalization applied.")

    adx = _raw_adx(indicators)
    result["adx"] = adx
    result["regime"], result["regime_strength"] = classify_regime(adx)
    if result["regime"] is None:
        result["missing_indicators"] = sorted(set(result["missing_indicators"]) | {"adx_direction"})
        result["warnings"].append("ADX unavailable; regime classification is unavailable.")

    rsi_normalized, _, rsi_available = _reading(indicators, "rsi14")
    macd_normalized, _, macd_available = _reading(indicators, "macd_histogram")
    result["leans"] = {
        "rsi": _lean(rsi_normalized) if rsi_available else "NEUTRAL",
        "macd": _lean(macd_normalized) if macd_available else "NEUTRAL",
        "trend": _lean(result["sub_scores"]["trend"]),
    }

    result["atr"] = _context_reading(context, "atr")
    result["natr"] = _context_reading(context, "natr")
    result["bb_squeeze"] = _context_reading(context, "bb_squeeze")
    missing_context = {name for name in ("atr", "natr", "bb_squeeze")
                       if result[name] is None}
    if missing_context:
        result["missing_indicators"] = sorted(
            set(result["missing_indicators"]) | missing_context)
    last_close = normalized_input.get("last_close")
    result["last_close"] = float(last_close) if _finite(last_close) else None
    if result["last_close"] is None:
        result["warnings"].append("last_close is not provided by the M2-A indicator result.")
    return result