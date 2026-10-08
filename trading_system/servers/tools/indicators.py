"""Deterministic OHLCV indicators for later scoring; no trading decisions."""

import math
from dataclasses import asdict, dataclass, field
from numbers import Real
from typing import Any


INDICATOR_NAMES = (
    "ema_spread", "adx_direction", "supertrend", "rsi14", "macd_histogram",
    "stochastic_rsi", "obv_slope", "cmf20", "relative_volume",
    "bollinger_percent_b", "zscore20", "donchian20", "vwap", "gap_percent",
    "relative_strength_spy",
)
CONTEXT_NAMES = ("atr", "natr", "bb_width", "bb_squeeze")
REQUIRED_COLUMNS = ("open", "high", "low", "close", "volume")


@dataclass(frozen=True)
class IndicatorSettings:
    rsi_period: int = 14
    atr_period: int = 14
    ema_short_period: int = 20
    ema_long_period: int = 50
    adx_period: int = 14
    supertrend_period: int = 10
    supertrend_multiplier: float = 3.0
    macd_fast_period: int = 12
    macd_slow_period: int = 26
    macd_signal_period: int = 9
    stochastic_rsi_period: int = 14
    stochastic_smooth_k: int = 3
    stochastic_smooth_d: int = 3
    obv_slope_period: int = 20
    cmf_period: int = 20
    relative_volume_period: int = 20
    bollinger_period: int = 20
    bollinger_std: float = 2.0
    squeeze_lookback: int = 120
    squeeze_quantile: float = 0.2
    zscore_period: int = 20
    donchian_period: int = 20


@dataclass
class IndicatorValue:
    raw: Any = None
    normalized: float | None = None
    available: bool = False


@dataclass
class IndicatorResult:
    symbol: str
    timeframe: str
    as_of: str | None
    indicators: dict[str, IndicatorValue] = field(default_factory=dict)
    context: dict[str, IndicatorValue] = field(default_factory=dict)
    missing_indicators: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def _pd():
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("pandas is required for M2-A indicator calculations.") from exc
    return pd


def _finite(value) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value)


def _number(series):
    if series is None:
        return None

    if _finite(series):
        return float(series)

    try:
        if len(series) == 0:
            return None
    except TypeError:
        return None

    try:
        value = series.iloc[-1]
    except AttributeError:
        try:
            value = series[-1]
        except (IndexError, KeyError, TypeError):
            return None
    except (IndexError, KeyError, TypeError):
        return None
    return float(value) if _finite(value) else None


def _normal_tanh(value) -> float | None:
    return math.tanh(value) if _finite(value) else None


def normalize_ema_spread(ema_short, ema_long, atr) -> float | None:
    if not all(_finite(value) for value in (ema_short, ema_long, atr)) or atr <= 0:
        return None
    return math.tanh((ema_short - ema_long) / atr)


def normalize_adx_direction(adx, plus_di, minus_di) -> float | None:
    if not all(_finite(value) for value in (adx, plus_di, minus_di)):
        return None
    return math.tanh((plus_di - minus_di) / 20.0) * min(max(adx / 20.0, 0.0), 1.0)


def normalize_macd_histogram(histogram, atr) -> float | None:
    if not all(_finite(value) for value in (histogram, atr)) or atr <= 0:
        return None
    return math.tanh(histogram / atr)


def normalize_obv_slope(slope, average_volume) -> float | None:
    if not all(_finite(value) for value in (slope, average_volume)) or average_volume <= 0:
        return None
    return math.tanh(slope / average_volume)


def normalize_cmf(value) -> float | None:
    return max(-1.0, min(1.0, value)) if _finite(value) else None


def normalize_relative_volume(value) -> float | None:
    if not _finite(value) or value < 0:
        return None
    if value == 0:
        return -1.0
    return math.tanh(math.log(value))


def normalize_bollinger_percent_b(value) -> float | None:
    if not _finite(value):
        return None
    return max(-1.0, min(1.0, (2.0 * value) - 1.0))


def normalize_zscore(value, clip: float = 3.0) -> float | None:
    if not _finite(value) or not _finite(clip) or clip <= 0:
        return None
    return max(-1.0, min(1.0, value / clip))


def normalize_donchian_position(value) -> float | None:
    if not _finite(value):
        return None
    return max(-1.0, min(1.0, value))


def normalize_relative_strength(asset_return, benchmark_return) -> float | None:
    if not all(_finite(value) for value in (asset_return, benchmark_return)):
        return None
    return math.tanh(asset_return - benchmark_return)


def _reading(raw=None, normalized=None) -> IndicatorValue:
    return IndicatorValue(raw=raw, normalized=normalized,
                          available=raw is not None)


def _empty_result(symbol, timeframe, warnings) -> IndicatorResult:
    return IndicatorResult(
        symbol=str(symbol).strip().upper(), timeframe=timeframe, as_of=None,
        indicators={name: IndicatorValue() for name in INDICATOR_NAMES},
        context={name: IndicatorValue() for name in CONTEXT_NAMES},
        missing_indicators=list(INDICATOR_NAMES) + list(CONTEXT_NAMES),
        warnings=list(warnings),
    )


def _prepare(frame, label: str):
    pd = _pd()
    warnings = []
    if not isinstance(frame, pd.DataFrame):
        return None, [f"{label} data must be a pandas DataFrame."]

    data = frame.copy(deep=True)
    data.columns = [str(column).strip().lower() for column in data.columns]
    missing = [column for column in REQUIRED_COLUMNS if column not in data.columns]
    if missing:
        return None, [f"{label} data is missing required columns: {', '.join(missing)}."]

    if "timestamp" in data.columns:
        timestamps = pd.to_datetime(data.pop("timestamp"), utc=True, errors="coerce")
    elif isinstance(data.index, pd.DatetimeIndex):
        timestamps = pd.to_datetime(data.index, utc=True, errors="coerce")
    else:
        return None, [f"{label} data requires a timestamp column or DatetimeIndex."]

    data.index = pd.DatetimeIndex(timestamps, name="timestamp")
    invalid_timestamp_count = int(data.index.isna().sum())
    if invalid_timestamp_count:
        warnings.append(f"{label} rows with invalid timestamps were discarded.")
    data = data.loc[~data.index.isna()].copy()
    data = data.sort_index(kind="stable")
    if data.index.has_duplicates:
        data = data.loc[~data.index.duplicated(keep="last")].copy()
        warnings.append(f"{label} duplicate timestamps were reduced to the last row.")

    for column in REQUIRED_COLUMNS:
        numeric = pd.to_numeric(data[column], errors="coerce")
        data[column] = numeric.where(numeric.map(_finite))
    invalid_values = data.loc[:, list(REQUIRED_COLUMNS)].isna().any(axis=1)
    if invalid_values.any():
        warnings.append(f"{label} invalid or missing OHLCV values were retained as missing.")
    return data, warnings


def _validate_settings(settings: IndicatorSettings) -> None:
    for name, value in vars(settings).items():
        if name.endswith("period") or name.endswith("lookback") or name.endswith("smooth_k") \
                or name.endswith("smooth_d"):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        elif not _finite(value) or value <= 0:
            raise ValueError(f"{name} must be a positive finite number.")
    if settings.macd_fast_period >= settings.macd_slow_period:
        raise ValueError("macd_fast_period must be less than macd_slow_period.")
    if not 0 < settings.squeeze_quantile < 1:
        raise ValueError("squeeze_quantile must be between zero and one.")


def _wilder(series, period: int):
    return series.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()


def _true_range(high, low, close):
    pd = _pd()
    previous = close.shift(1)
    ranges = pd.concat((high.sub(low), high.sub(previous).abs(),
                        low.sub(previous).abs()), axis=1)
    return ranges.max(axis=1)


def _atr(data, period: int):
    return _wilder(_true_range(data["high"], data["low"], data["close"]), period)


def _rsi(close, period: int):
    delta = close.diff()
    gains = _wilder(delta.clip(lower=0), period)
    losses = _wilder((-delta).clip(lower=0), period)
    rs = gains.div(losses.where(losses != 0))
    result = 100.0 - (100.0 / (1.0 + rs))
    result = result.mask((losses == 0) & (gains > 0), 100.0)
    result = result.mask((gains == 0) & (losses > 0), 0.0)
    result = result.mask((gains == 0) & (losses == 0), 50.0)
    return result


def _directional_movement(data, period: int):
    high, low, close = data["high"], data["low"], data["close"]
    up = high.diff()
    down = -low.diff()
    plus_dm = up.where((up > down) & (up > 0), 0.0)
    minus_dm = down.where((down > up) & (down > 0), 0.0)
    smoothed_tr = _wilder(_true_range(high, low, close), period)
    plus_di = 100.0 * _wilder(plus_dm, period).div(smoothed_tr.where(smoothed_tr != 0))
    minus_di = 100.0 * _wilder(minus_dm, period).div(smoothed_tr.where(smoothed_tr != 0))
    denominator = (plus_di + minus_di).where((plus_di + minus_di) != 0)
    dx = 100.0 * (plus_di - minus_di).abs().div(denominator)
    adx = _wilder(dx, period)
    return adx, plus_di, minus_di


def _supertrend(data, period: int, multiplier: float):
    pd = _pd()
    trend_atr = _atr(data, period)
    midpoint = (data["high"] + data["low"]) / 2.0
    upper_basic = midpoint + multiplier * trend_atr
    lower_basic = midpoint - multiplier * trend_atr
    upper = pd.Series(float("nan"), index=data.index)
    lower = pd.Series(float("nan"), index=data.index)
    direction = pd.Series(float("nan"), index=data.index)
    line = pd.Series(float("nan"), index=data.index)

    for position in range(len(data)):
        if not _finite(trend_atr.iloc[position]) or trend_atr.iloc[position] <= 0:
            continue
        current_upper = upper_basic.iloc[position]
        current_lower = lower_basic.iloc[position]
        if not _finite(current_upper) or not _finite(current_lower):
            continue
        if position == 0 or not _finite(upper.iloc[position - 1]):
            upper.iloc[position] = current_upper
            lower.iloc[position] = current_lower
            direction.iloc[position] = 1.0 if data["close"].iloc[position] >= midpoint.iloc[position] else -1.0
        else:
            previous_close = data["close"].iloc[position - 1]
            previous_upper = upper.iloc[position - 1]
            previous_lower = lower.iloc[position - 1]
            upper.iloc[position] = (min(current_upper, previous_upper)
                                    if previous_close <= previous_upper else current_upper)
            lower.iloc[position] = (max(current_lower, previous_lower)
                                    if previous_close >= previous_lower else current_lower)
            previous_direction = direction.iloc[position - 1]
            if previous_direction > 0:
                direction.iloc[position] = -1.0 if data["close"].iloc[position] < lower.iloc[position] else 1.0
            else:
                direction.iloc[position] = 1.0 if data["close"].iloc[position] > upper.iloc[position] else -1.0
        line.iloc[position] = lower.iloc[position] if direction.iloc[position] > 0 else upper.iloc[position]
    return line, direction


def _obv(close, volume):
    delta = close.diff()
    direction = delta.map(lambda value: 1.0 if value > 0 else -1.0 if value < 0 else 0.0)
    direction = direction.where(delta.notna())
    direction.iloc[0] = 0.0
    return (direction * volume).cumsum()


def _linear_slope(values) -> float | None:
    if len(values) < 2 or values.isna().any():
        return None
    x_mean = (len(values) - 1) / 2.0
    y_mean = float(values.mean())
    denominator = sum((index - x_mean) ** 2 for index in range(len(values)))
    if denominator == 0:
        return None
    return sum((index - x_mean) * (float(value) - y_mean)
               for index, value in enumerate(values)) / denominator


def compute_indicators(frame, symbol: str, timeframe: str, spy_frame=None,
                       settings: IndicatorSettings | None = None) -> IndicatorResult:
    """Compute deterministic indicators for completed caller-supplied OHLCV bars."""
    if timeframe not in {"1Day", "1Hour"}:
        raise ValueError("timeframe must be '1Day' or '1Hour'.")
    settings = settings or IndicatorSettings()
    _validate_settings(settings)
    result = _empty_result(symbol, timeframe, [])
    data, warnings = _prepare(frame, "OHLCV")
    result.warnings.extend(warnings)
    if data is None or data.empty:
        if data is not None:
            result.warnings.append("OHLCV data is empty after timestamp validation.")
        return result

    result.as_of = data.index[-1].isoformat()
    if data.loc[data.index[-1], list(REQUIRED_COLUMNS)].isna().any():
        result.warnings.append("Latest OHLCV bar is incomplete; current indicator readings are unavailable.")
        return result
    indicators = result.indicators
    context = result.context
    close, high, low, volume = (data[name] for name in REQUIRED_COLUMNS[1:])

    atr = _atr(data, settings.atr_period)
    atr_value = _number(atr)
    close_value = _number(close)
    natr = (100.0 * atr_value / close_value
            if atr_value is not None and close_value not in (None, 0.0) else None)
    context["atr"] = _reading(atr_value)
    context["natr"] = _reading(natr)

    ema_short = _number(close.ewm(span=settings.ema_short_period, min_periods=settings.ema_short_period,
                                  adjust=False).mean())
    ema_long = _number(close.ewm(span=settings.ema_long_period, min_periods=settings.ema_long_period,
                                 adjust=False).mean())
    indicators["ema_spread"] = _reading(
        {"ema_short": ema_short, "ema_long": ema_long} if ema_short is not None and ema_long is not None else None,
        normalize_ema_spread(ema_short, ema_long, atr_value))

    adx, plus_di, minus_di = _directional_movement(data, settings.adx_period)
    adx_value, plus_value, minus_value = _number(adx), _number(plus_di), _number(minus_di)
    indicators["adx_direction"] = _reading(
        {"adx": adx_value, "plus_di": plus_value, "minus_di": minus_value}
        if all(value is not None for value in (adx_value, plus_value, minus_value)) else None,
        normalize_adx_direction(adx_value, plus_value, minus_value))

    supertrend_line, supertrend_direction = _supertrend(
        data, settings.supertrend_period, settings.supertrend_multiplier)
    st_line, st_direction = _number(supertrend_line), _number(supertrend_direction)
    indicators["supertrend"] = _reading(
        {"line": st_line, "direction": int(st_direction)} if st_line is not None and st_direction is not None else None,
        float(int(st_direction)) if st_direction is not None else None)

    rsi = _rsi(close, settings.rsi_period)
    rsi_value = _number(rsi)
    indicators["rsi14"] = _reading(rsi_value,
                                   max(-1.0, min(1.0, (rsi_value - 50.0) / 50.0))
                                   if rsi_value is not None else None)

    macd_line = (close.ewm(span=settings.macd_fast_period, min_periods=settings.macd_fast_period,
                           adjust=False).mean()
                 - close.ewm(span=settings.macd_slow_period, min_periods=settings.macd_slow_period,
                             adjust=False).mean())
    macd_signal = macd_line.ewm(span=settings.macd_signal_period,
                                min_periods=settings.macd_signal_period, adjust=False).mean()
    histogram = _number(macd_line - macd_signal)
    indicators["macd_histogram"] = _reading(histogram, normalize_macd_histogram(histogram, atr_value))

    stoch_low = rsi.rolling(settings.stochastic_rsi_period,
                            min_periods=settings.stochastic_rsi_period).min()
    stoch_high = rsi.rolling(settings.stochastic_rsi_period,
                             min_periods=settings.stochastic_rsi_period).max()
    stoch_range = (stoch_high - stoch_low).where((stoch_high - stoch_low) != 0)
    stoch_raw = 100.0 * (rsi - stoch_low).div(stoch_range)
    stoch_k = stoch_raw.rolling(settings.stochastic_smooth_k,
                                min_periods=settings.stochastic_smooth_k).mean()
    stoch_d = stoch_k.rolling(settings.stochastic_smooth_d,
                              min_periods=settings.stochastic_smooth_d).mean()
    k_value, d_value = _number(stoch_k), _number(stoch_d)
    indicators["stochastic_rsi"] = _reading(
        {"k": k_value, "d": d_value} if k_value is not None and d_value is not None else None,
        max(-1.0, min(1.0, (k_value - 50.0) / 50.0)) if k_value is not None else None)

    obv = _obv(close, volume)
    obv_window = obv.tail(settings.obv_slope_period)
    volume_window = volume.tail(settings.obv_slope_period)
    obv_slope = _linear_slope(obv_window) if len(obv_window) == settings.obv_slope_period else None
    average_volume = _number(volume_window.mean()) if len(volume_window) == settings.obv_slope_period else None
    indicators["obv_slope"] = _reading(obv_slope, normalize_obv_slope(obv_slope, average_volume))

    spread = (close - low) - (high - close)
    high_low_range = (high - low).where((high - low) != 0)
    money_flow_multiplier = spread.div(high_low_range)
    money_flow_volume = money_flow_multiplier * volume
    cmf_numerator = money_flow_volume.rolling(settings.cmf_period,
                                              min_periods=settings.cmf_period).sum()
    cmf_denominator = volume.rolling(settings.cmf_period,
                                     min_periods=settings.cmf_period).sum()
    cmf_value = _number(cmf_numerator.div(cmf_denominator.where(cmf_denominator != 0)))
    indicators["cmf20"] = _reading(cmf_value, normalize_cmf(cmf_value))

    baseline_volume = volume.shift(1).rolling(settings.relative_volume_period,
                                              min_periods=settings.relative_volume_period).mean()
    relative_volume = _number(volume.div(baseline_volume.where(baseline_volume > 0)))
    indicators["relative_volume"] = _reading(relative_volume,
                                             normalize_relative_volume(relative_volume))

    bb_middle = close.rolling(settings.bollinger_period,
                              min_periods=settings.bollinger_period).mean()
    bb_deviation = close.rolling(settings.bollinger_period,
                                 min_periods=settings.bollinger_period).std(ddof=0)
    bb_upper = bb_middle + settings.bollinger_std * bb_deviation
    bb_lower = bb_middle - settings.bollinger_std * bb_deviation
    bb_width_series = (bb_upper - bb_lower).div(bb_middle.where(bb_middle != 0))
    bb_width = _number(bb_width_series)
    context["bb_width"] = _reading(bb_width)
    percent_b = _number((close - bb_lower).div((bb_upper - bb_lower).where((bb_upper - bb_lower) != 0)))
    indicators["bollinger_percent_b"] = _reading(percent_b, normalize_bollinger_percent_b(percent_b))

    squeeze_floor = bb_width_series.shift(1).rolling(
        settings.squeeze_lookback, min_periods=settings.squeeze_lookback
    ).quantile(settings.squeeze_quantile)
    squeeze_value = None
    current_width = _number(bb_width_series)
    current_floor = _number(squeeze_floor)
    if current_width is not None and current_floor is not None:
        squeeze_value = bool(current_width <= current_floor)
    context["bb_squeeze"] = _reading(squeeze_value)

    rolling_mean = close.rolling(settings.zscore_period,
                                 min_periods=settings.zscore_period).mean()
    rolling_std = close.rolling(settings.zscore_period,
                                min_periods=settings.zscore_period).std(ddof=0)
    zscore = _number((close - rolling_mean).div(rolling_std.where(rolling_std != 0)))
    indicators["zscore20"] = _reading(zscore, normalize_zscore(zscore))

    donchian_upper = high.shift(1).rolling(settings.donchian_period,
                                           min_periods=settings.donchian_period).max()
    donchian_lower = low.shift(1).rolling(settings.donchian_period,
                                          min_periods=settings.donchian_period).min()
    upper_value, lower_value = _number(donchian_upper), _number(donchian_lower)
    donchian_position = None
    if upper_value is not None and lower_value is not None and upper_value > lower_value and close_value is not None:
        donchian_position = 2.0 * ((close_value - lower_value) / (upper_value - lower_value)) - 1.0
    indicators["donchian20"] = _reading(
        {"upper": upper_value, "lower": lower_value, "position": donchian_position}
        if donchian_position is not None else None,
        normalize_donchian_position(donchian_position))

    if timeframe == "1Day":
        previous_close = _number(close.iloc[:-1])
        open_value = _number(data["open"])
        gap_percent = ((open_value / previous_close) - 1.0) * 100.0 \
            if open_value is not None and previous_close not in (None, 0.0) else None
        if not _finite(gap_percent):
            gap_percent = None
        indicators["gap_percent"] = _reading(
            gap_percent, math.tanh(gap_percent / 100.0) if gap_percent is not None else None)
    else:
        indicators["vwap"] = _hourly_vwap(data, atr_value)

    if spy_frame is not None:
        spy_data, spy_warnings = _prepare(spy_frame, "SPY")
        result.warnings.extend(spy_warnings)
        indicators["relative_strength_spy"] = _relative_strength(data, spy_data)
    else:
        result.warnings.append("SPY benchmark data not supplied; relative strength is unavailable.")

    for name, value in indicators.items():
        if not value.available:
            result.missing_indicators.append(name)
    for name, value in context.items():
        if not value.available:
            result.missing_indicators.append(name)
    result.missing_indicators = list(dict.fromkeys(result.missing_indicators))
    if result.missing_indicators:
        result.warnings.append("Some indicators are unavailable due to insufficient or invalid input data.")
    return result


def _hourly_vwap(data, atr_value) -> IndicatorValue:
    typical_price = (data["high"] + data["low"] + data["close"]) / 3.0
    session = data.index.date
    pv = typical_price * data["volume"]
    session_pv = pv.groupby(session).cumsum()
    session_volume = data["volume"].groupby(session).cumsum()
    vwap_series = session_pv.div(session_volume.where(session_volume > 0))
    value = _number(vwap_series)
    close_value = _number(data["close"])
    normalized = math.tanh((close_value - value) / atr_value) \
        if value is not None and close_value is not None and atr_value is not None and atr_value > 0 else None
    return _reading(value, normalized)


def _relative_strength(data, spy_data) -> IndicatorValue:
    if spy_data is None or spy_data.empty:
        return IndicatorValue()
    aligned = data["close"].rename("asset").to_frame().join(
        spy_data["close"].rename("spy"), how="inner").dropna()
    if len(aligned) < 2 or aligned.iloc[0]["asset"] == 0 or aligned.iloc[0]["spy"] == 0:
        return IndicatorValue()
    asset_return = float(aligned.iloc[-1]["asset"] / aligned.iloc[0]["asset"] - 1.0)
    spy_return = float(aligned.iloc[-1]["spy"] / aligned.iloc[0]["spy"] - 1.0)
    return _reading({"asset_return": asset_return, "spy_return": spy_return},
                    normalize_relative_strength(asset_return, spy_return))