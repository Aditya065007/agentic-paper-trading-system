# servers/tools/regime.py
# M3 - Market regime and events. Pure functions: numbers in, numbers out. No network, no MCP import
# (except register_regime_tools, which only receives an existing server object).
#
# compute_market_regime : SPY trend, SPY volatility, breadth -> risk_on / neutral / risk_off
# get_earnings_calendar : next earnings date per symbol from a manual list (fixtures/earnings.json)
#
# Bad input never raises: it gives "neutral" (regime) or None (earnings date) plus a warning.
# The architecture tool get_market_regime(timeframe) is a thin wrapper in the server file; it reads the
# cached SPY / watch-list bars and calls compute_market_regime. This module never fetches data.

import json
import math
from datetime import date, datetime, timezone
from pathlib import Path

_PACKAGE_DIR = Path(__file__).resolve().parents[2]
_FIXTURES = _PACKAGE_DIR / "fixtures"
_SETTINGS_PATH = _PACKAGE_DIR / "config" / "regime.yaml"

# Built-in copy of config/regime.yaml. Used only if the file is missing or invalid; a test keeps them equal.
DEFAULT_SETTINGS = {
    "spy_ema_period": 50,
    "vol_window": 20,
    "breadth_ema_period": 20,
    "risk_off_vol": 0.30,
    "breadth_risk_on_pct": 50.0,
    "breadth_risk_off_pct": 40.0,
    "block_new_buys_in_risk_off": False,
}
_WHOLE = ("spy_ema_period", "vol_window", "breadth_ema_period")
BARS_PER_YEAR = {"1Day": 252, "1Hour": 252 * 7}  # regular session has 7 hourly bars (09:30-16:00 ET)


def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def load_regime_settings(path=None) -> tuple:
    """Return (settings, warning). warning is None when config/regime.yaml was used. Never raises."""
    source = Path(path) if path is not None else _SETTINGS_PATH
    try:
        import yaml
        with open(source, "r", encoding="utf-8") as stream:
            raw = yaml.safe_load(stream)
        if not isinstance(raw, dict):
            raise ValueError("file is not a mapping")
        settings = {}
        for key, default in DEFAULT_SETTINGS.items():
            value = raw.get(key, default)
            if isinstance(default, bool):
                if not isinstance(value, bool):
                    raise ValueError(f"{key} must be true or false")
            elif not _is_number(value) or value <= 0:
                raise ValueError(f"{key} must be a positive number")
            elif key in _WHOLE and value != int(value):
                raise ValueError(f"{key} must be a whole number")
            settings[key] = value if isinstance(default, bool) else (int(value) if key in _WHOLE else float(value))
        if settings["breadth_risk_off_pct"] > settings["breadth_risk_on_pct"]:
            raise ValueError("breadth_risk_off_pct must not exceed breadth_risk_on_pct")
        return settings, None
    except Exception as exc:  # a setup mistake must fall back, never break the run
        return dict(DEFAULT_SETTINGS), (
            f"regime.yaml unusable ({type(exc).__name__}: {exc}); built-in default settings were used.")


def _clean_closes(values):
    """A list of finite positive numbers, or None."""
    if not isinstance(values, (list, tuple)) or not values:
        return None
    if not all(_is_number(v) and v > 0 for v in values):
        return None
    return [float(v) for v in values]


def _ema_last(values: list, period: int):
    """Last value of an EMA (alpha = 2 / (period + 1), seeded with the first close); None if too short."""
    if len(values) < period:
        return None
    alpha = 2.0 / (period + 1)
    ema = values[0]
    for value in values[1:]:
        ema = alpha * value + (1 - alpha) * ema
    return ema


def _realised_vol(closes: list, window: int, bars_per_year: int):
    """Annualised sample stdev of the last `window` simple returns; None if too few closes."""
    if len(closes) < window + 1 or window < 2:
        return None
    returns = [closes[i] / closes[i - 1] - 1 for i in range(len(closes) - window, len(closes))]
    mean = sum(returns) / len(returns)
    variance = sum((r - mean) ** 2 for r in returns) / (len(returns) - 1)
    return math.sqrt(variance) * math.sqrt(bars_per_year)


def compute_market_regime(spy_closes: list, watchlist_closes: dict = None, settings: dict = None,
                          timeframe: str = "1Day") -> dict:
    """Market backdrop from SPY closes (oldest first) and the watch-list's closes.

    Returns {spy_trend 'up'|'down'|None, spy_vol float|None, breadth_pct float|None,
             risk_mode 'risk_on'|'neutral'|'risk_off', timeframe, warnings}.
    Missing or unusable data gives risk_mode 'neutral' with a warning, never 'risk_on'."""
    cfg = dict(DEFAULT_SETTINGS)
    if isinstance(settings, dict):
        cfg.update({k: v for k, v in settings.items() if k in DEFAULT_SETTINGS})
    warnings = []
    result = {"spy_trend": None, "spy_vol": None, "breadth_pct": None, "risk_mode": "neutral",
              "timeframe": timeframe, "warnings": warnings}
    if timeframe not in BARS_PER_YEAR:
        warnings.append("timeframe must be '1Day' or '1Hour'; regime is neutral.")
        return result

    closes = _clean_closes(spy_closes)
    if closes is None:
        warnings.append("SPY closes are missing or invalid; regime is neutral.")
    else:
        ema = _ema_last(closes, cfg["spy_ema_period"])
        if ema is None:
            warnings.append(f"Need at least {cfg['spy_ema_period']} SPY closes for the trend.")
        else:
            result["spy_trend"] = "up" if closes[-1] > ema else "down"
        vol = _realised_vol(closes, cfg["vol_window"], BARS_PER_YEAR[timeframe])
        if vol is None:
            warnings.append(f"Need at least {cfg['vol_window'] + 1} SPY closes for volatility.")
        else:
            result["spy_vol"] = round(vol, 4)

    if isinstance(watchlist_closes, dict) and watchlist_closes:
        above = counted = 0
        for symbol, series in watchlist_closes.items():
            cleaned = _clean_closes(series)
            ema = _ema_last(cleaned, cfg["breadth_ema_period"]) if cleaned else None
            if ema is None:
                warnings.append(f"Breadth skipped {str(symbol).strip().upper()}: not enough valid closes.")
                continue
            counted += 1
            above += 1 if cleaned[-1] > ema else 0
        if counted:
            result["breadth_pct"] = round(100.0 * above / counted, 2)
        else:
            warnings.append("Breadth unavailable: no watch-list symbol had usable closes.")
    else:
        warnings.append("Breadth unavailable: no watch-list closes supplied.")

    trend, vol, breadth = result["spy_trend"], result["spy_vol"], result["breadth_pct"]
    if trend is None or vol is None:
        return result
    stressed = vol >= cfg["risk_off_vol"]
    if trend == "down" and (stressed or (breadth is not None and breadth < cfg["breadth_risk_off_pct"])):
        result["risk_mode"] = "risk_off"
    elif trend == "up" and not stressed and breadth is not None and breadth >= cfg["breadth_risk_on_pct"]:
        result["risk_mode"] = "risk_on"
    return result


def _parse_day(value):
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value.strip()[:10])
        except ValueError:
            return None
    return None


def get_earnings_calendar(symbols: list, manual_path=None, today=None) -> dict:
    """Next earnings date (ISO string) per symbol, or None when unknown or already past.

    v1 reads a manual list: fixtures/earnings.json = {"AAPL": "2026-10-29", ...}. A missing or
    unreadable file means every date is None (unknown); the guardian then warns that the earnings
    blackout was not checked. No network call is made."""
    if not isinstance(symbols, (list, tuple)):
        return {}
    cutoff = _parse_day(today) or datetime.now(timezone.utc).date()
    try:
        with open(Path(manual_path) if manual_path is not None else _FIXTURES / "earnings.json",
                  "r", encoding="utf-8") as stream:
            raw = json.load(stream)
    except Exception:
        raw = {}
    table = {str(k).strip().upper(): _parse_day(v) for k, v in raw.items()} if isinstance(raw, dict) else {}
    result = {}
    for symbol in symbols:
        if not isinstance(symbol, str) or not symbol.strip():
            continue
        key = symbol.strip().upper()
        day = table.get(key)
        result[key] = day.isoformat() if day is not None and day >= cutoff else None
    return result


def register_regime_tools(mcp) -> None:
    """Attach get_earnings_calendar to an existing MCP server. (get_market_regime is defined in the
    server file because it reads that file's bar cache.)"""
    mcp.tool()(get_earnings_calendar)
