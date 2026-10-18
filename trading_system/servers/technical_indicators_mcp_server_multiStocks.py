"""
# Last amended: 8th Oct, 2026
# MCP Server (stdio transport - started automatically by the client)
# Keep this file in the 'servers' folder below the client program.
# MCP Client: trading_crew.py
#
# Tools
# -----
#   fetch_ohlcv_batch(symbols)          : ONE Alpaca request for ALL symbols; caches each
#   calculate_indicators_batch(symbols) : RSI + MACD for ALL cached symbols
#   fetch_ohlcv(symbol)                 : single-symbol fetch
#   calculate_rsi(symbol)               : RSI for one symbol from the cache
#   calculate_macd(symbol)              : MACD for one symbol from the cache
#
# Cache is keyed by (symbol, timeframe).
# Credentials come from environment variables only.
"""

import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

# Allow this MCP server to import the project package when launched
# directly as a stdio subprocess.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import requests
from mcp.server.fastmcp import FastMCP

from trading_system.servers.tools.indicators import compute_indicators
from trading_system.servers.tools.scoring import score_indicators
from trading_system.servers.tools.news import register_news_tools
from trading_system.servers.tools.risk import register_risk_tools
from trading_system.servers.tools.broker import register_broker_tools
from trading_system.servers.tools.regime import (compute_market_regime, load_regime_settings,
                                                    register_regime_tools)

mcp = FastMCP("TechnicalIndicators")
register_news_tools(mcp)
register_risk_tools(mcp)
register_broker_tools(mcp)
register_regime_tools(mcp)

ALPACA_API_KEY = os.environ.get("ALPACA_API_KEY", "")
ALPACA_SECRET_KEY = os.environ.get("ALPACA_SECRET_KEY", "")
ALPACA_DATA_URL = "https://data.alpaca.markets/v2"
FEED = "iex"

HEADERS = {
    "APCA-API-KEY-ID": ALPACA_API_KEY,
    "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY,
}

# Cache is keyed by (symbol, timeframe) so Daily and Hourly data stay separate.
_cache: dict[tuple[str, str], dict] = {}

# The current run's timeframe is used by indicator tools whose MCP contract
# accepts only a symbol.
_active_timeframe: str | None = None


# ============================================================
# Helpers
# ============================================================

def _clean_symbols(symbols) -> list[str]:
    """Normalize symbols to uppercase, removing duplicates and blanks."""
    if isinstance(symbols, str):
        symbols = symbols.replace(",", " ").split()

    out = []

    for s in symbols:
        s = str(s).strip().upper()

        if s and s not in out:
            out.append(s)

    return out


def _fetch_bars(
    symbols: list[str],
    lookback_days: int,
    timeframe: str,
) -> dict:
    """Fetch paginated Alpaca OHLCV data for many symbols."""

    end_dt = datetime.now(timezone.utc)
    start_dt = end_dt - timedelta(days=lookback_days)

    params = {
        "symbols": ",".join(symbols),
        "timeframe": timeframe,
        "start": start_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "end": end_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "limit": 10_000,
        "feed": FEED,
    }

    url = f"{ALPACA_DATA_URL}/stocks/bars"

    rows: dict[str, list] = {}

    while True:

        max_retries = 3

        for attempt in range(max_retries):

            try:
                resp = requests.get(
                    url,
                    headers=HEADERS,
                    params=params,
                    timeout=20,
                )

                resp.raise_for_status()
                payload = resp.json()
                break

            except requests.RequestException:

                if attempt == max_retries - 1:
                    raise

                time.sleep(2 ** attempt)

        for sym, bars in (payload.get("bars") or {}).items():
            rows.setdefault(sym, []).extend(bars)

        token = payload.get("next_page_token")

        if not token:
            break

        params["page_token"] = token

    frames = {}

    for sym, bars in rows.items():

        df = pd.DataFrame(bars)

        df["t"] = pd.to_datetime(df["t"])

        df = (
            df.set_index("t")
            .sort_index()
            .rename(
                columns={
                    "o": "open",
                    "h": "high",
                    "l": "low",
                    "c": "close",
                    "v": "volume",
                }
            )[[
                "open",
                "high",
                "low",
                "close",
                "volume",
            ]]
        )

        frames[sym] = df

    return frames


def _get_df(symbol: str):
    """Return cached data for the active timeframe."""

    symbol = symbol.upper()

    if _active_timeframe is None:
        raise ValueError(
            f"No active timeframe for '{symbol}'. "
            "Run fetch_ohlcv_batch first."
        )

    key = (symbol, _active_timeframe)

    if key not in _cache:
        raise ValueError(
            f"No cached data for '{symbol}' at timeframe "
            f"'{_active_timeframe}'. Run fetch_ohlcv_batch first."
        )

    return _cache[key]["df"]


def _summary(
    symbol: str,
    df: pd.DataFrame,
    timeframe: str,
) -> dict:
    """Return a compact summary of fetched OHLCV data."""

    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "start": str(df.index[0]),
        "end": str(df.index[-1]),
        "bars_fetched": len(df),
        "last_close": round(
            float(df["close"].iloc[-1]),
            4,
        ),
    }


# ============================================================
# Technical indicator calculations
# ============================================================

def _rsi_value(
    closes: pd.Series,
    period: int,
) -> float:
    """Calculate RSI using Wilder-style smoothing."""

    delta = closes.diff()

    avg_gain = delta.clip(
        lower=0
    ).ewm(
        com=period - 1,
        min_periods=period,
    ).mean()

    avg_loss = (-delta).clip(
        lower=0
    ).ewm(
        com=period - 1,
        min_periods=period,
    ).mean()

    g = float(avg_gain.iloc[-1])
    l = float(avg_loss.iloc[-1])

    if l == 0:
        return 100.0 if g > 0 else 50.0

    return round(
        100 - 100 / (1 + g / l),
        2,
    )


def _rsi_reading(
    v: float,
) -> tuple[str, str, str]:
    """Return signal, lean and interpretation for RSI."""

    if v >= 70:
        return (
            "Overbought",
            "BEARISH",
            "Overbought: stretched to the upside, "
            "pullback risk is high.",
        )

    if v <= 30:
        return (
            "Oversold",
            "NEUTRAL",
            "Oversold: a rebound is possible but "
            "needs trend confirmation.",
        )

    if v >= 50:
        return (
            "Neutral",
            "BULLISH",
            "Above 50: momentum leans positive, "
            "not yet overbought.",
        )

    return (
        "Neutral",
        "NEUTRAL",
        "Below 50: momentum is soft with no "
        "clear upward bias.",
    )


def _macd_values(
    closes,
    fast,
    slow,
    sig,
) -> dict:
    """Calculate MACD values and crossover."""

    macd_line = (
        closes.ewm(
            span=fast,
            adjust=False,
        ).mean()
        -
        closes.ewm(
            span=slow,
            adjust=False,
        ).mean()
    )

    signal_line = macd_line.ewm(
        span=sig,
        adjust=False,
    ).mean()

    hist = macd_line - signal_line

    h = float(hist.iloc[-1])
    prev = (
        float(hist.iloc[-2])
        if len(hist) > 1
        else 0.0
    )

    if prev <= 0 < h:
        crossover = "Bullish"

    elif prev >= 0 > h:
        crossover = "Bearish"

    else:
        crossover = "None"

    return {
        "macd_line": round(
            float(macd_line.iloc[-1]),
            4,
        ),
        "signal_line": round(
            float(signal_line.iloc[-1]),
            4,
        ),
        "histogram": round(
            h,
            4,
        ),
        "crossover": crossover,
    }


def _macd_reading(
    m: dict,
) -> tuple[str, str]:
    """Return lean and interpretation for MACD."""

    h = m["histogram"]
    c = m["crossover"]

    if c == "Bullish":
        return (
            "BULLISH",
            "Fresh bullish crossover: MACD moved above "
            "its signal line.",
        )

    if c == "Bearish":
        return (
            "BEARISH",
            "Fresh bearish crossover: MACD moved below "
            "its signal line.",
        )

    if h > 0:
        return (
            "BULLISH",
            "MACD above signal line: upward trend "
            "momentum persists.",
        )

    if h < 0:
        return (
            "BEARISH",
            "MACD below signal line: downward trend "
            "momentum persists.",
        )

    return (
        "NEUTRAL",
        "MACD and signal line are level: no clear trend.",
    )


# ============================================================
# Tools
# ============================================================

@mcp.tool()
def fetch_ohlcv_batch(
    symbols: list[str],
    lookback_days: int = 180,
    timeframe: str = "1Day",
) -> dict:
    """
    Fetch OHLCV bars for all given symbols in one Alpaca request.

    Caches each symbol's data for the indicator tools.
    """

    global _active_timeframe

    syms = _clean_symbols(symbols)

    frames = _fetch_bars(
        syms,
        lookback_days,
        timeframe,
    )

    fetched = []
    missing = []

    for s in syms:

        df = frames.get(s)

        if df is None or df.empty:
            missing.append(s)
            continue

        _cache[(s, timeframe)] = {
            "timeframe": timeframe,
            "df": df,
        }

        fetched.append(
            _summary(
                s,
                df,
                timeframe,
            )
        )

    # Set active timeframe only after the entire batch
    # has been cached.
    _active_timeframe = timeframe

    return {
        "fetched": fetched,
        "missing": missing,
        "status": "ok",
    }


@mcp.tool()
def calculate_indicators_batch(
    symbols: list[str],
    rsi_period: int = 14,
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal: int = 9,
) -> dict:
    """
    Calculate RSI and MACD for ALL requested symbols from the
    OHLCV data already cached by fetch_ohlcv_batch.

    All calculations are deterministic Python calculations.
    No LLM is involved.

    Required workflow:

        1. fetch_ohlcv_batch(...)
        2. calculate_indicators_batch(...)

    Returns one result per symbol plus a missing/error list.
    """

    results = []
    missing = []

    for raw_symbol in symbols:

        symbol = str(
            raw_symbol
        ).strip().upper()

        try:

            df = _get_df(symbol)

            # --------------------------------------------
            # Validate sufficient history
            # --------------------------------------------

            if len(df) < rsi_period + 1:
                raise ValueError(
                    f"{symbol}: only {len(df)} bars; "
                    f"need {rsi_period + 1} for RSI."
                )

            if len(df) < macd_slow:
                raise ValueError(
                    f"{symbol}: only {len(df)} bars; "
                    f"need {macd_slow} for MACD."
                )

            # --------------------------------------------
            # RSI
            # --------------------------------------------

            rsi_value = _rsi_value(
                df["close"],
                rsi_period,
            )

            (
                rsi_signal,
                rsi_lean,
                rsi_interpretation,
            ) = _rsi_reading(
                rsi_value
            )

            # --------------------------------------------
            # MACD
            # --------------------------------------------

            macd_result = _macd_values(
                df["close"],
                macd_fast,
                macd_slow,
                macd_signal,
            )

            (
                macd_lean,
                macd_interpretation,
            ) = _macd_reading(
                macd_result
            )

            # --------------------------------------------
            # Combined result
            # --------------------------------------------

            results.append(
                {
                    "symbol": symbol,

                    "latest_close": round(
                        float(
                            df["close"].iloc[-1]
                        ),
                        4,
                    ),

                    "bars_used": len(df),

                    "as_of": str(
                        df.index[-1]
                    ),

                    "rsi": {
                        "period": rsi_period,
                        "value": rsi_value,
                        "signal": rsi_signal,
                        "lean": rsi_lean,
                        "interpretation": rsi_interpretation,
                    },

                    "macd": {
                        "fast_period": macd_fast,
                        "slow_period": macd_slow,
                        "signal_period": macd_signal,
                        **macd_result,
                        "lean": macd_lean,
                        "interpretation": macd_interpretation,
                    },
                }
            )

        except Exception as e:

            missing.append(
                {
                    "symbol": symbol,
                    "error": str(e),
                }
            )

    return {
        "status": "ok",
        "results": results,
        "missing": missing,
    }


@mcp.tool()
def fetch_ohlcv(
    symbol: str,
    lookback_days: int = 180,
    timeframe: str = "1Day",
) -> dict:
    """
    Fetch OHLCV bars for one symbol and cache them.
    """

    global _active_timeframe

    symbol = symbol.upper()

    frames = _fetch_bars(
        [symbol],
        lookback_days,
        timeframe,
    )

    if symbol not in frames:
        raise ValueError(
            f"No bar data returned for '{symbol}'."
        )

    _cache[(symbol, timeframe)] = {
        "timeframe": timeframe,
        "df": frames[symbol],
    }

    _active_timeframe = timeframe

    return _summary(
        symbol,
        frames[symbol],
        timeframe,
    )


@mcp.tool()
def calculate_rsi(
    symbol: str,
    period: int = 14,
) -> dict:
    """Calculate RSI for one symbol from cached data."""

    symbol = symbol.upper()

    df = _get_df(symbol)

    if len(df) < period + 1:
        raise ValueError(
            f"{symbol}: only {len(df)} bars; "
            f"need {period + 1}."
        )

    value = _rsi_value(
        df["close"],
        period,
    )

    signal, lean, interp = _rsi_reading(
        value
    )

    return {
        "symbol": symbol,
        "period": period,
        "rsi": value,
        "signal": signal,
        "lean": lean,
        "interpretation": interp,
        "latest_close": round(
            float(df["close"].iloc[-1]),
            4,
        ),
        "bars_used": len(df),
        "as_of": str(df.index[-1]),
    }


@mcp.tool()
def calculate_macd(
    symbol: str,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
) -> dict:
    """Calculate MACD for one symbol from cached data."""

    symbol = symbol.upper()

    df = _get_df(symbol)

    if len(df) < slow_period:
        raise ValueError(
            f"{symbol}: only {len(df)} bars; "
            f"need {slow_period}."
        )

    m = _macd_values(
        df["close"],
        fast_period,
        slow_period,
        signal_period,
    )

    lean, interp = _macd_reading(
        m
    )

    return {
        "symbol": symbol,
        "fast_period": fast_period,
        "slow_period": slow_period,
        "signal_period": signal_period,
        **m,
        "lean": lean,
        "interpretation": interp,
        "latest_close": round(
            float(df["close"].iloc[-1]),
            4,
        ),
        "bars_used": len(df),
        "as_of": str(df.index[-1]),
    }


@mcp.tool()
def calculate_m2_batch(
    symbols: list[str],
    timeframe: str = "1Day",
    sleeve: str = "core",
    sector: str | None = None,
    news_scores: dict[str, float | None] | None = None,
    sleeves: dict[str, str] | None = None,
    sectors: dict[str, str] | None = None,
) -> dict:
    """
    Run M2-A indicators and M2-B deterministic scoring for all
    requested symbols using only the canonical MCP cache.

    news_scores is an optional {SYMBOL: score or None} map supplied by
    the orchestrator (M4). A missing or invalid score is excluded by the
    scorer and the remaining weights are renormalized. No news is
    fetched here.

    sleeves / sectors are optional {SYMBOL: value} maps. A symbol found
    in `sleeves` is scored with that sleeve ("core" or "high_beta"); every
    other symbol uses the `sleeve` argument (default "core"). `sectors`
    works the same way for the sector label. An unknown sleeve value is
    not guessed: the scorer rejects it and the symbol gets a fail-safe HOLD.

    M1 fetch_ohlcv_batch must be called first in the same MCP
    session/process. SPY must also be present in that cache so
    relative-strength scoring can be computed.
    """

    if timeframe not in {"1Day", "1Hour"}:
        raise ValueError("timeframe must be '1Day' or '1Hour'.")

    clean_symbols = []
    for raw_symbol in symbols:
        symbol = str(raw_symbol).strip().upper()
        if symbol and symbol not in clean_symbols:
            clean_symbols.append(symbol)

    if not clean_symbols:
        return {
            "status": "ok",
            "results": [],
            "missing": [],
        }

    try:
        spy_df = _get_df("SPY")
    except Exception:
        spy_df = None

    supplied_news = {}
    if isinstance(news_scores, dict):
        for raw_symbol, value in news_scores.items():
            supplied_news[str(raw_symbol).strip().upper()] = value

    def _per_symbol(mapping):
        out = {}
        if isinstance(mapping, dict):
            for raw_symbol, value in mapping.items():
                out[str(raw_symbol).strip().upper()] = value
        return out

    sleeve_by_symbol = _per_symbol(sleeves)
    sector_by_symbol = _per_symbol(sectors)

    results = []
    missing = []

    for symbol in clean_symbols:
        try:
            df = _get_df(symbol)

            indicator_result = compute_indicators(
                frame=df,
                symbol=symbol,
                timeframe=timeframe,
                spy_frame=spy_df,
            )

            indicator_dict = indicator_result.to_dict()

            score_result = score_indicators(
                symbol=symbol,
                timeframe=timeframe,
                sleeve=sleeve_by_symbol.get(symbol, sleeve),
                sector=sector_by_symbol.get(symbol, sector),
                indicator_result=indicator_dict,
                news_score=supplied_news.get(symbol),
            )

            results.append({
                "symbol": symbol,
                "indicators": indicator_dict,
                "score": score_result,
            })

        except Exception as exc:
            missing.append({
                "symbol": symbol,
                "error": str(exc),
            })

    return {
        "status": "ok",
        "timeframe": timeframe,
        "sleeve": sleeve,
        "sector": sector,
        "spy_available": spy_df is not None,
        "results": results,
        "missing": missing,
    }



@mcp.tool()
def get_market_regime(symbols: list[str] | None = None, timeframe: str = "1Day") -> dict:
    """
    M3 market backdrop from the bars already cached by fetch_ohlcv_batch:
    SPY trend (last close vs EMA), SPY volatility and breadth (share of
    `symbols` above their EMA) -> risk_mode risk_on | neutral | risk_off.

    No data is fetched here. Missing SPY or watch-list bars give
    risk_mode "neutral" with warnings, never risk_on. Limits come from
    config/regime.yaml.
    """

    if timeframe not in {"1Day", "1Hour"}:
        raise ValueError("timeframe must be '1Day' or '1Hour'.")

    def _closes(symbol):
        entry = _cache.get((symbol, timeframe))
        if not entry:
            return None
        try:
            return [float(v) for v in entry["df"]["close"].tolist()]
        except Exception:
            return None

    watchlist = {}
    for raw_symbol in symbols or []:
        symbol = str(raw_symbol).strip().upper()
        if symbol and symbol != "SPY":
            closes = _closes(symbol)
            if closes is not None:
                watchlist[symbol] = closes

    settings, settings_warning = load_regime_settings()
    result = compute_market_regime(
        _closes("SPY"), watchlist, settings, timeframe
    )
    if settings_warning:
        result["warnings"].append(settings_warning)
    result["block_new_buys_in_risk_off"] = bool(
        settings.get("block_new_buys_in_risk_off")
    )
    return result


# ============================================================
# MCP server startup
# ============================================================

if __name__ == "__main__":
    mcp.run(transport="stdio")
