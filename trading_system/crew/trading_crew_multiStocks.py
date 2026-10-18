"""
Deterministic Multi-Stock Paper Trading Orchestrator
====================================================

Architecture:

    Python Orchestrator
        |
        +--> ONE MCP stdio process
        |       |
        |       +--> fetch_ohlcv_batch()
        |       |
        |       +--> calculate_m2_batch()
        |               |
        |               +--> M2-A Indicators
        |               +--> M2-B Deterministic Scoring
        |
        +--> Deterministic Python Decision Engine
        |       |
        |       +--> BUY
        |       +--> SELL
        |       +--> HOLD
        |
        +--> Starter Position Sizing
        |
        +--> Reporting

Important:
- No CrewAI LLM is used for M1/M2.
- No Groq/LiteLLM calls occur in this file.
- The same MCP process/session is kept alive for both tool calls.
- This preserves the MCP server's process-local OHLCV cache.
- SPY is fetched in the same M1 batch for relative-strength context.
- SPY is NOT part of the trading universe.
- Orders are NOT submitted by this client.
- Final M5/M6 risk sizing is not implemented yet.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


# ============================================================================
# PATHS / ENVIRONMENT
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

ENV_FILE = PROJECT_ROOT / ".env"

load_dotenv(ENV_FILE)

MCP_SERVER_PATH = (
    PROJECT_ROOT
    / "trading_system"
    / "servers"
    / "technical_indicators_mcp_server_multiStocks.py"
)


# ============================================================================
# STARTER CONFIGURATION
# ============================================================================

CAPITAL = 100_000.0

# Intentionally preserved from the current starter implementation.
# Final M5/M6 risk configuration will replace these.
MAX_POSITION_PCT = 0.20
CASH_RESERVE_PCT = 0.10

DEFAULT_LOOKBACK_DAYS = 180
DEFAULT_TIMEFRAME = "1Day"

# Benchmark used by M2-A relative-strength calculations.
BENCHMARK_SYMBOL = "SPY"


# ============================================================================
# DIRECT MCP CLIENT
# ============================================================================


class DirectMCPClient:
    """
    One persistent MCP server process + one persistent ClientSession.

    Required by the current MCP server architecture:

        fetch_ohlcv_batch()
            -> populates process-local OHLCV cache

        calculate_m2_batch()
            -> reads that same cache

    Therefore, both calls MUST happen inside the same
    MCP process/session.
    """

    def __init__(self, server_path: Path):

        self.server_path = server_path

        self._stdio_cm = None
        self._session_cm = None

        self.read_stream = None
        self.write_stream = None
        self.session: ClientSession | None = None

    async def __aenter__(self) -> "DirectMCPClient":

        if not self.server_path.exists():
            raise FileNotFoundError(
                f"MCP server file not found:\n{self.server_path}"
            )

        server_params = StdioServerParameters(
            command=sys.executable,
            args=[str(self.server_path)],
            env=os.environ.copy(),
        )

        self._stdio_cm = stdio_client(server_params)

        self.read_stream, self.write_stream = (
            await self._stdio_cm.__aenter__()
        )

        self._session_cm = ClientSession(
            self.read_stream,
            self.write_stream,
        )

        self.session = await self._session_cm.__aenter__()

        await self.session.initialize()

        return self

    async def __aexit__(
        self,
        exc_type,
        exc_value,
        traceback,
    ):

        session_error = None
        stdio_error = None

        if self._session_cm is not None:

            try:
                await self._session_cm.__aexit__(
                    exc_type,
                    exc_value,
                    traceback,
                )

            except Exception as exc:
                session_error = exc

        if self._stdio_cm is not None:

            try:
                await self._stdio_cm.__aexit__(
                    exc_type,
                    exc_value,
                    traceback,
                )

            except Exception as exc:
                stdio_error = exc

        self.session = None
        self._session_cm = None
        self._stdio_cm = None

        # Never hide the original exception.
        if exc_value is None:

            if session_error is not None:
                raise session_error

            if stdio_error is not None:
                raise stdio_error

    async def list_tools(self) -> list[str]:

        if self.session is None:
            raise RuntimeError(
                "MCP session is not initialized."
            )

        result = await self.session.list_tools()

        tools = getattr(
            result,
            "tools",
            None,
        ) or []

        names: list[str] = []

        for tool in tools:

            name = getattr(
                tool,
                "name",
                None,
            )

            if name:
                names.append(str(name))

        return names

    async def call_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> Any:

        if self.session is None:
            raise RuntimeError(
                "MCP session is not initialized."
            )

        result = await self.session.call_tool(
            tool_name,
            arguments,
        )

        return extract_mcp_result(result)


# ============================================================================
# MCP RESULT EXTRACTION
# ============================================================================


def extract_mcp_result(result: Any) -> Any:
    """
    Extract a usable Python value from an MCP CallToolResult.

    Supports:
        - structured_content
        - structuredContent
        - text JSON
        - text output
        - model_dump()
    """

    # ------------------------------------------------------------------
    # Error flag
    # ------------------------------------------------------------------

    is_error = getattr(
        result,
        "is_error",
        None,
    )

    if is_error is None:
        is_error = getattr(
            result,
            "isError",
            False,
        )

    if is_error:
        raise RuntimeError(
            f"MCP tool returned an error: {result}"
        )

    # ------------------------------------------------------------------
    # Preferred structured content
    # ------------------------------------------------------------------

    structured = getattr(
        result,
        "structured_content",
        None,
    )

    if structured is not None:
        return structured

    # ------------------------------------------------------------------
    # Compatibility structured content
    # ------------------------------------------------------------------

    structured = getattr(
        result,
        "structuredContent",
        None,
    )

    if structured is not None:
        return structured

    # ------------------------------------------------------------------
    # Text content
    # ------------------------------------------------------------------

    content = getattr(
        result,
        "content",
        None,
    )

    if content:

        text_parts: list[str] = []

        for item in content:

            item_type = getattr(
                item,
                "type",
                None,
            )

            if item_type == "text":

                item_text = getattr(
                    item,
                    "text",
                    None,
                )

                if item_text:
                    text_parts.append(
                        str(item_text)
                    )

        combined = "\n".join(
            text_parts
        ).strip()

        if combined:

            try:
                return json.loads(
                    combined
                )

            except json.JSONDecodeError:
                return combined

    # ------------------------------------------------------------------
    # Pydantic/model fallback
    # ------------------------------------------------------------------

    if hasattr(result, "model_dump"):

        dumped = result.model_dump()

        if dumped:

            structured = dumped.get(
                "structured_content"
            )

            if structured is None:
                structured = dumped.get(
                    "structuredContent"
                )

            if structured is not None:
                return structured

            dumped_content = dumped.get(
                "content"
            )

            if dumped_content:

                for item in dumped_content:

                    if not isinstance(
                        item,
                        dict,
                    ):
                        continue

                    if item.get("type") != "text":
                        continue

                    text_value = item.get(
                        "text"
                    )

                    if not text_value:
                        continue

                    try:
                        return json.loads(
                            text_value
                        )

                    except json.JSONDecodeError:
                        return text_value

    raise RuntimeError(
        "Unable to extract usable data "
        "from MCP tool result."
    )


# ============================================================================
# SYMBOL NORMALIZATION
# ============================================================================


def normalize_symbols(
    symbols: list[str],
) -> list[str]:

    result: list[str] = []
    seen: set[str] = set()

    for raw_symbol in symbols:

        symbol = str(
            raw_symbol
        ).strip().upper()

        if not symbol:
            continue

        if symbol not in seen:

            seen.add(symbol)
            result.append(symbol)

    if not result:

        raise ValueError(
            "At least one stock symbol is required."
        )

    return result


# ============================================================================
# CONFIG VALIDATION
# ============================================================================


def validate_config(
    symbols: list[str],
    lookback_days: int,
    timeframe: str,
) -> None:

    if not symbols:

        raise ValueError(
            "No symbols supplied."
        )

    if lookback_days <= 0:

        raise ValueError(
            "lookback_days must be greater than zero."
        )

    if timeframe not in {
        "1Day",
        "1Hour",
    }:

        raise ValueError(
            f"Unsupported timeframe: {timeframe}. "
            "Expected 1Day or 1Hour."
        )


# ============================================================================
# ALPACA
# ============================================================================


def get_alpaca_base_url() -> str:
    """
    Normalize ALPACA_TRADING_URL.

    Accepted examples:

        https://paper-api.alpaca.markets
        https://paper-api.alpaca.markets/
        https://paper-api.alpaca.markets/v2
        https://paper-api.alpaca.markets/v2/
    """

    raw_url = (
        os.getenv(
            "ALPACA_TRADING_URL"
        )
        or "https://paper-api.alpaca.markets"
    ).strip()

    raw_url = raw_url.rstrip("/")

    if raw_url.endswith("/v2"):
        return raw_url

    return f"{raw_url}/v2"


def get_alpaca_headers() -> dict[str, str]:

    api_key = os.getenv(
        "ALPACA_API_KEY"
    )

    secret_key = os.getenv(
        "ALPACA_SECRET_KEY"
    )

    if not api_key or not secret_key:

        raise RuntimeError(
            "ALPACA_API_KEY / "
            "ALPACA_SECRET_KEY are not configured."
        )

    return {
        "APCA-API-KEY-ID": api_key,
        "APCA-API-SECRET-KEY": secret_key,
    }


def get_holdings() -> list[str]:
    """
    Read current Alpaca paper positions.

    If the endpoint fails, fail-safe behaviour is:
        assume no holdings.

    This function DOES NOT submit orders.
    """

    try:

        base_url = get_alpaca_base_url()

        headers = get_alpaca_headers()

        response = requests.get(
            f"{base_url}/positions",
            headers=headers,
            timeout=15,
        )

        response.raise_for_status()

        positions = response.json()

        if not isinstance(
            positions,
            list,
        ):

            raise RuntimeError(
                "Unexpected Alpaca positions response."
            )

        holdings: list[str] = []

        for position in positions:

            if not isinstance(
                position,
                dict,
            ):
                continue

            symbol = position.get(
                "symbol"
            )

            if symbol:
                holdings.append(
                    str(symbol).upper()
                )

        return sorted(
            set(holdings)
        )

    except Exception as exc:

        print(
            "WARNING: could not read "
            f"positions ({exc}); assuming NO current holdings."
        )

        return []


# ============================================================================
# M1 PRICE EXTRACTION
# ============================================================================


def build_m1_price_map(
    fetch_report: dict[str, Any],
) -> dict[str, float]:
    """
    Build a symbol -> latest_close map from M1.

    This is a fallback for the sizing layer.

    M2-B remains authoritative for the technical decision.
    M1 is only being used here to ensure sizing has a valid price.
    """

    price_map: dict[str, float] = {}

    fetched = fetch_report.get(
        "fetched",
        [],
    )

    if not isinstance(
        fetched,
        list,
    ):
        return price_map

    for item in fetched:

        if not isinstance(
            item,
            dict,
        ):
            continue

        symbol = str(
            item.get(
                "symbol",
                "",
            )
        ).upper()

        raw_price = item.get(
            "last_close"
        )

        if not symbol:
            continue

        if not isinstance(
            raw_price,
            (int, float),
        ):
            continue

        try:
            price = float(
                raw_price
            )
        except (
            TypeError,
            ValueError,
        ):
            continue

        if not math.isfinite(
            price
        ):
            continue

        if price <= 0:
            continue

        price_map[symbol] = price

    return price_map


def apply_m1_price_fallback(
    decisions: list[dict[str, Any]],
    fetch_report: dict[str, Any],
) -> None:
    """
    Populate missing latest_close values from M1.

    This mutates only the decision dictionaries' latest_close field.
    """

    price_map = build_m1_price_map(
        fetch_report
    )

    for decision in decisions:

        current_price = decision.get(
            "latest_close"
        )

        if isinstance(
            current_price,
            (int, float),
        ):

            try:

                if (
                    math.isfinite(
                        float(current_price)
                    )
                    and float(current_price) > 0
                ):
                    continue

            except (
                TypeError,
                ValueError,
            ):
                pass

        symbol = str(
            decision.get(
                "symbol",
                "",
            )
        ).upper()

        fallback_price = price_map.get(
            symbol
        )

        if fallback_price is not None:

            decision[
                "latest_close"
            ] = fallback_price


# ============================================================================
# DETERMINISTIC DECISION ENGINE — M2-B
# ============================================================================


def make_decisions(
    m2_report: dict[str, Any],
    held_symbols: list[str] | None,
) -> list[dict[str, Any]]:
    """Convert M2 output into deterministic BUY/SELL/HOLD decisions.

    Invalid or unusable M2 data always fails safe to HOLD. The first row for
    each symbol wins so malformed duplicate rows cannot override a decision.
    """
    if not isinstance(m2_report, dict):
        raise ValueError("M2 report must be a dictionary.")

    held = {
        str(symbol).upper()
        for symbol in (held_symbols or [])
    }
    results = m2_report.get("results", [])
    missing = m2_report.get("missing", [])
    if not isinstance(results, list):
        results = []
    if not isinstance(missing, list):
        missing = []

    decisions: list[dict[str, Any]] = []
    represented: set[str] = set()

    for item in results:
        if not isinstance(item, dict):
            continue
        try:
            symbol = str(item.get("symbol", "")).strip().upper()
        except Exception:
            # Without a usable symbol there is no safe decision row to emit.
            continue
        if not symbol or symbol in represented:
            continue
        # Reserve the symbol on its first row, even if that row is malformed.
        represented.add(symbol)

        try:
            score = item.get("score")
            if not isinstance(score, dict):
                score = {}
            indicators = item.get("indicators")
            if not isinstance(indicators, dict):
                indicators = {}
            context = indicators.get("context")
            if not isinstance(context, dict):
                context = {}

            composite = score.get("composite")
            valid_composite = (
                isinstance(composite, (int, float))
                and not isinstance(composite, bool)
                and math.isfinite(composite)
            )
            triage = str(score.get("triage", "hold")).lower()
            last_close = context.get("last_close")
            if last_close is None:
                last_close = score.get("last_close")

            if not valid_composite:
                decision = "HOLD"
                if composite is None:
                    reason = "M2-B composite unavailable; fail-safe HOLD."
                else:
                    reason = (
                        "M2-B composite is not a finite real number; "
                        "fail-safe HOLD."
                    )
            elif triage == "clear_buy":
                decision = "BUY"
                reason = f"M2-B deterministic BUY; composite={composite:.4f}."
            elif triage == "clear_sell":
                if symbol in held:
                    decision = "SELL"
                    reason = f"M2-B deterministic SELL; composite={composite:.4f}."
                else:
                    decision = "HOLD"
                    reason = "Long-only policy: cannot sell a symbol that is not held."
            elif triage == "borderline":
                decision = "HOLD"
                reason = "borderline, awaiting M7 debate"
            else:
                decision = "HOLD"
                reason = (
                    "M2-B deterministic HOLD; "
                    f"triage={triage}, composite={composite:.4f}."
                )

            decisions.append({
                "symbol": symbol,
                "decision": decision,
                "reason": reason,
                "held": symbol in held,
                "latest_close": last_close,
                "as_of": score.get("as_of"),
                "composite": composite,
                "triage": triage,
                "regime": score.get("regime"),
                "regime_strength": score.get("regime_strength"),
                "adx": score.get("adx"),
                "atr": score.get("atr"),
                "sub_scores": score.get("sub_scores", {}),
                "missing_indicators": indicators.get("missing_indicators", []),
                "warnings": indicators.get("warnings", []),
            })
        except Exception as exc:
            decisions.append({
                "symbol": symbol,
                "decision": "HOLD",
                "reason": (
                    f"M2 result row unusable ({type(exc).__name__}); "
                    "fail-safe HOLD."
                ),
                "held": symbol in held,
                "latest_close": None,
                "as_of": None,
                "composite": None,
                "triage": "hold",
                "regime": None,
                "regime_strength": None,
                "adx": None,
                "atr": None,
                "sub_scores": {},
                "missing_indicators": [],
                "warnings": [],
            })

    # Symbols M2 could not process: fail-safe HOLD, without duplicate output.
    for item in missing:
        if not isinstance(item, dict):
            continue
        symbol = str(item.get("symbol", "")).strip().upper()
        if not symbol or symbol in represented:
            continue
        represented.add(symbol)
        decisions.append({
            "symbol": symbol,
            "decision": "HOLD",
            "reason": "M2 returned no usable result; fail-safe HOLD.",
            "held": symbol in held,
            "latest_close": None,
            "as_of": None,
            "composite": None,
            "triage": "hold",
            "regime": None,
            "regime_strength": None,
            "adx": None,
            "atr": None,
            "sub_scores": {},
            "missing": item,
        })

    return decisions


# ============================================================================
# STARTER POSITION SIZING
# ============================================================================


def make_plan(
    decisions: list[dict[str, Any]],
    capital: float = CAPITAL,
) -> list[dict[str, Any]]:
    """
    Starter sizing only.

    BUY:
        maximum allocation =
        MAX_POSITION_PCT * capital

    Cash reserve:
        CASH_RESERVE_PCT * capital

    No order is submitted here.

    Final M5/M6 risk sizing will replace this logic.
    """

    if (
        not math.isfinite(
            float(capital)
        )
        or capital <= 0
    ):

        raise ValueError(
            "Capital must be a finite positive number."
        )

    max_position_value = (
        float(capital)
        * MAX_POSITION_PCT
    )

    cash_reserve = (
        float(capital)
        * CASH_RESERVE_PCT
    )

    plan: list[
        dict[str, Any]
    ] = []

    for decision in decisions:

        symbol = decision.get(
            "symbol"
        )

        action = decision.get(
            "decision",
            "HOLD",
        )

        latest_close = decision.get(
            "latest_close"
        )

        planned_value = 0.0
        planned_quantity = 0

        valid_price = (
            isinstance(
                latest_close,
                (int, float),
            )
            and math.isfinite(
                float(latest_close)
            )
            and float(latest_close) > 0
        )

        if (
            action == "BUY"
            and valid_price
        ):

            planned_value = (
                max_position_value
            )

            planned_quantity = int(
                planned_value
                / float(latest_close)
            )

        plan.append(
            {
                "symbol": symbol,
                "decision": action,
                "latest_close": latest_close,
                "planned_value": round(
                    planned_value,
                    2,
                ),
                "planned_quantity": planned_quantity,
                "max_position_value": round(
                    max_position_value,
                    2,
                ),
                "cash_reserve": round(
                    cash_reserve,
                    2,
                ),
            }
        )

    return plan


# ============================================================================
# REPORTING
# ============================================================================


def print_report(
    symbols: list[str],
    lookback_days: int,
    timeframe: str,
    holdings: list[str],
    fetch_report: dict[str, Any],
    m2_report: dict[str, Any],
    decisions: list[dict[str, Any]],
    plan: list[dict[str, Any]],
) -> None:

    print()
    print("=" * 90)
    print("DETERMINISTIC PAPER TRADING REPORT")
    print("=" * 90)

    print()
    print(
        f"Symbols: {symbols}"
    )

    print(
        f"Lookback: {lookback_days} days"
    )

    print(
        f"Timeframe: {timeframe}"
    )

    print(
        f"Holdings: {holdings}"
    )

    print()
    print(
        "Strategy engine: DETERMINISTIC PYTHON"
    )

    print(
        "LLM decision-making: DISABLED"
    )

    print(
        "Order submission: DISABLED"
    )

    # ------------------------------------------------------------------
    # M1 — MARKET DATA
    # ------------------------------------------------------------------

    print()
    print("-" * 90)
    print("M1 — MARKET DATA")
    print("-" * 90)

    # Actual fetch_ohlcv_batch contract:
    #
    # {
    #     "status": "ok",
    #     "fetched": [...],
    #     "missing": [...]
    # }
    #
    fetch_results = fetch_report.get(
        "fetched",
        [],
    )

    fetch_missing = fetch_report.get(
        "missing",
        [],
    )

    if not isinstance(
        fetch_results,
        list,
    ):
        fetch_results = []

    for item in fetch_results:

        if not isinstance(
            item,
            dict,
        ):
            continue

        bars = item.get("bars_fetched", item.get("bars"))
        as_of = item.get("end", item.get("as_of"))

        print(
            f"{str(item.get('symbol', '')):>6} | "
            f"bars={bars} | "
            f"last_close={item.get('last_close')} | "
            f"as_of={as_of}"
        )

    if fetch_missing:

        print()
        print("Missing:")

        for item in fetch_missing:

            print(
                f"  {item}"
            )

    # ------------------------------------------------------------------
    # M2 — INDICATORS + DETERMINISTIC SCORING
    # ------------------------------------------------------------------

    print()
    print("-" * 90)
    print(
        "M2 — INDICATORS + DETERMINISTIC SCORING"
    )
    print("-" * 90)

    m2_results = m2_report.get(
        "results",
        [],
    )

    if not isinstance(
        m2_results,
        list,
    ):
        m2_results = []

    for item in m2_results:

        if not isinstance(
            item,
            dict,
        ):
            continue

        score = item.get(
            "score"
        ) or {}

        print(
            f"{str(item.get('symbol', '')):>6} | "
            f"regime={str(score.get('regime')):<10} | "
            f"ADX={score.get('adx')} | "
            f"composite={score.get('composite')} | "
            f"triage={score.get('triage')}"
        )

    m2_missing = m2_report.get(
        "missing",
        [],
    )

    if m2_missing:

        print()
        print("M2 missing:")

        for item in m2_missing:

            print(
                f"  {item}"
            )

    # ------------------------------------------------------------------
    # DETERMINISTIC DECISIONS
    # ------------------------------------------------------------------

    print()
    print("-" * 90)
    print(
        "DETERMINISTIC DECISIONS"
    )
    print("-" * 90)

    for item in decisions:

        print(
            f"{str(item.get('symbol', '')):>6} | "
            f"{str(item.get('decision', 'HOLD')):<4} | "
            f"composite={str(item.get('composite')):<10} | "
            f"triage={str(item.get('triage', 'hold')):<5} | "
            f"regime={str(item.get('regime')):<10} | "
            f"{item.get('reason', '')}"
        )

    # ------------------------------------------------------------------
    # STARTER ACTION PLAN
    # ------------------------------------------------------------------

    print()
    print("-" * 90)
    print(
        "STARTER ACTION PLAN"
    )
    print("-" * 90)

    for item in plan:

        print(
            f"{str(item.get('symbol', '')):>6} | "
            f"{str(item.get('decision', 'HOLD')):<4} | "
            f"qty={item.get('planned_quantity', 0):<5} | "
            f"value={float(item.get('planned_value', 0.0)):.2f} | "
            f"price={item.get('latest_close')}"
        )

    print()
    print("=" * 90)
    print(
        "END REPORT"
    )
    print("=" * 90)
    print()


# ============================================================================
# M4 NEWS SCORES (optional, non-fatal)
# ============================================================================


async def fetch_news_scores(
    mcp: Any,
    symbols: list[str],
    available_tools: list[str],
) -> tuple[dict[str, float | None], list[str]]:
    """
    Fetch M4 news sentiment for M2 scoring.

    Calls get_news and then score_headlines in the same MCP session and
    returns ({SYMBOL: news_score or None}, warnings).

    News is optional. Any failure returns ({}, [warning]) so the scoring
    layer excludes the news group and renormalizes the remaining weights.
    This function never raises and never prints credentials.
    """

    missing_tools = {
        "get_news",
        "score_headlines",
    } - set(available_tools)

    if missing_tools:
        return {}, [
            "News tools unavailable: "
            + ", ".join(sorted(missing_tools))
        ]

    try:
        news = await mcp.call_tool(
            "get_news",
            {"symbols": list(symbols)},
        )

        sentiment = await mcp.call_tool(
            "score_headlines",
            {"headlines": news},
        )

    except Exception as exc:
        return {}, [
            f"News unavailable: {type(exc).__name__}: {exc}"
        ]

    warnings: list[str] = []

    if isinstance(news, dict):
        for key in ("warnings", "errors"):
            values = news.get(key)
            if isinstance(values, (list, tuple)):
                warnings.extend(
                    str(value) for value in values if value
                )

    by_symbol = (
        sentiment.get("by_symbol")
        if isinstance(sentiment, dict)
        else None
    )

    if not isinstance(by_symbol, dict):
        warnings.append(
            "News unavailable: score_headlines returned no by_symbol data."
        )
        return {}, warnings

    scores: dict[str, float | None] = {}

    for symbol in symbols:
        entry = by_symbol.get(symbol)
        scores[symbol] = (
            entry.get("news_score")
            if isinstance(entry, dict)
            else None
        )

    return scores, warnings


# ============================================================================
# MAIN STRATEGY
# ============================================================================


async def _run_strategy_impl(
    symbols: list[str],
    lookback_days: int,
    timeframe: str,
    advisory_band: str | None = None,
    guarded: bool = False,
) -> dict[str, Any]:

    symbols = normalize_symbols(
        symbols
    )

    validate_config(
        symbols=symbols,
        lookback_days=lookback_days,
        timeframe=timeframe,
    )

    print()
    print("=" * 90)
    print(
        "STARTING DIRECT-MCP TRADING ORCHESTRATOR"
    )
    print("=" * 90)

    print(
        f"Symbols: {symbols}"
    )

    print(
        f"Lookback: {lookback_days} days"
    )

    print(
        f"Timeframe: {timeframe}"
    )

    # ------------------------------------------------------------------
    # Current holdings
    # ------------------------------------------------------------------

    holdings = get_holdings()

    print(
        f"Holdings: {holdings}"
    )

    print(
        "Strategy engine: DETERMINISTIC PYTHON"
    )

    print(
        "LLM decision-making: DISABLED"
    )

    if advisory_band:

        print(
            "LLM advisory: ENABLED "
            f"(M7 debate, band={advisory_band}; the Judge may move a "
            "composite by at most the configured cap and code rechecks "
            "the thresholds)"
        )

    print(
        "MCP execution: DIRECT ClientSession / stdio"
    )

    # ------------------------------------------------------------------
    # ONE MCP PROCESS / SESSION
    # ------------------------------------------------------------------

    async with DirectMCPClient(
        MCP_SERVER_PATH
    ) as mcp:

        available_tools = (
            await mcp.list_tools()
        )

        print()
        print(
            "MCP tools:",
            ", ".join(
                available_tools
            ),
        )

        required_tools = {
            "fetch_ohlcv_batch",
            "calculate_m2_batch",
        }

        missing_tools = (
            required_tools
            - set(available_tools)
        )

        if missing_tools:

            raise RuntimeError(
                "Required MCP tools are missing: "
                + ", ".join(
                    sorted(
                        missing_tools
                    )
                )
            )

        # ==============================================================
        # M1 — BATCH OHLCV
        # ==============================================================

        # SPY is fetched in the same M1 call so that M2-A can calculate
        # relative strength. It is NOT passed to M2 as a trading symbol.
        fetch_symbols = list(symbols)

        if BENCHMARK_SYMBOL not in fetch_symbols:

            fetch_symbols.append(
                BENCHMARK_SYMBOL
            )

        fetch_args = {
            "lookback_days": lookback_days,
            "symbols": fetch_symbols,
            "timeframe": timeframe,
        }

        print()
        print(
            "MCP → fetch_ohlcv_batch"
        )

        print(
            f"Arguments: {fetch_args}"
        )

        fetch_report = (
            await mcp.call_tool(
                "fetch_ohlcv_batch",
                fetch_args,
            )
        )

        if not isinstance(
            fetch_report,
            dict,
        ):

            raise RuntimeError(
                "fetch_ohlcv_batch returned "
                f"{type(fetch_report).__name__}, "
                "expected dict."
            )

        if fetch_report.get(
            "status"
        ) not in {
            None,
            "ok",
        }:

            raise RuntimeError(
                "fetch_ohlcv_batch failed:\n"
                + json.dumps(
                    fetch_report,
                    indent=2,
                    default=str,
                )
            )

        print()
        print(
            "M1 fetch completed."
        )

        # ==============================================================
        # M4 — NEWS SCORES (optional; failure never stops the run)
        # ==============================================================

        news_scores, news_warnings = await fetch_news_scores(
            mcp=mcp,
            symbols=symbols,
            available_tools=available_tools,
        )

        print()
        print(
            f"M4 news scores: {news_scores}"
        )

        for warning in news_warnings:
            print(
                f"M4 warning: {warning}"
            )

        # ==============================================================
        # M2 — INDICATORS + DETERMINISTIC SCORING
        # ==============================================================

        m2_args = {
            "symbols": symbols,
            "timeframe": timeframe,
            "sleeve": "core",
            "sector": None,
            "news_scores": news_scores,
        }

        print()
        print(
            "MCP → calculate_m2_batch"
        )

        print(
            f"Arguments: {m2_args}"
        )

        m2_report = await mcp.call_tool(
            "calculate_m2_batch",
            m2_args,
        )

        if not isinstance(
            m2_report,
            dict,
        ):

            raise RuntimeError(
                "calculate_m2_batch returned "
                f"{type(m2_report).__name__}, "
                "expected dict."
            )

        if m2_report.get(
            "status"
        ) not in {
            None,
            "ok",
        }:

            raise RuntimeError(
                "calculate_m2_batch failed:\n"
                + json.dumps(
                    m2_report,
                    indent=2,
                    default=str,
                )
            )

        print()
        print(
            "M2 indicators + scoring completed."
        )

    # ==================================================================
    # DETERMINISTIC DECISION ENGINE
    # ==================================================================

    print()
    print(
        "Running deterministic decision engine..."
    )

    decisions = make_decisions(
        m2_report=m2_report,
        held_symbols=holdings,
    )

    # ------------------------------------------------------------------
    # M1 latest-close fallback
    #
    # This does NOT influence BUY/SELL/HOLD.
    # It only guarantees the sizing layer has a usable price when
    # M2-B does not expose last_close.
    # ------------------------------------------------------------------

    apply_m1_price_fallback(
        decisions=decisions,
        fetch_report=fetch_report,
    )

    # ==================================================================
    # M7 ADVISORY DEBATE (opt-in; failure never changes a decision)
    # ==================================================================

    advisory_outcomes = None
    advisory_warnings: list[str] = []

    if advisory_band:

        (
            advisory_outcomes,
            advisory_warnings,
        ) = await _run_advisory_step(
            decisions=decisions,
            timeframe=timeframe,
            band_mode=advisory_band,
        )

    # ==================================================================
    # STARTER SIZING
    # ==================================================================

    plan = make_plan(
        decisions=decisions,
        capital=CAPITAL,
    )

    # ==================================================================
    # M5/M6 GUARDED PLAN (opt-in; proposes only, never places)
    # ==================================================================

    guarded_plan = None

    if guarded:

        guarded_plan = _run_guarded_step(decisions)

        # The guarded sizing replaces the starter sizing in the report
        # and in M9 "actions"; the starter plan stays the default.
        plan = guarded_plan["plan"]

    # ==================================================================
    # REPORT
    # ==================================================================

    print_report(
        symbols=symbols,
        lookback_days=lookback_days,
        timeframe=timeframe,
        holdings=holdings,
        fetch_report=fetch_report,
        m2_report=m2_report,
        decisions=decisions,
        plan=plan,
    )

    if guarded_plan is not None:

        _print_guarded_summary(guarded_plan)

    result = {
        "status": "ok",
        "symbols": symbols,
        "lookback_days": lookback_days,
        "timeframe": timeframe,
        "holdings": holdings,
        "fetch_report": fetch_report,
        "m2_report": m2_report,
        "news_scores": news_scores,
        "news_warnings": news_warnings,
        "decisions": decisions,
        "plan": plan,
    }

    if guarded_plan is not None:

        result["guarded_plan"] = guarded_plan

    if advisory_band:

        result["advisory"] = advisory_outcomes
        result["advisory_warnings"] = advisory_warnings

    return result


def _run_guarded_step(
    decisions: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Size and guardian-check the decisions (M5) against the broker
    snapshot (M6). Proposes only. If the broker cannot be read safely
    nothing is proposed: fail closed.
    """

    from trading_system.crew import guarded_plan as gp
    from trading_system.servers.tools import broker

    broker_mode = (
        os.environ.get("BROKER_MODE") or "offline"
    ).strip().lower()

    try:

        sector_map = gp.load_sector_map()

        snapshot = broker.get_portfolio_snapshot(
            sector_map=sector_map
        )

        # The guardian needs the sector of NEW buys as well, not only
        # of stocks that are already held.
        snapshot["sectors"] = sector_map

    except Exception as exc:

        empty = gp.build_guarded_plan(decisions, {})
        empty["proposals"] = []
        empty["approved"] = []
        empty["blocked"] = []
        empty["warnings"].append(
            "Broker snapshot unavailable "
            f"({type(exc).__name__}); no order proposed."
        )
        empty["broker_error"] = type(exc).__name__
        empty["broker_mode"] = broker_mode
        return empty

    guarded = gp.build_guarded_plan(
        decisions,
        snapshot,
        earnings_dates=gp.load_earnings_dates(),
    )

    guarded["broker_mode"] = broker_mode

    if broker_mode == "offline":

        guarded["warnings"].insert(
            0,
            "OFFLINE FIXTURES: equity, cash and holdings used for "
            "sizing are sample data, not your Alpaca account. "
            "Set BROKER_MODE=alpaca_paper to use your paper account.",
        )

    return guarded


def _print_guarded_summary(
    guarded_plan: dict[str, Any],
) -> None:

    print()
    print("=" * 90)
    print("M5/M6 GUARDIAN (proposals only - nothing is sent anywhere)")
    print("=" * 90)

    print(
        "Account data source: "
        f"{guarded_plan.get('broker_mode', 'unknown')}"
    )

    for order in guarded_plan["approved"]:

        print(
            f"APPROVED {order['side'].upper():<4} {order['symbol']:<6} "
            f"qty={order['qty']} limit={order['limit_price']} "
            f"stop={order.get('stop_price')}"
        )

    for item in guarded_plan["blocked"]:

        order = item.get("order") or {}
        print(
            f"BLOCKED  {str(order.get('side')).upper():<4} "
            f"{order.get('symbol')} by {item['rule']}: {item['detail']}"
        )

    for item in guarded_plan["skipped"]:

        print(
            f"SKIPPED  {item['symbol']} {item['decision']}: "
            f"{item['reason']}"
        )

    for warning in guarded_plan["warnings"]:

        print(f"WARNING: {warning}")

    if not (
        guarded_plan["approved"]
        or guarded_plan["blocked"]
        or guarded_plan["skipped"]
    ):

        print("No BUY or SELL decision to size.")


async def _run_advisory_step(
    decisions: list[dict[str, Any]],
    timeframe: str,
    band_mode: str,
) -> tuple[list[dict[str, Any]], list[str]]:
    """
    Run the M7 debate. Any failure is reported and the deterministic
    decisions stay exactly as M2/M8 produced them.
    """

    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    try:
        from trading_system.crew.advisory import (
            run_advisory_debate,
        )

        # CrewAI kickoff blocks, so it runs in a worker thread.
        outcomes = await asyncio.to_thread(
            run_advisory_debate,
            decisions,
            timeframe,
            band_mode,
        )

        warnings = [
            warning
            for outcome in outcomes
            for warning in outcome.get("warnings", [])
        ]

        return outcomes, warnings

    except Exception as exc:

        key = os.environ.get("GROQ_API_KEY", "").strip()
        detail = f"{type(exc).__name__}: {exc}"

        if key:
            detail = detail.replace(key, "[REDACTED]")

        message = (
            f"M7 advisory unavailable ({detail[:200]}); "
            "deterministic decisions unchanged."
        )

        print()
        print(message)

        return [], [message]


# ============================================================================
# M9 — RUN REPORT + JOURNAL (never changes the run outcome)
# ============================================================================


def _utc_now_iso() -> str:

    return datetime.now(
        timezone.utc
    ).isoformat()


def _new_run_id() -> str:

    return (
        "run-"
        + datetime.now(
            timezone.utc
        ).strftime("%Y%m%dT%H%M%S")
        + "-"
        + uuid.uuid4().hex[:10]
    )


def _is_degraded(
    result: dict[str, Any],
) -> bool:
    """
    A completed run is degraded when some input was unavailable.

    score.warnings is intentionally NOT used: the scoring layer appends
    [VERIFY] warnings to every row, so they would always be present.
    """

    def has_missing(report: Any) -> bool:

        return (
            not isinstance(report, dict)
            or bool(report.get("missing"))
        )

    return bool(
        result.get("news_warnings")
        or result.get("advisory_warnings")
        or has_missing(result.get("m2_report"))
        or has_missing(result.get("fetch_report"))
        or any(
            isinstance(item, dict)
            and item.get("composite") is None
            for item in result.get("decisions") or []
        )
    )


def _build_m9_payload(
    result: dict[str, Any],
    run_id: str,
    timestamp: str,
    degraded: bool,
) -> dict[str, Any]:
    """Compact report payload: no raw M1/M2 reports."""

    decisions = [
        {
            key: item.get(key)
            for key in (
                "symbol",
                "decision",
                "reason",
                "held",
                "triage",
                "composite",
                "latest_close",
                "regime",
                "missing_indicators",
            )
        }
        for item in result.get("decisions") or []
        if isinstance(item, dict)
    ]

    warnings = list(
        result.get("news_warnings") or []
    ) + list(
        result.get("advisory_warnings") or []
    )

    for label, key in (
        ("M1", "fetch_report"),
        ("M2", "m2_report"),
    ):
        report = result.get(key)
        if isinstance(report, dict):
            for item in report.get("missing") or []:
                warnings.append(
                    f"{label} missing: "
                    + (
                        str(item.get("symbol"))
                        if isinstance(item, dict)
                        else str(item)
                    )
                )

    payload = {
        "run_id": run_id,
        "timestamp": timestamp,
        "status": result.get("status"),
        "degraded": degraded,
        "symbols": result.get("symbols"),
        "timeframe": result.get("timeframe"),
        "lookback_days": result.get("lookback_days"),
        "holdings": result.get("holdings"),
        "decisions": decisions,
        "actions": result.get("plan"),
        "warnings": warnings,
    }

    if result.get("guarded_plan") is not None:

        guarded_plan = result["guarded_plan"]
        payload["risk_results"] = {
            "approved": guarded_plan.get("approved"),
            "blocked": guarded_plan.get("blocked"),
            "skipped": guarded_plan.get("skipped"),
            "warnings": guarded_plan.get("warnings"),
        }

    if result.get("advisory") is not None:

        payload["advisory"] = [
            {
                key: item.get(key)
                for key in (
                    "symbol",
                    "band_mode",
                    "composite",
                    "adjustment",
                    "adjusted_composite",
                    "decision_before",
                    "decision_after",
                    "bull",
                    "bear",
                    "reason",
                    "status",
                    "models",
                )
            }
            for item in result["advisory"]
            if isinstance(item, dict)
        ]

    return payload


def _write_m9_artifacts(
    payload: dict[str, Any],
    reports_dir: Any = None,
    memory_dir: Any = None,
) -> tuple[str | None, str | None]:
    """
    Write one run report and one journal entry.

    The M9 modules are imported lazily so that importing this file has
    no side effects and so that launching it as a script still works.
    A writer failure is printed to stderr and never changes the run.
    """

    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    report_path = None
    journal_path = None

    try:
        from trading_system.crew.reporting import record_run_report

        report_path = str(
            record_run_report(payload, reports_dir)
        )

    except Exception as exc:
        print(
            f"M9 report writer warning: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )

    try:
        from trading_system.crew.memory import record_memory

        decisions = payload.get("decisions") or []

        counts = {
            name: sum(
                1
                for item in decisions
                if isinstance(item, dict)
                and item.get("decision") == name
            )
            for name in ("BUY", "SELL", "HOLD")
        }

        journal_path = str(
            record_memory(
                {
                    "run_id": payload.get("run_id"),
                    "timestamp": payload.get("timestamp"),
                    "agent": "M8 orchestrator",
                    "status": payload.get("status"),
                    "symbols": payload.get("symbols"),
                    "timeframe": payload.get("timeframe"),
                    "result": {
                        "degraded": payload.get("degraded"),
                        "decision_counts": counts,
                    },
                    "warnings": payload.get("warnings"),
                    "errors": payload.get("errors"),
                },
                memory_dir,
            )
        )

    except Exception as exc:
        print(
            f"M9 journal writer warning: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )

    return report_path, journal_path


def _write_advisory_journals(
    result: dict[str, Any],
    run_id: str,
    timestamp: str,
    memory_dir: Any = None,
) -> list[str]:
    """
    One timestamped journal entry per agent (Bull, Bear, Judge) per
    debated symbol. A writer failure is printed and never changes the run.
    """

    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    paths: list[str] = []

    try:
        from trading_system.crew.memory import record_memory
    except Exception as exc:
        print(
            f"M9 advisory journal warning: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return paths

    for outcome in result.get("advisory") or []:

        if not isinstance(outcome, dict):
            continue

        models = outcome.get("models") or {}
        symbol = outcome.get("symbol")

        entries = (
            (
                f"M7 Bull {symbol}",
                "ok" if outcome.get("bull") else "failed",
                {
                    "symbol": symbol,
                    "bullets": outcome.get("bull"),
                    "model": models.get("bull"),
                },
            ),
            (
                f"M7 Bear {symbol}",
                "ok" if outcome.get("bear") else "failed",
                {
                    "symbol": symbol,
                    "bullets": outcome.get("bear"),
                    "model": models.get("bear"),
                },
            ),
            (
                f"M7 Judge {symbol}",
                "ok" if outcome.get("status") == "ok" else "failed",
                {
                    "symbol": symbol,
                    "adjustment": outcome.get("adjustment"),
                    "reason": outcome.get("reason"),
                    "composite": outcome.get("composite"),
                    "adjusted_composite": outcome.get("adjusted_composite"),
                    "decision_before": outcome.get("decision_before"),
                    "decision_after": outcome.get("decision_after"),
                    "model": models.get("judge"),
                },
            ),
        )

        for agent, status, body in entries:

            try:
                paths.append(
                    str(
                        record_memory(
                            {
                                "run_id": run_id,
                                "timestamp": timestamp,
                                "agent": agent,
                                "status": status,
                                "symbols": [symbol],
                                "timeframe": result.get("timeframe"),
                                "result": body,
                                "warnings": outcome.get("warnings"),
                            },
                            memory_dir,
                        )
                    )
                )
            except Exception as exc:
                print(
                    f"M9 advisory journal warning: "
                    f"{type(exc).__name__}: {exc}",
                    file=sys.stderr,
                )

    return paths


async def run_strategy(
    symbols: list[str],
    lookback_days: int,
    timeframe: str,
    advisory_band: str | None = None,
    guarded: bool = False,
) -> dict[str, Any]:
    """
    Public entry point: run the strategy, then record M9 artifacts.

    status stays "ok" for a completed run (the CLI exit code depends on
    it); degraded is reported separately. A failed run records a failure
    report and journal entry and then re-raises.
    """

    run_id = _new_run_id()
    started_at = _utc_now_iso()

    try:
        # advisory_band is only passed when requested, so the default
        # call is exactly the pre-advisory call.
        impl_extra = (
            {"advisory_band": advisory_band}
            if advisory_band
            else {}
        )

        if guarded:

            impl_extra["guarded"] = True

        result = await _run_strategy_impl(
            symbols=symbols,
            lookback_days=lookback_days,
            timeframe=timeframe,
            **impl_extra,
        )

    except Exception as exc:
        _write_m9_artifacts(
            {
                "run_id": run_id,
                "timestamp": started_at,
                "status": "failed",
                "degraded": True,
                "symbols": symbols,
                "timeframe": timeframe,
                "lookback_days": lookback_days,
                "errors": [
                    f"{type(exc).__name__}: {exc}"
                ],
            }
        )
        raise

    degraded = _is_degraded(result)

    report_path, journal_path = _write_m9_artifacts(
        _build_m9_payload(
            result,
            run_id,
            started_at,
            degraded,
        )
    )

    result["run_id"] = run_id
    result["degraded"] = degraded
    result["report_path"] = report_path
    result["journal_path"] = journal_path

    if result.get("advisory"):

        result["advisory_journal_paths"] = _write_advisory_journals(
            result,
            run_id,
            started_at,
        )

    return result


# ============================================================================
# CLI
# ============================================================================


def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Deterministic multi-stock paper "
            "trading orchestrator using direct MCP."
        )
    )

    parser.add_argument(
        "symbols",
        nargs="+",
        help=(
            "Stock symbols, e.g. "
            "AAPL MSFT NVDA JPM XOM"
        ),
    )

    parser.add_argument(
        "--lookback",
        type=int,
        default=DEFAULT_LOOKBACK_DAYS,
        help=(
            "Historical lookback in days "
            f"(default: {DEFAULT_LOOKBACK_DAYS})"
        ),
    )

    parser.add_argument(
        "--timeframe",
        choices=[
            "1Day",
            "1Hour",
        ],
        default=DEFAULT_TIMEFRAME,
        help=(
            "Trading timeframe "
            f"(default: {DEFAULT_TIMEFRAME})"
        ),
    )

    parser.add_argument(
        "--advisory",
        action="store_true",
        help=(
            "Run the M7 Bull/Bear/Judge debate (CrewAI + Groq) for "
            "symbols inside the debate band. Needs GROQ_API_KEY. "
            "Off by default; code keeps the final decision."
        ),
    )

    parser.add_argument(
        "--debate-band",
        choices=[
            "spec",
            "borderline",
        ],
        default="spec",
        help=(
            "Which symbols are debated with --advisory: 'spec' = within "
            "0.10 of a clear threshold on either side (spec 8.4); "
            "'borderline' = only rows the scorer labelled borderline "
            "(default: spec)"
        ),
    )

    parser.add_argument(
        "--guarded",
        action="store_true",
        help=(
            "Use the M5 guardian and M6 broker snapshot for sizing "
            "(proposals only; no order is placed). Off by default."
        ),
    )

    return parser.parse_args()


# ============================================================================
# ASYNC ENTRYPOINT
# ============================================================================


async def async_main() -> int:

    args = parse_args()

    try:

        run_extra = (
            {"advisory_band": args.debate_band}
            if args.advisory
            else {}
        )

        if args.guarded:

            run_extra["guarded"] = True

        result = await run_strategy(
            symbols=args.symbols,
            lookback_days=args.lookback,
            timeframe=args.timeframe,
            **run_extra,
        )

        if result.get(
            "status"
        ) == "ok":

            return 0

        return 1

    except KeyboardInterrupt:

        print()
        print(
            "Interrupted by user."
        )

        return 130

    except Exception as exc:

        print()
        print("=" * 90)
        print(
            "FATAL ERROR"
        )
        print("=" * 90)

        print(
            f"{type(exc).__name__}: {exc}"
        )

        return 1


def main() -> int:

    return asyncio.run(
        async_main()
    )


if __name__ == "__main__":

    raise SystemExit(
        main()
    )