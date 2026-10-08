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
    held_symbols: list[str],
) -> list[dict[str, Any]]:
    """
    Deterministic M2-B decision engine.

    M2-B is authoritative for technical triage.

    BUY / SELL / HOLD are derived from the deterministic
    composite score and triage output.

    If M2-B cannot produce a composite score:
        HOLD

    LLM/M4/M5 are intentionally not involved here yet.
    """

    if not isinstance(
        m2_report,
        dict,
    ):

        raise ValueError(
            "M2 report must be a dictionary."
        )

    held = {
        str(symbol).upper()
        for symbol in held_symbols
    }

    results = m2_report.get(
        "results",
        [],
    )

    missing = m2_report.get(
        "missing",
        [],
    )

    if not isinstance(
        results,
        list,
    ):
        results = []

    if not isinstance(
        missing,
        list,
    ):
        missing = []

    decisions: list[
        dict[str, Any]
    ] = []

    for item in results:

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

        if not symbol:
            continue

        score = item.get(
            "score"
        ) or {}

        indicators = item.get(
            "indicators"
        ) or {}

        # Defensive context lookup.
        context = indicators.get(
            "context"
        ) or {}

        composite = score.get(
            "composite"
        )

        triage = str(
            score.get(
                "triage",
                "hold",
            )
        ).lower()

        # M2-A may not expose last_close in context.
        # Try context first, then M2-B score.
        last_close = context.get(
            "last_close"
        )

        if last_close is None:

            last_close = score.get(
                "last_close"
            )

        # --------------------------------------------------------------
        # Fail-safe when composite is unavailable.
        #
        # This is expected until M4/news is connected if the scoring
        # configuration requires the news group.
        # --------------------------------------------------------------

        if composite is None:

            decision = "HOLD"

            reason = (
                "M2-B composite unavailable; "
                "fail-safe HOLD."
            )

        elif triage == "buy":

            decision = "BUY"

            reason = (
                "M2-B deterministic BUY; "
                f"composite={float(composite):.4f}."
            )

        elif triage == "sell":

            decision = "SELL"

            reason = (
                "M2-B deterministic SELL; "
                f"composite={float(composite):.4f}."
            )

        else:

            decision = "HOLD"

            reason = (
                "M2-B deterministic HOLD; "
                f"triage={triage}, "
                f"composite={float(composite):.4f}."
            )

        decisions.append(
            {
                "symbol": symbol,
                "decision": decision,
                "reason": reason,
                "held": symbol in held,
                "latest_close": last_close,
                "as_of": score.get(
                    "as_of"
                ),
                "composite": composite,
                "triage": triage,
                "regime": score.get(
                    "regime"
                ),
                "regime_strength": score.get(
                    "regime_strength"
                ),
                "adx": score.get(
                    "adx"
                ),
                "sub_scores": score.get(
                    "sub_scores",
                    {},
                ),
                "missing_indicators": indicators.get(
                    "missing_indicators",
                    [],
                ),
                "warnings": indicators.get(
                    "warnings",
                    [],
                ),
            }
        )

    # --------------------------------------------------------------
    # Symbols M2 could not process.
    # Fail-safe HOLD.
    # --------------------------------------------------------------

    represented = {
        item["symbol"]
        for item in decisions
        if item.get("symbol")
    }

    for item in missing:

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

        if (
            not symbol
            or symbol in represented
        ):
            continue

        decisions.append(
            {
                "symbol": symbol,
                "decision": "HOLD",
                "reason": (
                    "M2 returned no usable result; "
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
                "sub_scores": {},
                "missing": item,
            }
        )

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

        print(
            f"{str(item.get('symbol', '')):>6} | "
            f"bars={item.get('bars')} | "
            f"last_close={item.get('last_close')} | "
            f"as_of={item.get('as_of')}"
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
# MAIN STRATEGY
# ============================================================================


async def run_strategy(
    symbols: list[str],
    lookback_days: int,
    timeframe: str,
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
        # M2 — INDICATORS + DETERMINISTIC SCORING
        # ==============================================================

        m2_args = {
            "symbols": symbols,
            "timeframe": timeframe,
            "sleeve": "core",
            "sector": None,
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
    # STARTER SIZING
    # ==================================================================

    plan = make_plan(
        decisions=decisions,
        capital=CAPITAL,
    )

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

    return {
        "status": "ok",
        "symbols": symbols,
        "lookback_days": lookback_days,
        "timeframe": timeframe,
        "holdings": holdings,
        "fetch_report": fetch_report,
        "m2_report": m2_report,
        "decisions": decisions,
        "plan": plan,
    }


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

    return parser.parse_args()


# ============================================================================
# ASYNC ENTRYPOINT
# ============================================================================


async def async_main() -> int:

    args = parse_args()

    try:

        result = await run_strategy(
            symbols=args.symbols,
            lookback_days=args.lookback,
            timeframe=args.timeframe,
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