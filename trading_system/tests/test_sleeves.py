"""G1: per-symbol sleeve and sector from config/universe.yaml reach the scorer. Fully offline.

Real code under test: the MCP server's calculate_m2_batch, scoring.py, guarded_plan loaders and the
orchestrator's M2 call. Bars and indicators are fakes (harness from test_pipeline_offline.py)."""

import asyncio
import sys
from pathlib import Path

import pytest

REPOSITORY_DIR = Path(__file__).resolve().parents[2]
if str(REPOSITORY_DIR) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_DIR))

import yaml

from trading_system.crew import guarded_plan
from trading_system.crew import trading_crew_multiStocks as crew
from trading_system.servers.tools import scoring
from trading_system.tests.test_pipeline_offline import (  # noqa: F401  (fixtures + harness)
    OfflineMCP, SYMBOLS, isolated_artifact_dirs, pipeline, run_pipeline)

UNIVERSE = Path(__file__).resolve().parents[1] / "config" / "universe.yaml"


# ------------------------------------------------------------------ loader

def write(tmp_path, text):
    path = tmp_path / "universe.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_loader_reads_sleeves_and_normalises_case(tmp_path):
    path = write(tmp_path, "sleeves:\n  nvda: High_Beta\n  AAPL: core\n")
    assert guarded_plan.load_sleeve_map(path) == {"NVDA": "high_beta", "AAPL": "core"}


def test_loader_drops_unknown_sleeve_names_so_a_typo_stays_core(tmp_path):
    path = write(tmp_path, "sleeves:\n  NVDA: hihg_beta\n  TSLA: high_beta\n")
    assert guarded_plan.load_sleeve_map(path) == {"TSLA": "high_beta"}


@pytest.mark.parametrize("text", ["", "sectors:\n  AAPL: Technology\n", "sleeves: [a, b]\n", "sleeves: [\n"])
def test_loader_gives_empty_map_when_file_or_block_is_missing_or_bad(tmp_path, text):
    assert guarded_plan.load_sleeve_map(write(tmp_path, text)) == {}
    assert guarded_plan.load_sleeve_map(tmp_path / "nope.yaml") == {}


def test_shipped_universe_file_is_valid_and_meets_the_30_percent_rule():
    data = yaml.safe_load(UNIVERSE.read_text(encoding="utf-8"))
    assert set(data["sleeves"].values()) <= {"core", "high_beta"}
    stocks = [s for s, sector in data["sectors"].items() if sector != "Index"]
    high = [s for s in stocks if data["sleeves"].get(s) == "high_beta"]
    assert set(data["sleeves"]) <= set(data["sectors"])           # every tagged ticker has a sector
    assert len(high) / len(stocks) >= 0.30                        # design decision D10
    assert guarded_plan.load_sleeve_map() == data["sleeves"]


# ------------------------------------------------------------------ server

def test_server_scores_each_symbol_with_its_own_sleeve(pipeline):
    out = pipeline.calculate_m2_batch(
        symbols=["AAPL", "XOM"], timeframe="1Day", sleeves={"XOM": "high_beta"},
        sectors={"AAPL": "Technology"})
    rows = {r["symbol"]: r["score"] for r in out["results"]}
    assert rows["AAPL"]["sleeve"] == "core" and rows["XOM"]["sleeve"] == "high_beta"
    assert rows["AAPL"]["sector"] == "Technology" and rows["XOM"]["sector"] is None


def test_server_without_maps_behaves_as_before(pipeline):
    out = pipeline.calculate_m2_batch(symbols=["AAPL", "XOM"], timeframe="1Day")
    assert {r["score"]["sleeve"] for r in out["results"]} == {"core"}


def test_server_map_keys_are_case_insensitive_and_override_the_default(pipeline):
    out = pipeline.calculate_m2_batch(symbols=["aapl", "XOM"], timeframe="1Day", sleeve="high_beta",
                                      sleeves={"aapl": "core"})
    rows = {r["symbol"]: r["score"]["sleeve"] for r in out["results"]}
    assert rows == {"AAPL": "core", "XOM": "high_beta"}


def test_server_unknown_sleeve_is_not_guessed_the_symbol_gets_no_score(pipeline):
    out = pipeline.calculate_m2_batch(symbols=["AAPL"], timeframe="1Day", sleeves={"AAPL": "wild"})
    score = out["results"][0]["score"]
    assert score["composite"] is None
    assert any("sleeve must be" in w for w in score["warnings"])


def test_the_two_sleeves_really_weight_the_same_inputs_differently():
    assert scoring.WEIGHTS["core"] != scoring.WEIGHTS["high_beta"]
    assert scoring.WEIGHTS["high_beta"]["news"] < scoring.WEIGHTS["core"]["news"]


# ------------------------------------------------------------------ orchestrator

def test_orchestrator_sends_only_the_requested_symbols_maps(monkeypatch, pipeline):
    monkeypatch.setattr(guarded_plan, "load_sleeve_map", lambda: {"AAPL": "high_beta", "NVDA": "high_beta"})
    monkeypatch.setattr(guarded_plan, "load_sector_map", lambda: {"AAPL": "Technology", "ZZZZ": "Other"})
    seen = {}
    original = OfflineMCP.call_tool

    async def spy(self, name, arguments):
        if name == "calculate_m2_batch":
            seen.update(arguments)
        return await original(self, name, arguments)

    monkeypatch.setattr(OfflineMCP, "call_tool", spy)
    result, _ = run_pipeline(monkeypatch, pipeline)
    assert seen["sleeves"] == {"AAPL": "high_beta"}            # NVDA/ZZZZ are not in this run
    assert seen["sectors"] == {"AAPL": "Technology"}
    assert seen["sleeve"] == "core" and seen["sector"] is None  # defaults unchanged
    assert result["status"] == "ok"


def test_orchestrator_still_runs_when_the_universe_file_is_missing(monkeypatch, pipeline):
    monkeypatch.setattr(guarded_plan, "load_sleeve_map", lambda: {})
    monkeypatch.setattr(guarded_plan, "load_sector_map", lambda: {})
    result, _ = run_pipeline(monkeypatch, pipeline)
    assert result["status"] == "ok"
