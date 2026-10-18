"""Keep every test away from the real memory/ and run_reports/ folders.

M9 writes decisions.jsonl, trade_journal.md, lessons.md and the per-agent md files whenever a cycle runs.
Tests that run a cycle without redirecting the folders would otherwise add fake records to the
user's real journal. Individual tests may still point the folders elsewhere themselves."""

import pytest

from trading_system.crew import memory, reporting


@pytest.fixture(autouse=True)
def _isolated_m9_folders(tmp_path_factory, monkeypatch):
    root = tmp_path_factory.mktemp("m9")
    monkeypatch.setattr(memory, "MEMORY_DIR", root / "memory")
    monkeypatch.setattr(reporting, "REPORTS_DIR", root / "run_reports")
