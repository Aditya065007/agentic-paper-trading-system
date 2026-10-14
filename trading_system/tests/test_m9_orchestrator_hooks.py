"""T4: the orchestrator records a run report and a journal entry (offline, mocked)."""

import asyncio
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_DIR = Path(__file__).resolve().parents[1]
REPOSITORY_DIR = PROJECT_DIR.parent
if str(REPOSITORY_DIR) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_DIR))

from trading_system.crew import memory, reporting
from trading_system.crew import trading_crew_multiStocks as crew

CREW_FILE = PROJECT_DIR / "crew" / "trading_crew_multiStocks.py"
SECRET = "sk-test-SECRET-1234567890"


@pytest.fixture(autouse=True)
def isolated_artifact_dirs(tmp_path, monkeypatch):
    """Never write reports or journal entries into the real repository folders."""
    monkeypatch.setattr(reporting, "REPORTS_DIR", tmp_path / "run_reports")
    monkeypatch.setattr(memory, "MEMORY_DIR", tmp_path / "memory")
    return tmp_path


class FakeMCP:
    def __init__(self, responses):
        self.responses, self.calls = responses, []

    async def list_tools(self):
        return ["fetch_ohlcv_batch", "calculate_m2_batch", "get_news", "score_headlines"]

    async def call_tool(self, name, arguments):
        self.calls.append((name, arguments))
        response = self.responses[name]
        if isinstance(response, Exception):
            raise response
        return response


def install(monkeypatch, responses):
    mcp = FakeMCP(responses)

    class Client:
        def __init__(self, path):
            pass

        async def __aenter__(self):
            return mcp

        async def __aexit__(self, *exc):
            return False

    monkeypatch.setattr(crew, "DirectMCPClient", Client)
    monkeypatch.setattr(crew, "get_holdings", lambda: [])
    return mcp


def responses(**overrides):
    base = {
        "fetch_ohlcv_batch": {"status": "ok", "missing": [], "fetched": [
            {"symbol": "AAPL", "bars_fetched": 126, "end": "2026-10-09", "last_close": 190.0}]},
        "calculate_m2_batch": {"status": "ok", "results": [], "missing": []},
        "get_news": {"headlines": [], "warnings": [], "errors": []},
        "score_headlines": {"by_symbol": {"AAPL": {"news_score": 0.6}}},
    }
    base.update(overrides)
    return base


def run_strategy():
    return asyncio.run(crew.run_strategy(symbols=["AAPL"], lookback_days=180, timeframe="1Day"))


PAYLOAD = {"run_id": "run-1", "timestamp": "2026-10-12T10:15:00+00:00", "status": "ok",
           "symbols": ["AAPL"], "decisions": [{"symbol": "AAPL", "decision": "BUY"}], "warnings": []}


# --------------------------- the writer helper ---------------------------

def test_helper_writes_a_report_and_a_journal_entry(tmp_path):
    report, journal = crew._write_m9_artifacts(PAYLOAD, tmp_path / "r", tmp_path / "m")
    assert Path(report).is_file() and Path(journal).is_file()
    assert ":" not in Path(report).name and ":" not in Path(journal).name
    assert "run-1" in Path(report).read_text(encoding="utf-8")
    assert "BUY" in Path(journal).read_text(encoding="utf-8")


def test_secrets_never_reach_the_files(tmp_path, monkeypatch):
    monkeypatch.setenv("NEWSAPI_KEY", SECRET)
    payload = dict(PAYLOAD, warnings=[f"upstream said key={SECRET}"], errors=[f"NEWSAPI_KEY={SECRET}"])
    report, journal = crew._write_m9_artifacts(payload, tmp_path / "r", tmp_path / "m")
    for path in (report, journal):
        text = Path(path).read_text(encoding="utf-8")
        assert SECRET not in text and "[REDACTED]" in text


def test_a_failing_report_writer_never_raises(tmp_path, monkeypatch, capsys):
    def boom(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(reporting, "record_run_report", boom)
    report, journal = crew._write_m9_artifacts(PAYLOAD, tmp_path / "r", tmp_path / "m")
    assert report is None and Path(journal).is_file()
    assert "M9 report writer warning: OSError: disk full" in capsys.readouterr().err


def test_a_failing_journal_writer_never_raises(tmp_path, monkeypatch, capsys):
    def boom(*args, **kwargs):
        raise OSError("read-only")

    monkeypatch.setattr(memory, "record_memory", boom)
    report, journal = crew._write_m9_artifacts(PAYLOAD, tmp_path / "r", tmp_path / "m")
    assert Path(report).is_file() and journal is None
    assert "M9 journal writer warning: OSError: read-only" in capsys.readouterr().err


# --------------------------- run_strategy integration ---------------------------

def test_a_clean_run_writes_artifacts_and_stays_ok(monkeypatch, isolated_artifact_dirs):
    install(monkeypatch, responses())
    result = run_strategy()
    assert result["status"] == "ok" and result["degraded"] is False
    assert re.fullmatch(r"run-\d{8}T\d{6}-[0-9a-f]{10}", result["run_id"])
    report = Path(result["report_path"]).read_text(encoding="utf-8")
    assert result["run_id"] in report and "Status: ok" in report
    assert "m2_report" not in report and "fetch_report" not in report   # compact payload only
    assert Path(result["journal_path"]).is_file()
    assert Path(result["report_path"]).parent == isolated_artifact_dirs / "run_reports"


def test_a_missing_symbol_marks_the_run_degraded_but_keeps_status_ok(monkeypatch):
    install(monkeypatch, responses(calculate_m2_batch={
        "status": "ok", "results": [], "missing": [{"symbol": "AAPL", "error": "no cache"}]}))
    result = run_strategy()
    assert result["status"] == "ok" and result["degraded"] is True
    assert "M2 missing: AAPL" in Path(result["report_path"]).read_text(encoding="utf-8")


def test_missing_news_marks_the_run_degraded(monkeypatch):
    install(monkeypatch, responses(get_news=RuntimeError("feed down")))
    result = run_strategy()
    assert result["status"] == "ok" and result["degraded"] is True
    assert "News unavailable: RuntimeError: feed down" in result["news_warnings"]


def test_exit_code_is_zero_for_a_degraded_run(monkeypatch):
    install(monkeypatch, responses(get_news=RuntimeError("feed down")))
    monkeypatch.setattr(sys, "argv", ["crew", "AAPL"])
    assert asyncio.run(crew.async_main()) == 0


def test_a_failed_run_records_a_failure_and_reraises(monkeypatch, isolated_artifact_dirs):
    install(monkeypatch, responses(fetch_ohlcv_batch=RuntimeError("alpaca down")))
    with pytest.raises(RuntimeError, match="alpaca down"):
        run_strategy()
    reports = list((isolated_artifact_dirs / "run_reports").glob("*.md"))
    journals = list((isolated_artifact_dirs / "memory").glob("*.md"))
    assert len(reports) == 1 and len(journals) == 1
    text = reports[0].read_text(encoding="utf-8")
    assert "Status: failed" in text and "alpaca down" in text


def test_exit_code_is_one_for_a_failed_run(monkeypatch):
    install(monkeypatch, responses(fetch_ohlcv_batch=RuntimeError("alpaca down")))
    monkeypatch.setattr(sys, "argv", ["crew", "AAPL"])
    assert asyncio.run(crew.async_main()) == 1


def test_a_writer_failure_does_not_change_the_run(monkeypatch):
    install(monkeypatch, responses())

    def boom(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(reporting, "record_run_report", boom)
    result = run_strategy()
    assert result["status"] == "ok" and result["report_path"] is None
    assert Path(result["journal_path"]).is_file()


# --------------------------- launching as a script ---------------------------

def clean_env():
    """Current import path without the repository root, so the test proves the file is self-sufficient."""
    kept = [entry for entry in sys.path if entry and Path(entry).resolve() != REPOSITORY_DIR]
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(kept)
    return env


def test_the_crew_file_can_still_be_launched_as_a_script(tmp_path):
    done = subprocess.run([sys.executable, str(CREW_FILE), "--help"], cwd=tmp_path,
                          env=clean_env(), capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr
    assert "usage" in done.stdout.lower()


def test_artifacts_can_be_written_when_the_file_is_run_as_a_script(tmp_path):
    code = (
        "import json, runpy, sys\n"
        "namespace = runpy.run_path(sys.argv[1], run_name='crew_script_probe')\n"
        "payload = {'run_id': 'r1', 'timestamp': '2026-10-12T10:15:00+00:00', 'status': 'ok',\n"
        "           'symbols': ['AAPL'], 'decisions': [], 'warnings': []}\n"
        "print(json.dumps(namespace['_write_m9_artifacts'](payload, sys.argv[2], sys.argv[3])))\n"
    )
    done = subprocess.run(
        [sys.executable, "-c", code, str(CREW_FILE), str(tmp_path / "r"), str(tmp_path / "m")],
        cwd=tmp_path, env=clean_env(), capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr
    report, journal = json.loads(done.stdout.strip().splitlines()[-1])
    assert Path(report).is_file() and Path(journal).is_file()
