"""The thin CrewAI crew is wired into run_cycle as an opt-in, read-only "crew review". Fully offline.

Real code under test: run_cycle (_crew_review_step, config, CLI) and trading_crew's read-only tools.
The Groq client and the crew run are stubbed; nothing touches the network. The point of these tests is
that the review can never change a decision, an order or the cycle's outcome."""

import asyncio
import copy
import sys
from pathlib import Path

import pytest

REPOSITORY_DIR = Path(__file__).resolve().parents[2]
if str(REPOSITORY_DIR) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_DIR))

from trading_system.crew import advisory, llm as llm_module, run_cycle as cycle, trading_crew
from trading_system.servers.tools import broker
from trading_system.tests.test_guarded_cycle import SNAPSHOT, cycle_env, decision, run  # noqa: F401
from trading_system.tests.test_pipeline_offline import isolated_artifact_dirs, pipeline  # noqa: F401


@pytest.fixture()
def stubs(monkeypatch):
    """Stand-ins for the snapshot, the Groq client and the crew. Records what the crew was given."""
    seen = {}
    monkeypatch.setattr(broker, "get_portfolio_snapshot", lambda sector_map=None: dict(SNAPSHOT))
    monkeypatch.setattr(advisory, "install_groq_message_compat", lambda *a, **k: True)
    monkeypatch.setattr(advisory, "_disable_tracing_by_default", lambda: None)

    class FakeClient:
        framework_client = "framework-llm"

    monkeypatch.setattr(llm_module, "create_llm_client", lambda role, *a, **k: seen.setdefault("role", role) and FakeClient())

    def fake_crew(decisions, snapshot, llm, rules=None, **kw):
        seen["rows"] = copy.deepcopy(decisions)
        seen["llm"] = llm
        return {"status": "ok", "report": "AAPL: BUY approved.", "tasks": 2 * len(decisions)}

    monkeypatch.setattr(trading_crew, "run_thin_crew", fake_crew)
    return seen


def result_with(*rows):
    return {"decisions": list(rows)}


def test_only_buy_and_sell_rows_are_reviewed_and_decisions_are_untouched(stubs):
    rows = [decision("AAPL", "BUY"), decision("KO", "HOLD"), decision("TSLA", "SELL", held=True)]
    result = result_with(*rows)
    before = copy.deepcopy(result)
    out = cycle._crew_review_step({}, result)
    assert out["status"] == "ok" and out["reviewed"] == ["AAPL", "TSLA"]
    assert out["report"] == "AAPL: BUY approved." and out["tasks"] == 4
    assert [r["symbol"] for r in stubs["rows"]] == ["AAPL", "TSLA"]
    assert stubs["llm"] == "framework-llm" and stubs["role"] == "news"
    assert result == before


def test_no_buy_or_sell_means_no_llm_call(stubs):
    out = cycle._crew_review_step({}, result_with(decision("KO", "HOLD")))
    assert out["status"] == "skipped" and out["report"] is None
    assert "rows" not in stubs and out["warnings"]


def test_review_is_capped_to_bound_the_token_cost(stubs):
    rows = [decision(f"S{i}", "BUY") for i in range(cycle.MAX_CREW_REVIEW_ROWS + 4)]
    out = cycle._crew_review_step({}, result_with(*rows))
    assert len(out["reviewed"]) == cycle.MAX_CREW_REVIEW_ROWS


def test_any_failure_becomes_a_warning_and_hides_the_key(stubs, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gsk_secret_value")

    def boom(*a, **k):
        raise RuntimeError("rate limit for gsk_secret_value")

    monkeypatch.setattr(trading_crew, "run_thin_crew", boom)
    out = cycle._crew_review_step({}, result_with(decision("AAPL", "BUY")))
    assert out["status"] == "unavailable" and out["report"] is None
    text = " ".join(out["warnings"])
    assert "gsk_secret_value" not in text and "[REDACTED]" in text and "unchanged" in text


def test_missing_groq_key_is_a_warning_not_a_crash(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setattr(broker, "get_portfolio_snapshot", lambda sector_map=None: dict(SNAPSHOT))
    out = cycle._crew_review_step({}, result_with(decision("AAPL", "BUY")))
    assert out["status"] == "unavailable" and out["warnings"]


def test_cli_flag_and_config_validation():
    assert cycle._parse_cli(["AAPL", "--crew-review"])["crew_review"] is True
    assert "crew_review" not in cycle._parse_cli(["AAPL"])
    assert cycle._validate({"symbols": ["aapl"], "crew_review": True})["crew_review"] is True
    assert cycle._validate({"symbols": ["aapl"]})["crew_review"] is False


def test_default_cycle_has_no_review_and_runs_no_crew(cycle_env, stubs):
    assert run({})["crew_review"] is None
    assert "rows" not in stubs


def test_the_review_never_changes_what_the_cycle_proposes(cycle_env, stubs):
    plain = run({})
    reviewed = run({"crew_review": True})
    assert reviewed["crew_review"]["status"] == "ok"
    for key in ("status", "approved", "blocked", "skipped", "proposals", "decisions", "placed"):
        assert reviewed[key] == plain[key], key


def test_a_failing_review_does_not_fail_or_alter_the_cycle(cycle_env, stubs, monkeypatch):
    plain = run({})
    monkeypatch.setattr(trading_crew, "run_thin_crew", lambda *a, **k: (_ for _ in ()).throw(ValueError("x")))
    reviewed = run({"crew_review": True})
    assert reviewed["crew_review"]["status"] == "unavailable"
    assert reviewed["approved"] == plain["approved"] and reviewed["status"] == plain["status"]
