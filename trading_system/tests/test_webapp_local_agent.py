import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from trading_system.webapp import local_agent as agent


def sample_result():
    return {
        "status": "awaiting_approval",
        "run_id": "run-test-1",
        "timeframe": "1Day",
        "symbols": ["AAPL", "TSLA"],
        "decisions": [
            {"symbol": "AAPL", "decision": "BUY", "latest_close": 100.0, "composite": 0.8},
            {"symbol": "TSLA", "decision": "SELL", "latest_close": 200.0, "composite": -0.6},
        ],
        "approved": [
            {"symbol": "AAPL", "side": "buy", "qty": 3, "limit_price": 100.2, "stop_price": 93.6},
            {"symbol": "TSLA", "side": "sell", "qty": 2, "limit_price": 199.8, "stop_price": None},
        ],
        "blocked": [],
        "skipped": [],
        "warnings": [],
        "placed": [],
    }


def test_validate_symbols_normalizes_deduplicates_and_limits():
    assert agent.validate_symbols("aapl, MSFT, AAPL") == ["AAPL", "MSFT"]
    with pytest.raises(agent.RequestError):
        agent.validate_symbols("AAPL; rm -rf")
    with pytest.raises(agent.RequestError):
        agent.validate_symbols(",".join(f"S{i}" for i in range(16)))


def test_validate_cycle_config_rejects_bad_timeframe_and_lookback():
    with pytest.raises(agent.RequestError):
        agent.validate_cycle_request({"symbols": ["AAPL"], "timeframe": "5Min"})
    with pytest.raises(agent.RequestError):
        agent.validate_cycle_request({"symbols": ["AAPL"], "lookback_days": 2})


def test_snapshot_redacts_environment_secrets_and_sensitive_fields(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test_do_not_leak")
    clean = agent.safe_snapshot({
        "reason": "call failed gsk_test_do_not_leak",
        "credentials": {"api_key": "should-not-appear"},
        "approved": [{"symbol": "AAPL", "qty": 1}],
    })
    rendered = str(clean)
    assert "gsk_test_do_not_leak" not in rendered
    assert "should-not-appear" not in rendered
    assert "[REDACTED]" in rendered
    assert clean["approved"][0]["symbol"] == "AAPL"


def test_update_local_settings_keeps_values_in_process_only(monkeypatch):
    for name in agent.KEY_NAMES:
        monkeypatch.delenv(name, raising=False)
    result = agent.update_local_settings({
        "ALPACA_API_KEY": "paper-key-example",
        "GROQ_API_KEY": "groq-key-example",
        "broker_mode": "offline",
    })
    assert os.environ["ALPACA_API_KEY"] == "paper-key-example"
    assert result["keys"]["ALPACA_API_KEY"]["configured"] is True
    assert "paper-key-example" not in str(result)
    assert not agent.STATE_DIR.exists()


def test_cycle_is_saved_as_pending_and_approval_uses_saved_orders_without_rerun(monkeypatch, tmp_path):
    monkeypatch.setattr(agent, "STATE_DIR", tmp_path)
    calls = []
    monkeypatch.setattr(agent.run_cycle, "run_cycle_sync", lambda config: calls.append(config) or sample_result())
    monkeypatch.setattr(agent.journal, "append_order_results", lambda *args, **kwargs: {})
    placed_calls = []
    monkeypatch.setattr(agent.guarded_plan, "place_approved",
                        lambda orders, only_symbols=None: placed_calls.append((orders, only_symbols)) or [
                            {"status": "accepted_dry_run", "submitted": False,
                             "details": {"symbol": o["symbol"], "side": o["side"], "qty": o["qty"],
                                         "limit_price": o["limit_price"]}}
                            for o in orders if o["symbol"] in set(only_symbols or [])
                        ])
    snapshot = agent.execute_cycle({"symbols": "AAPL,TSLA", "timeframe": "1Day"})
    assert snapshot["approval_state"] == "pending"
    assert len(calls) == 1
    saved = agent._load_snapshot(snapshot["snapshot_id"])
    assert saved["approved"] == sample_result()["approved"]

    result = agent.approve_saved_orders({
        "snapshot_id": snapshot["snapshot_id"], "symbols": ["AAPL"], "mode": "dry_run"
    })
    assert len(calls) == 1, "approval must not rerun the analysis"
    assert result["approval_state"] == "completed"
    assert result["approved_symbols"] == ["AAPL"]
    assert [p[0]["symbol"] for p in placed_calls[0][0]] == ["AAPL"]


def test_approval_rejects_symbols_not_in_saved_guardian_orders(monkeypatch, tmp_path):
    monkeypatch.setattr(agent, "STATE_DIR", tmp_path)
    monkeypatch.setattr(agent.run_cycle, "run_cycle_sync", lambda config: sample_result())
    snapshot = agent.execute_cycle({"symbols": ["AAPL", "TSLA"]})
    with pytest.raises(agent.RequestError):
        agent.approve_saved_orders({
            "snapshot_id": snapshot["snapshot_id"], "symbols": ["MSFT"], "mode": "dry_run"
        })


def test_snapshot_cannot_be_approved_twice(monkeypatch, tmp_path):
    monkeypatch.setattr(agent, "STATE_DIR", tmp_path)
    monkeypatch.setattr(agent.run_cycle, "run_cycle_sync", lambda config: sample_result())
    monkeypatch.setattr(agent.journal, "append_order_results", lambda *args, **kwargs: {})
    monkeypatch.setattr(agent.guarded_plan, "place_approved", lambda orders, only_symbols=None: [])
    snapshot = agent.execute_cycle({"symbols": ["AAPL", "TSLA"]})
    request = {"snapshot_id": snapshot["snapshot_id"], "symbols": ["AAPL"], "mode": "dry_run"}
    agent.approve_saved_orders(request)
    with pytest.raises(agent.RequestError):
        agent.approve_saved_orders(request)


def test_live_approval_is_blocked_when_master_gate_is_off(monkeypatch, tmp_path):
    monkeypatch.setattr(agent, "STATE_DIR", tmp_path)
    monkeypatch.setattr(agent.run_cycle, "run_cycle_sync", lambda config: sample_result())
    monkeypatch.setenv("BROKER_MODE", "alpaca_paper")
    monkeypatch.setenv("ALPACA_API_KEY", "paper-key")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "paper-secret")
    monkeypatch.delenv("ALPACA_ORDERS_ENABLED", raising=False)
    snapshot = agent.execute_cycle({"symbols": ["AAPL", "TSLA"]})
    with pytest.raises(agent.RequestError, match="safety gate"):
        agent.approve_saved_orders({
            "snapshot_id": snapshot["snapshot_id"], "symbols": ["AAPL"], "mode": "alpaca_paper"
        })


def test_dashboard_cannot_enable_order_or_stop_safety_gates(monkeypatch):
    monkeypatch.delenv("ALPACA_ORDERS_ENABLED", raising=False)
    monkeypatch.delenv("ALPACA_STOPS_ENABLED", raising=False)
    for name in ("ALPACA_ORDERS_ENABLED", "ALPACA_STOPS_ENABLED"):
        with pytest.raises(agent.RequestError, match="cannot be changed"):
            agent.update_local_settings({name: "true"})
    assert "ALPACA_ORDERS_ENABLED" not in os.environ
    assert "ALPACA_STOPS_ENABLED" not in os.environ


def test_stale_snapshot_expires_before_approval(monkeypatch, tmp_path):
    monkeypatch.setattr(agent, "STATE_DIR", tmp_path)
    monkeypatch.setattr(agent.run_cycle, "run_cycle_sync", lambda config: sample_result())
    snapshot = agent.execute_cycle({"symbols": ["AAPL", "TSLA"]})
    snapshot["created_at"] = "2020-01-01T00:00:00+00:00"
    agent._save_snapshot(snapshot)
    with pytest.raises(agent.RequestError, match="expired"):
        agent.approve_saved_orders({
            "snapshot_id": snapshot["snapshot_id"], "symbols": ["AAPL"], "mode": "dry_run"
        })
    assert agent._load_snapshot(snapshot["snapshot_id"])["approval_state"] == "expired"
