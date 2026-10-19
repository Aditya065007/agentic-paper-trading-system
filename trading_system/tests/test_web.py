"""Offline tests for local dashboard validation, key handling, and saved-order approval."""
import time
from pathlib import Path

import pytest
from starlette.testclient import TestClient

from trading_system.web import keys, server, store


def test_symbols_are_normalized_and_deduplicated():
    assert server.normalize_symbols([" aapl ", "MSFT", "AAPL"]) == ["AAPL", "MSFT"]


@pytest.mark.parametrize("symbols", [[], ["bad symbol"], ["", "AAPL"], [123], ["A" * 11], ["AAPL"] * 26])
def test_invalid_symbol_lists_are_rejected(symbols):
    with pytest.raises(ValueError):
        server.normalize_symbols(symbols)


def test_key_settings_cannot_change_order_gates_or_broker_url():
    for name in ("ALPACA_ORDERS_ENABLED", "ALPACA_STOPS_ENABLED", "ALPACA_TRADING_URL"):
        with pytest.raises(ValueError):
            keys.save_settings({name: "true"})


def test_key_settings_reject_newlines_and_invalid_broker_mode():
    with pytest.raises(ValueError):
        keys.save_settings({"GROQ_API_KEY": "abc\nALPACA_ORDERS_ENABLED=true"})
    with pytest.raises(ValueError):
        keys.save_settings({"BROKER_MODE": "live"})


def test_public_settings_mask_secret_values(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gsk_example_secret")
    public = keys.public_settings()["GROQ_API_KEY"]
    assert public["configured"] is True
    assert public["masked"].endswith("cret")
    assert "gsk_example_secret" not in str(public)


def test_run_store_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "STATE_DIR", tmp_path / "state")
    monkeypatch.setattr(store, "RUNS_FILE", tmp_path / "state" / "runs.json")
    record = {"id": "test-run", "status": "completed"}
    store.save_run(record)
    assert store.get_run("test-run") == record


def test_approval_uses_exact_saved_orders_and_never_reruns_pipeline(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "STATE_DIR", tmp_path / "state")
    monkeypatch.setattr(store, "RUNS_FILE", tmp_path / "state" / "runs.json")
    monkeypatch.setattr(server, "LAUNCH_TOKEN", "test-launch-token")
    monkeypatch.setattr(server, "LIVE_MODE", False)
    orders = [{"symbol": "AAPL", "side": "buy", "qty": 2, "limit_price": 100.0}]
    store.save_run({
        "id": "run-1", "status": "completed", "pipeline_status": "awaiting_approval",
        "approval_expires_at": time.time() + 60, "approved_orders": orders,
        "result": {"run_id": "pipeline-1", "timeframe": "1Day"},
    })
    captured = {}
    def place(saved_orders, only_symbols=None):
        captured["orders"] = saved_orders
        captured["symbols"] = only_symbols
        return [{"symbol": "AAPL", "submitted": False, "status": "dry_run"}]
    from trading_system.crew import guarded_plan
    monkeypatch.setattr(guarded_plan, "place_approved", place)
    client = TestClient(server.app, base_url="http://127.0.0.1:8765")
    response = client.post(
        "/api/runs/run-1/approve",
        headers={"X-Launch-Token": "test-launch-token"},
        json={"symbols": ["AAPL"], "confirm": True},
    )
    assert response.status_code == 200, response.text
    assert captured == {"orders": orders, "symbols": ["AAPL"]}
    assert store.get_run("run-1")["status"] == "approved"
    second = client.post(
        "/api/runs/run-1/approve",
        headers={"X-Launch-Token": "test-launch-token"},
        json={"symbols": ["AAPL"], "confirm": True},
    )
    assert second.status_code == 409


def test_local_server_binds_only_to_loopback():
    source = Path(server.__file__).read_text(encoding="utf-8")
    assert 'uvicorn.run(app, host="127.0.0.1"' in source
    assert "PAPER_WEB_LIVE_MODE" in source
