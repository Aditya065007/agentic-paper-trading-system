"""Offline tests for local dashboard validation, key handling, and saved-order approval."""
import json
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
        "key_fingerprint": keys.fingerprint(),
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


# ---------------------------------------------------------------------------------------------------------------
# Live placement, key rotation, request guard and lock behaviour (added with the live/rotation work).
# Every test uses temporary files: the real .env and run history are never touched.
# ---------------------------------------------------------------------------------------------------------------
import os

from trading_system.crew import journal as journal_module
from trading_system.crew import run_cycle as run_cycle_module
from trading_system.servers.tools import live_orders

TOKEN = {"X-Launch-Token": "test-launch-token"}
ORDERS = [{"symbol": "AAPL", "side": "buy", "qty": 2, "limit_price": 100.0},
          {"symbol": "MSFT", "side": "buy", "qty": 1, "limit_price": 200.0}]


@pytest.fixture()
def sandbox(tmp_path, monkeypatch):
    """Temp store and .env, a known token, dry-run mode, and fake Alpaca keys."""
    monkeypatch.setattr(store, "STATE_DIR", tmp_path / "state")
    monkeypatch.setattr(store, "RUNS_FILE", tmp_path / "state" / "runs.json")
    monkeypatch.setattr(keys, "ENV_PATH", tmp_path / ".env")
    (tmp_path / ".env").write_text("# comment\nOTHER=keep\n", encoding="utf-8")
    monkeypatch.setattr(server, "LAUNCH_TOKEN", "test-launch-token")
    monkeypatch.setattr(server, "PORT", 8765)
    monkeypatch.setattr(server, "LIVE_MODE", False)
    monkeypatch.setattr(server, "LAUNCH_ORDERS_GATE", False)
    for name, value in {"ALPACA_API_KEY": "PKAAAA1111", "ALPACA_SECRET_KEY": "secretAAAA1111",
                        "GROQ_API_KEY": "gsk_aaaa", "NEWSAPI_KEY": "newsaaaa", "BROKER_MODE": "alpaca_paper"}.items():
        monkeypatch.setenv(name, value)
    monkeypatch.delenv("TRADER_NAME", raising=False)
    calls = {"journal": [], "live_journal": []}
    monkeypatch.setattr(journal_module, "append_order_results",
                        lambda run_id, timeframe, placed, live=False, **kw: calls["journal"].append((run_id, timeframe, placed, live)))
    monkeypatch.setattr(run_cycle_module, "_journal_live_orders",
                        lambda result, placed: calls["live_journal"].append((result, placed)) or [])
    return TestClient(server.app, base_url="http://127.0.0.1:8765"), calls


def seed(run_id="run-1", fingerprint=None, orders=None, expires=60):
    store.save_run({"id": run_id, "status": "completed", "pipeline_status": "awaiting_approval",
                    "approval_expires_at": time.time() + expires, "approved_orders": orders or ORDERS,
                    "key_fingerprint": keys.fingerprint() if fingerprint is None else fingerprint,
                    "result": {"run_id": "pipeline-1", "timeframe": "1Day"}})


def approve(client, symbols=("AAPL",), run_id="run-1"):
    return client.post(f"/api/runs/{run_id}/approve", headers=TOKEN, json={"symbols": list(symbols), "confirm": True})


def go_live(monkeypatch):
    monkeypatch.setattr(server, "LIVE_MODE", True)
    monkeypatch.setattr(server, "LAUNCH_ORDERS_GATE", True)
    monkeypatch.setenv("BROKER_MODE", "alpaca_paper")


# ------------------------------------------------------------------------------------------------ request guard
def test_guard_rejects_wrong_host_missing_token_and_foreign_origin(sandbox):
    client, _ = sandbox
    assert client.get("/api/status", headers=TOKEN).status_code == 200
    assert client.get("/api/status").status_code == 403                                    # no token
    assert client.get("/api/status", headers={"X-Launch-Token": "wrong"}).status_code == 403
    rebind = TestClient(server.app, base_url="http://evil.example:8765")
    assert rebind.get("/api/status", headers=TOKEN).status_code == 400                      # DNS rebinding
    foreign = client.post("/api/keys/clear", headers={**TOKEN, "Origin": "http://evil.example"}, json={})
    assert foreign.status_code == 403                                                       # cross-site POST
    assert client.get("/").status_code == 403                                               # page needs ?token=
    assert client.get("/?token=test-launch-token").status_code == 200


# ------------------------------------------------------------------------------------------------ live placement
def test_live_approve_sends_exactly_the_saved_selected_orders_and_journals_them(sandbox, monkeypatch):
    client, calls = sandbox
    go_live(monkeypatch)
    sent = {}
    def fake_submit(orders, only_symbols=None):
        sent["orders"], sent["only"] = orders, only_symbols
        return [{"order_id": "abc", "status": "pending_new", "submitted": True, "live": True,
                 "details": {"symbol": "AAPL", "side": "buy", "qty": 2}}]
    monkeypatch.setattr(live_orders, "submit_orders", fake_submit)
    seed()
    response = approve(client)
    assert response.status_code == 200, response.text
    assert sent == {"orders": ORDERS, "only": ["AAPL"]}
    run = response.json()["run"]
    assert run["approval_mode"] == "alpaca_paper" and run["status"] == "approved"
    assert "key_fingerprint" not in run
    # the same audit trail the command line writes
    assert calls["journal"] and calls["journal"][0][0] == "pipeline-1" and calls["journal"][0][3] is True
    assert calls["live_journal"]
    assert approve(client).status_code == 409                                              # one use only


def test_dry_run_mode_never_calls_the_broker(sandbox, monkeypatch):
    client, calls = sandbox
    monkeypatch.setattr(live_orders, "submit_orders", lambda *a, **k: pytest.fail("broker must not be called"))
    from trading_system.crew import guarded_plan
    monkeypatch.setattr(guarded_plan, "place_approved", lambda orders, only_symbols=None: [{"symbol": "AAPL", "status": "dry_run"}])
    seed()
    response = approve(client)
    assert response.status_code == 200
    assert response.json()["run"]["approval_mode"] == "dry_run"
    assert calls["journal"][0][3] is False and not calls["live_journal"]


def test_live_mode_with_a_closed_gate_is_refused_and_can_be_retried(sandbox, monkeypatch):
    client, _ = sandbox
    go_live(monkeypatch)
    monkeypatch.setattr(server, "LAUNCH_ORDERS_GATE", False)
    monkeypatch.setattr(live_orders, "submit_orders", lambda *a, **k: pytest.fail("must not send"))
    seed()
    response = approve(client)
    assert response.status_code == 403
    assert store.get_run("run-1")["status"] == "completed"                                 # not consumed
    monkeypatch.setattr(server, "LAUNCH_ORDERS_GATE", True)
    monkeypatch.setattr(live_orders, "submit_orders", lambda orders, only_symbols=None: [{"submitted": True}])
    assert approve(client).status_code == 200


def test_broker_blocking_error_is_reported_as_403_not_a_generic_failure(sandbox, monkeypatch):
    client, _ = sandbox
    go_live(monkeypatch)
    def blocked(orders, only_symbols=None):
        raise live_orders.OrderSubmissionBlocked("the paper host check failed")
    monkeypatch.setattr(live_orders, "submit_orders", blocked)
    seed()
    response = approve(client)
    assert response.status_code == 403 and "paper host" in response.json()["error"]
    assert store.get_run("run-1")["status"] == "completed"


def test_approval_refuses_symbols_that_were_not_guardian_approved(sandbox):
    client, _ = sandbox
    seed()
    assert approve(client, symbols=("TSLA",)).status_code == 400
    assert approve(client, symbols=()).status_code == 400


# ------------------------------------------------------------------------------------------------ key rotation
def test_changed_alpaca_keys_block_approval_of_an_older_analysis(sandbox, monkeypatch):
    client, _ = sandbox
    seed()
    monkeypatch.setenv("ALPACA_API_KEY", "PKBBBB2222")                                       # someone else signed in
    response = approve(client)
    assert response.status_code == 409 and "keys changed" in response.json()["error"]
    assert store.get_run("run-1")["status"] == "completed"
    monkeypatch.setenv("ALPACA_API_KEY", "PKAAAA1111")                                       # back to the same keys
    assert approve(client).status_code == 200


def test_fingerprint_changes_with_either_key_and_is_empty_without_keys(monkeypatch):
    monkeypatch.setenv("ALPACA_API_KEY", "k1"); monkeypatch.setenv("ALPACA_SECRET_KEY", "s1")
    first = keys.fingerprint()
    monkeypatch.setenv("ALPACA_SECRET_KEY", "s2")
    assert keys.fingerprint() != first and "s1" not in first and "k1" not in first
    monkeypatch.delenv("ALPACA_API_KEY"); monkeypatch.delenv("ALPACA_SECRET_KEY")
    assert keys.fingerprint() == ""


def test_rotating_keys_writes_env_and_process_keeps_other_lines_and_masks_output(sandbox, monkeypatch):
    client, _ = sandbox
    body = {"values": {"ALPACA_API_KEY": "PKNEW99887766", "ALPACA_SECRET_KEY": "newsecret5544", "TRADER_NAME": "Vasundhara"}}
    response = client.post("/api/keys", headers=TOKEN, json=body)
    assert response.status_code == 200, response.text
    text = keys.ENV_PATH.read_text(encoding="utf-8")
    assert "ALPACA_API_KEY=PKNEW99887766" in text and "TRADER_NAME=Vasundhara" in text
    assert "OTHER=keep" in text and "# comment" in text
    assert os.environ["ALPACA_API_KEY"] == "PKNEW99887766"                                   # next run uses it
    assert "PKNEW99887766" not in response.text and "newsecret5544" not in response.text
    status = client.get("/api/status", headers=TOKEN).json()
    assert status["trader"] == "Vasundhara"
    assert "PKNEW99887766" not in json.dumps(status)


def test_trader_name_is_validated(sandbox):
    client, _ = sandbox
    for bad in ("x" * 41, "a\nB=1", "<script>"):
        assert client.post("/api/keys", headers=TOKEN, json={"values": {"TRADER_NAME": bad}}).status_code == 400


def test_sign_out_clears_secrets_and_name_from_file_and_process(sandbox, monkeypatch):
    client, _ = sandbox
    client.post("/api/keys", headers=TOKEN, json={"values": {"TRADER_NAME": "Ashmita"}})
    response = client.post("/api/keys/clear", headers=TOKEN, json={})
    assert response.status_code == 200
    for name in ("ALPACA_API_KEY", "ALPACA_SECRET_KEY", "GROQ_API_KEY", "NEWSAPI_KEY", "TRADER_NAME"):
        assert name not in os.environ
    text = keys.ENV_PATH.read_text(encoding="utf-8")
    assert "PKAAAA1111" not in text and "gsk_aaaa" not in text and "OTHER=keep" in text
    assert keys.fingerprint() == ""


def test_account_endpoint_masks_the_account_number_and_never_returns_keys(sandbox, monkeypatch):
    client, _ = sandbox
    class Reply:
        status_code = 200
        def json(self): return {"account_number": "PA1234567890", "status": "ACTIVE", "equity": "100000",
                                "buying_power": "200000", "cash": "100000", "api_key": "leak"}
    seen = {}
    def fake_get(url, headers=None, timeout=None, **kw):
        seen["url"] = url
        return Reply()
    monkeypatch.setattr(keys.requests, "get", fake_get)
    body = client.get("/api/account", headers=TOKEN).json()
    assert body["ok"] is True and body["account"]["number"].endswith("7890") and "PA123456" not in json.dumps(body)
    assert "leak" not in json.dumps(body) and "PKAAAA1111" not in json.dumps(body)
    assert seen["url"].startswith("https://paper-api.alpaca.markets/")                      # paper host only


def test_account_endpoint_reports_rejected_keys_plainly(sandbox, monkeypatch):
    client, _ = sandbox
    class Reply:
        status_code = 401
    monkeypatch.setattr(keys.requests, "get", lambda *a, **k: Reply())
    body = client.get("/api/account", headers=TOKEN).json()
    assert body["ok"] is False and "PAPER" in body["message"]


def test_broker_mode_is_locked_in_live_mode(sandbox, monkeypatch):
    client, _ = sandbox
    go_live(monkeypatch)
    assert client.post("/api/keys", headers=TOKEN, json={"values": {"BROKER_MODE": "offline"}}).status_code == 400
    assert client.post("/api/keys", headers=TOKEN, json={"values": {"BROKER_MODE": "alpaca_paper"}}).status_code == 200


# ------------------------------------------------------------------------------------------------ locks
def test_approval_and_key_changes_wait_politely_while_a_run_holds_the_lock(sandbox):
    client, _ = sandbox
    seed()
    assert server._RUN_LOCK.acquire(blocking=False)
    try:
        assert approve(client).status_code == 409
        assert client.post("/api/keys", headers=TOKEN, json={"values": {"TRADER_NAME": "X"}}).status_code == 409
        assert client.post("/api/keys/clear", headers=TOKEN, json={}).status_code == 409
        assert client.get("/api/status", headers=TOKEN).status_code == 200                  # page stays responsive
        assert store.get_run("run-1")["status"] == "completed"                              # nothing was claimed
    finally:
        server._RUN_LOCK.release()
    assert approve(client).status_code in (200, 502)


# ------------------------------------------------------------------------------------------------ launch flag
def test_live_flag_is_a_terminal_switch_that_sets_the_gate_for_this_process(monkeypatch):
    monkeypatch.setattr(server, "LIVE_MODE", False)
    monkeypatch.setattr(server, "LAUNCH_ORDERS_GATE", False)
    monkeypatch.setattr(server, "PORT", 8765)
    monkeypatch.delenv("ALPACA_ORDERS_ENABLED", raising=False)
    monkeypatch.setenv("BROKER_MODE", "offline")
    server.apply_launch_flags(server._parse_args([]))
    assert server.LIVE_MODE is False and os.environ.get("ALPACA_ORDERS_ENABLED") != "true"
    server.apply_launch_flags(server._parse_args(["--live", "--port", "9001"]))
    assert server.LIVE_MODE and server.LAUNCH_ORDERS_GATE and server.PORT == 9001
    assert os.environ["ALPACA_ORDERS_ENABLED"] == "true" and os.environ["BROKER_MODE"] == "alpaca_paper"
    assert "--live" not in Path(server.__file__).with_name("static").joinpath("index.html").read_text(encoding="utf-8").split("<script>")[1].split("api(")[0]


def test_status_reports_whether_live_placement_is_possible(sandbox, monkeypatch):
    client, _ = sandbox
    assert client.get("/api/status", headers=TOKEN).json()["can_place_live"] is False
    go_live(monkeypatch)
    status = client.get("/api/status", headers=TOKEN).json()
    assert status["can_place_live"] is True and status["broker_locked"] is True
