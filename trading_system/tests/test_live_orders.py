"""Live PAPER order submission (servers/tools/live_orders.py + run_cycle submit="alpaca_paper").

Fully offline: a fake Alpaca replaces `requests`, records every call, and the tests assert exactly what
was (and was not) sent. No network, no real keys. The gates are tested one by one: a closed gate must
mean ZERO http calls."""

import asyncio
import sys
import uuid
from pathlib import Path

import pytest
import requests

REPOSITORY_DIR = Path(__file__).resolve().parents[2]
if str(REPOSITORY_DIR) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_DIR))

from trading_system.crew import memory, run_cycle as cycle
from trading_system.crew import trading_crew_multiStocks as crew
from trading_system.servers.tools import broker, live_orders

KEY, SECRET = "PKTESTKEY123456", "SECRETVALUE987654"
PAPER = "https://paper-api.alpaca.markets/v2/"


class Resp:
    def __init__(self, status_code, payload):
        self.status_code, self._payload = status_code, payload

    def json(self):
        if self._payload is None:
            raise ValueError("no json")
        return self._payload


class FakeAlpaca:
    def __init__(self, buying_power=200_000.0, positions=None, orders=None, post_status=200,
                 post_message="rejected", post_raises=False, create_despite_raise=False,
                 lookup_status=None):
        self.buying_power, self.positions, self.orders = buying_power, positions or [], orders or []
        self.post_status, self.post_message = post_status, post_message
        self.post_raises, self.create_despite_raise = post_raises, create_despite_raise
        self.lookup_status = lookup_status          # force the by_client_order_id lookup to this status
        self.gets, self.posts, self.urls = [], [], []

    def get(self, url, headers=None, params=None, timeout=None):
        self.urls.append(url)
        self.gets.append((url, params))
        assert headers and headers.get("APCA-API-KEY-ID") == KEY
        if url.endswith("/account"):
            return Resp(200, {"equity": "100000", "cash": "60000", "buying_power": str(self.buying_power),
                              "last_equity": "100000"})
        if url.endswith("/positions"):
            return Resp(200, self.positions)
        if url.endswith("/orders:by_client_order_id"):
            if self.lookup_status is not None:
                return Resp(self.lookup_status, {"message": "x"})
            for order in self.orders:
                if order.get("client_order_id") == params["client_order_id"]:
                    return Resp(200, order)
            return Resp(404, {"message": "order not found"})
        if url.endswith("/orders"):
            return Resp(200, [o for o in self.orders if o.get("status") in ("new", "accepted")
                              and o.get("symbol") == params.get("symbols")])
        raise AssertionError(f"unexpected GET {url}")

    def post(self, url, headers=None, json=None, timeout=None):
        self.urls.append(url)
        self.posts.append(json)
        assert url == PAPER + "orders"
        created = {"id": str(uuid.uuid4()), "status": "accepted", "client_order_id": json["client_order_id"],
                   "symbol": json["symbol"], "side": json["side"], "created_at": "2026-10-19T03:00:00Z"}
        if self.post_raises:
            if self.create_despite_raise:
                self.orders.append(created)
            raise requests.ConnectionError(f"boom {KEY}")
        if self.post_status != 200:
            return Resp(self.post_status, {"message": self.post_message})
        self.orders.append(created)
        return Resp(200, created)

    @property
    def calls(self):
        return len(self.gets) + len(self.posts)


@pytest.fixture()
def env(monkeypatch, tmp_path):
    monkeypatch.setenv("ALPACA_ORDERS_ENABLED", "true")
    monkeypatch.setenv("BROKER_MODE", "alpaca_paper")
    monkeypatch.setenv("ALPACA_API_KEY", KEY)
    monkeypatch.setenv("ALPACA_SECRET_KEY", SECRET)
    monkeypatch.delenv("ALPACA_TRADING_URL", raising=False)
    monkeypatch.setattr(memory, "MEMORY_DIR", tmp_path / "memory")
    return monkeypatch


@pytest.fixture()
def alpaca(env):
    fake = FakeAlpaca()
    env.setattr(requests, "get", fake.get)
    env.setattr(requests, "post", fake.post)
    return fake


def buy(symbol="MSFT", qty=22, price=530.05, **extra):
    return dict({"symbol": symbol, "side": "buy", "qty": qty, "order_type": "limit",
                 "limit_price": price, "stop_price": 505.21, "time_in_force": "day"}, **extra)


def sell(symbol="TSLA", qty=10, price=99.8):
    return {"symbol": symbol, "side": "sell", "qty": qty, "order_type": "limit",
            "limit_price": price, "stop_price": None, "time_in_force": "day"}


# ------------------------------------------------------------------ the gates

@pytest.mark.parametrize("value", [None, "", "false", "1", "yes", "on", "truee"])
def test_the_master_switch_must_be_exactly_true(env, alpaca, value):
    if value is None:
        env.delenv("ALPACA_ORDERS_ENABLED")
    else:
        env.setenv("ALPACA_ORDERS_ENABLED", value)
    with pytest.raises(live_orders.OrderSubmissionBlocked):
        live_orders.submit_orders([buy()])
    assert alpaca.calls == 0


def test_offline_broker_mode_closes_the_gate(env, alpaca):
    env.setenv("BROKER_MODE", "offline")
    with pytest.raises(live_orders.OrderSubmissionBlocked, match="alpaca_paper"):
        live_orders.submit_orders([buy()])
    assert alpaca.calls == 0


def test_a_non_paper_url_closes_the_gate(env, alpaca):
    env.setenv("ALPACA_TRADING_URL", "https://api.alpaca.markets")
    with pytest.raises(live_orders.OrderSubmissionBlocked, match="PAPER"):
        live_orders.submit_orders([buy()])
    assert alpaca.calls == 0


def test_missing_keys_close_the_gate(env, alpaca):
    env.delenv("ALPACA_SECRET_KEY")
    with pytest.raises(live_orders.OrderSubmissionBlocked):
        live_orders.submit_orders([buy()])
    assert alpaca.calls == 0


def test_more_than_the_per_submission_limit_is_refused(alpaca):
    orders = [buy(symbol=f"S{i}") for i in range(live_orders.MAX_ORDERS_PER_SUBMISSION + 1)]
    with pytest.raises(live_orders.OrderSubmissionBlocked, match="limit"):
        live_orders.submit_orders(orders)
    assert alpaca.calls == 0


def test_the_dry_run_broker_still_refuses_live_mode_and_the_server_never_imports_live_orders():
    assert broker.place_order("MSFT", "buy", 1, limit_price=10.0, dry_run=False)["status"] == "rejected"
    server = (Path(__file__).resolve().parents[1] / "servers"
              / "technical_indicators_mcp_server_multiStocks.py").read_text(encoding="utf-8")
    assert "live_orders" not in server


# ------------------------------------------------------------------ one order

def test_a_valid_buy_is_posted_once_with_the_exact_body(alpaca):
    (result,) = live_orders.submit_orders([buy()])
    assert alpaca.posts == [{"symbol": "MSFT", "qty": "22", "side": "buy", "type": "limit",
                             "time_in_force": "day", "limit_price": "530.05",
                             "client_order_id": live_orders.make_client_order_id("MSFT", "buy")}]
    assert result["submitted"] is True and result["status"] == "accepted" and result["live"] is True
    assert result["order_id"] and result["details"]["limit_price"] == 530.05
    assert all(u.startswith(PAPER) for u in alpaca.urls)


def test_the_stop_price_is_not_sent_to_the_broker(alpaca):
    live_orders.submit_orders([buy()])
    assert "stop_price" not in alpaca.posts[0] and "order_class" not in alpaca.posts[0]


def test_only_the_chosen_symbols_are_sent(alpaca):
    results = live_orders.submit_orders([buy("MSFT"), buy("AAPL", 34, 344.08)], only_symbols=["aapl"])
    assert [r["details"]["symbol"] for r in results] == ["AAPL"]
    assert [p["symbol"] for p in alpaca.posts] == ["AAPL"]


def test_the_same_order_twice_is_a_duplicate_and_is_not_resent(alpaca):
    live_orders.submit_orders([buy()])
    (second,) = live_orders.submit_orders([buy()])
    assert len(alpaca.posts) == 1
    assert second["submitted"] is False and second["reason_code"] == "duplicate"


def test_an_open_order_on_the_same_side_blocks_a_new_one(alpaca):
    alpaca.orders.append({"id": "o1", "status": "new", "symbol": "MSFT", "side": "buy",
                          "client_order_id": "manual-1"})
    (result,) = live_orders.submit_orders([buy()])
    assert alpaca.posts == [] and result["status"] == "skipped"
    assert result["reason_code"] == "open_order_exists"


def test_an_open_order_on_the_other_side_does_not_block(alpaca):
    alpaca.orders.append({"id": "o1", "status": "new", "symbol": "MSFT", "side": "sell",
                          "client_order_id": "manual-1"})
    (result,) = live_orders.submit_orders([buy()])
    assert result["submitted"] is True and len(alpaca.posts) == 1


def test_not_enough_buying_power_rejects_without_sending(alpaca):
    alpaca.buying_power = 1_000.0
    (result,) = live_orders.submit_orders([buy()])
    assert result["status"] == "rejected" and "buying power" in result["reason"]
    assert alpaca.posts == []


def test_selling_more_than_held_is_rejected_without_sending(alpaca):
    alpaca.positions = [{"symbol": "TSLA", "qty": "4", "side": "long", "avg_entry_price": "190",
                         "market_value": "800", "unrealized_plpc": "0.01"}]
    (result,) = live_orders.submit_orders([sell(qty=10)])
    assert result["status"] == "rejected" and "only 4 held" in result["reason"]
    assert alpaca.posts == []


def test_selling_what_is_held_is_sent(alpaca):
    alpaca.positions = [{"symbol": "TSLA", "qty": "10", "side": "long", "avg_entry_price": "190",
                         "market_value": "1900", "unrealized_plpc": "0.01"}]
    (result,) = live_orders.submit_orders([sell()])
    assert result["submitted"] is True and alpaca.posts[0]["side"] == "sell"


@pytest.mark.parametrize("bad", [
    buy(order_type="market"), buy(qty=2.5), buy(qty=0), buy(qty=True), buy(price=0), buy(price=-1),
    buy(price=float("nan")), buy(time_in_force="gtc"), buy(symbol="NOT A TICKER"), buy(symbol=None),
    buy(symbol="AAPL;DROP"), dict(buy(), side="short"), "not a dict", None])
def test_malformed_orders_are_rejected_before_any_http_call(alpaca, bad):
    (result,) = live_orders.submit_orders([bad])
    assert result["status"] == "rejected" and result["submitted"] is False
    assert alpaca.calls == 0


def test_an_alpaca_rejection_is_reported_and_secrets_are_redacted(alpaca):
    alpaca.post_status, alpaca.post_message = 422, f"bad request for {KEY} and {SECRET}"
    (result,) = live_orders.submit_orders([buy()])
    assert result["status"] == "rejected" and "HTTP 422" in result["reason"]
    assert KEY not in repr(result) and SECRET not in repr(result)


def test_an_unclear_duplicate_lookup_fails_closed_and_sends_nothing(alpaca):
    alpaca.lookup_status = 500
    (result,) = live_orders.submit_orders([buy()])
    assert result["status"] == "rejected" and "nothing was sent" in result["reason"]
    assert alpaca.posts == []


# ---------------------------------------------------- the send fails in flight

def test_a_failed_send_that_did_reach_alpaca_is_found_and_not_resent(alpaca):
    alpaca.post_raises, alpaca.create_despite_raise = True, True
    (result,) = live_orders.submit_orders([buy()])
    assert result["submitted"] is True and result.get("reconciled") is True
    assert len(alpaca.posts) == 1
    assert KEY not in repr(result)


def test_a_failed_send_that_did_not_reach_alpaca_is_reported_as_not_sent(alpaca):
    alpaca.post_raises = True
    (result,) = live_orders.submit_orders([buy()])
    assert result["submitted"] is False and result["status"] == "rejected"
    assert "no such order" in result["reason"] and len(alpaca.posts) == 1


def test_a_failed_send_that_cannot_be_confirmed_says_so(alpaca, env):
    real_get = alpaca.get
    state = {"posted": False}
    original_post = alpaca.post

    def post(*a, **k):
        state["posted"] = True
        return original_post(*a, **k)

    def get(url, headers=None, params=None, timeout=None):
        if state["posted"] and url.endswith("/orders:by_client_order_id"):
            return Resp(500, None)
        return real_get(url, headers=headers, params=params, timeout=timeout)

    alpaca.post_raises = True
    env.setattr(requests, "post", post)
    env.setattr(requests, "get", get)
    (result,) = live_orders.submit_orders([buy()])
    assert result["status"] == "unconfirmed" and result["submitted"] is False
    assert "Check the Alpaca dashboard" in result["reason"]


# ------------------------------------------------------------------ run_cycle

def fake_strategy(approved):
    async def run(**kwargs):
        return {"run_id": "run-test", "degraded": False, "timeframe": "1Day", "symbols": ["MSFT", "AAPL"],
                "decisions": [], "plan": [],
                "guarded_plan": {"approved": approved, "proposals": approved, "blocked": [], "skipped": [],
                                 "warnings": [], "broker_mode": "alpaca_paper"},
                "report_path": None, "journal_path": None}
    return run


@pytest.fixture()
def strategy(env):
    def install(approved):
        env.setattr(crew, "run_strategy", fake_strategy(approved))
    return install


@pytest.mark.parametrize("approval", [None, "all", [], ["  "]])
def test_live_submission_needs_an_explicit_symbol_list(strategy, alpaca, approval):
    strategy([buy()])
    with pytest.raises(cycle.CycleConfigError):
        asyncio.run(cycle.run_cycle({"symbols": ["MSFT"], "submit": "alpaca_paper", "approval": approval}))
    assert alpaca.calls == 0


def test_an_unknown_submit_value_is_refused(strategy, alpaca):
    strategy([buy()])
    with pytest.raises(cycle.CycleConfigError):
        asyncio.run(cycle.run_cycle({"symbols": ["MSFT"], "submit": "live", "approval": ["MSFT"]}))


def test_a_ui_cannot_switch_live_orders_on_through_env(strategy, alpaca):
    strategy([buy()])
    with pytest.raises(cycle.CycleConfigError, match="not allowed"):
        asyncio.run(cycle.run_cycle({"symbols": ["MSFT"], "approval": ["MSFT"], "submit": "alpaca_paper",
                                     "env": {"ALPACA_ORDERS_ENABLED": "true"}}))
    assert alpaca.calls == 0


def test_default_cycle_stays_dry_run_even_with_the_switch_on(strategy, alpaca):
    strategy([buy()])
    result = asyncio.run(cycle.run_cycle({"symbols": ["MSFT"], "approval": ["MSFT"]}))
    assert result["dry_run"] is True and result["submission"] == "dry_run"
    assert result["placed"][0]["status"] == "accepted_dry_run"
    assert alpaca.posts == []


def test_a_closed_gate_reports_submission_blocked_and_sends_nothing(strategy, alpaca, env):
    env.delenv("ALPACA_ORDERS_ENABLED")
    strategy([buy()])
    result = asyncio.run(cycle.run_cycle({"symbols": ["MSFT"], "approval": ["MSFT"], "submit": "alpaca_paper"}))
    assert result["status"] == "submission_blocked" and result["placed"] == []
    assert "switched off" in result["submission_error"] and result["dry_run"] is True
    assert alpaca.calls == 0


def test_approved_live_cycle_sends_only_the_approved_symbols_and_journals_them(strategy, alpaca):
    strategy([buy("MSFT"), buy("AAPL", 34, 344.08)])
    result = asyncio.run(cycle.run_cycle({"symbols": ["MSFT", "AAPL"], "approval": ["MSFT"],
                                          "submit": "alpaca_paper"}))
    assert result["status"] == "ok" and result["submission"] == "alpaca_paper"
    assert result["dry_run"] is False
    assert [p["details"]["symbol"] for p in result["placed"]] == ["MSFT"]
    assert [p["symbol"] for p in alpaca.posts] == ["MSFT"]
    assert any("No protective stop" in w for w in result["warnings"])
    (journal,) = result["order_journal_paths"]
    text = Path(journal).read_text(encoding="utf-8")
    assert "MSFT" in text and KEY not in text and SECRET not in text
    assert KEY not in repr(result) and SECRET not in repr(result)


def test_a_cycle_with_nothing_approved_sends_nothing(strategy, alpaca):
    strategy([])
    result = asyncio.run(cycle.run_cycle({"symbols": ["MSFT"], "approval": ["MSFT"], "submit": "alpaca_paper"}))
    assert result["status"] == "no_orders" and alpaca.calls == 0
