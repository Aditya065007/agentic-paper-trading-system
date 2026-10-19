"""G8 follow-up: orders can be placed on consecutive days, and the CLI prints the approve command. Offline."""

import datetime as dt

from trading_system.crew import run_cycle as rc
from trading_system.servers.tools import live_orders
from trading_system.tests.test_live_orders import FakeAlpaca, alpaca, buy, env, sell  # noqa: F401

DAY1, DAY2, DAY3 = dt.date(2026, 10, 19), dt.date(2026, 10, 20), dt.date(2026, 10, 21)


def test_the_same_buy_can_be_placed_again_on_the_next_day(alpaca):
    (first,) = live_orders.submit_orders([buy()], run_date=DAY1)
    assert first["submitted"] is True
    alpaca.orders[0]["status"] = "expired"                       # a DAY order that was never filled
    (second,) = live_orders.submit_orders([buy()], run_date=DAY2)
    assert second["submitted"] is True and len(alpaca.posts) == 2
    assert alpaca.posts[0]["client_order_id"] != alpaca.posts[1]["client_order_id"]
    assert alpaca.posts[1]["client_order_id"] == "pt-20261020-MSFT-buy"


def test_a_filled_order_from_yesterday_does_not_block_todays(alpaca):
    live_orders.submit_orders([buy()], run_date=DAY1)
    alpaca.orders[0]["status"] = "filled"
    (second,) = live_orders.submit_orders([buy(qty=5)], run_date=DAY2)
    assert second["submitted"] is True and second["status"] == "accepted"


def test_the_same_day_still_cannot_be_ordered_twice(alpaca):
    live_orders.submit_orders([buy()], run_date=DAY1)
    (again,) = live_orders.submit_orders([buy()], run_date=DAY1)
    assert again["submitted"] is False and again["reason_code"] == "duplicate" and len(alpaca.posts) == 1


def test_three_days_in_a_row_three_orders(alpaca):
    for day in (DAY1, DAY2, DAY3):
        for order in alpaca.orders:
            order["status"] = "expired"
        (result,) = live_orders.submit_orders([buy()], run_date=day)
        assert result["submitted"] is True
    assert [p["client_order_id"] for p in alpaca.posts] == [
        "pt-20261019-MSFT-buy", "pt-20261020-MSFT-buy", "pt-20261021-MSFT-buy"]


def test_a_buy_and_an_exit_on_consecutive_days_both_go_out_with_stops_off(alpaca):
    alpaca.positions = [{"symbol": "MSFT", "qty": "22", "side": "long", "avg_entry_price": "500",
                        "market_value": "11000", "unrealized_plpc": "0.01"}]
    live_orders.submit_orders([buy()], run_date=DAY1)
    alpaca.orders[0]["status"] = "filled"
    (exit_order,) = live_orders.submit_orders([sell("MSFT", 22, 520.0)], run_date=DAY2)
    assert exit_order["submitted"] is True and alpaca.posts[1]["side"] == "sell"


# ------------------------------------------------------------------ the hint printed by the CLI
CONFIG = {"symbols": ["AAPL", "MSFT"], "advisory": False}


def approved(*symbols):
    return {"status": "awaiting_approval", "approved": [{"symbol": s, "side": "buy"} for s in symbols]}


def test_the_hint_names_the_exact_command_with_the_approved_symbols():
    hint = rc.next_step_hint(CONFIG, approved("AAPL", "MSFT"))
    assert "run_cycle AAPL MSFT --approve AAPL,MSFT --submit" in hint
    assert "ALPACA_ORDERS_ENABLED=true BROKER_MODE=alpaca_paper" in hint


def test_the_hint_keeps_timeframe_and_advisory_and_lists_each_symbol_once():
    hint = rc.next_step_hint({"symbols": ["AAPL"], "timeframe": "BOTH", "advisory": True},
                             approved("AAPL", "AAPL"))
    assert "--timeframe BOTH --advisory --approve AAPL --submit" in hint


def test_no_hint_when_nothing_is_waiting_or_approved():
    assert rc.next_step_hint(CONFIG, approved()) is None
    assert rc.next_step_hint(CONFIG, {"status": "ok", "approved": [{"symbol": "AAPL"}]}) is None
    assert rc.next_step_hint(CONFIG, {"status": "no_orders"}) is None
    assert rc.next_step_hint(CONFIG, None) is None
    assert rc.next_step_hint(CONFIG, {"status": "awaiting_approval", "approved": [None, {"symbol": 5}]}) is None


def test_the_hint_is_only_text_and_the_gates_are_not_in_it_by_accident():
    hint = rc.next_step_hint(CONFIG, approved("AAPL"))
    assert "--approve all" not in hint and hint.count("--submit") == 1
