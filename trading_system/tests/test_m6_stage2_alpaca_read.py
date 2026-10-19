"""Stage 2 acceptance tests: M6 reads the Alpaca PAPER account (read-only) and still never sends
an order. Every HTTP call is mocked: no network, no real keys. Written by the integration owner."""

import json
import os
import unittest
from unittest import mock

from trading_system.servers.tools import broker, risk

BASE_RULES = {
    "position": {"max_single_position_pct": 0.12, "cash_reserve_pct": 0.10,
                 "max_sector_exposure_pct": 0.30},
    "sizing": {"risk_per_trade_pct": 0.01, "stop_loss_atr_mult": 2.0,
               "trailing_exit_atr_mult": 3.0},
    "limits": {"daily_loss_limit_pct": 0.03, "max_new_orders_per_day": 5,
               "earnings_blackout_days": 2},
    "orders": {"order_type": "limit", "whole_shares": True, "limit_buffer_pct": 0.002},
}


def sell(symbol, qty, price):
    return {"symbol": symbol, "side": "sell", "qty": qty, "limit_price": price}


KEY, SECRET = "DUMMYKEY123456", "DUMMYSECRET654321"
BASE = "https://paper-api.alpaca.markets/v2"

ACCOUNT = {"equity": "100000.50", "cash": "50000.25", "buying_power": "200000", "last_equity": "99000"}
POSITIONS = [
    {"symbol": "aapl", "qty": "5", "avg_entry_price": "100.0", "market_value": "600.0",
     "unrealized_plpc": "0.20", "side": "long"},
    {"symbol": "XOM", "qty": "80", "avg_entry_price": "110.0", "market_value": "9000.0",
     "unrealized_plpc": "-0.0028035244307129", "side": "long"},
]
ORDERS = [{"symbol": "AAPL", "side": "buy", "status": "filled"},
          {"symbol": "AMD", "side": "buy", "status": "canceled"},
          {"symbol": "TSLA", "side": "buy", "status": "new"},
          {"symbol": "XOM", "side": "sell", "status": "filled"}]


class FakeResponse:
    def __init__(self, status=200, payload=None, bad_json=False):
        self.status_code, self._payload, self._bad_json = status, payload, bad_json

    def json(self):
        if self._bad_json:
            raise ValueError("not json")
        return self._payload


class AlpacaReadCase(unittest.TestCase):
    routes = {"/account": ACCOUNT, "/positions": POSITIONS, "/orders": ORDERS}

    def setUp(self):
        self.calls = []
        env = {k: v for k, v in os.environ.items() if not k.startswith(("ALPACA_", "BROKER_"))}
        env.update(BROKER_MODE="alpaca_paper", ALPACA_API_KEY=KEY, ALPACA_SECRET_KEY=SECRET)
        patches = [mock.patch.dict(os.environ, env, clear=True),
                   mock.patch.object(broker.requests, "get", side_effect=self.fake_get),
                   mock.patch.object(broker.requests, "post", side_effect=AssertionError("order sent!")),
                   mock.patch.object(broker.requests, "delete", side_effect=AssertionError("delete sent!"),
                                     create=True)]
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)
        broker._orders.clear()
        self.addCleanup(broker._orders.clear)

    def fake_get(self, url, headers=None, params=None, timeout=None):
        self.calls.append({"url": url, "headers": headers, "params": params, "timeout": timeout})
        path = url[len(BASE):] if url.startswith(BASE) else url
        if path not in self.routes:
            raise AssertionError(f"unexpected URL {url}")
        payload = self.routes[path]
        return payload if isinstance(payload, FakeResponse) else FakeResponse(200, payload)


class AlpacaReadTests(AlpacaReadCase):
    def test_account_is_read_from_the_paper_url_with_the_right_headers(self):
        account = broker.get_account()
        call = self.calls[0]
        self.assertEqual(call["url"], f"{BASE}/account")
        self.assertEqual(call["headers"], {"APCA-API-KEY-ID": KEY, "APCA-API-SECRET-KEY": SECRET})
        self.assertTrue(call["timeout"])
        self.assertEqual(account, {"equity": 100000.5, "cash": 50000.25, "buying_power": 200000.0,
                                   "last_equity": 99000.0})
        json.dumps(account)

    def test_positions_are_mapped_to_our_shape(self):
        positions = broker.get_portfolio()["positions"]
        self.assertEqual(self.calls[0]["url"], f"{BASE}/positions")
        first = positions[0]
        self.assertEqual((first["symbol"], first["qty"], first["avg_entry"], first["market_value"]),
                         ("AAPL", 5.0, 100.0, 600.0))
        self.assertAlmostEqual(first["pl_pct"], 20.0)  # Alpaca sends a fraction; we keep percent
        self.assertAlmostEqual(positions[1]["pl_pct"], -0.2804, places=3)
        json.dumps(positions)

    def test_a_short_position_gets_a_negative_quantity_and_stops_the_guardian(self):
        self.routes = dict(self.routes, **{"/positions": [
            {"symbol": "TSLA", "qty": "10", "avg_entry_price": "400", "market_value": "4000",
             "unrealized_plpc": "0", "side": "short"}]})
        snapshot = broker.get_portfolio_snapshot()
        self.assertEqual(snapshot["positions"]["TSLA"]["qty"], -10.0)
        result = risk.check_rules([sell("TSLA", 1, 400.0)], snapshot, BASE_RULES)
        self.assertEqual(result["approved"], [])  # a short breaks long-only: fail closed
        self.assertEqual(result["blocked"][0]["rule"], "bad_snapshot")

    def test_the_snapshot_includes_todays_buys_and_the_day_start_equity(self):
        snapshot = broker.get_portfolio_snapshot({"AAPL": "Technology", "XOM": "Energy"})
        urls = [c["url"] for c in self.calls]
        self.assertIn(f"{BASE}/orders", urls)
        orders_call = next(c for c in self.calls if c["url"].endswith("/orders"))
        self.assertEqual(orders_call["params"]["status"], "all")
        self.assertIn("after", orders_call["params"])
        self.assertEqual(snapshot["new_buys_today"], 2)  # filled + new; canceled and sell not counted
        self.assertEqual(snapshot["day_start_equity"], 99000.0)
        self.assertEqual(snapshot["positions"]["AAPL"]["sector"], "Technology")
        self.assertEqual(snapshot["equity"], 100000.5)
        json.dumps(snapshot)

    def test_dry_run_orders_are_added_to_the_buy_count(self):
        broker.place_order("AMD", "buy", 1, limit_price=160.0)
        self.assertEqual(broker.get_portfolio_snapshot()["new_buys_today"], 3)

    def test_url_variants_are_accepted_and_normalised(self):
        for url in (BASE, BASE + "/", "https://paper-api.alpaca.markets",
                    "https://paper-api.alpaca.markets/"):
            with mock.patch.dict(os.environ, {"ALPACA_TRADING_URL": url}):
                self.calls.clear()
                broker.get_account()
                self.assertEqual(self.calls[0]["url"], f"{BASE}/account", url)

    def test_orders_are_never_sent_even_when_asked(self):
        result = broker.place_order("AAPL", "buy", 1, limit_price=190.0, dry_run=False)
        self.assertEqual(result["status"], "rejected")
        accepted = broker.place_order("AAPL", "buy", 1, limit_price=190.0)
        self.assertEqual(accepted["status"], "accepted_dry_run")
        broker.cancel_order(accepted["order_id"])  # post/delete are rigged to fail the test if used


class AlpacaSafetyTests(AlpacaReadCase):
    def refuses(self, expected_text=None):
        with self.assertRaises(broker.BrokerError) as caught:
            broker.get_account()
        self.assertEqual(self.calls, [])
        if expected_text:
            self.assertIn(expected_text, str(caught.exception))

    def test_a_live_or_unknown_url_is_refused_before_any_request(self):
        for url in ("https://api.alpaca.markets", "https://api.alpaca.markets/v2",
                    "http://paper-api.alpaca.markets", "https://evil.example.com",
                    "https://paper-api.alpaca.markets.evil.example.com"):
            with mock.patch.dict(os.environ, {"ALPACA_TRADING_URL": url}):
                self.refuses("PAPER")

    def test_missing_keys_are_refused_before_any_request(self):
        for missing in ("ALPACA_API_KEY", "ALPACA_SECRET_KEY"):
            env = {k: v for k, v in os.environ.items() if k != missing}
            with mock.patch.dict(os.environ, env, clear=True):
                self.refuses("ALPACA_API_KEY")

    def test_an_unknown_mode_is_refused(self):
        with mock.patch.dict(os.environ, {"BROKER_MODE": "live"}):
            self.refuses("BROKER_MODE")

    def test_failures_raise_broker_error_and_never_leak_the_keys(self):
        failures = [FakeResponse(401, {"message": "forbidden"}), FakeResponse(500, {}),
                    FakeResponse(200, bad_json=True)]
        for failure in failures:
            self.routes = dict(self.routes, **{"/account": failure})
            with self.assertRaises(broker.BrokerError) as caught:
                broker.get_account()
            text = str(caught.exception)
            self.assertNotIn(KEY, text)
            self.assertNotIn(SECRET, text)

    def test_network_errors_raise_broker_error_without_the_original_text(self):
        with mock.patch.object(broker.requests, "get",
                               side_effect=ConnectionError(f"boom {KEY} {SECRET}")):
            with self.assertRaises(broker.BrokerError) as caught:
                broker.get_account()
        self.assertNotIn(KEY, str(caught.exception))
        self.assertNotIn(SECRET, str(caught.exception))

    def test_malformed_numbers_and_shapes_raise_broker_error(self):
        cases = [("/account", {"equity": "abc", "cash": "1", "buying_power": "1"}),
                 ("/account", {"equity": "NaN", "cash": "1", "buying_power": "1"}),
                 ("/account", {"cash": "1", "buying_power": "1"}),
                 ("/account", ["not", "an", "object"]),
                 ("/positions", {"not": "a list"}),
                 ("/positions", [{"qty": "1", "avg_entry_price": "1", "market_value": "1",
                                  "unrealized_plpc": "0"}]),
                 ("/positions", [{"symbol": "AAPL", "qty": "x", "avg_entry_price": "1",
                                  "market_value": "1", "unrealized_plpc": "0"}])]
        for path, payload in cases:
            self.routes = dict(self.routes, **{path: payload})
            call = broker.get_account if path == "/account" else broker.get_portfolio
            with self.assertRaises(broker.BrokerError, msg=(path, payload)):
                call()
            self.routes = AlpacaReadCase.routes


if __name__ == "__main__":
    unittest.main()
