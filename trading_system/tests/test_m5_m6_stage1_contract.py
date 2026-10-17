"""Stage 1 acceptance tests: M5 (risk/sizing/guardian) and M6 (offline broker) must fit the
pipeline exactly. Fully offline: no network, no real keys. Written by the integration owner;
Ashmita's own unit tests live in separate files."""

import copy
import json
import os
import tempfile
import unittest
from pathlib import Path
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


def rules_with(section, key, value):
    changed = copy.deepcopy(BASE_RULES)
    changed[section][key] = value
    return changed


def buy(symbol, qty, price, **extra):
    return dict({"symbol": symbol, "side": "buy", "qty": qty, "limit_price": price}, **extra)


def sell(symbol, qty, price):
    return {"symbol": symbol, "side": "sell", "qty": qty, "limit_price": price}


SNAPSHOT = {"equity": 100_000, "cash": 50_000,
            "positions": {"MSFT": {"market_value": 11_000, "qty": 20}}}


def check(orders, snapshot=None, rules=None):
    return risk.check_rules(orders, copy.deepcopy(snapshot or SNAPSHOT), rules or BASE_RULES)


def rule_of(result, index=0):
    return result["blocked"][index]["rule"]


class RulesFileTests(unittest.TestCase):
    def test_the_shipped_rules_file_loads_and_validates(self):
        loaded = risk.load_rules()
        for section in ("position", "sizing", "limits", "orders"):
            self.assertIn(section, loaded)

    def test_invalid_values_fail_clearly(self):
        bad_values = [("position", "max_single_position_pct", 1.5),
                      ("position", "max_single_position_pct", 0),
                      ("position", "cash_reserve_pct", "0.1"),
                      ("sizing", "stop_loss_atr_mult", -1),
                      ("limits", "max_new_orders_per_day", 2.5),
                      ("limits", "daily_loss_limit_pct", float("nan")),
                      ("orders", "limit_buffer_pct", True),
                      ("orders", "order_type", "market"),
                      ("orders", "whole_shares", False)]
        for section, key, value in bad_values:
            with self.assertRaises(ValueError, msg=(section, key, value)):
                risk.size_position("AAPL", 190, 3.2, 100_000, rules=rules_with(section, key, value))

    def test_a_missing_section_fails_clearly(self):
        broken = copy.deepcopy(BASE_RULES)
        del broken["limits"]
        with self.assertRaises(ValueError):
            risk.check_rules([], SNAPSHOT, broken)

    def test_rules_can_be_loaded_from_a_path_and_bad_files_fail(self):
        import yaml
        with tempfile.TemporaryDirectory() as folder:
            good = Path(folder) / "good.yaml"
            good.write_text(yaml.safe_dump(BASE_RULES), encoding="utf-8")
            self.assertEqual(risk.load_rules(good)["position"]["cash_reserve_pct"], 0.10)
            with self.assertRaises(ValueError):
                risk.load_rules(Path(folder) / "missing.yaml")
            junk = Path(folder) / "junk.yaml"
            junk.write_text("position: [unclosed", encoding="utf-8")
            with self.assertRaises(ValueError):
                risk.load_rules(junk)


class SizingTests(unittest.TestCase):
    def size(self, **kwargs):
        args = dict(symbol="AAPL", price=190.0, atr=3.2, equity=100_000, rules=BASE_RULES)
        args.update(kwargs)
        return risk.size_position(**args)

    def test_the_documented_example(self):
        result = self.size()
        self.assertEqual(result["qty"], 63)
        self.assertEqual(result["stop_price"], 183.6)
        self.assertEqual(result["notional"], 11_970.0)
        self.assertIn("Trimmed to the position cap.", result["warnings"])
        self.assertEqual(set(result), {"symbol", "qty", "notional", "stop_price", "reason", "warnings"})
        json.dumps(result)

    def test_explicit_limits_override_the_rules(self):
        self.assertEqual(self.size(max_position_pct=0.20)["qty"], 105)

    def test_risk_budget_binds_when_the_cap_does_not(self):
        result = self.size(price=20.0, atr=5.0)
        self.assertEqual(result["qty"], 100)
        self.assertEqual(result["warnings"], [])

    def test_existing_holding_reduces_the_room_under_the_cap(self):
        self.assertEqual(self.size(existing_value=6_000)["qty"], 31)
        none_left = self.size(existing_value=11_900)
        self.assertEqual(none_left["qty"], 0)
        self.assertEqual(none_left["notional"], 0.0)

    def test_bad_input_means_no_trade_not_an_exception(self):
        for bad in ({"price": 0}, {"price": -5}, {"atr": 0}, {"equity": 0},
                    {"price": float("nan")}, {"atr": None}, {"equity": "100000"},
                    {"existing_value": -1}, {"existing_value": None}):
            result = self.size(**bad)
            self.assertEqual(result["qty"], 0, bad)
            self.assertTrue(result["reason"].startswith("No trade"), bad)
            self.assertTrue(result["warnings"], bad)

    def test_a_stop_at_or_below_zero_is_no_trade(self):
        result = self.size(price=5.0, atr=4.0)
        self.assertEqual(result["qty"], 0)
        self.assertGreaterEqual(result["stop_price"], 0)

    def test_the_orchestrator_sizes_on_the_limit_price(self):
        # M8 passes the LIMIT price into size_position, so qty * limit_price never exceeds the cap.
        limit = risk.make_limit_price("buy", 214.125, BASE_RULES)
        result = self.size(symbol="NVDA", price=limit, atr=7.5)
        self.assertLessEqual(result["qty"] * limit, 100_000 * 0.12)


class LimitPriceTests(unittest.TestCase):
    def test_buys_get_a_cushion_above_and_sells_below(self):
        self.assertEqual(risk.make_limit_price("buy", 100.0, BASE_RULES), 100.2)
        self.assertEqual(risk.make_limit_price("sell", 100.0, BASE_RULES), 99.8)
        self.assertEqual(risk.make_limit_price("buy", 190.0, BASE_RULES), 190.38)

    def test_bad_inputs_raise_value_error(self):
        for side, price in (("hold", 100.0), ("buy", 0), ("buy", -1), ("buy", None),
                            ("buy", float("nan")), ("buy", True)):
            with self.assertRaises(ValueError, msg=(side, price)):
                risk.make_limit_price(side, price, BASE_RULES)


class GuardianTests(unittest.TestCase):
    def test_the_documented_example(self):
        result = check([buy("AAPL", 63, 190.0), buy("NVDA", 200, 900.0),
                        sell("TSLA", 10, 400.0), sell("MSFT", 20, 550.0)])
        self.assertEqual([o["symbol"] for o in result["approved"]], ["AAPL", "MSFT"])
        self.assertEqual([(b["order"]["symbol"], b["rule"]) for b in result["blocked"]],
                         [("NVDA", "position_cap"), ("TSLA", "long_only")])
        self.assertEqual(result["new_buys_approved"], 1)
        json.dumps(result)

    def test_selling_more_than_is_held_is_blocked(self):
        self.assertEqual(rule_of(check([sell("MSFT", 21, 550.0)])), "oversell")
        self.assertEqual(len(check([sell("MSFT", 20, 550.0)])["approved"]), 1)

    def test_positions_may_be_a_list_or_a_dict(self):
        as_list = dict(SNAPSHOT, positions=[{"symbol": "AAPL", "qty": 40, "market_value": 7600.0}])
        result = check([sell("AAPL", 10, 190.0)], as_list)
        self.assertEqual(len(result["approved"]), 1)
        self.assertEqual(result["blocked"], [])

    def test_symbols_are_matched_case_insensitively(self):
        self.assertEqual(len(check([sell("msft", 5, 550.0)])["approved"]), 1)

    def test_malformed_orders_are_blocked_and_never_crash(self):
        weird = [None, "AAPL", 5, {}, buy("AAPL", 2.5, 190.0), buy("AAPL", True, 190.0),
                 buy("AAPL", "10", 190.0), buy("AAPL", 0, 190.0), buy("AAPL", -3, 190.0),
                 buy("AAPL", 10, 0), buy("AAPL", 10, float("nan")), buy("", 10, 190.0),
                 {"symbol": "AAPL", "side": "hold", "qty": 1, "limit_price": 10.0},
                 {"symbol": 7, "side": "buy", "qty": 1, "limit_price": 10.0}]
        result = check(weird)
        self.assertEqual(result["approved"], [])
        self.assertEqual({b["rule"] for b in result["blocked"]}, {"bad_order"})
        self.assertEqual(len(result["blocked"]), len(weird))

    def test_integer_valued_float_quantities_are_accepted(self):
        self.assertEqual(len(check([buy("AAPL", 5.0, 190.0)])["approved"]), 1)

    def test_an_unusable_snapshot_blocks_everything(self):
        for snapshot in (None, {}, "x", {"equity": 100000}, {"equity": 0, "cash": 10},
                         {"equity": 100000, "cash": -1}, {"equity": 100000, "cash": 10, "positions": 5},
                         {"equity": 100000, "cash": 10, "positions": {"AAPL": "lots"}},
                         {"equity": 100000, "cash": 10, "positions": [{"qty": 1}]},
                         {"equity": 100000, "cash": 10, "positions": {"AAPL": {"qty": -5}}}):
            result = risk.check_rules([buy("AAPL", 1, 10.0), sell("AAPL", 1, 10.0)], snapshot, BASE_RULES)
            self.assertEqual(result["approved"], [], snapshot)
            self.assertEqual([b["rule"] for b in result["blocked"]], ["bad_snapshot"] * 2, snapshot)
            self.assertTrue(result["warnings"], snapshot)

    def test_daily_new_buy_cap_counts_buys_already_made_today(self):
        orders = [buy(f"S{i}", 1, 10.0) for i in range(6)]
        result = check(orders)
        self.assertEqual(len(result["approved"]), 5)
        self.assertEqual(rule_of(result), "max_new_orders")
        self.assertEqual(result["new_buys_approved"], 5)
        later = check(orders[:4], dict(SNAPSHOT, new_buys_today=3))
        self.assertEqual(len(later["approved"]), 2)
        self.assertEqual(later["new_buys_approved"], 2)
        self.assertEqual(rule_of(later), "max_new_orders")

    def test_sells_do_not_use_up_the_buy_cap_or_cash(self):
        snapshot = dict(SNAPSHOT, positions={f"S{i}": {"qty": 5, "market_value": 100.0} for i in range(6)})
        result = check([sell(f"S{i}", 5, 20.0) for i in range(6)], snapshot)
        self.assertEqual(len(result["approved"]), 6)
        self.assertEqual(result["new_buys_approved"], 0)

    def test_position_cap_includes_what_is_already_held(self):
        held = dict(SNAPSHOT, positions={"AAPL": {"qty": 35, "market_value": 11_900.0}})
        self.assertEqual(rule_of(check([buy("AAPL", 1, 343.0)], held)), "position_cap")
        held = dict(SNAPSHOT, positions={"AAPL": {"qty": 31, "market_value": 6_000.0}})
        self.assertEqual(len(check([buy("AAPL", 1, 190.0)], held)["approved"]), 1)

    def test_two_buys_of_the_same_stock_in_one_batch_share_one_cap(self):
        result = check([buy("AAPL", 35, 200.0), buy("AAPL", 35, 200.0)])
        self.assertEqual(len(result["approved"]), 1)
        self.assertEqual(rule_of(result), "position_cap")

    def test_the_cash_reserve_is_never_spent(self):
        snapshot = {"equity": 100_000, "cash": 15_000, "positions": {}}
        result = check([buy("AAA", 20, 190.0), buy("BBB", 10, 190.0)], snapshot)
        self.assertEqual([o["symbol"] for o in result["approved"]], ["AAA"])
        self.assertEqual(rule_of(result), "cash_reserve")

    def test_the_first_failing_rule_is_reported(self):
        snapshot = {"equity": 100_000, "cash": 1_000, "positions": {}}
        self.assertEqual(rule_of(check([buy("NVDA", 200, 900.0)], snapshot)), "position_cap")

    def test_the_result_is_json_serialisable_and_inputs_are_not_mutated(self):
        orders = [buy("AAPL", 10, 190.0), sell("MSFT", 5, 550.0)]
        snapshot = copy.deepcopy(SNAPSHOT)
        before = copy.deepcopy((orders, snapshot))
        result = risk.check_rules(orders, snapshot, BASE_RULES)
        json.dumps(result)
        self.assertEqual((orders, snapshot), before)


class FixtureCase(unittest.TestCase):
    """Runs the broker against a private fixtures folder with the network switched off."""

    ACCOUNT = {"equity": 100000, "cash": 50000, "buying_power": 100000}
    PORTFOLIO = {"positions": [
        {"symbol": "XOM", "qty": 80, "avg_entry": 110.0, "market_value": 9000.0, "pl_pct": 2.3,
         "sector": "Energy"},
        {"symbol": "AAPL", "qty": 40, "avg_entry": 190.0, "market_value": 7600.0, "pl_pct": 2.1,
         "sector": "Technology"}]}

    def setUp(self):
        self._folder = tempfile.TemporaryDirectory()
        folder = Path(self._folder.name)
        (folder / "account.json").write_text(json.dumps(self.ACCOUNT), encoding="utf-8")
        (folder / "portfolio.json").write_text(json.dumps(self.PORTFOLIO), encoding="utf-8")
        env = {k: v for k, v in os.environ.items() if k != "BROKER_MODE"}
        patches = [mock.patch.object(broker, "_FIXTURES", folder),
                   mock.patch.dict(os.environ, env, clear=True),
                   mock.patch.object(broker.requests, "get", side_effect=AssertionError("network used")),
                   mock.patch.object(broker.requests, "post", side_effect=AssertionError("network used"))]
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)
        broker._orders.clear()
        self.addCleanup(broker._orders.clear)
        self.addCleanup(self._folder.cleanup)


class BrokerOfflineTests(FixtureCase):
    def test_account_and_portfolio_have_the_documented_shape(self):
        account = broker.get_account()
        self.assertEqual({k: account[k] for k in ("equity", "cash", "buying_power")}, self.ACCOUNT)
        positions = broker.get_portfolio()["positions"]
        self.assertEqual({p["symbol"] for p in positions}, {"XOM", "AAPL"})
        for position in positions:
            for key in ("symbol", "qty", "market_value"):
                self.assertIn(key, position)
        json.dumps([account, positions])

    def test_the_snapshot_is_in_the_shape_the_guardian_expects(self):
        snapshot = broker.get_portfolio_snapshot({"aapl": "Tech"})
        self.assertEqual(set(snapshot["positions"]), {"XOM", "AAPL"})
        self.assertEqual(snapshot["positions"]["XOM"]["qty"], 80)
        self.assertEqual(snapshot["positions"]["XOM"]["sector"], "Energy")
        self.assertEqual(snapshot["positions"]["AAPL"]["sector"], "Tech")  # the map wins over the file
        self.assertEqual((snapshot["equity"], snapshot["cash"]), (100000, 50000))
        self.assertEqual(snapshot["new_buys_today"], 0)
        json.dumps(snapshot)
        guarded = risk.check_rules([sell("XOM", 80, 109.78)], snapshot, BASE_RULES)
        self.assertEqual(len(guarded["approved"]), 1)

    def test_new_buys_today_counts_open_dry_run_buys_only(self):
        first = broker.place_order("AAPL", "buy", 10, limit_price=190.0)
        broker.place_order("MSFT", "buy", 5, limit_price=500.0)
        broker.place_order("XOM", "sell", 5, limit_price=110.0)
        self.assertEqual(broker.get_portfolio_snapshot()["new_buys_today"], 2)
        broker.cancel_order(first["order_id"])
        self.assertEqual(broker.get_portfolio_snapshot()["new_buys_today"], 1)

    def test_a_good_order_is_accepted_as_a_dry_run(self):
        order = broker.place_order("aapl", "buy", 63, limit_price=190.0)
        self.assertEqual(order["status"], "accepted_dry_run")
        details = order["details"]
        self.assertEqual((details["symbol"], details["side"], details["qty"], details["limit_price"]),
                         ("AAPL", "buy", 63, 190.0))
        for key in ("order_type", "time_in_force", "client_order_id", "placed_at"):
            self.assertIn(key, details)
        json.dumps(order)

    def test_bad_orders_are_rejected_without_being_recorded(self):
        bad = [dict(symbol="AAPL", side="buy", qty=0, limit_price=190.0),
               dict(symbol="AAPL", side="buy", qty=-1, limit_price=190.0),
               dict(symbol="AAPL", side="buy", qty=2.5, limit_price=190.0),
               dict(symbol="AAPL", side="buy", qty=True, limit_price=190.0),
               dict(symbol="AAPL", side="buy", qty="3", limit_price=190.0),
               dict(symbol="AAPL", side="buy", qty=None, limit_price=190.0),
               dict(symbol="AAPL", side="hold", qty=1, limit_price=190.0),
               dict(symbol="AAPL", side="buy", qty=1, limit_price=None),
               dict(symbol="AAPL", side="buy", qty=1, limit_price=0),
               dict(symbol="AAPL", side="buy", qty=1, limit_price=float("nan")),
               dict(symbol="", side="buy", qty=1, limit_price=190.0),
               dict(symbol=None, side="buy", qty=1, limit_price=190.0)]
        for kwargs in bad:
            result = broker.place_order(**kwargs)
            self.assertEqual(result["status"], "rejected", kwargs)
            self.assertIsNone(result["order_id"], kwargs)
            self.assertTrue(result["details"]["reason"], kwargs)
        self.assertEqual(broker.get_open_orders()["open_orders"], [])

    def test_live_orders_are_refused(self):
        result = broker.place_order("AAPL", "buy", 1, limit_price=190.0, dry_run=False)
        self.assertEqual(result["status"], "rejected")
        self.assertEqual(broker.get_open_orders()["open_orders"], [])

    def test_a_repeated_client_order_id_is_rejected(self):
        first = broker.place_order("AAPL", "buy", 1, limit_price=190.0, client_order_id="run1-AAPL")
        again = broker.place_order("AAPL", "buy", 1, limit_price=190.0, client_order_id="run1-AAPL")
        self.assertEqual(first["status"], "accepted_dry_run")
        self.assertEqual(again["status"], "rejected")
        self.assertEqual(len(broker.get_open_orders()["open_orders"]), 1)

    def test_order_ids_are_unique_and_the_lifecycle_works(self):
        a = broker.place_order("AAPL", "buy", 1, limit_price=190.0)
        b = broker.place_order("AMD", "buy", 2, limit_price=160.0)
        self.assertNotEqual(a["order_id"], b["order_id"])
        self.assertEqual(len(broker.get_open_orders()["open_orders"]), 2)
        self.assertEqual(broker.cancel_order(a["order_id"]), {"order_id": a["order_id"], "status": "cancelled"})
        self.assertEqual([o["details"]["symbol"] for o in broker.get_open_orders()["open_orders"]], ["AMD"])
        self.assertEqual(broker.cancel_order("nope")["status"], "not_found")

    def test_an_unknown_broker_mode_is_refused(self):
        with mock.patch.dict(os.environ, {"BROKER_MODE": "live"}):
            with self.assertRaises(broker.BrokerError):
                broker.get_account()


class FakeMCP:
    def __init__(self):
        self.names = []

    def tool(self, *args, **kwargs):
        def decorate(function):
            self.names.append(function.__name__)
            return function
        return decorate


class RegistrationTests(unittest.TestCase):
    def test_tools_attach_to_an_existing_mcp_instance_without_changing_the_functions(self):
        server = FakeMCP()
        risk.register_risk_tools(server)
        broker.register_broker_tools(server)
        self.assertEqual(sorted(server.names), sorted(
            ["size_position", "check_rules", "get_account", "get_portfolio", "place_order",
             "get_open_orders", "cancel_order"]))


class OrchestratorHandshakeTests(FixtureCase):
    """Replays exactly what the orchestrator (M8) will do after the debate."""

    DECISIONS = [{"symbol": "AAPL", "decision": "BUY", "latest_close": 343.39},
                 {"symbol": "MSFT", "decision": "HOLD", "latest_close": 528.99},
                 {"symbol": "XOM", "decision": "SELL", "latest_close": 110.0},
                 {"symbol": "NVDA", "decision": "BUY", "latest_close": 214.125}]
    ATR = {"AAPL": 6.0, "NVDA": 7.5, "XOM": 2.0}
    SECTORS = {"AAPL": "Technology", "NVDA": "Technology", "XOM": "Energy"}

    def run_cycle(self):
        snapshot = broker.get_portfolio_snapshot(self.SECTORS)
        orders = []
        for decision in self.DECISIONS:
            symbol, close = decision["symbol"], decision["latest_close"]
            held = snapshot["positions"].get(symbol)
            if decision["decision"] == "BUY":
                limit = risk.make_limit_price("buy", close, BASE_RULES)
                sized = risk.size_position(
                    symbol, limit, self.ATR[symbol], snapshot["equity"],
                    existing_value=held["market_value"] if held else 0.0, rules=BASE_RULES)
                if sized["qty"] > 0:
                    orders.append(buy(symbol, sized["qty"], limit, sector=self.SECTORS[symbol]))
            elif decision["decision"] == "SELL" and held:
                orders.append(sell(symbol, int(held["qty"]),
                                   risk.make_limit_price("sell", close, BASE_RULES)))
        verdict = risk.check_rules(orders, snapshot, BASE_RULES)
        placed = [broker.place_order(o["symbol"], o["side"], o["qty"], limit_price=o["limit_price"],
                                     dry_run=True) for o in verdict["approved"]]
        return orders, verdict, placed

    def test_decisions_become_guarded_dry_run_orders(self):
        orders, verdict, placed = self.run_cycle()
        self.assertEqual([o["symbol"] for o in orders], ["AAPL", "XOM", "NVDA"])
        self.assertEqual(verdict["blocked"], [])
        self.assertEqual([p["status"] for p in placed], ["accepted_dry_run"] * 3)
        by_symbol = {o["symbol"]: o for o in verdict["approved"]}
        # AAPL is already held ($7,600 of a $12,000 cap), so only 12 more shares fit.
        self.assertEqual((by_symbol["AAPL"]["qty"], by_symbol["AAPL"]["limit_price"]), (12, 344.08))
        self.assertEqual((by_symbol["NVDA"]["qty"], by_symbol["NVDA"]["limit_price"]), (55, 214.55))
        self.assertEqual((by_symbol["XOM"]["qty"], by_symbol["XOM"]["limit_price"]), (80, 109.78))
        for order in verdict["approved"]:
            if order["side"] == "buy":
                self.assertLessEqual(order["qty"] * order["limit_price"], 100_000 * 0.12)
        self.assertEqual(verdict["new_buys_approved"], 2)
        json.dumps([orders, verdict, placed])

    def test_a_second_run_the_same_day_sees_the_first_runs_buys(self):
        self.run_cycle()
        snapshot = broker.get_portfolio_snapshot(self.SECTORS)
        self.assertEqual(snapshot["new_buys_today"], 2)


if __name__ == "__main__":
    unittest.main()
