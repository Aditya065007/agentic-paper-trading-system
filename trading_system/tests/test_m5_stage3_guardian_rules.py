"""Stage 3 acceptance tests: the remaining guardian rules (sector cap, daily-loss brake,
earnings blackout, cooldown). The data for each rule is SUPPLIED by the caller through optional
snapshot keys, so these rules work today with offline data and later with live M3 data.
Fully offline. Written by the integration owner."""

import copy
import json
import unittest

from trading_system.servers.tools import risk

BASE_RULES = {
    "position": {"max_single_position_pct": 0.12, "cash_reserve_pct": 0.10,
                 "max_sector_exposure_pct": 0.30},
    "sizing": {"risk_per_trade_pct": 0.01, "stop_loss_atr_mult": 2.0,
               "trailing_exit_atr_mult": 3.0},
    "limits": {"daily_loss_limit_pct": 0.03, "max_new_orders_per_day": 5,
               "earnings_blackout_days": 2},
    "orders": {"order_type": "limit", "whole_shares": True, "limit_buffer_pct": 0.002},
}
WITH_COOLDOWN = copy.deepcopy(BASE_RULES)
WITH_COOLDOWN["limits"]["cooldown_days"] = 3


def buy(symbol, qty, price, **extra):
    return dict({"symbol": symbol, "side": "buy", "qty": qty, "limit_price": price}, **extra)


def sell(symbol, qty, price):
    return {"symbol": symbol, "side": "sell", "qty": qty, "limit_price": price}


def snapshot(**extra):
    base = {"equity": 100_000, "cash": 60_000, "positions": {}, "today": "2026-10-14"}
    base.update(extra)
    return base


def check(orders, snap, rules=BASE_RULES):
    return risk.check_rules(orders, copy.deepcopy(snap), rules)


def rules_hit(result):
    return [b["rule"] for b in result["blocked"]]


class SectorCapTests(unittest.TestCase):
    TECH = {"AAPL": {"qty": 70, "market_value": 25_000.0, "sector": "Technology"}}

    def test_a_buy_that_pushes_a_sector_over_30_percent_is_blocked(self):
        result = check([buy("AMD", 40, 150.0, sector="Technology")], snapshot(positions=self.TECH))
        self.assertEqual(rules_hit(result), ["sector_cap"])  # 25,000 + 6,000 > 30,000
        self.assertIn("Technology", result["blocked"][0]["detail"])

    def test_another_sector_is_unaffected(self):
        result = check([buy("XOM", 40, 150.0, sector="Energy")], snapshot(positions=self.TECH))
        self.assertEqual(len(result["approved"]), 1)

    def test_buys_in_one_batch_share_the_sector_cap(self):
        held = {"MSFT": {"qty": 30, "market_value": 15_000.0, "sector": "Technology"}}
        result = check([buy("AAA", 45, 200.0, sector="Technology"),
                        buy("BBB", 45, 200.0, sector="Technology")], snapshot(positions=held))
        self.assertEqual([o["symbol"] for o in result["approved"]], ["AAA"])  # 24,000 then 33,000
        self.assertEqual(rules_hit(result), ["sector_cap"])

    def test_the_sector_can_come_from_the_snapshot_map_or_the_held_position(self):
        by_map = check([buy("AMD", 40, 150.0)],
                       snapshot(positions=self.TECH, sectors={"amd": "Technology"}))
        self.assertEqual(rules_hit(by_map), ["sector_cap"])
        held = {"AMD": {"qty": 1, "market_value": 100.0, "sector": "Technology"}, **self.TECH}
        by_position = check([buy("AMD", 40, 150.0)], snapshot(positions=held))
        self.assertEqual(rules_hit(by_position), ["sector_cap"])

    def test_an_unknown_sector_is_counted_together_and_warned_about(self):
        result = check([buy("ZZZ", 10, 100.0)], snapshot())
        self.assertEqual(len(result["approved"]), 1)
        self.assertTrue(any("ZZZ" in w and "sector" in w.lower() for w in result["warnings"]))
        crowded = {"QQQ": {"qty": 1, "market_value": 29_500.0}}  # no sector -> UNKNOWN bucket
        blocked = check([buy("ZZZ", 10, 100.0)], snapshot(positions=crowded))
        self.assertEqual(rules_hit(blocked), ["sector_cap"])

    def test_selling_is_never_blocked_by_the_sector_cap(self):
        result = check([sell("AAPL", 70, 355.0)], snapshot(positions=self.TECH))
        self.assertEqual(len(result["approved"]), 1)


class DailyLossTests(unittest.TestCase):
    def test_down_three_percent_or_more_blocks_new_buys_but_not_sells(self):
        held = {"MSFT": {"qty": 20, "market_value": 11_000.0}}
        for equity in (97_000, 96_000):  # exactly -3.0% and worse
            result = check([buy("AAPL", 10, 190.0), sell("MSFT", 20, 550.0)],
                           snapshot(equity=equity, day_start_equity=100_000, positions=held))
            self.assertEqual(rules_hit(result), ["daily_loss_limit"], equity)
            self.assertEqual([o["symbol"] for o in result["approved"]], ["MSFT"], equity)

    def test_down_less_than_the_limit_is_fine(self):
        result = check([buy("AAPL", 10, 190.0)], snapshot(equity=97_100, day_start_equity=100_000))
        self.assertEqual(len(result["approved"]), 1)
        self.assertFalse(any("daily_loss" in w for w in result["warnings"]))

    def test_missing_day_start_equity_warns_but_does_not_block(self):
        result = check([buy("AAPL", 10, 190.0)], snapshot())
        self.assertEqual(len(result["approved"]), 1)
        self.assertEqual(sum("daily_loss" in w for w in result["warnings"]), 1)

    def test_no_daily_loss_warning_when_there_are_no_buys(self):
        result = check([], snapshot())
        self.assertEqual(result["warnings"], [])

    def test_a_bad_day_start_equity_blocks_everything(self):
        for bad in (0, -5, "100000", float("nan"), True):
            result = check([buy("AAPL", 1, 10.0)], snapshot(day_start_equity=bad))
            self.assertEqual(rules_hit(result), ["bad_snapshot"], bad)


class EarningsBlackoutTests(unittest.TestCase):
    def blocked_for(self, earnings_day):
        result = check([buy("AAPL", 10, 190.0)], snapshot(earnings_dates={"aapl": earnings_day}))
        return rules_hit(result)

    def test_buys_within_two_days_of_earnings_are_blocked(self):
        self.assertEqual(self.blocked_for("2026-10-14"), ["earnings_blackout"])  # today
        self.assertEqual(self.blocked_for("2026-10-15"), ["earnings_blackout"])  # +1
        self.assertEqual(self.blocked_for("2026-10-16"), ["earnings_blackout"])  # +2

    def test_earnings_further_away_or_already_past_do_not_block(self):
        self.assertEqual(self.blocked_for("2026-10-17"), [])  # +3
        self.assertEqual(self.blocked_for("2026-10-13"), [])  # yesterday

    def test_other_symbols_and_sells_are_unaffected(self):
        snap = snapshot(positions={"AAPL": {"qty": 10, "market_value": 1_900.0}},
                        earnings_dates={"AAPL": "2026-10-15"})
        result = check([buy("MSFT", 10, 190.0), sell("AAPL", 10, 190.0)], snap)
        self.assertEqual(len(result["approved"]), 2)

    def test_a_longer_blackout_in_the_rules_is_respected(self):
        longer = copy.deepcopy(BASE_RULES)
        longer["limits"]["earnings_blackout_days"] = 5
        result = check([buy("AAPL", 10, 190.0)], snapshot(earnings_dates={"AAPL": "2026-10-19"}), longer)
        self.assertEqual(rules_hit(result), ["earnings_blackout"])


class CooldownTests(unittest.TestCase):
    def test_a_recent_exit_blocks_a_rebuy_inside_the_cooldown(self):
        for exit_day in ("2026-10-14", "2026-10-12"):  # 0 and 2 days ago, cooldown is 3
            result = check([buy("AAPL", 10, 190.0)], snapshot(recent_exits={"AAPL": exit_day}),
                           WITH_COOLDOWN)
            self.assertEqual(rules_hit(result), ["cooldown"], exit_day)

    def test_after_the_cooldown_a_rebuy_is_allowed(self):
        result = check([buy("AAPL", 10, 190.0)], snapshot(recent_exits={"AAPL": "2026-10-11"}),
                       WITH_COOLDOWN)
        self.assertEqual(len(result["approved"]), 1)

    def test_without_cooldown_days_in_the_rules_the_rule_is_off(self):
        result = check([buy("AAPL", 10, 190.0)], snapshot(recent_exits={"AAPL": "2026-10-14"}))
        self.assertEqual(len(result["approved"]), 1)

    def test_bad_cooldown_days_fail_clearly(self):
        for bad in (-1, 1.5, "3", True):
            broken = copy.deepcopy(BASE_RULES)
            broken["limits"]["cooldown_days"] = bad
            with self.assertRaises(ValueError, msg=bad):
                risk.check_rules([], snapshot(), broken)


class MalformedExtraDataTests(unittest.TestCase):
    def test_unreadable_dates_or_maps_block_everything_instead_of_guessing(self):
        bad_snapshots = [snapshot(today="soon"), snapshot(today=12345),
                         snapshot(earnings_dates={"AAPL": "not-a-date"}),
                         snapshot(earnings_dates=["AAPL"]),
                         snapshot(recent_exits={"AAPL": None}),
                         snapshot(sectors=["Technology"]),
                         snapshot(new_buys_today=-1), snapshot(new_buys_today=True),
                         snapshot(new_buys_today=1.5)]
        for snap in bad_snapshots:
            result = check([buy("AAPL", 1, 10.0), sell("AAPL", 1, 10.0)], snap, WITH_COOLDOWN)
            self.assertEqual(result["approved"], [], snap)
            self.assertEqual(rules_hit(result), ["bad_snapshot"] * 2, snap)

    def test_rules_apply_in_the_documented_order(self):
        # daily_loss_limit < earnings_blackout < cooldown < max_new_orders < position_cap < sector_cap < cash_reserve
        snap = snapshot(equity=90_000, day_start_equity=100_000, cash=1_000,
                        earnings_dates={"AAPL": "2026-10-15"}, recent_exits={"AAPL": "2026-10-13"},
                        new_buys_today=5)
        order = [buy("AAPL", 500, 190.0)]
        self.assertEqual(rules_hit(check(order, snap, WITH_COOLDOWN)), ["daily_loss_limit"])
        snap.pop("day_start_equity")
        self.assertEqual(rules_hit(check(order, snap, WITH_COOLDOWN)), ["earnings_blackout"])
        snap.pop("earnings_dates")
        self.assertEqual(rules_hit(check(order, snap, WITH_COOLDOWN)), ["cooldown"])
        snap.pop("recent_exits")
        self.assertEqual(rules_hit(check(order, snap, WITH_COOLDOWN)), ["max_new_orders"])
        snap["new_buys_today"] = 0
        self.assertEqual(rules_hit(check(order, snap, WITH_COOLDOWN)), ["position_cap"])

    def test_everything_returned_is_json_serialisable(self):
        result = check([buy("AAPL", 10, 190.0)], snapshot(earnings_dates={"AAPL": "2026-10-15"}))
        json.dumps(result)


if __name__ == "__main__":
    unittest.main()
