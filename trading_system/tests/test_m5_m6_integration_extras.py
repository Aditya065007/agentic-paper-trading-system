"""Integration extras for M5/M6: checks the three contract files cannot make.
They cover (1) registration on the REAL FastMCP class, (2) the MCP server file exposing the M5/M6
tools, and (3) the shipped config and fixture files. Fully offline: no network, no keys.
Written by the integration owner. Do not weaken these tests to make them pass."""

import asyncio
import json
import sys
import unittest
from pathlib import Path

REPOSITORY_DIR = Path(__file__).resolve().parents[2]  # the folder that contains trading_system/
if str(REPOSITORY_DIR) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_DIR))

from trading_system.servers.tools import broker, risk

PACKAGE_DIR = REPOSITORY_DIR / "trading_system"
M5_M6_TOOLS = {"size_position", "check_rules", "get_account", "get_portfolio", "place_order",
               "get_open_orders", "cancel_order"}


def listed_tool_names(server):
    return {tool.name for tool in asyncio.run(server.list_tools())}


class RealFastMCPRegistrationTests(unittest.TestCase):
    def test_the_tools_register_on_a_real_fastmcp_server(self):
        from mcp.server.fastmcp import FastMCP
        server = FastMCP("m5m6-extras-test")
        risk.register_risk_tools(server)
        broker.register_broker_tools(server)
        self.assertEqual(listed_tool_names(server), M5_M6_TOOLS)

    def test_the_registered_functions_still_work_as_plain_functions(self):
        from mcp.server.fastmcp import FastMCP
        server = FastMCP("m5m6-extras-test")
        risk.register_risk_tools(server)
        self.assertEqual(risk.size_position("AAPL", 190.0, 3.2, 100_000)["qty"], 63)


class McpServerFileTests(unittest.TestCase):
    def test_the_project_server_exposes_the_m5_m6_tools_next_to_the_existing_ones(self):
        from trading_system.servers import technical_indicators_mcp_server_multiStocks as server_file
        names = listed_tool_names(server_file.mcp)
        self.assertTrue(M5_M6_TOOLS <= names, M5_M6_TOOLS - names)
        for existing in ("fetch_ohlcv_batch", "calculate_m2_batch"):
            self.assertIn(existing, names)


class ShippedFilesTests(unittest.TestCase):
    def test_the_shipped_rules_hold_the_architecture_starting_values(self):
        rules = risk.load_rules()
        self.assertEqual(rules["position"], {"max_single_position_pct": 0.12, "cash_reserve_pct": 0.10,
                                             "max_sector_exposure_pct": 0.30})
        self.assertEqual(rules["sizing"], {"risk_per_trade_pct": 0.01, "stop_loss_atr_mult": 2.0,
                                           "trailing_exit_atr_mult": 3.0})
        self.assertEqual(rules["limits"]["daily_loss_limit_pct"], 0.03)
        self.assertEqual(rules["limits"]["max_new_orders_per_day"], 5)
        self.assertEqual(rules["limits"]["earnings_blackout_days"], 2)
        self.assertEqual(rules["orders"]["order_type"], "limit")
        self.assertIs(rules["orders"]["whole_shares"], True)
        self.assertEqual(rules["orders"]["limit_buffer_pct"], 0.002)

    def test_the_shipped_fixtures_have_the_documented_shape(self):
        account = json.loads((PACKAGE_DIR / "fixtures" / "account.json").read_text(encoding="utf-8"))
        for key in ("equity", "cash", "buying_power"):
            self.assertGreater(account[key], 0, key)
        positions = json.loads((PACKAGE_DIR / "fixtures" / "portfolio.json").read_text(encoding="utf-8"))["positions"]
        self.assertTrue(positions)
        for position in positions:
            for key in ("symbol", "qty", "avg_entry", "market_value", "pl_pct", "sector"):
                self.assertIn(key, position)

    def test_the_shipped_files_work_together_offline(self):
        import os
        from unittest import mock
        env = {k: v for k, v in os.environ.items() if k != "BROKER_MODE"}
        with mock.patch.dict(os.environ, env, clear=True):
            snapshot = broker.get_portfolio_snapshot()
        self.assertEqual(risk.check_rules([], snapshot)["blocked"], [])


class PurityTests(unittest.TestCase):
    TOOLS = PACKAGE_DIR / "servers" / "tools"

    def test_risk_is_pure_and_broker_has_no_hard_coded_secrets(self):
        risk_text = (self.TOOLS / "risk.py").read_text(encoding="utf-8")
        for forbidden in ("import requests", "import mcp", "from mcp", "import socket", "os.environ"):
            self.assertNotIn(forbidden, risk_text, forbidden)
        broker_text = (self.TOOLS / "broker.py").read_text(encoding="utf-8")
        for pattern in ("gsk_", "PKTEST", "AKIA", "https://api.alpaca.markets"):
            self.assertNotIn(pattern, broker_text, pattern)


if __name__ == "__main__":
    unittest.main()
