"""G7: Reflection agent (offline: fake runner, stub clients, fake fill lookup)."""

import json
from pathlib import Path

import pytest

from trading_system.crew import advisory, agent_files, journal, reflection
from trading_system.crew import run_cycle as rc
from trading_system.crew.llm import load_llm_config
from trading_system.tests.test_advisory import stub_agent_factory, stub_client_factory

SECRET = "sk-test-SECRET-1234567890"
GOOD = '{"lesson": "Momentum led the entry but volume faded; weight volume more.", "assessment": "bad"}'


def t(day, hour=15):
    return f"2026-10-{day:02d}T{hour:02d}:00:00+00:00"


def dry(side, symbol="AAPL", qty=10, price=100.0, oid="d1", ts=None, run="r1", status="accepted_dry_run"):
    return {"type": "order", "timestamp": ts or t(12), "run_id": run, "timeframe": "1Day", "symbol": symbol,
            "side": side, "qty": qty, "limit_price": price, "order_id": oid, "status": status,
            "submitted": None, "live": False, "account": "offline"}


def live(side, symbol="AAPL", qty=10, price=100.0, oid="11111111-aaaa", ts=None, run="r1",
         status="accepted", submitted=True):
    rec = dry(side, symbol, qty, price, oid, ts, run, status)
    rec.update(live=True, submitted=submitted, account="alpaca_paper")
    return rec


def filled(price, qty=10, status="filled"):
    return {"status": status, "filled_qty": str(qty), "filled_avg_price": str(price)}


def roundtrip(buy_price=100.0, sell_price=110.0, qty=10):
    return [dry("buy", qty=qty, price=buy_price, oid="b1", ts=t(12), run="r-buy"),
            dry("sell", qty=qty, price=sell_price, oid="s1", ts=t(15), run="r-sell")]


# ----------------------------------------------------------------- finding closed trades
def test_a_buy_then_sell_is_one_closed_trade_with_correct_numbers():
    (trade,) = reflection.find_closed_trades(roundtrip(100.0, 110.0, 10), include_dry_run=True)
    assert (trade["symbol"], trade["qty"], trade["entry_price"], trade["exit_price"]) == ("AAPL", 10, 100.0, 110.0)
    assert trade["pnl"] == 100.0 and trade["pnl_pct"] == 10.0 and trade["days_held"] == 3.0
    assert trade["entry_run_id"] == "r-buy" and trade["exit_run_id"] == "r-sell"
    assert trade["price_basis"] == "dry_run_limit" and trade["exit_key"] == "offline|AAPL|s1"


def test_dry_run_orders_are_ignored_unless_asked():
    assert reflection.find_closed_trades(roundtrip()) == []


def test_a_buy_alone_is_not_a_closed_trade():
    assert reflection.find_closed_trades([dry("buy")], include_dry_run=True) == []


def test_fifo_pairing_with_partial_sells_and_a_weighted_entry():
    orders = [dry("buy", qty=10, price=100, oid="b1", ts=t(10)), dry("buy", qty=10, price=110, oid="b2", ts=t(11)),
              dry("sell", qty=15, price=120, oid="s1", ts=t(14)), dry("sell", qty=5, price=90, oid="s2", ts=t(16))]
    first, second = reflection.find_closed_trades(orders, include_dry_run=True)
    assert first["qty"] == 15 and first["entry_price"] == round((10 * 100 + 5 * 110) / 15, 4)
    assert first["entry_order_ids"] == ["b1", "b2"] and first["days_held"] == 4.0
    assert second["qty"] == 5 and second["entry_price"] == 110.0 and second["pnl"] == -100.0


def test_symbols_and_accounts_are_paired_separately():
    orders = [dry("buy", "AAPL", oid="a"), dry("buy", "MSFT", oid="m"), dry("sell", "MSFT", price=120, oid="ms")]
    (trade,) = reflection.find_closed_trades(orders, include_dry_run=True)
    assert trade["symbol"] == "MSFT"


def test_a_sell_without_a_logged_buy_is_skipped_with_a_warning():
    notes = []
    assert reflection.find_closed_trades([dry("sell")], include_dry_run=True, warnings=notes) == []
    assert any("no logged buy" in n for n in notes)


def test_an_oversized_sell_reflects_only_the_matched_part():
    notes = []
    orders = [dry("buy", qty=4, oid="b"), dry("sell", qty=10, price=120, oid="s")]
    (trade,) = reflection.find_closed_trades(orders, include_dry_run=True, warnings=notes)
    assert trade["qty"] == 4 and any("only 4 of 10" in n for n in notes)


def test_rejected_and_unsubmitted_orders_never_count():
    orders = [dry("buy", oid="b"), dry("sell", oid="s", status="rejected")]
    assert reflection.find_closed_trades(orders, include_dry_run=True) == []
    orders = [live("buy", oid="1" * 8, submitted=False), live("sell", oid="2" * 8, submitted=False)]
    assert reflection.find_closed_trades(orders, fill_lookup=lambda i: filled(100)) == []


def test_live_orders_need_a_fill_lookup():
    notes = []
    orders = [live("buy", oid="b" * 8), live("sell", oid="s" * 8, price=110)]
    assert reflection.find_closed_trades(orders, fill_lookup=None, warnings=notes) == []
    assert any("no fill confirmation" in n for n in notes)


def test_live_trades_use_the_confirmed_fill_prices():
    orders = [live("buy", oid="buy-0001", price=100, ts=t(12)), live("sell", oid="sell-001", price=110, ts=t(15))]
    fills = {"buy-0001": filled(99.5), "sell-001": filled(111.25)}
    (trade,) = reflection.find_closed_trades(orders, fill_lookup=fills.__getitem__)
    assert (trade["entry_price"], trade["exit_price"], trade["price_basis"]) == (99.5, 111.25, "fill")
    assert trade["account"] == "alpaca_paper" and trade["pnl"] == 117.5


def test_an_unfilled_or_cancelled_live_buy_is_not_a_position():
    orders = [live("buy", oid="buy-0001"), live("sell", oid="sell-001", price=110)]
    fills = {"buy-0001": filled(100, status="canceled"), "sell-001": filled(110)}
    assert reflection.find_closed_trades(orders, fill_lookup=fills.__getitem__) == []


def test_a_failed_lookup_defers_live_trades_but_not_dry_run_ones():
    notes = []

    def boom(order_id):
        raise TimeoutError("slow")

    orders = roundtrip() + [live("buy", "MSFT", oid="m-buy-001"), live("sell", "MSFT", oid="m-sell-01")]
    trades = reflection.find_closed_trades(orders, include_dry_run=True, fill_lookup=boom, warnings=notes)
    assert [x["symbol"] for x in trades] == ["AAPL"] and any("fill lookup failed" in n for n in notes)


def test_one_failed_lookup_stops_live_pairing_so_a_sell_is_never_matched_to_the_wrong_lot():
    notes = []
    orders = [live("buy", oid="buy-0001", ts=t(10)), live("buy", oid="buy-0002", price=120, ts=t(11)),
              live("sell", oid="sell-001", price=130, ts=t(14))]

    def lookup(order_id):
        if order_id == "buy-0002":
            raise TimeoutError("slow")
        return filled(100 if order_id == "buy-0001" else 130)

    assert reflection.find_closed_trades(orders, fill_lookup=lookup, warnings=notes) == []
    assert any("fill lookup failed" in n for n in notes)
    # once the lookup works again the trade appears, paired with the right lot
    (trade,) = reflection.find_closed_trades(orders[:1] + orders[2:], fill_lookup=lookup)
    assert trade["entry_price"] == 100.0


def test_no_sell_means_no_lookup_at_all():
    calls = []
    reflection.find_closed_trades([live("buy")], fill_lookup=lambda i: calls.append(i))
    assert calls == []


def test_a_reflected_trade_is_not_reflected_again_and_failures_retry_once():
    ok = {"type": "outcome", "exit_key": "offline|AAPL|s1", "status": "ok"}
    assert reflection.find_closed_trades(roundtrip() + [ok], include_dry_run=True) == []
    bad = {"type": "outcome", "exit_key": "offline|AAPL|s1", "status": "fallback"}
    assert len(reflection.find_closed_trades(roundtrip() + [bad], include_dry_run=True)) == 1
    assert reflection.find_closed_trades(roundtrip() + [bad, bad], include_dry_run=True) == []


# ----------------------------------------------------------------- context, prompt, validation
def test_context_is_compact_and_scalar_only():
    (trade,) = reflection.find_closed_trades(roundtrip(), include_dry_run=True)
    entry = {"decision": "BUY", "triage": "clear_buy", "composite": 0.123456789, "regime": "trending",
             "news_score": 0.1, "market_risk_mode": "neutral", "stop_price": 93.8, "bars": list(range(500)),
             "sub_scores": {"trend": 0.5, "momentum": float("nan")}, "advisory": {"adjustment": 0.05}}
    context = reflection.build_reflection_context(trade, entry, {"decision": "SELL", "reason": "x" * 400})
    assert set(context) == {"symbol", "outcome", "entry", "exit"}
    assert context["entry"]["composite"] == 0.1235 and context["entry"]["sub_scores"]["momentum"] is None
    assert "bars" not in context["entry"] and context["entry"]["debate_adjustment"] == 0.05
    assert len(context["exit"]["reason"]) == 160 and len(json.dumps(context)) < 1500


def test_prompt_carries_the_rules_and_the_recent_lessons():
    prompt = reflection.format_reflection_prompt({"symbol": "AAPL"}, ["l1", "l2", "l3", "l4"])
    assert "never determine actions or orders" in prompt and "Do not recommend a trade" in prompt
    assert '"l4"' in prompt and '"l1"' not in prompt and "AAPL" in prompt


def test_validate_accepts_json_fenced_json_and_objects():
    expect = {"lesson": "ok", "assessment": "good"}
    assert reflection.validate_reflection_output('{"lesson": "ok", "assessment": "good"}') == expect
    assert reflection.validate_reflection_output('```json\n{"lesson": "ok", "assessment": "good"}\n```') == expect
    assert reflection.validate_reflection_output(expect) == expect


@pytest.mark.parametrize("bad", [
    "not json", "[]", '{"lesson": "a"}', '{"lesson": "a", "assessment": "good", "x": 1}',
    '{"lesson": "a", "assessment": "great"}', '{"lesson": "", "assessment": "good"}',
    '{"lesson": "two\\nlines", "assessment": "good"}', json.dumps({"lesson": "x" * 301, "assessment": "bad"}),
    '{"lesson": 5, "assessment": "bad"}', None, 42])
def test_validate_rejects_malformed_output(bad):
    with pytest.raises(ValueError):
        reflection.validate_reflection_output(bad)


def test_validate_redacts_secrets(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", SECRET)
    out = reflection.validate_reflection_output({"lesson": f"key {SECRET}", "assessment": "good"})
    assert SECRET not in out["lesson"]


# ----------------------------------------------------------------- the LLM step
def run(trades, runner, decisions=(), lessons=()):
    return reflection.run_reflection(trades, list(decisions), lessons, client_factory=stub_client_factory,
                                     agent_factory=stub_agent_factory, crew_runner=runner, printer=lambda *a: None)


def two_trades():
    orders = roundtrip() + [dry("buy", "MSFT", oid="mb", run="rm1"), dry("sell", "MSFT", price=90, oid="ms", run="rm2")]
    return reflection.find_closed_trades(orders, include_dry_run=True)


def test_one_call_per_trade_with_the_reflection_agent_and_model():
    calls = []

    def runner(agent, prompt, expected):
        calls.append((agent.role, prompt, expected))
        return GOOD

    outcomes = run(two_trades(), runner, lessons=["an older lesson"])
    assert len(calls) == 2 and {c[0] for c in calls} == {"Reflection"}
    assert "JSON only" in calls[0][2] and "an older lesson" in calls[0][1]
    assert [o["status"] for o in outcomes] == ["ok", "ok"] and outcomes[0]["model"] == "groq/openai/gpt-oss-20b"
    assert outcomes[0]["assessment"] == "bad" and outcomes[0]["lesson"].startswith("Momentum")


def test_an_invalid_reply_is_retried_once_then_succeeds():
    replies = iter(["garbage", GOOD])
    (outcome,) = run(two_trades()[:1], lambda a, p, e: next(replies))
    assert outcome["status"] == "ok" and len(outcome["warnings"]) == 1


def test_two_failures_give_a_fallback_with_no_lesson():
    def runner(agent, prompt, expected):
        raise RuntimeError(f"provider down {SECRET}")

    (outcome,) = run(two_trades()[:1], runner)
    assert outcome["status"] == "fallback" and outcome["lesson"] is None and outcome["assessment"] is None
    assert len(outcome["warnings"]) == 2 and outcome["pnl"] == 100.0


def test_context_comes_from_the_entry_and_exit_decision_records():
    prompts = []
    decisions = [{"type": "decision", "run_id": "r-buy", "symbol": "AAPL", "composite": 0.4111, "decision": "BUY"},
                 {"type": "decision", "run_id": "r-sell", "symbol": "AAPL", "composite": -0.5222, "decision": "SELL"}]
    run(two_trades()[:1], lambda a, p, e: prompts.append(p) or GOOD, decisions)
    assert "0.4111" in prompts[0] and "-0.5222" in prompts[0]


def test_no_trades_means_no_agent_is_built():
    def factory(role, config=None):
        raise AssertionError("no client should be created")

    assert reflection.run_reflection([], [], client_factory=factory, printer=lambda *a: None) == []


# ----------------------------------------------------------------- reflect() end to end
def seed(tmp_path, orders=None):
    journal.record_run({"timeframe": "1Day", "decisions": [
        {"symbol": "AAPL", "decision": "BUY", "triage": "clear_buy", "composite": 0.45, "sub_scores": {"trend": 0.4}}]},
        "r-buy", t(12), tmp_path)
    for order in orders if orders is not None else roundtrip():
        placed = [{"order_id": order["order_id"], "status": order["status"], "details": {
            "symbol": order["symbol"], "side": order["side"], "qty": order["qty"],
            "limit_price": order["limit_price"]}}]
        journal.append_order_results(order["run_id"], "1Day", placed, live=order["live"],
                                     timestamp=order["timestamp"], memory_dir=tmp_path)


def do_reflect(tmp_path, runner, **kw):
    return reflection.reflect("run-x", "2026-10-16T14:00:00+00:00", ["AAPL"], "1Day", include_dry_run=True,
                              memory_dir=tmp_path, reports_dir=tmp_path / "reports",
                              client_factory=stub_client_factory, agent_factory=stub_agent_factory,
                              crew_runner=runner, printer=lambda *a: None, **kw)


def test_reflect_writes_lesson_outcome_line_and_agent_file_then_does_not_repeat(tmp_path):
    seed(tmp_path)
    out = do_reflect(tmp_path, lambda a, p, e: GOOD)
    assert out["status"] == "ok" and out["closed_trades"] == 1 and out["reflections"][0]["assessment"] == "bad"
    assert journal.load_recent_lessons(3, tmp_path) == ["Momentum led the entry but volume faded; weight volume more."]
    (outcome,) = [r for r in journal.read_decisions(tmp_path) if r["type"] == "outcome"]
    assert outcome["symbol"] == "AAPL" and outcome["pnl"] == 100.0 and outcome["status"] == "ok"
    assert outcome["run_id"] == "run-x" and outcome["entry_run_id"] == "r-buy"
    text = Path(out["agent_file"]).read_text(encoding="utf-8")
    assert Path(out["agent_file"]).name == "agentReflection_16102026-10-00_1D.md"
    assert "Run ID: run-x" in text and "groq/openai/gpt-oss-20b" in text and "weight volume more" in text
    assert "Outcome | " in (tmp_path / "trade_journal.md").read_text(encoding="utf-8")
    calls = []
    again = do_reflect(tmp_path, lambda a, p, e: calls.append(p) or GOOD)
    assert again["status"] == "no_closed_trades" and calls == []


def test_fallback_records_the_attempt_but_no_lesson_and_stops_after_two(tmp_path):
    seed(tmp_path)

    def broken(agent, prompt, expected):
        raise RuntimeError("down")

    for _ in range(2):
        assert do_reflect(tmp_path, broken)["status"] == "degraded"
    assert journal.load_recent_lessons(3, tmp_path) == []
    assert len([r for r in journal.read_decisions(tmp_path) if r["type"] == "outcome"]) == 2
    assert do_reflect(tmp_path, broken)["status"] == "no_closed_trades"


def test_a_setup_problem_writes_nothing_and_keeps_the_trade_pending(tmp_path):
    seed(tmp_path)

    def no_key(role, config=None):
        raise RuntimeError("GROQ_API_KEY is not set")

    out = reflection.reflect("run-x", t(16), ["AAPL"], "1Day", include_dry_run=True, memory_dir=tmp_path,
                             reports_dir=tmp_path / "reports", client_factory=no_key,
                             agent_factory=stub_agent_factory, printer=lambda *a: None)
    assert out["status"] == "unavailable" and any("Reflection unavailable" in w for w in out["warnings"])
    assert not [r for r in journal.read_decisions(tmp_path) if r["type"] == "outcome"]
    assert do_reflect(tmp_path, lambda a, p, e: GOOD)["status"] == "ok"


def test_no_closed_trades_makes_no_llm_call_and_no_files(tmp_path):
    seed(tmp_path, orders=[dry("buy")])
    out = do_reflect(tmp_path, lambda a, p, e: pytest.fail("the LLM must not be called"))
    assert out == {"status": "no_closed_trades", "closed_trades": 0, "reflections": [], "warnings": [],
                   "agent_file": None}
    assert not (tmp_path / "reports").exists()


def test_the_per_run_cap_defers_the_rest(tmp_path):
    orders = []
    for i in range(7):
        sym = f"S{i}"
        orders += [dry("buy", sym, oid=f"b{i}", ts=t(10)), dry("sell", sym, price=110, oid=f"s{i}", ts=t(11))]
    seed(tmp_path, orders=orders)
    out = do_reflect(tmp_path, lambda a, p, e: GOOD)
    assert len(out["reflections"]) == 5 and out["closed_trades"] == 7
    assert any("2 more closed trade" in w for w in out["warnings"])
    assert do_reflect(tmp_path, lambda a, p, e: GOOD)["closed_trades"] == 2


def test_reflect_never_raises(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(journal, "read_decisions", lambda *a, **k: (_ for _ in ()).throw(OSError("gone")))
    out = reflection.reflect("r", t(16), memory_dir=tmp_path)
    assert out["status"] == "unavailable" and "M7 reflection warning" in capsys.readouterr().err


def test_lessons_written_by_reflection_reach_the_next_judge(tmp_path, monkeypatch):
    from trading_system.crew import memory
    seed(tmp_path)
    do_reflect(tmp_path, lambda a, p, e: GOOD)
    monkeypatch.setattr(memory, "MEMORY_DIR", tmp_path)
    from trading_system.tests.test_advisory import make_runner, row, run_debate
    calls = []
    run_debate([row("AAPL", 0.38, "borderline", "HOLD")], runner=make_runner(calls=calls))
    assert any("weight volume more" in prompt for role, prompt in calls)


# ----------------------------------------------------------------- fills, journal, agent file
class FakeResponse:
    def __init__(self, code, data):
        self.status_code, self._data = code, data

    def json(self):
        return self._data


def test_alpaca_fill_lookup_reads_one_order(monkeypatch):
    from trading_system.servers.tools import live_orders
    seen = []
    monkeypatch.setattr(live_orders, "_request", lambda m, p, params=None, body=None: seen.append((m, p)) or
                        FakeResponse(200, {"status": "filled", "filled_qty": "10", "filled_avg_price": "101.5", "id": "x"}))
    assert reflection.alpaca_fill_lookup("abcd1234-ef56") == {"status": "filled", "filled_qty": "10",
                                                              "filled_avg_price": "101.5"}
    assert seen == [("GET", "/orders/abcd1234-ef56")]


@pytest.mark.parametrize("bad_id", [None, "", "../account", "short", "a b c d e f g h", "x" * 80])
def test_alpaca_fill_lookup_rejects_odd_ids_without_any_request(monkeypatch, bad_id):
    from trading_system.servers.tools import live_orders
    monkeypatch.setattr(live_orders, "_request", lambda *a, **k: pytest.fail("no request expected"))
    with pytest.raises(ValueError):
        reflection.alpaca_fill_lookup(bad_id)


def test_alpaca_fill_lookup_raises_on_a_bad_reply(monkeypatch):
    from trading_system.servers.tools import live_orders
    monkeypatch.setattr(live_orders, "_request", lambda *a, **k: FakeResponse(404, {"message": "no"}))
    with pytest.raises(RuntimeError):
        reflection.alpaca_fill_lookup("abcd1234-ef56")


def test_append_outcome_writes_a_line_and_redacts(tmp_path, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", SECRET)
    assert journal.append_outcome({"symbol": "AAPL", "pnl": 12.5, "lesson": f"leak {SECRET}", "status": "ok"}, tmp_path)
    (line,) = journal.read_decisions(tmp_path)
    assert line["type"] == "outcome" and line["pnl"] == 12.5
    assert SECRET not in (tmp_path / "decisions.jsonl").read_text() + (tmp_path / "trade_journal.md").read_text()
    assert journal.append_outcome("junk", tmp_path) is False


def test_append_outcome_never_raises(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(journal, "_append", lambda *a, **k: (_ for _ in ()).throw(OSError("ro")))
    assert journal.append_outcome({"symbol": "A"}, tmp_path) is False
    assert "M9 journal warning" in capsys.readouterr().err


def test_reflection_is_in_the_roster_but_the_cycle_writer_skips_it(tmp_path):
    assert agent_files.AGENTS[-1] == "Reflection"
    from trading_system.tests.test_g6_journal import STAMP, full_result
    names = {Path(p).name.split("_")[0] for p in agent_files.write_agent_files(full_result(), "r", STAMP, tmp_path)}
    assert "agentReflection" not in names
    assert agent_files.write_agent_file("Reflection", full_result(), "r", STAMP, tmp_path) is None


def test_write_agent_file_never_raises(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(agent_files, "render_agent_file", lambda *a, **k: (_ for _ in ()).throw(OSError("full")))
    assert agent_files.write_agent_file("Data", {}, "r", "2026-10-12T14:15:00+00:00", tmp_path) is None
    assert "M9 agent file warning (Data)" in capsys.readouterr().err


# ----------------------------------------------------------------- run_cycle
def fake_cycle(monkeypatch, reflect_impl=None):
    from trading_system.tests.test_g6_journal import full_result
    result = full_result()
    result.update(run_id="run-9", degraded=False, agent_report_paths=["a.md"], journal_files={"decisions_log": "d"})

    async def fake_run_strategy(**kwargs):
        return result

    monkeypatch.setattr(rc.crew, "run_strategy", fake_run_strategy)
    monkeypatch.setattr(rc.gp, "place_approved", lambda orders, only_symbols=None: [])
    if reflect_impl is not None:
        monkeypatch.setattr(reflection, "reflect", reflect_impl)


def test_reflection_is_off_by_default_and_the_new_keys_are_returned(monkeypatch):
    fake_cycle(monkeypatch, lambda **kw: pytest.fail("reflection must not run by default"))
    out = rc.run_cycle_sync({"symbols": ["AAPL"]})
    assert out["reflections"] is None and out["agent_report_paths"] == ["a.md"]
    assert out["journal_files"] == {"decisions_log": "d"}


def test_reflect_true_runs_after_the_cycle_with_dry_run_off_and_no_fills_offline(monkeypatch):
    seen = {}
    fake_cycle(monkeypatch, lambda **kw: seen.update(kw) or {"status": "no_closed_trades", "reflections": []})
    monkeypatch.delenv("BROKER_MODE", raising=False)
    out = rc.run_cycle_sync({"symbols": ["aapl"], "reflect": True})
    assert out["reflections"]["status"] == "no_closed_trades"
    assert seen["run_id"] == "run-9" and seen["include_dry_run"] is False and seen["fill_lookup"] is None
    assert seen["symbols"] == ["AAPL"] and seen["timeframe"] == "1Day"


def test_reflect_dry_run_and_alpaca_mode_pass_the_right_options(monkeypatch):
    seen = {}
    fake_cycle(monkeypatch, lambda **kw: seen.update(kw) or {"status": "ok", "reflections": []})
    monkeypatch.setenv("BROKER_MODE", "alpaca_paper")
    rc.run_cycle_sync({"symbols": ["AAPL"], "reflect": True, "reflect_dry_run": True})
    assert seen["include_dry_run"] is True and seen["fill_lookup"] is reflection.alpaca_fill_lookup


def test_a_failing_reflection_never_changes_the_cycle(monkeypatch):
    fake_cycle(monkeypatch, lambda **kw: (_ for _ in ()).throw(RuntimeError("boom")))
    out = rc.run_cycle_sync({"symbols": ["AAPL"], "reflect": True})
    assert out["reflections"]["status"] == "unavailable" and out["run_id"] == "run-9"
    assert out["status"] in ("no_orders", "awaiting_approval", "ok")


def test_reflect_dry_run_needs_reflect():
    with pytest.raises(rc.CycleConfigError):
        rc.run_cycle_sync({"symbols": ["AAPL"], "reflect_dry_run": True})


def test_the_cli_parses_the_new_flags():
    assert rc._parse_cli([]) == {"symbols": ["AAPL"], "advisory": False, "reflect": False, "reflect_dry_run": False}
    assert rc._parse_cli(["AAPL", "MSFT", "--advisory", "--reflect"]) == {
        "symbols": ["AAPL", "MSFT"], "advisory": True, "reflect": True, "reflect_dry_run": False}
    assert rc._parse_cli(["NVDA", "--reflect-dry-run"])["reflect"] is True
