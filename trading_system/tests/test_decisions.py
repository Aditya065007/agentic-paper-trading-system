from trading_system.crew.trading_crew_multiStocks import make_decisions


def make_report(symbol="AAPL", triage="hold", composite=0.5):
    return {
        "results": [
            {
                "symbol": symbol,
                "score": {
                    "triage": triage,
                    "composite": composite,
                },
                "indicators": {},
            }
        ],
        "missing": [],
    }


def test_clear_buy_when_held():
    result = make_decisions(make_report(triage="clear_buy"), ["AAPL"])[0]
    assert result["decision"] == "BUY"


def test_clear_buy_when_not_held():
    result = make_decisions(make_report(triage="clear_buy"), [])[0]
    assert result["decision"] == "BUY"


def test_clear_sell_when_held():
    result = make_decisions(make_report(triage="clear_sell"), ["AAPL"])[0]
    assert result["decision"] == "SELL"


def test_clear_sell_when_not_held():
    result = make_decisions(make_report(triage="clear_sell"), [])[0]
    assert result["decision"] == "HOLD"
    assert "long-only" in result["reason"].lower()


def test_borderline_waits_for_m7():
    result = make_decisions(make_report(triage="borderline"), [])[0]
    assert result["decision"] == "HOLD"
    assert result["reason"] == "borderline, awaiting M7 debate"


def test_hold_triage():
    result = make_decisions(make_report(triage="hold"), [])[0]
    assert result["decision"] == "HOLD"


def test_missing_composite_fails_safe():
    result = make_decisions(
        make_report(triage="clear_buy", composite=None), []
    )[0]
    assert result["decision"] == "HOLD"
    assert "fail-safe HOLD" in result["reason"]


def test_symbol_only_in_missing_fails_safe():
    report = {"results": [], "missing": [{"symbol": "MSFT"}]}
    result = make_decisions(report, [])[0]
    assert result["symbol"] == "MSFT"
    assert result["decision"] == "HOLD"
    assert result["composite"] is None


import math
import pytest


@pytest.mark.parametrize("bad_composite", [float("nan"), float("inf"), -float("inf"), "0.5", "n/a", True, []])
@pytest.mark.parametrize("triage,held", [("clear_buy", []), ("clear_sell", ["AAPL"])])
def test_invalid_composite_fails_safe(bad_composite, triage, held):
    result = make_decisions(make_report(triage=triage, composite=bad_composite), held)[0]
    assert result["decision"] == "HOLD"
    assert "fail-safe HOLD" in result["reason"]


def test_score_list_fails_safe():
    report = {"results": [{"symbol": "AAPL", "score": [], "indicators": {}}], "missing": []}
    result = make_decisions(report, [])[0]
    assert result["decision"] == "HOLD"
    assert result["composite"] is None


def test_context_list_is_handled_safely():
    report = {"results": [{"symbol": "AAPL", "score": {"triage": "clear_buy", "composite": 0.7}, "indicators": {"context": []}}], "missing": []}
    result = make_decisions(report, [])[0]
    assert result["decision"] == "BUY"
    assert result["latest_close"] is None


def test_bad_row_does_not_block_later_good_row():
    class BadString:
        def __str__(self):
            raise RuntimeError("bad symbol")
    report = {"results": [
        {"symbol": BadString(), "score": {}, "indicators": {}},
        {"symbol": "AAPL", "score": {"triage": "clear_buy", "composite": 0.8}, "indicators": {}},
    ], "missing": []}
    result = make_decisions(report, [])[0]
    assert result["symbol"] == "AAPL"
    assert result["decision"] == "BUY"


def test_none_holdings_means_no_holdings():
    result = make_decisions(make_report(triage="clear_sell"), None)[0]
    assert result["decision"] == "HOLD"
    assert "long-only" in result["reason"].lower()


def test_duplicate_symbols_keep_first_row_only():
    report = {"results": [
        {"symbol": "AAPL", "score": {"triage": "hold", "composite": 0.5}, "indicators": {}},
        {"symbol": "AAPL", "score": {"triage": "clear_buy", "composite": 0.9}, "indicators": {}},
    ], "missing": [{"symbol": "AAPL"}]}
    results = make_decisions(report, [])
    assert len(results) == 1
    assert results[0]["decision"] == "HOLD"
