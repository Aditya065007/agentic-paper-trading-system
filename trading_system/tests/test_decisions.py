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
