
from trading_system.crew.trading_crew_multiStocks import print_report


def test_print_report_uses_m1_fetch_field_names(capsys):
    print_report(
        symbols=["AAPL"],
        lookback_days=180,
        timeframe="1Day",
        holdings=[],
        fetch_report={
            "fetched": [
                {
                    "symbol": "AAPL",
                    "bars_fetched": 126,
                    "end": "2026-10-09",
                    "last_close": 190.0,
                }
            ],
            "missing": [],
        },
        m2_report={"results": [], "missing": []},
        decisions=[],
        plan=[],
    )

    output = capsys.readouterr().out

    assert "bars=126" in output
    assert "as_of=2026-10-09" in output
    assert "bars=None" not in output
    assert "as_of=None" not in output
