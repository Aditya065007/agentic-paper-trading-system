"""G8: BOTH timeframes = daily decides, hourly may only VETO a BUY.

Rule (a deliberate, permissive choice so that daily signals still lead to orders):
  1. Run the hourly pipeline first (decisions only: no debate, no sizing, no orders).
  2. A symbol is vetoed only if its hourly triage is clear_sell (composite below the hourly sell line).
  3. Run the daily pipeline as usual (debate, guardian, sizing). Each daily BUY whose symbol is vetoed is
     downgraded to HOLD before sizing. The hourly run never creates, enlarges or reverses anything.
  4. If the hourly run fails, the daily run proceeds with no veto and the failure is recorded as a warning
     and marks the run degraded. This knowingly departs from the "missing data -> HOLD" rule because the
     hourly signal is timing-only; an unavailable timing filter should not stop a valid daily entry.

The hourly timeframe has never run in a live cycle; its bars were verified on real data (G4) but the
full hourly scoring path is still [UNVERIFIED] live."""

import sys
from collections.abc import Mapping

HOURLY_LOOKBACK_DAYS = 30        # profiles.yaml 1Hour history_days
HOURLY_VETO_TRIAGE = "clear_sell"


def hourly_vetoes(hourly_decisions) -> dict:
    """{SYMBOL: reason} for every hourly decision row triaged clear_sell. Pure."""
    vetoes = {}
    for row in hourly_decisions or []:
        if not isinstance(row, Mapping) or not row.get("symbol"):
            continue
        if str(row.get("triage", "")).lower() == HOURLY_VETO_TRIAGE:
            composite = row.get("composite")
            shown = (f"{composite:.4f}" if isinstance(composite, (int, float))
                     and not isinstance(composite, bool) else "n/a")
            vetoes[str(row["symbol"]).upper()] = f"hourly composite {shown} is a clear SELL"
    return vetoes


async def run_both(run_strategy, symbols, lookback_days, advisory_band=None, guarded=True) -> dict:
    """Run hourly then daily and return the DAILY result with the BOTH fields added.

    `run_strategy` is crew.run_strategy (injected so tests can fake it). Never raises for an hourly
    failure; a daily failure raises exactly as a normal daily run would."""
    warnings, veto, hourly_run_id = [], {}, None
    try:
        hourly = await run_strategy(symbols=symbols, lookback_days=HOURLY_LOOKBACK_DAYS, timeframe="1Hour")
        hourly_run_id = hourly.get("run_id")
        veto = hourly_vetoes(hourly.get("decisions"))
        if hourly.get("degraded"):
            warnings.append("Hourly run was degraded (some inputs unavailable); its veto may be incomplete.")
    except Exception as exc:
        veto = {}
        warnings.append(f"Hourly run failed ({type(exc).__name__}); the daily run proceeds with no veto.")
        print(f"BOTH timeframes warning: {type(exc).__name__}: {exc}", file=sys.stderr)

    extra = {"advisory_band": advisory_band} if advisory_band else {}
    result = await run_strategy(symbols=symbols, lookback_days=lookback_days, timeframe="1Day",
                                guarded=guarded, hourly_veto=veto, timeframe_mode="BOTH",
                                hourly_warnings=warnings, **extra)
    result["hourly_run_id"] = hourly_run_id
    return result
