# G7 / G8 runbook (commands, bash)

Run from the repo root with the virtual environment active. Nothing here changes a safety gate. Live paper
orders still need ALL of: `--approve SYM,SYM` (an explicit list), `--submit`, `ALPACA_ORDERS_ENABLED=true` and
`BROKER_MODE=alpaca_paper` on the command line (a UI cannot set them).

## The daily routine (two steps, you approve)
Run it after the US close (about 02:00 IST or later), so the signal uses the finished daily bar. Orders are DAY
limit orders: sent after the close they wait and are submitted at the next open.

1. Look at what the system proposes (nothing is sent anywhere):

       BROKER_MODE=alpaca_paper python -m trading_system.crew.run_cycle AAPL MSFT NVDA JPM

   If the guardian approved orders, the command ends by printing the exact line for step 2 on stderr.

2. Approve and place (paper account only):

       ALPACA_ORDERS_ENABLED=true BROKER_MODE=alpaca_paper python -m trading_system.crew.run_cycle AAPL MSFT NVDA JPM --approve AAPL,MSFT --submit

Why this works every day: the order id is `pt-YYYYMMDD-SYMBOL-side` (UTC date), so a new day is a new order; a DAY
order that was not filled has expired by the next day, so it cannot block it. Step 2 re-runs the analysis, so only
symbols the guardian approves in that run are placed; with `--advisory` the debate can answer differently the
second time. What can still stop an order on any day: the guardian limits (5 new buys a day, 12% per position, 30%
per sector, 10% cash reserve, 3% daily loss), not enough buying power, an open order for the same symbol and side,
or no valid signal (HOLD).

## Other switches
* Without `--approve` the cycle stops at proposals (nothing placed). `--approve` without `--submit` places
  DRY-RUN orders only. `--approve all` is refused together with `--submit`.
* `--timeframe BOTH`: the hourly run may only veto a daily BUY (clear_sell hourly -> HOLD). Hourly has never run
  in a live cycle, so check the first BOTH run by eye.
* Debate mode defaults to `consensus` (advisory.DEFAULT_DEBATE_MODE); `report` and `as_built` exist as function options.
* SPY is never an order candidate.
* Protective stop: add `ALPACA_STOPS_ENABLED=true` to step 2 to attach a stop to each BUY (OTO order). OFF by default.
  With it on, a BUY without a valid stop below its limit is rejected, never sent unprotected.
  [VERIFY on the first paper order] what time in force Alpaca gives the stop leg of a DAY entry: if it is DAY, the
  stop expires at that day's close. And while the stop leg is open, a later exit SELL for that symbol is skipped
  (an open sell order exists), so with stops on an exit needs the stop cancelled in the Alpaca dashboard first.
  With stops off (the default) exits are placed normally, but nothing watches the stop price at the broker.

## Reflection (G7)
    python -m trading_system.crew.run_cycle AAPL --reflect            # after a cycle: lessons for new closed trades
    python -m trading_system.crew.run_cycle AAPL --reflect-dry-run    # also counts DRY-RUN trades (not real fills)
A closed trade is a SELL matched first-in-first-out against earlier BUYs. Live trades count only once Alpaca
confirms the fill (status `filled`, with the real fill price); dry-run trades use limit prices.

## Not built (decisions for the team)
* Automatic daily placement with no human step (a scheduler plus auto-approval). The spec has an approval toggle,
  but no code auto-approves, and the explicit symbol list is what keeps a person in the loop.
* A per-run check of held positions against their stop price (spec 9: "otherwise checked each run").
