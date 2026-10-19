# Experiments log

Course requirement (architecture spec section 12): for every change to an indicator, weight, model or
threshold, record what changed, why, the result, and why it worked or failed.
Design decisions (what the code does) are in `design_decisions.md`; this file records what happened when
we tried things.

## How to add an entry

Copy the template at the bottom, number it `E<next>`, and fill every line. Rules:

1. **Config hash.** Every agent file in `run_reports/YYYYMMDD/` starts with a 12-character config hash
   (data, llm_verified, profiles, regime, rules, universe, weights). Quote the hash of the run you are
   describing, so two runs can be compared. If the hash is unknown, write `unknown`.
2. **Only numbers you saw.** Copy numbers from a run output. Never turn a number into a story and never
   fill a gap with a guess; write `not measured` instead.
3. **Why it worked or failed** is a cause you can point to in the code or data. If you do not know the
   cause yet, write `not yet known` and say what would show it.
4. **Status** is one of:
   * `SEEN`: the number comes from a run the user made and pasted into a working session.
   * `REPORTED`: recorded in the team's context notes from an earlier session; not re-run when this file
     was written.
   * `CODE-TESTED`: the behaviour is proven by unit tests with fakes only; no live run yet.
   * `UNVERIFIED`: nobody has checked it.

All numbers below come from the user's own runs between 14 and 19 October 2026 as recorded in the team's
context notes. They were not re-run when this file was written (G9).

---

## E01. Per-symbol sleeve tags (G1)

* **Date:** 18 Oct 2026. **Config hash:** unknown (the hash header arrived with G6).
* **What changed:** each stock is scored with its own sleeve from `config/universe.yaml`. NVDA, AMD, TSLA,
  META, AMZN and BAC are `high_beta`; the rest are `core`. Before this, every run scored every stock as `core`.
* **Why:** the course needs a high-beta sleeve (at least 30% of the stocks), and the spec gives high-beta
  stocks less weight on news and more on momentum and volume.
* **Result:** NVDA composite -0.3896 -> -0.3977. AAPL, MSFT and JPM unchanged (they stay `core`). All four
  stayed HOLD. NVDA ended 0.002 above the -0.40 sell line.
* **Why it worked or failed:** the weights did change (high-beta: news 0.10 against 0.15, momentum 0.25
  against 0.20, volume 0.20 against 0.15, in the trending column). Which sub-scores produced the -0.008 move
  was not recorded. Not yet known; the `sub_scores` in `memory/decisions.jsonl` of a run with the same data
  would show it.
* **Status:** SEEN. **Decisions:** 37, 38, 39. The sleeve assignment is a judgement call that the team has
  not confirmed.

## E02. Regime blend of the weights (G2)

* **Date:** 18-19 Oct 2026. **Config hash:** unknown.
* **What changed:** weights = t x trending + (1 - t) x ranging, with t = clip((ADX - 20) / 5, 0, 1), read
  from `config/weights.yaml`. Before this only the trending column was used.
* **Why:** spec 8.3. In a ranging market trend-following indicators mislead and mean reversion should count
  for more.
* **Result:** NVDA -0.398 -> -0.502 and AAPL 0.3835 -> 0.3949; MSFT and JPM unchanged by the blend at that
  run. A later run gave AAPL 0.3893 (ranging), NVDA -0.5237 (ranging, clear sell) and JPM -0.0744 (ADX 23.96,
  blend strength 0.79).
* **Why it worked or failed:** NVDA's ranging high-beta column puts 0.35 on mean reversion (0.05 in the
  trending column), which fits the large move, but the sub-scores were not recorded, so the cause is not
  proven. MSFT stayed trending (ADX 37.6), so the blend left it alone. Whether the blend gives better trades
  is not yet known: no trade has been closed.
* **Status:** REPORTED. **Decisions:** 40, 41.

## E03. One headline moved a score to -0.93 (G5 confidence guard)

* **Date:** 18-19 Oct 2026. **Config hash:** unknown before the guard; 762e0f7592a9 for the 19 Oct run.
* **What changed:** the news score is multiplied by min(1, scored headlines / 3).
* **Why:** JPM scored -0.9318 from a single headline, a cautionary Jamie Dimon quote that FinBERT read as
  strongly negative. One headline is too thin to carry a signal.
* **Result:** JPM -0.9318 -> -0.3106 (still one headline, now counted at one third). JPM's composite stayed
  HOLD both times. 19 Oct run: AAPL +0.1224, MSFT +0.1312, NVDA -0.3546, JPM -0.3106.
* **Why it worked or failed:** the guard limits the weight of thin evidence. It does not remove the cause:
  FinBERT scores the tone of the text, so a single cautionary headline still leaves a visible negative score.
* **Status:** SEEN. **Decision:** 57.

## E04. News scores drift between runs on the free NewsAPI plan

* **Date:** 18-19 Oct 2026.
* **What changed:** nothing; an observation.
* **Result:** AAPL 0.122 -> 0.160 -> 0.122 and NVDA -0.355 -> -0.135 -> -0.355 across runs.
* **Why:** the notes say the free plan returns few headlines, delayed by about 24 hours, so the set of
  articles changes from run to run. The earlier batched query also shared one 100-article page across all
  symbols (`pageSize` 100 in `get_news`).
* **Action taken:** per-symbol queries with a 120-minute disk cache and a daily request budget (E10).
* **Status:** REPORTED. **Decision:** 58.

## E05. The debate flipped a decision (M7 Bull, Bear, Judge)

* **Date:** before 18 Oct 2026.
* **What happened:** a small Judge adjustment moved a borderline stock across a threshold and changed the
  decision, and the Judge's written reason contradicted its own adjustment. The notes name MSFT (flipped to
  BUY) in one place and AAPL (BUY -> HOLD) in another; which one is not confirmed.
* **Why it matters:** the LLM is meant to advise only, and a nudge of a few hundredths should not decide a
  trade.
* **Action taken:** G8 made `consensus` the default debate mode: a decision changes only when the Judge
  output is valid, Bull and Bear are both present, and the adjustment agrees with the threshold crossing.
  `report` (never changes a decision) and `as_built` still exist.
* **Result of the fix:** CODE-TESTED only. Not yet seen in a live run.
* **Also reported, not verified:** the same inputs gave different adjustments 14 minutes apart. The
  temperature is set to 0.0 in `config/llm_verified.yaml`, so a difference would be surprising; it was not
  checked.
* **Status:** REPORTED for the flip; UNVERIFIED for the repeatability claim. **Decision:** 70.

## E06. The 09:00 hourly bar mixes in pre-market (G4)

* **Date:** 18 Oct 2026.
* **What changed:** hourly bars are kept if they overlap 09:30-16:00 New York time, so the 09:00 bar stays.
* **Why:** spec 5.5 asks for regular hours only. Alpaca's hourly bars are aligned to the clock hour.
* **Result:** real hourly bars: 7 per regular day (09:00-15:00). The 09:00 bar on Oct 9 for AAPL had
  834,589 shares against 257,976 for the 10:00 bar, so pre-market volume is mixed into it.
* **Decision:** kept on purpose, because dropping it would lose the 09:30-10:00 half hour. Volume-based
  hourly indicators will see that bar as unusually large.
* **Status:** SEEN. **Decision:** 50 (its `[VERIFY]` is answered by this entry). Hourly has never run in a
  live cycle.

## E07. Groq models for the agents

* **Date:** 14 Oct 2026.
* **What changed:** News, Bull, Bear and Reflection use `openai/gpt-oss-20b`; the Judge uses
  `openai/gpt-oss-120b`. Settings are in `config/llm_verified.yaml`.
* **Why:** spec 10.1 suggests a small, fast model for News and Reflection and a larger one for the Judge.
* **Result:** both models answered a test call on the user's free Groq plan (30 requests per minute and
  8,000 tokens per minute per model).
* **Why `max_output_tokens` stays at 1024:** reasoning tokens count as output. A low cap gives an empty
  reply, which fails validation and falls back to adjustment 0. Do not cut it much.
* **Status:** SEEN (the call); REPORTED (the limits).

## E08. Market regime on 19 Oct 2026 (G3)

* **Date:** 19 Oct 2026. **Config hash:** 762e0f7592a9.
* **Result:** `neutral`: SPY trend down, SPY volatility 0.1048, breadth 75%.
* **Why:** by the rules in `config/regime.yaml`, a down trend gives `risk_off` only with volatility at or
  above 0.30 or breadth below 40%. Neither held, and `risk_on` needs an up trend.
* **Effect on trading:** none. `block_new_buys_in_risk_off` is `false`, so the regime is reported only.
* **Status:** SEEN. **Decisions:** 43, 44, 45. Whether the team switches the pause on is open.

## E09. `missing_indicators` fixed, and what it revealed (G5)

* **Date:** 19 Oct 2026.
* **What changed:** the list used to show every indicator as missing on every run; it now shows real gaps.
* **Result:** `bb_squeeze` is missing for every stock. JPM is also missing `cmf20` (not investigated).
* **Why `bb_squeeze` is missing:** it needs the 20-bar band width plus a 120-bar window, about 140 bars. A
  180-day daily fetch gives about 124 bars. It is context only and does not enter the score.
* **Open:** why JPM's `cmf20` is missing. Start with its inputs (high, low, close, volume).
* **Status:** SEEN. **Decision:** 56.

## E10. Per-symbol news with a cache and a daily budget (G5)

* **Date:** 19 Oct 2026.
* **What changed:** one NewsAPI request per symbol, cached 120 minutes on disk, counted against
  `NEWSAPI_DAILY_LIMIT` (default 80).
* **Why:** a busy symbol cannot crowd out a quiet one, and repeated runs stop spending requests (E04).
* **Result:** CODE-TESTED. The live check (a second run showing `cache_hits` equal to the number of symbols)
  was handed to the user; its output has not been recorded.
* **Status:** UNVERIFIED live. **Decision:** 58. The 100-requests-a-day plan limit and the UTC day boundary
  are assumptions.

## E11. Baseline run of 19 Oct 2026 (daily, no debate)

* **Run id:** run-20261018T201711-6fc39c0eb1. **Config hash:** 762e0f7592a9. Broker: offline fixtures.
* **Result:** AAPL 0.3893 ranging, borderline. MSFT 0.3282 trending (ADX 37.6), borderline. NVDA -0.5237
  ranging, clear sell (not held, so a long-only HOLD). JPM -0.0744 blend, hold. All four HOLD. AAPL and MSFT
  are the debate candidates. Last bars were from 16 Oct.
* **Use:** compare later runs against this one when a weight, threshold or model changes.
* **Status:** SEEN.

## E12. Reflection agent on closed trades (G7)

* **What changed:** a post-mortem lesson for each closed trade, saved to `lessons.md`; the Judge receives
  the last 3 lessons.
* **Result:** not run live. The user has no closed paper trade yet (the one live order, placed on a Sunday,
  was cancelled).
* **Why it worked or failed:** not yet known (not run live). A lesson that names a real cause, and a Judge
  reason that uses it, would show whether it works.
* **Status:** CODE-TESTED. **Decisions:** 64-68.

## E13. BOTH timeframes: hourly may only veto a daily BUY (G8)

* **What changed:** `--timeframe BOTH` runs hourly first; a symbol whose hourly triage is `clear_sell` has its
  daily BUY downgraded to HOLD.
* **Why:** a permissive rule, so daily signals still lead to orders; the hourly signal is timing only.
* **Result:** not run live. Hourly bars were checked on real data (E06) but no live cycle has used them.
* **Why it worked or failed:** not yet known (not run live).
* **Status:** CODE-TESTED. **Decision:** 69.

## E14. Broker-side stop on a BUY (G8)

* **What changed:** with `ALPACA_STOPS_ENABLED=true`, each live BUY carries a stop (entry minus 2 x ATR,
  from the guardian) as an OTO order. Off by default.
* **Result:** not run live.
* **Why it worked or failed:** not yet known (not run live).
* **To verify on the first paper order:** the time in force of the stop leg of a DAY entry, and that an open
  stop blocks a later exit SELL until it is cancelled.
* **Status:** CODE-TESTED. **Decision:** 71.

---

## Template

```
## E<n>. <short title>

* **Date:** <day month year>. **Config hash:** <12 characters or unknown>.
* **What changed:** <indicator, weight, model or threshold; old value -> new value>
* **Why:** <the reason>
* **Result:** <numbers copied from a run>
* **Why it worked or failed:** <a cause you can point to, or "not yet known">
* **Status:** <SEEN | REPORTED | CODE-TESTED | UNVERIFIED>. **Decision:** <number in design_decisions.md>
```
