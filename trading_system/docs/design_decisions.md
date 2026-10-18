# Design Decisions

Decisions about the paper-trading system's scoring, decision engine and orchestration. Architecture contract numbers (7.x) and decision IDs (D1-D11) refer to the architecture file v2.

## Scoring, decision-engine and orchestration decisions (added 12 Oct 2026)

Status key: DECIDED = implemented and tested. OPEN = needs a team decision.

### Decided

1. **Missing indicators are excluded inside their group.** A group score is the mean of its available indicators. A group is missing only when none of its indicators is available. (`scoring.py`, `_group_score`; `tests/test_scoring.py`.)

2. **30% missing-weight gate.** The gate counts whole missing groups. Exactly 30% still produces a composite. More than 30% gives composite `None` and triage `hold`. This follows architecture spec 7.2.

3. **Missing news is renormalized, never zero-filled.** `None` means unavailable and `0.0` means neutral; they give different composites. A news outage no longer blanks every decision.

4. **Decision mapping.** `clear_buy` is BUY. `clear_sell` is SELL only for a held symbol, otherwise HOLD (D2, long-only). `borderline` is HOLD until the M7 debate exists. `hold` is HOLD. A missing composite is HOLD (fail-safe). (`make_decisions`; `tests/test_decisions.py`.)

5. **News is fetched by the orchestrator, not by the scoring tool.** The orchestrator calls `get_news` then `score_headlines` and passes a `{SYMBOL: score or None}` map to `calculate_m2_batch`. The server makes no hidden network calls. A news failure is non-fatal: it becomes a warning and the run continues. (`fetch_news_scores`; `tests/test_news_wiring.py`.)

6. **Run status versus degraded.** A completed run keeps `status = "ok"` because the CLI exit code depends on it. A separate `degraded` flag is true when news, M1 or M2 data is missing, or a composite is unavailable. `score.warnings` is not used for this, because the scoring layer appends `[VERIFY]` warnings to every row.

7. **M9 artifacts.** Every run writes one run report and one journal entry, including failed runs (status `failed`, then the exception is re-raised). The report payload is compact (decisions, actions, warnings; no raw M1/M2 reports). The writers are imported lazily, so launching the file as a script still works, and a writer failure never changes the run outcome. (`tests/test_m9_orchestrator_hooks.py`.)

8. **Offline end-to-end test.** `tests/test_pipeline_offline.py` runs the real server code, scoring, decisions, plan and M9 writers with the data source, news tools and MCP transport faked, and with network access blocked. It deliberately does not assert `make_plan` sizing numbers.

### Open

9. **OPEN: borderline band.** The code treats only the 0.10 interval inside each clear threshold as borderline (0.30 to 0.40 daily). Spec 8.4 says within 0.10 of the threshold (0.30 to 0.50). A composite of 0.45 currently skips the debate. Decide before building the M7 debate step.

10. **OPEN: regime-blended weights.** `scoring.py` uses only the trending weight column. Spec 8.3 wants `t * trending + (1 - t) * ranging`, with weights in `config/weights.yaml`.

11. **OPEN: position cap.** The starter code uses 20% per position. The spec proposal is 12%. Owner: M5/M6.

12. **OPEN: news contract alignment (M4 owner).** Spec 7.1 expects a server-side `news_scores` store, 3-day decay, event flags, a requests-left counter and a daily disk cache. The current `news.py` has different tool signatures and an in-process cache only.

13. **OPEN: starter sizing (M5/M6 owner).** `make_plan` buys again for an already-held symbol, does not bound the number of BUYs against capital, does not enforce the cash reserve, and gives SELL rows no quantity because holdings carry symbols only.

## M5/M6 integration decisions (added 17 Oct 2026)

Ashmita's M5 (`risk.py`) and M6 (`broker.py`) are the base; the fixes and additions were made by the integration owner and tested against `tests/test_m5_m6_stage1_contract.py`, `test_m6_stage2_alpaca_read.py` and `test_m5_stage3_guardian_rules.py`.

### Decided

14. **Position cap is 12% of equity** (architecture section 9). This settles open item 11.
15. **Earnings blackout and cooldown count calendar days in v1.** The architecture says trading days; refining this needs a decision and a test change.
16. **Unknown sector:** a buy with no known sector is counted in a shared UNKNOWN bucket and a warning names the symbol.
17. **Missing data warns, it does not block:** a missing `day_start_equity` gives one `daily_loss unchecked` warning; missing `earnings_dates` gives one `earnings_blackout unchecked` warning. Unreadable data blocks everything (`bad_snapshot`).
18. **Long-only is enforced on the snapshot:** a short position makes every order `bad_snapshot`.
19. **M6 is dry-run only.** `BROKER_MODE` is `offline` (default) or `alpaca_paper` (read-only GETs, paper host only). `place_order(dry_run=False)` is rejected and no code path can send an order. **Amended by decision 29:** the M6 broker and its MCP tools stay dry-run only; live paper submission is a separate, gated module.

### Still open

13. Starter sizing in `make_plan` is replaced only when the orchestrator calls M5/M6 (not done yet).
20. **OPEN: earnings dates, sector map and cooldown days have no source yet** (M3, `universe.yaml` and `profiles.yaml` are empty).

## Orchestrator wiring and run_cycle (added 18 Oct 2026)

### Decided

21. **M5/M6 are opt-in in the orchestrator.** `--guarded` on the CLI (or `guarded=True` in `run_strategy`) sizes with `size_position`, checks with `check_rules`, and replaces the starter plan in the report and M9 `actions`. Without the flag the run is exactly as before. This closes open item 13 for the guarded path; the starter `make_plan` stays as the default.
22. **Decision rows carry `atr`** (from M2-B `score["atr"]`). A BUY without ATR is skipped with a reason, never sized by guess.
23. **Buys are fitted to the position cap at the limit price.** The guardian checks the limit price (last price plus 0.2%), so a buy sized to the cap at the last price would be blocked. `guarded_plan` reduces the quantity so it fits at the limit price. `risk.py` is unchanged.
24. **SELL exits the whole position** held according to the broker snapshot. A SELL for something the broker does not show is skipped.
25. **Fail closed.** If the broker snapshot cannot be read, nothing is proposed and a warning says why.
26. **Approval boundary in `run_cycle(config)`.** Default `approval=None`: proposals are returned (`status: awaiting_approval`) and nothing is placed. `approval="all"` or a list of symbols places only guardian-approved orders for those symbols, dry-run only. A repeat of the same order on the same day is rejected as a duplicate.
27. **Temporary env overrides** (`config["env"]`, allow-list: Alpaca keys, Groq key, `BROKER_MODE`) exist so a UI can pass keys for one cycle. They are restored afterwards and never returned, logged or written to a report.
28. **Sector map in `config/universe.yaml`; earnings dates from optional `fixtures/earnings.json`.** Both are starter data owned by the team. Missing earnings dates give the guardian's `earnings_blackout unchecked` warning.

### Still open

- M3 (market regime, live earnings calendar) is not built; cooldown has no source of recent exits.
- Dry-run orders live in memory and reset per process.
- Regime-blended weights (item 10) and the Reflection agent are untouched.
- Live Alpaca read (`BROKER_MODE=alpaca_paper`) is not verified here: no keys in this environment.

## Live paper orders (added 18 Oct 2026)

29. **Live submission to the Alpaca PAPER account exists, behind gates.** It lives in `servers/tools/live_orders.py`, which is not an MCP tool and is never imported by the MCP server, so no agent can call it. `broker.place_order(dry_run=False)` is still rejected. Nothing is sent unless all of these hold: `ALPACA_ORDERS_ENABLED=true` in the real environment (default off; `run_cycle`'s `env` override cannot set it); `BROKER_MODE=alpaca_paper`; the paper host and keys are present; the caller passed `submit="alpaca_paper"` and an explicit, non-empty list of approved symbols (`"all"` and `None` are refused); and the order is guardian-approved.
30. **Order shape:** whole-share DAY limit orders only. Pre-flight against the real account: enough buying power for a buy, enough shares held for a sell. At most 10 orders per submission.
31. **Duplicate protection:** `client_order_id = pt-YYYYMMDD-SYMBOL-side` (one live order per symbol, side and UTC day), looked up at Alpaca before sending, plus a refusal when any open order for that symbol and side exists. A cancelled order still counts as "already submitted today"; resubmitting the same symbol and side on the same day needs a deliberate change.
32. **Fail closed, never resend blindly.** If Alpaca cannot be asked before sending, nothing is sent. If a send fails in flight, the order is looked up by `client_order_id`: found means reported as submitted, absent means reported as not sent, and unknown means `unconfirmed` with a note to check the Alpaca dashboard.
33. **No protective stop at the broker.** `stop_price` stays advisory and is not sent. Bracket or stop orders are a later decision; a warning says so in the result.
34. **Audit trail.** Each live order writes one M9 journal entry with the Alpaca order id and status. Keys are never logged, returned or journaled (redacted from error text).
35. **Where to set the switch.** Set `ALPACA_ORDERS_ENABLED=true` on the command line for one run, not in `.env`, because `.env` is loaded into the environment and would leave live paper orders on for every run.
36. **Unverified against live Alpaca:** the "order not found" reply to the by-client-order-id lookup is treated as HTTP 404 (anything unclear fails closed). First live attempt should be one small order, checked in the Alpaca dashboard.
