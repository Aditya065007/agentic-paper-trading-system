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
