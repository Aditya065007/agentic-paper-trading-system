# WHERE IS WHAT: export of a paper-trading run

- Exported: 2026-11-11T12:39:14 (latest dashboard run)
- Run id: run-20261102T160827-fe1ba66c76   Timeframe: 1Hour   Symbols: ['NVDA', 'COIN', 'CCL', 'MSFT', 'BAC', 'COP', 'SLB', 'PFE', 'KO', 'UNP']
- Settings hash (same as in every agent file): 07dd1b2a7c2f
- Broker mode: alpaca_paper
- Dashboard status: completed   Live mode: True

## If the professor asks...

| Question | Open this file | Which module and code |
|---|---|---|
| Where are the agent .md files? | run_reports/YYYYMMDD/agent<Name>_DDMMYYYY-HH-MM_1D.md | run_reports/ in this export. Written by crew/agent_files.py : write_agent_files |
| Where is the scoring? | 01_decisions_and_scores.csv (+ run_reports/.../agentScoring*.md) | M2: servers/tools/scoring.py : score_indicators. Sub-scores = trend, momentum, volume, mean reversion, breakout, news |
| Where are the Bull and Bear cases? | 03_debate_cases.md (+ run_reports/.../agentBull*.md, agentBear*.md) | M7: crew/advisory.py : run_advisory_debate. Only borderline stocks are debated |
| Where is the Judge adjustment? | 03_debate_cases.md, 01_decisions_and_scores.csv, memory/decisions.jsonl | M7: crew/agents.py : validate_judge_output (cap +/-0.15) |
| Where is the market regime? | 02_market_regime.json (+ run_reports/.../agentRegime*.md) | M3: servers/tools/regime.py : compute_market_regime |
| Where are the news scores? | 07_news_scores.json (fresh runs), run_reports/.../agentNews*.md, raw_data_cache/news* | M4: servers/tools/news.py : get_news_by_symbol, score_headlines (FinBERT) |
| Where are the raw prices? | raw_data_cache/ (raw Alpaca rows) | M1: servers/technical_indicators_mcp_server_multiStocks.py : fetch_ohlcv_batch; cleaned by servers/tools/data.py |
| Where are the indicator values (RSI, MACD ...)? | 08_indicators_full.csv (ONLY from --fresh) | M2: servers/tools/indicators.py : compute_indicators. Not stored by any normal run file |
| Where are the risk checks and order proposals? | 04_guardian_orders.csv (+ run_reports/.../agentRisk*.md) | M5: servers/tools/risk.py : check_rules; crew/guarded_plan.py : build_guarded_plan |
| Where is the placed order? | 05_orders_placed.json, memory/trade_journal.md, memory/decisions.jsonl | M6: servers/tools/live_orders.py (live) or crew/guarded_plan.py : place_approved (dry run) |
| Where is the journal / history? | memory/trade_journal.md, memory/decisions.jsonl, memory/lessons.md | M9: crew/journal.py |
| Which settings were used? | settings_used/ and config_hash.txt | config/*.yaml; the same hash appears in every agent file header |
| What did the dashboard store? | full_run_record.json (dashboard runs only) | web/store.py -> web/.local_state/runs.json |

## Readable files made from this run

- 01_decisions_and_scores.csv
- 02_market_regime.json
- 03_debate_cases.md
- 04_guardian_orders.csv
- 05_orders_placed.json
- 06_warnings.txt
- full_run_record.json

## Folders copied

- run_reports/ (agent files and run report: Data, Scoring, News, Regime, Bull, Bear, Judge, Risk): 6 files
- memory/ (decisions.jsonl, trade_journal.md, lessons.md): 641 files
- settings_used/ + fixtures_used/ (the exact settings and sample account): 11 files
- project_docs/ (design decisions and experiments log): 3 files
- raw_data_cache/ (raw Alpaca price rows and NewsAPI headlines): 242 files

## Honest notes

- Individual indicator values (RSI, MACD, ...) are not saved by a normal run. Only `--fresh` captures them.
- The raw price cache holds the unprocessed Alpaca rows; cleaning happens in servers/tools/data.py.
- API keys are never copied: .env is skipped and key values are replaced by [REDACTED].
- If advisory (the AI debate) was off, there are no Bull/Bear files and 03_debate_cases.md says so.
