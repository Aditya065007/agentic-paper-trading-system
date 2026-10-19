# Agentic paper-trading system

A long-only paper-trading system for a course project. Code scores stocks and sizes orders; LLM agents
(CrewAI on Groq) only advise; a person approves every order. It uses an Alpaca **paper** account, a local
MCP server for data and tools, and writes a timestamped Markdown file for every agent on every run.

Team: Aditya (M1 data, M2 indicators and scoring, M8 orchestrator), Ashmita (M3 regime, M5 risk and sizing,
M6 broker), Vasundhara (M4 news, M7 LLM agents, M9 memory and reports, M10 viewer).

The architecture spec (`1_trading_system_architecture.txt`) is kept outside this repository; the docs cite its
section numbers. What the code does and why is in
`trading_system/docs/design_decisions.md`. What we tried and what happened is in
`trading_system/docs/experiments.md`. Commands for the daily routine are in `trading_system/docs/G7_G8_RUNBOOK.md`.

## Principles

1. **Code decides, the LLM advises.** Scores, thresholds, sizing and risk rules are plain Python. The Judge
   can move a composite by at most +-0.15 and thresholds are re-checked in code; any failure means no change.
2. **Missing data fails safe.** No usable score means HOLD, never a guess.
3. **A person approves.** Nothing is sent to the broker without an explicit `--approve` list and `--submit`.
4. **Paper only.** The broker module refuses any host that is not the Alpaca paper host.
5. **No secrets in files.** Reports, journals and logs redact keys.

## Setup (Windows with WSL, or Linux/macOS)

Run everything from the repository root (the folder that contains `trading_system/`).

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r trading_system/requirements.txt
cp .env.example .env          # then fill in your own keys; .env is git-ignored
```

News sentiment uses the FinBERT model, loaded from the local cache only. Download it once (it also needs
`torch`, which is large):

```bash
python -c "from transformers import AutoTokenizer, AutoModelForSequenceClassification as M; AutoTokenizer.from_pretrained('ProsusAI/finbert'); M.from_pretrained('ProsusAI/finbert')"
```

Without `transformers`, `torch` and that download, news is skipped with a warning and the other signals still
work. **Do not `source .env`**; the programs read it themselves, and a value with a space breaks the shell.

Keys needed: `ALPACA_API_KEY` and `ALPACA_SECRET_KEY` (paper), `NEWSAPI_KEY`, `GROQ_API_KEY`. See `.env.example`
for every setting. With no keys the system still runs on sample data from `trading_system/fixtures`
(`BROKER_MODE=offline`, the default) and skips what needs a key.

## Run

All commands start from the repository root with the virtual environment active.

**Look at what the system proposes (nothing is sent anywhere):**

```bash
BROKER_MODE=alpaca_paper python -m trading_system.crew.run_cycle AAPL MSFT NVDA JPM
```

Options of `run_cycle`:

| Flag | What it does |
|---|---|
| `--timeframe 1Day\|1Hour\|BOTH` | Daily is the default. `BOTH`: daily decides, hourly may only veto a daily BUY. |
| `--advisory` | Run the Bull, Bear and Judge debate on borderline stocks (uses Groq). |
| `--approve all\|SYM,SYM` | The human approval. Without it the cycle stops at proposals. With it alone, orders are DRY-RUN. |
| `--submit` | Send the approved orders to the Alpaca paper account. Needs `--approve SYM,SYM` (an explicit list). |
| `--reflect` | After the cycle, write a lesson for each newly closed trade (confirmed live fills only). |
| `--reflect-dry-run` | Also count dry-run trades (limit prices, not real fills). |

**Place paper orders** (every gate must be open, and only for the symbols you name):

```bash
ALPACA_ORDERS_ENABLED=true BROKER_MODE=alpaca_paper python -m trading_system.crew.run_cycle AAPL MSFT NVDA JPM --approve AAPL,MSFT --submit
```

Add `ALPACA_STOPS_ENABLED=true` to attach a protective stop to each BUY. Both switches belong on the command
line only, never in `.env` and never in a UI.

**The orchestrator alone** (older entry point, no approval or placement):

```bash
python -m trading_system.crew.trading_crew_multiStocks AAPL MSFT --lookback 180 --timeframe 1Day --advisory --debate-band spec --guarded
```

**Viewer for saved reports** (read-only): `streamlit run trading_system/ui/app.py`

**Tests** (offline; no network, no keys): `python -m pytest -q`

## How a run works

1. **M1 data** fetches daily or hourly bars for the whole watch-list in one batched Alpaca call (split
   adjusted, forming bar dropped, hourly kept to regular hours, disk cache).
2. **M3 regime** reads SPY trend, SPY volatility and breadth: `risk_on`, `neutral` or `risk_off`. Reported only
   unless `block_new_buys_in_risk_off` is switched on in `config/regime.yaml`.
3. **M4 news** gets headlines per symbol from NewsAPI and scores them with FinBERT; a score counts fully
   only with three or more headlines.
4. **M2 scoring** turns about 14 indicators into six group scores and a composite from -1 to +1. Weights
   depend on the sleeve (`core` or `high_beta`) and blend by the ADX regime. Triage: clear BUY, clear SELL,
   borderline, HOLD.
5. **M7 debate** (optional) has Bull and Bear argue borderline stocks and the Judge give a bounded adjustment.
   The default `consensus` mode changes a decision only when Judge, Bull and Bear all produced valid output
   and the adjustment agrees with the threshold crossing.
6. **M5 guardian** sizes each order from ATR and checks the risk limits in `config/rules.yaml`: at most 12%
   of equity in one stock, 30% in a sector, 10% cash kept, 5 new buys a day, no new buys after a 3% daily
   loss or within 2 days of earnings. Orders are whole-share limit orders.
7. **Approval and M6 broker.** You approve by name. Live orders are DAY limit orders to the paper account,
   with duplicate-order checks.
8. **M9 records** write the agent files, the decisions log and the journal (see Outputs).
9. **Reflection** (optional, `--reflect`) writes a lesson for each closed trade; the Judge reads the last
   three next time.

## Project layout

```
.env.example                    settings template (copy to .env)
trading_system/
  config/                       universe.yaml (sectors, sleeves), weights.yaml, profiles.yaml, rules.yaml,
                                regime.yaml, data.yaml, llm_verified.yaml (models), llm.yaml
  servers/
    technical_indicators_mcp_server_multiStocks.py    the local MCP server and its tools
    tools/                      data.py (M1), indicators.py + scoring.py (M2), regime.py (M3), news.py (M4),
                                risk.py (M5), broker.py + live_orders.py (M6)
  crew/
    run_cycle.py                the single entry point for one cycle (CLI and, later, the UI)
    trading_crew_multiStocks.py the deterministic orchestrator (M8); calls the MCP tools directly
    both_timeframes.py          BOTH-timeframe logic
    guarded_plan.py             sizing and guardian step, order placement
    agents.py, llm.py, advisory.py    M7: agent definitions, Groq wiring, the debate
    reflection.py               Reflection agent for closed trades
    trading_crew.py             thin CrewAI crew (Scoring and Risk agents as read-only tool callers)
    memory.py, reporting.py, agent_files.py, journal.py   M9: reports, per-agent files, journal, lessons
  ui/app.py                     Streamlit viewer of saved reports (M10, read-only)
  fixtures/                     sample account and portfolio for offline mode
  docs/                         design_decisions.md, experiments.md, G7_G8_RUNBOOK.md
  tests/                        offline tests
```

MCP tools on the local server: `fetch_ohlcv_batch`, `fetch_ohlcv`, `calculate_m2_batch`,
`calculate_indicators_batch`, `get_market_regime`, `get_earnings_calendar`, `get_news`, `get_news_by_symbol`,
`score_headlines`, `size_position`, `check_rules`, `get_account`, `get_portfolio`, `get_open_orders`,
`place_order`, `cancel_order`, `calculate_rsi`, `calculate_macd`.

## Outputs (all git-ignored)

| Where | What |
|---|---|
| `trading_system/run_reports/YYYYMMDD/agent<Name>_DDMMYYYY-HH-MM_<1D\|1H\|BOTH>.md` | One file per agent per run: Data, Scoring, News, Regime, Bull, Bear, Judge, Risk, Reflection. Each starts with run id, timestamp, watch-list, timeframe, LLM provider and model, and a config hash. A file is written only when that agent produced something. |
| `trading_system/run_reports/*_run.md` | One summary report per run. |
| `trading_system/memory/decisions.jsonl` | One JSON line per decision per run, one per order result, and one per reflection outcome. |
| `trading_system/memory/trade_journal.md` | The same story for people, one section per run. |
| `trading_system/memory/lessons.md` | Reflection lessons; the last three go to the Judge. |
| `trading_system/data_cache/` | Cached Alpaca bars and news answers. |

File names use `HH-MM` and New York time, because a colon is not allowed in Windows file names; the
course brief shows `HH:MM`. The professor has not yet been asked to accept this.

## Course requirements: where we stand

| Requirement | Status |
|---|---|
| 10-15 stocks, one batched OHLCV call | The batched call is built. `universe.yaml` maps 19 stocks plus SPY; the final watch-list is not chosen. |
| At least 30% high-beta | 6 of 19 mapped stocks (31.6%). Check again when the final list is chosen. |
| 15-20 day trading window | A run schedule, not code. No live trading window has been run yet. |
| CrewAI and a local MCP server (not Alpaca's MCP) | Local MCP server is built. CrewAI runs the debate, the Reflection agent and the thin Scoring/Risk crew; the main pipeline is deterministic Python calling MCP directly. |
| Every agent output saved to a timestamped md file | Built (`agent_files.py`); `HH-MM` deviation above. |
| Record of why each experiment worked or failed | `trading_system/docs/experiments.md`. |
| Human approval before orders | Built (`--approve`). |
| Web interface (stock list, indicators, keys, results) | Not built. The Streamlit viewer only reads saved reports. |

## Known gaps and open decisions

* **Not run live yet:** the debate in `consensus` mode, Reflection, BOTH timeframes, broker-side stops, the
  per-symbol news cache, and the hourly timeframe in any cycle. Each is covered by offline tests only.
* **Stops:** with stops off (default) nothing watches the stop price at the broker. With stops on, an open
  stop blocks a later exit SELL until you cancel it in the Alpaca dashboard.
* **Earnings blackout** needs a hand-made `trading_system/fixtures/earnings.json` (`{"AAPL": "2026-10-29"}`);
  there is no live earnings source. Without the file the guardian warns that it was not checked.
* **Position cap:** the spec says 12% per stock (`rules.yaml`); the starter code had 20%. Confirm with the team.
* **Sleeve tags** in `universe.yaml` are a judgement call, not data.
* **News** comes from the free NewsAPI plan: few headlines, about a day late, so scores drift between runs.
* **No scheduler** and no automatic placement; this is deliberate, a person approves each day.
* **Control web UI** (the last step of the plan) is not built.

## Before you submit or share

* Rotate the Groq key (and any other key that was ever pasted into a chat).
* Check the pull request's "Files changed" for `.env`, `.crewai/`, `run_reports/`, `memory/` and `data_cache/`.
  None of them belong in the repository.
* The pull request goes into `integration/m1-m4-m8-m9`, not `main`.
