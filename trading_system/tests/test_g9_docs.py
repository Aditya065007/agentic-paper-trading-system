"""G9: the documents match the code. Offline; reads files only.

Fails when someone adds an environment variable, a command-line flag, a config file or a module without
documenting it, or when a safety gate or a key-like value ends up in .env.example."""

import re
from pathlib import Path

import pytest
import yaml

PACKAGE = Path(__file__).resolve().parents[1]
ROOT = PACKAGE.parent
README = (ROOT / "README.md").read_text(encoding="utf-8")
ENV_EXAMPLE = (ROOT / ".env.example").read_text(encoding="utf-8")
EXPERIMENTS = (PACKAGE / "docs" / "experiments.md").read_text(encoding="utf-8")
DECISIONS = (PACKAGE / "docs" / "design_decisions.md").read_text(encoding="utf-8")

GATES = ("ALPACA_ORDERS_ENABLED", "ALPACA_STOPS_ENABLED")
# Read by code but not a user setting: CrewAI tracing is switched off by default inside advisory.py.
INTERNAL_ENV = {"CREWAI_TRACING_ENABLED"}


def code_files():
    return [p for p in PACKAGE.rglob("*.py") if "tests" not in p.parts and "__pycache__" not in p.parts]


def env_names_read_by_code():
    pattern = re.compile(r"""(?:environ\.get|environ\[|getenv|environ\.setdefault)\(?\s*["']([A-Z][A-Z0-9_]+)["']""")
    found = set()
    for path in code_files():
        found.update(pattern.findall(path.read_text(encoding="utf-8")))
    return found - INTERNAL_ENV


def active_lines():
    return [line for line in ENV_EXAMPLE.splitlines() if line.strip() and not line.lstrip().startswith("#")]


def active_assignments():
    return dict(line.split("=", 1) for line in active_lines() if "=" in line)


# ------------------------------------------------------------------ .env.example
def test_the_scan_really_finds_the_known_variables():
    assert {"ALPACA_API_KEY", "GROQ_API_KEY", "BROKER_MODE", "ALPACA_ORDERS_ENABLED"} <= env_names_read_by_code()


def test_every_variable_the_code_reads_is_in_env_example():
    missing = [name for name in sorted(env_names_read_by_code()) if name not in ENV_EXAMPLE]
    assert not missing, f"undocumented in .env.example: {missing}"


def test_env_example_has_no_variable_that_no_code_reads():
    unknown = [name for name in active_assignments() if name not in env_names_read_by_code()]
    assert not unknown, f"not read by any code: {unknown}"


def test_the_order_gates_are_comments_only():
    names = set(active_assignments())
    for gate in GATES:
        assert gate not in names, f"{gate} must not be an active setting in .env.example"
        assert gate in ENV_EXAMPLE  # but they are explained
    assert "DO NOT PUT THESE IN THIS FILE" in ENV_EXAMPLE


def test_env_example_holds_no_secret_values_and_no_live_mode():
    values = active_assignments()
    for key in ("ALPACA_API_KEY", "ALPACA_SECRET_KEY", "NEWSAPI_KEY", "GROQ_API_KEY", "ALPACA_TRADING_URL"):
        assert values.get(key, "").strip() == "", f"{key} must be empty in the example"
    assert values.get("BROKER_MODE", "offline").strip() == "offline"
    assert not re.search(r"(gsk_|sk-|PK[A-Z0-9]{16})", ENV_EXAMPLE)
    assert all(len(value.strip()) < 30 for value in values.values())


def test_env_is_git_ignored_and_the_example_says_not_to_source_it():
    lines = [line.strip() for line in (ROOT / ".gitignore").read_text().splitlines()]
    assert ".env" in lines
    assert "never \"source\"" in ENV_EXAMPLE


# ------------------------------------------------------------------ README
def cli_flags(path):
    return set(re.findall(r"""add_argument\(\s*["'](--[a-z][a-z-]*)["']""", path.read_text(encoding="utf-8")))


def test_every_run_cycle_flag_is_in_the_readme():
    flags = cli_flags(PACKAGE / "crew" / "run_cycle.py")
    assert {"--approve", "--submit", "--advisory", "--timeframe", "--reflect"} <= flags
    assert not [f for f in sorted(flags) if f not in README], "undocumented run_cycle flags"


def test_every_orchestrator_flag_is_in_the_readme():
    flags = set(re.findall(r"""["'](--[a-z][a-z-]*)["']""",
                           (PACKAGE / "crew" / "trading_crew_multiStocks.py").read_text(encoding="utf-8")))
    flags = {f for f in flags if f in {"--lookback", "--timeframe", "--advisory", "--debate-band", "--guarded"}}
    assert len(flags) == 5
    assert not [f for f in sorted(flags) if f not in README]


def test_every_config_file_and_module_is_in_the_readme():
    names = [p.name for p in (PACKAGE / "config").glob("*.yaml")]
    for folder in ("crew", "servers/tools"):
        names += [p.name for p in (PACKAGE / folder).glob("*.py") if p.name != "__init__.py"]
    names.append("technical_indicators_mcp_server_multiStocks.py")
    missing = [n for n in sorted(set(names)) if n not in README and n.removesuffix(".yaml").removesuffix(".py") not in README]
    assert not missing, f"not mentioned in the README: {missing}"


def test_every_server_tool_is_in_the_readme():
    server = (PACKAGE / "servers" / "technical_indicators_mcp_server_multiStocks.py").read_text(encoding="utf-8")
    tools = set(re.findall(r"@mcp\.tool\(\)\s*\n\s*(?:async )?def (\w+)", server))
    assert "get_market_regime" in tools
    # tools registered from other modules are listed by name there; the README must list the ones defined here
    assert not [t for t in sorted(tools) if t not in README]


def test_the_readme_shows_the_gates_on_the_command_line_and_the_safety_checklist():
    for text in ("ALPACA_ORDERS_ENABLED=true BROKER_MODE=alpaca_paper python -m trading_system.crew.run_cycle",
                 "Rotate the Groq key", "integration/m1-m4-m8-m9", "Files changed", "Do not `source .env`",
                 "HH-MM", "python -m pytest -q"):
        assert text in README, text


def test_readme_commands_name_real_modules():
    for module in re.findall(r"python -m (trading_system[\w.]+)", README):
        assert (ROOT / (module.replace(".", "/") + ".py")).is_file(), module
    assert (ROOT / "trading_system" / "ui" / "app.py").is_file() and "streamlit run trading_system/ui/app.py" in README


def test_the_readme_high_beta_figure_matches_the_universe():
    universe = yaml.safe_load((PACKAGE / "config" / "universe.yaml").read_text())
    stocks = [s for s in universe["sectors"] if s != "SPY"]
    high = [s for s, v in universe["sleeves"].items() if v == "high_beta" and s in stocks]
    assert f"{len(high)} of {len(stocks)} mapped stocks" in README
    assert len(high) / len(stocks) >= 0.30


def test_the_package_readme_points_to_the_real_one():
    assert "../README.md" in (PACKAGE / "README.md").read_text()


# ------------------------------------------------------------------ experiments log and decisions
def entries():
    parts = re.split(r"^## (E\d+)\. ", EXPERIMENTS, flags=re.M)
    return {parts[i]: parts[i + 1] for i in range(1, len(parts) - 1, 2)}


def test_the_experiments_log_has_entries_numbered_without_gaps():
    found = entries()
    assert len(found) >= 14
    assert list(found) == [f"E{i:02d}" for i in range(1, len(found) + 1)] or \
        list(found) == [f"E{i}" for i in range(1, len(found) + 1)]


@pytest.mark.parametrize("label", ["**Status:**"])
def test_every_entry_has_a_valid_status(label):
    for name, body in entries().items():
        assert label in body, name
        status = re.search(r"\*\*Status:\*\* ([A-Z -]+?)[ (.;]", body)
        assert status and status.group(1) in {"SEEN", "REPORTED", "CODE-TESTED", "UNVERIFIED"}, (name, status)


def test_every_entry_has_what_why_and_result_lines():
    for name, body in entries().items():
        # E04, E05, E08 and E11 record an observation or a baseline run, not a change of ours.
        needs_change = name not in ("E04", "E05", "E08", "E11")
        assert "**What changed:**" in body or not needs_change, name
        assert "**Result" in body or "**What happened:**" in body, name
        assert "**Why" in body or (name == "E11" and "**Use:**" in body), name


def test_every_decision_number_cited_in_the_experiments_exists():
    present = {int(n) for n in re.findall(r"^(\d+)\. \*\*", DECISIONS, flags=re.M)}
    cited = set()
    for body in entries().values():
        for line in re.findall(r"\*\*Decisions?:\*\* ([0-9, \-]+)", body):
            for part in re.split(r",\s*", line.strip(" ,")):
                if "-" in part:
                    low, high = (int(x) for x in part.split("-"))
                    cited.update(range(low, high + 1))
                elif part.strip():
                    cited.add(int(part))
    assert cited and cited <= present, sorted(cited - present)


def test_the_experiments_log_records_the_known_baseline_and_the_hash():
    for fact in ("762e0f7592a9", "-0.9318", "-0.3106", "834,589", "257,976", "run-20261018T201711-6fc39c0eb1"):
        assert fact in EXPERIMENTS, fact


def test_the_experiments_log_claims_no_live_result_for_unrun_features():
    for name in ("E12", "E13", "E14"):
        body = entries()[name]
        assert "not run live" in body and "CODE-TESTED" in body


def test_design_decisions_are_numbered_once_each_and_in_order():
    numbers = [int(n) for n in re.findall(r"^(\d+)\. \*\*", DECISIONS, flags=re.M)]
    tail = numbers[numbers.index(1):] if 1 in numbers else numbers
    assert len(set(tail)) == len(tail)
    assert tail[-4:] == [75, 76, 77, 78]
