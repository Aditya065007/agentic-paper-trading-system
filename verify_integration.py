#!/usr/bin/env python3
"""
Offline integration audit for agentic-paper-trading-system.

Place this file in the repository root and run:
    .venv/bin/python verify_integration.py

This script does not modify source files, call external APIs, or place orders.
It runs the existing unittest suite, compiles project Python files, checks
integration wiring in the current orchestration, validates M9/M10 paths, and
checks whitespace errors in the Git diff.
"""
from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SYSTEM = ROOT / "trading_system"
TESTS = SYSTEM / "tests"
ORCHESTRATOR = SYSTEM / "crew" / "trading_crew_multiStocks.py"
MEMORY = SYSTEM / "crew" / "memory.py"
REPORTING = SYSTEM / "crew" / "reporting.py"
UI = SYSTEM / "ui" / "app.py"

results: list[tuple[str, str, str]] = []  # status, check, detail


def report(status: str, check: str, detail: str) -> None:
    results.append((status, check, detail))
    icon = {"PASS": "[PASS]", "WARN": "[WARN]", "FAIL": "[FAIL]"}[status]
    print(f"{icon} {check}: {detail}")


def run_command(args: list[str], *, cwd: Path = ROOT) -> subprocess.CompletedProcess:
    return subprocess.run(
        args, cwd=cwd, text=True, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, check=False
    )


def parse_python(path: Path) -> ast.AST | None:
    try:
        source = path.read_text(encoding="utf-8")
        return ast.parse(source, filename=str(path))
    except (OSError, UnicodeError, SyntaxError) as exc:
        report("FAIL", f"Python parse: {path.relative_to(ROOT)}", str(exc))
        return None


def call_names(node: ast.AST) -> set[str]:
    names: set[str] = set()
    for item in ast.walk(node):
        if isinstance(item, ast.Call):
            func = item.func
            if isinstance(func, ast.Name):
                names.add(func.id)
            elif isinstance(func, ast.Attribute):
                names.add(func.attr)
    return names


def function_node(tree: ast.AST, name: str) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    for item in ast.walk(tree):
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == name:
            return item
    return None


def function_source(path: Path, node: ast.AST) -> str:
    lines = path.read_text(encoding="utf-8").splitlines()
    return "\n".join(lines[node.lineno - 1 : getattr(node, "end_lineno", node.lineno)])


def main() -> int:
    print("=" * 72)
    print("PAPER-TRADING SYSTEM: OFFLINE INTEGRATION AUDIT")
    print(f"Repository: {ROOT}")
    print("Safety: no external API calls; no broker orders; no source modifications")
    print("=" * 72)

    # 1. Repository and expected module checks.
    required = [TESTS, ORCHESTRATOR, MEMORY, REPORTING, UI]
    missing = [str(p.relative_to(ROOT)) for p in required if not p.exists()]
    if missing:
        report("FAIL", "Expected project files", "Missing: " + ", ".join(missing))
        print_summary()
        return 1
    report("PASS", "Expected project files", "Orchestrator, M9 writers, M10 UI, and tests found")

    # 2. Compile every Python source file under trading_system.
    py_files = [p for p in SYSTEM.rglob("*.py") if ".venv" not in p.parts and "__pycache__" not in p.parts]
    parse_failed = False
    for path in py_files:
        try:
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (OSError, UnicodeError, SyntaxError) as exc:
            parse_failed = True
            report("FAIL", f"Python syntax: {path.relative_to(ROOT)}", str(exc))
    if not parse_failed:
        report("PASS", "Python syntax", f"{len(py_files)} Python files parsed successfully")

    # 3. Run all existing tests.
    print("\n--- Existing unit tests ---")
    test_run = run_command([sys.executable, "-m", "unittest", "discover", "-s", "trading_system/tests", "-v"])
    print(test_run.stdout.rstrip())
    if test_run.returncode == 0:
        report("PASS", "Unit test suite", "All discovered unittest tests passed")
    else:
        report("FAIL", "Unit test suite", f"Exit code {test_run.returncode}; inspect test output above")

    # 4. Inspect the real orchestration function, not just module-level existence.
    tree = parse_python(ORCHESTRATOR)
    run_node = function_node(tree, "run_strategy") if tree else None
    if run_node is None:
        report("FAIL", "M8 run_strategy", "Could not find run_strategy() in the orchestrator")
        orchestrator_text = ""
        run_calls: set[str] = set()
    else:
        orchestrator_text = function_source(ORCHESTRATOR, run_node)
        run_calls = call_names(run_node)
        report("PASS", "M8 run_strategy", "Found the orchestration entry point for static inspection")

    # M4 news path: this is a static wiring check; it does not call NewsAPI.
    news_markers = ("fetch_news", "get_news", "news_sentiment", "news_score", "fetch_news_batch")
    if any(marker in orchestrator_text.lower() for marker in news_markers):
        report("PASS", "M4 → M8 news wiring", "News-related call/reference appears inside run_strategy(); verify runtime payload in integration testing")
    else:
        report("WARN", "M4 → M8 news wiring", "No recognizable news-fetch/scoring call inside run_strategy(); M4 may exist only as a standalone tool")

    # M7 advisory path: advisory text must actually be referenced by orchestration.
    advisory_markers = ("build_advisory", "run_advisory", "advisory_agent", "llm_advisory", "judge_advisory")
    if any(marker in orchestrator_text.lower() for marker in advisory_markers):
        report("PASS", "M7 → M8 advisory wiring", "Advisory-related call/reference appears inside run_strategy(); validate fallback and output at runtime")
    else:
        report("WARN", "M7 → M8 advisory wiring", "No recognizable advisory-agent call inside run_strategy(); advisory is not demonstrably integrated")

    if "LLM decision-making: DISABLED" in orchestrator_text or "LLM decision-making" in orchestrator_text and "DISABLED" in orchestrator_text:
        report("WARN", "LLM execution mode", "Orchestrator explicitly indicates LLM decision-making is disabled")
    elif any("llm" in name.lower() or "advisory" in name.lower() for name in run_calls):
        report("PASS", "LLM execution mode", "LLM/advisory symbols are referenced by run_strategy(); this does not prove provider credentials or runtime success")
    else:
        report("WARN", "LLM execution mode", "No direct LLM/advisory calls detected in run_strategy()")

    # M9 wiring: check calls in the actual run_strategy body, not just definitions elsewhere.
    m9_targets = {
        "run report": {"record_run_report"},
        "journal": {"record_memory"},
        "experiment record": {"record_experiment", "record_experiment_report"},
    }
    for label, expected in m9_targets.items():
        hits = sorted(run_calls & expected)
        if hits:
            report("PASS", f"M8 → M9 {label}", "Direct call(s): " + ", ".join(hits))
        else:
            report("WARN", f"M8 → M9 {label}", "No direct writer call inside run_strategy(); artifacts may not be created during a normal run")

    # M9/M10 directory agreement.
    try:
        memory_text = MEMORY.read_text(encoding="utf-8")
        reporting_text = REPORTING.read_text(encoding="utf-8")
        ui_text = UI.read_text(encoding="utf-8")
        paths_match = (
            ' / "memory"' in memory_text
            and ' / "run_reports"' in reporting_text
            and 'PROJECT_DIR / "memory"' in ui_text
            and 'PROJECT_DIR / "run_reports"' in ui_text
        )
        if paths_match:
            report("PASS", "M9 ↔ M10 artifact paths", "M9 writers and M10 viewer point to trading_system/memory and trading_system/run_reports")
        else:
            report("WARN", "M9 ↔ M10 artifact paths", "Could not confirm expected directory strings; review paths in memory.py, reporting.py, and ui/app.py")
    except OSError as exc:
        report("FAIL", "M9 ↔ M10 artifact paths", str(exc))

    # Check M10 import is possible without launching Streamlit or contacting services.
    import_check = run_command([
        sys.executable, "-c",
        "import trading_system.ui.app; print('M10 module import succeeded')"
    ])
    if import_check.returncode == 0:
        report("PASS", "M10 import smoke test", import_check.stdout.strip().splitlines()[-1])
    else:
        report("FAIL", "M10 import smoke test", import_check.stdout.strip()[-1200:])

    # Safety tripwire: no order-placement call should live directly in run_strategy.
    dangerous = {"submit_order", "place_order", "submit_orders", "create_order", "market_order"}
    order_hits = sorted(run_calls & dangerous)
    if order_hits:
        report("FAIL", "No broker-order calls in run_strategy", "Potential order calls found: " + ", ".join(order_hits))
    else:
        report("PASS", "No broker-order calls in run_strategy", "No common order-placement calls found directly in run_strategy()")

    # Git whitespace check is useful but not mandatory if this is not a Git checkout.
    git_dir = ROOT / ".git"
    if git_dir.exists():
        diff_check = run_command(["git", "diff", "--check"])
        if diff_check.returncode == 0:
            report("PASS", "Git diff --check", "No whitespace errors in tracked changes")
        else:
            report("WARN", "Git diff --check", diff_check.stdout.strip() or f"Exit code {diff_check.returncode}")
    else:
        report("WARN", "Git diff --check", "No .git directory found at script location; skipped")

    print_summary()
    fails = sum(status == "FAIL" for status, _, _ in results)
    warns = sum(status == "WARN" for status, _, _ in results)
    print("\nInterpretation: PASS means the check succeeded; WARN means a likely integration gap or a static check requiring manual confirmation.")
    print("This script cannot prove live API/model behavior; it deliberately makes no network calls.")
    return 1 if fails else (2 if warns else 0)


def print_summary() -> None:
    passed = sum(status == "PASS" for status, _, _ in results)
    warned = sum(status == "WARN" for status, _, _ in results)
    failed = sum(status == "FAIL" for status, _, _ in results)
    print("\n" + "=" * 72)
    print(f"SUMMARY: {passed} PASS | {warned} WARN | {failed} FAIL")
    print("=" * 72)


if __name__ == "__main__":
    raise SystemExit(main())
