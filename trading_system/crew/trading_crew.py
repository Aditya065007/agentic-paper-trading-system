"""G8: the thin CrewAI crew (course requirement: CrewAI agents calling tools).

Design: the working pipeline stays deterministic Python (trading_crew_multiStocks.py). This module wraps the
SAME deterministic functions as CrewAI tools so the Scoring and Risk agents are real tool callers, with one
task per stock. The crew only READS and REPORTS:

  * scoring_lookup(symbol)  returns the already computed decision row (composite, triage, regime, reason).
  * risk_check(symbol)      runs the M5 guardian (guarded_plan.build_guarded_plan) on that one row against the
                            broker snapshot it was given, and returns approved / blocked / skipped.

Nothing here scores, sizes, approves or places anything on its own, and run_thin_crew() asserts that the decision
rows are unchanged afterwards. Its output is a text report; the guarded plan from the orchestrator remains the
only source of orders. Running it needs an LLM client (the Groq client from crew/llm.py), so it costs tokens;
building it and the tools is offline and tested with stand-ins. [UNVERIFIED live: a full crew run on Groq.]"""

import copy
import json
from collections.abc import Mapping

from . import guarded_plan as gp


def _row_map(decisions) -> dict:
    return {str(r["symbol"]).upper(): r for r in decisions or [] if isinstance(r, Mapping) and r.get("symbol")}


def make_tool_functions(decisions, snapshot, rules=None) -> dict:
    """{"scoring": fn, "risk": fn}: plain functions (symbol -> JSON text) over deterministic results."""
    rows = _row_map(decisions)

    def scoring_lookup(symbol: str) -> str:
        """Return the deterministic score and decision already computed for one stock."""
        row = rows.get(str(symbol).strip().upper())
        if row is None:
            return json.dumps({"error": f"no decision row for {symbol}"})
        keep = ("symbol", "decision", "triage", "composite", "regime", "held", "latest_close", "reason")
        return json.dumps({k: row.get(k) for k in keep}, default=str)

    def risk_check(symbol: str) -> str:
        """Run the deterministic risk guardian on one stock and return approved, blocked and skipped orders."""
        row = rows.get(str(symbol).strip().upper())
        if row is None:
            return json.dumps({"error": f"no decision row for {symbol}"})
        plan = gp.build_guarded_plan([copy.deepcopy(row)], snapshot or {}, rules)
        return json.dumps({k: plan.get(k) for k in ("approved", "blocked", "skipped")}, default=str)

    return {"scoring": scoring_lookup, "risk": risk_check}


def build_stock_tasks(symbols) -> list:
    """One task spec per stock, in order. Pure."""
    return [{"symbol": str(s).upper(),
             "scoring": f"Call scoring_lookup for {str(s).upper()} and state its decision, triage and composite. "
                        "Report only what the tool returns.",
             "risk": f"Call risk_check for {str(s).upper()} and state what was approved or blocked and why. "
                     "Report only what the tool returns."} for s in symbols or []]


def build_crew(decisions, snapshot, llm, rules=None, agent_factory=None, task_factory=None,
               crew_factory=None, tool_factory=None):
    """Build the Crew (Scoring agent, Risk agent, two tasks per stock). The factories exist for tests."""
    if llm is None:
        raise ValueError("the thin crew needs an LLM client (agents cannot call tools without one).")
    if agent_factory is None or task_factory is None or crew_factory is None or tool_factory is None:
        from crewai import Agent, Crew, Process, Task
        from crewai.tools import tool
        agent_factory = agent_factory or Agent
        task_factory = task_factory or Task
        tool_factory = tool_factory or tool
        crew_factory = crew_factory or (lambda **kw: Crew(process=Process.sequential, verbose=False, **kw))
    funcs = make_tool_functions(decisions, snapshot, rules)
    scoring_tool = tool_factory("scoring_lookup")(funcs["scoring"])
    risk_tool = tool_factory("risk_check")(funcs["risk"])
    scoring = agent_factory(role="Scoring", goal="Report the deterministic score and decision for a stock.",
                            backstory="You only read results from tools; you never calculate or change a score.",
                            tools=[scoring_tool], llm=llm, allow_delegation=False, verbose=False)
    risk = agent_factory(role="Risk", goal="Report the deterministic risk-guardian verdict for a stock.",
                         backstory="You only read results from tools; you never size, approve or place orders.",
                         tools=[risk_tool], llm=llm, allow_delegation=False, verbose=False)
    tasks = []
    for spec in build_stock_tasks([r for r in _row_map(decisions)]):
        tasks.append(task_factory(description=spec["scoring"], expected_output="One short line.", agent=scoring))
        tasks.append(task_factory(description=spec["risk"], expected_output="One short line.", agent=risk))
    return crew_factory(agents=[scoring, risk], tasks=tasks)


def run_thin_crew(decisions, snapshot, llm, rules=None, **factories) -> dict:
    """Kick off the crew and return {"status", "report", "tasks"}. Decisions are asserted unchanged."""
    before = copy.deepcopy(decisions)
    crew = build_crew(decisions, snapshot, llm, rules, **factories)
    output = crew.kickoff()
    if decisions != before:
        raise RuntimeError("the thin crew must never change a decision row.")
    return {"status": "ok", "report": getattr(output, "raw", None) or str(output),
            "tasks": len(getattr(crew, "tasks", []) or [])}
