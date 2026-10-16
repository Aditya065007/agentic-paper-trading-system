"""M8 <-> M7 bridge: run the Bull -> Bear -> Judge debate with real CrewAI crews.

Design rules (architecture decisions D3/D9):
- Code decides, the LLM only advises. The Judge can move a composite by at most
  llm_influence_cap (0.15); thresholds are then rechecked HERE, in code.
- Any failure (config, key, provider, malformed output) leaves the deterministic
  decision untouched. Nothing in this module places orders or touches risk limits.
- Only M7's own helpers are used for prompts and validation (agents.py / llm.py).

Per debated symbol ONE CrewAI Crew runs the named advisory agents (News when headlines
exist, then Bull, Bear, Judge) as sequential tasks; the Judge task receives the Bull and
Bear task outputs as context. Data, Scoring and Risk are deterministic tool-callers: they
stay code (M1/M2/M5) and appear in the printed crew roster, not as LLM agents.
"""

import functools
import math
import os
from collections.abc import Mapping
from pathlib import Path

from .agents import (
	create_agents,
	format_agent_prompt,
	get_judge_adjustment,
	materialize_advisory_agent,
	validate_debate_bullets,
)
from .llm import load_llm_config

# Verified live by the user on 2026-10-14 (CrewAI LLM call OK for both models).
ADVISORY_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "llm_verified.yaml"

# Same values as TRIAGE_THRESHOLDS in servers/tools/scoring.py (a test keeps them equal).
TRIAGE_THRESHOLDS = {
	"1Day": {"buy": 0.40, "sell": -0.40},
	"1Hour": {"buy": 0.45, "sell": -0.45},
}
BAND_BUFFER = 0.10
# "spec": within 0.10 of a clear threshold on EITHER side (architecture spec 8.4).
# "borderline": only rows the scorer already labelled borderline (0.10 inside the threshold).
BAND_MODES = ("spec", "borderline")
DEFAULT_BAND_MODE = "spec"

JUDGE_FALLBACK_PREFIX = "Invalid or unavailable Judge output"
EXPECTED_BULLETS = "At most three short bullet lines, each starting with '- '. No other text."
EXPECTED_JUDGE = 'JSON only: {"adjustment": <number>, "reason": "<one short line>"}'
EXPECTED_NEWS = "At most three short bullet lines on what the supplied headlines imply. No other text."
EXPECTED = {"news": EXPECTED_NEWS, "bull": EXPECTED_BULLETS, "bear": EXPECTED_BULLETS,
			"judge": EXPECTED_JUDGE}
# Seen on the first real run: the Bull cited earnings and acquisitions that were never supplied.
GROUNDING_RULE = ("Use ONLY the facts in the compact context JSON and in the other agents' "
				  "outputs. Do not cite earnings, products, news, prices or any other outside facts.")
ROSTER_CODE_AGENTS = ("Data", "Scoring", "Risk")


def _finite(value) -> bool:
	return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _redact(text, limit: int = 200) -> str:
	"""Short, secret-free error text for warnings and reports."""
	message = str(text)
	key = os.environ.get("GROQ_API_KEY", "").strip()
	if key:
		message = message.replace(key, "[REDACTED]")
	return " ".join(message.split())[:limit]


def is_debate_candidate(row, timeframe: str, band_mode: str = DEFAULT_BAND_MODE) -> bool:
	if band_mode not in BAND_MODES:
		raise ValueError(f"band_mode must be one of {BAND_MODES}.")
	if not isinstance(row, Mapping):
		return False
	composite = row.get("composite")
	thresholds = TRIAGE_THRESHOLDS.get(timeframe)
	if not _finite(composite) or thresholds is None:
		return False
	if band_mode == "borderline":
		return str(row.get("triage", "")).lower() == "borderline"
	return any(abs(composite - thresholds[side]) <= BAND_BUFFER + 1e-12 for side in ("buy", "sell"))


def recheck_decision(composite: float, timeframe: str, held: bool) -> str:
	"""Long-only threshold rule, applied in code after the Judge adjustment."""
	thresholds = TRIAGE_THRESHOLDS[timeframe]
	if composite >= thresholds["buy"]:
		return "BUY"
	if composite <= thresholds["sell"]:
		return "SELL" if held else "HOLD"
	return "HOLD"


# CrewAI (1.15.x) marks system messages with a "cache_breakpoint" flag and only strips it
# for its native providers. Groq goes through LiteLLM, which forwards the flag, and Groq
# rejects the request ("property 'cache_breakpoint' is unsupported"). Found on the first
# live run; removed here, for Groq-bound calls only.
UNSUPPORTED_MESSAGE_KEYS = ("cache_breakpoint",)


def strip_unsupported_message_keys(messages):
	if not isinstance(messages, list):
		return messages
	cleaned = []
	for message in messages:
		if isinstance(message, dict) and any(key in message for key in UNSUPPORTED_MESSAGE_KEYS):
			message = {key: value for key, value in message.items()
					   if key not in UNSUPPORTED_MESSAGE_KEYS}
		cleaned.append(message)
	return cleaned


def _clean_groq_kwargs(kwargs):
	if str(kwargs.get("model", "")).startswith("groq/") and "messages" in kwargs:
		kwargs["messages"] = strip_unsupported_message_keys(kwargs["messages"])
	return kwargs


def install_groq_message_compat(litellm_module=None) -> bool:
	"""Idempotently wrap litellm.completion/acompletion; returns False if LiteLLM is absent."""
	if litellm_module is None:
		try:
			import litellm as litellm_module
		except ImportError:
			return False

	original = getattr(litellm_module, "completion", None)
	if original is not None and not getattr(original, "_groq_compat", False):
		@functools.wraps(original)
		def completion(*args, **kwargs):
			return original(*args, **_clean_groq_kwargs(kwargs))
		completion._groq_compat = True
		litellm_module.completion = completion

	original_async = getattr(litellm_module, "acompletion", None)
	if original_async is not None and not getattr(original_async, "_groq_compat", False):
		@functools.wraps(original_async)
		async def acompletion(*args, **kwargs):
			return await original_async(*args, **_clean_groq_kwargs(kwargs))
		acompletion._groq_compat = True
		litellm_module.acompletion = acompletion
	return True


def default_crew_runner(agent, prompt: str, expected_output: str):
	"""One real CrewAI crew with one task for one agent."""
	from crewai import Crew, Process, Task

	task = Task(description=prompt, expected_output=expected_output, agent=agent)
	crew = Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=False)
	return crew.kickoff()


def _text(output) -> str:
	if isinstance(output, str):
		return output
	raw = getattr(output, "raw", None)
	return raw if isinstance(raw, str) else str(output)


def _model_of(definition) -> str | None:
	client = getattr(definition, "llm_client", None)
	return getattr(client, "model", None)


def _disable_tracing_by_default() -> None:
	"""CrewAI asks to upload execution traces (prompts and outputs) to its cloud. Off unless
	the person sets CREWAI_TRACING_ENABLED themselves; the variable beats any saved consent."""
	os.environ.setdefault("CREWAI_TRACING_ENABLED", "false")


def _grounded(prompt: str) -> str:
	return f"{prompt}\n{GROUNDING_RULE}"


def default_debate_runner(agents: Mapping, prompts: Mapping, expected: Mapping) -> dict:
	"""ONE real CrewAI crew: News (optional) -> Bull -> Bear -> Judge, sequential tasks.

	Bull and Bear are independent (explicit empty context, otherwise CrewAI would feed Bull's
	answer to Bear); the Judge receives News, Bull and Bear outputs as task context.
	Returns {"outputs": {role: text}, "errors": {...}} and keeps the outputs of tasks that
	finished before a failure.
	"""
	from crewai import Crew, Process, Task

	order = [role for role in ("news", "bull", "bear", "judge") if role in agents]
	tasks = {}
	for role in order:
		if role == "judge":
			context = [tasks[r] for r in ("news", "bull", "bear") if r in tasks]
		else:
			context = [tasks["news"]] if "news" in tasks and role in ("bull", "bear") else []
		tasks[role] = Task(description=prompts[role], expected_output=expected[role],
						   agent=agents[role], context=context)
	crew = Crew(agents=[agents[role] for role in order], tasks=[tasks[role] for role in order],
				process=Process.sequential, verbose=False)
	errors = {}
	try:
		crew.kickoff()
	except Exception as exc:
		errors["crew"] = _redact(exc)
	outputs = {}
	for role, task in tasks.items():
		raw = getattr(getattr(task, "output", None), "raw", None)
		if isinstance(raw, str) and raw.strip():
			outputs[role] = raw
	return {"outputs": outputs, "errors": errors}


def sequential_debate_runner(runner) -> "callable":
	"""Same contract as default_debate_runner but one agent at a time through `runner`
	(agent, prompt, expected) -> output. Used with injected stubs and for tests."""
	def run(agents, prompts, expected):
		outputs, errors = {}, {}
		for role in ("news", "bull", "bear"):
			if role in agents:
				try:
					outputs[role] = _text(runner(agents[role], prompts[role], expected[role]))
				except Exception as exc:
					errors[role] = _redact(exc)
		judge_prompt = prompts["judge"] + "".join(
			f"\n{role.capitalize()} case:\n{outputs.get(role) or '- (unavailable)'}"
			for role in ("bull", "bear"))
		try:
			outputs["judge"] = _text(runner(agents["judge"], judge_prompt, expected["judge"]))
		except Exception as exc:
			errors["judge"] = _redact(exc)
		return {"outputs": outputs, "errors": errors}
	return run


def _print_roster(printer, definitions, news_idle: bool) -> None:
	parts = [f"{name}[code]" for name in ROSTER_CODE_AGENTS]
	for role in ("news", "bull", "bear", "judge", "reflection"):
		model = _model_of(definitions[role]) or "no client"
		if role == "reflection":
			parts.append("Reflection[LLM: not run]")
		elif role == "news" and news_idle:
			parts.append("News[LLM: idle, no headlines]")
		else:
			parts.append(f"{role.capitalize()}[{model}]")
	printer("  Crew roster (spec 5.3): " + " | ".join(parts))


def run_advisory_debate(decisions, timeframe: str, band_mode: str = DEFAULT_BAND_MODE,
						config=None, client_factory=None, agent_factory=None,
						crew_runner=None, printer=print, debate_runner=None) -> list[dict]:
	"""Debate eligible rows. Mutates only those decision rows; returns one outcome per row.

	Raises only for setup problems (unverified config, missing key, no CrewAI); the
	orchestrator catches that and keeps every deterministic decision.
	crew_runner: single-agent runner (agent, prompt, expected) used for the Judge's one
	retry, and for the whole debate when debate_runner is not supplied (stubs/tests).
	"""
	loaded = load_llm_config(ADVISORY_CONFIG_PATH if config is None else config)
	candidates = [row for row in decisions if is_debate_candidate(row, timeframe, band_mode)]
	thresholds = TRIAGE_THRESHOLDS.get(timeframe, {})
	printer()
	printer("-" * 90)
	printer(f"M7 ADVISORY DEBATE (band={band_mode}, thresholds={thresholds}, "
			f"cap=+/-{loaded['llm_influence_cap']})")
	printer("-" * 90)
	if not candidates:
		printer("  No symbol is inside the debate band; deterministic decisions stand.")
		return []

	_disable_tracing_by_default()
	if client_factory is None:
		install_groq_message_compat()
	definitions = create_agents(config=loaded, client_factory=client_factory,
								enable_advisory_llm=True)
	agents = {role: materialize_advisory_agent(definitions[role], agent_factory)
			  for role in ("bull", "bear", "judge")}
	models = {role: _model_of(definitions[role]) for role in agents}
	single_runner = crew_runner or default_crew_runner
	if debate_runner is None:
		debate_runner = (sequential_debate_runner(crew_runner) if crew_runner is not None
						 else default_debate_runner)
	max_bullets = loaded["debate"]["max_bullets"]
	news_used = any(row.get("headlines") for row in candidates)
	_print_roster(printer, definitions, news_idle=not news_used)
	outcomes = []

	for row in candidates:
		symbol = row.get("symbol")
		composite = float(row["composite"])
		warnings = []
		headlines = [item for item in (row.get("headlines") or []) if item]
		context = {
			"score_row": {"symbol": symbol, "composite": composite,
						  "sub_scores": row.get("sub_scores") or {}},
			"regime": row.get("regime"),
			"headlines": headlines,
			"recent_lessons": [],
		}
		printer(f"\n  {symbol} | composite={composite:.4f} | triage={row.get('triage')} "
				f"| decision={row.get('decision')}")

		crew_roles = (("news",) if headlines else ()) + ("bull", "bear", "judge")
		if headlines and "news" not in agents:
			agents["news"] = materialize_advisory_agent(definitions["news"], agent_factory)
			models["news"] = _model_of(definitions["news"])
		prompts = {role: _grounded(format_agent_prompt(role, context, loaded))
				   for role in crew_roles}
		try:
			debate = debate_runner({role: agents[role] for role in crew_roles}, prompts, EXPECTED)
		except Exception as exc:
			debate = {"outputs": {}, "errors": {"crew": _redact(exc)}}
		outputs = debate.get("outputs") or {}
		errors = debate.get("errors") or {}

		def why(role):
			return errors.get(role) or errors.get("crew") or "no output from the crew"

		bullets = {}
		for role in ("bull", "bear"):
			try:
				if role not in outputs:
					raise ValueError(why(role))
				bullets[role] = validate_debate_bullets(outputs[role], max_bullets)
			except Exception as exc:
				bullets[role] = []
				warnings.append(f"{symbol} {role} unavailable: {_redact(exc)}")
			printer(f"    {role.capitalize():5s}: " + (" | ".join(bullets[role]) or "(unavailable)"))

		judge_prompt = prompts["judge"]
		for role in ("bull", "bear"):
			judge_prompt += f"\n{role.capitalize()} case:\n" + (
				"\n".join(f"- {item}" for item in bullets[role]) or "- (unavailable)")
		judge_errors = []
		attempts = {"count": 0}

		def generate(prompt):
			attempts["count"] += 1
			try:
				if attempts["count"] == 1:  # first answer comes from the multi-agent crew
					if "judge" not in outputs:
						raise ValueError(why("judge"))
					return outputs["judge"]
				return single_runner(agents["judge"], prompt, EXPECTED_JUDGE)  # the one retry
			except Exception as exc:
				judge_errors.append(_redact(exc))
				raise

		incomplete = not (bullets["bull"] and bullets["bear"])
		if incomplete:
			# A one-sided debate must not move a score: any failure -> adjustment 0.
			judged = {"adjustment": 0.0,
					  "reason": f"{JUDGE_FALLBACK_PREFIX}; debate incomplete, no adjustment."}
			warnings.append(f"{symbol} debate incomplete (Bull or Bear unavailable); "
							"no adjustment applied.")
		else:
			judged = get_judge_adjustment(generate, judge_prompt, loaded)
		fell_back = str(judged["reason"]).startswith(JUDGE_FALLBACK_PREFIX)
		adjustment = 0.0 if fell_back else float(judged["adjustment"])
		if fell_back and not incomplete:
			detail = judge_errors[-1] if judge_errors else "output failed validation"
			warnings.append(f"{symbol} judge unavailable or invalid ({detail}); "
							"no adjustment applied.")

		adjusted = composite + adjustment
		decision_before = row.get("decision")
		if fell_back:
			decision_after = decision_before  # failure never changes a deterministic decision
		else:
			decision_after = recheck_decision(adjusted, timeframe, bool(row.get("held")))
			row["decision"] = decision_after
			row["reason"] = (f"M7 debate: adjustment {adjustment:+.4f} ({judged['reason']}); "
							 f"adjusted composite {adjusted:.4f}; thresholds rechecked in code "
							 f"-> {decision_after}.")
		row["advisory_adjustment"] = adjustment
		row["adjusted_composite"] = adjusted

		outcome = {
			"symbol": symbol, "band_mode": band_mode, "timeframe": timeframe,
			"composite": composite, "adjustment": adjustment, "adjusted_composite": adjusted,
			"decision_before": decision_before, "decision_after": decision_after,
			"bull": bullets["bull"], "bear": bullets["bear"], "reason": judged["reason"],
			"status": "fallback" if fell_back else "ok", "models": models,
			"crew": [name.capitalize() for name in crew_roles],
			"warnings": warnings,
		}
		row["debate"] = outcome
		outcomes.append(outcome)
		for warning in warnings:
			printer(f"    ! {warning}")
		printer(f"    Judge: adjustment={adjustment:+.4f} | {judged['reason']}")
		printer(f"    Result: composite {composite:.4f} -> {adjusted:.4f} | "
				f"{decision_before} -> {decision_after}")
	return outcomes
