"""Narrow M7 advisory-agent definitions with deterministic roles kept code-only."""

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass

from .llm import ADVISORY_ROLES, create_llm_client, load_llm_config


ROLE_GUIDANCE = {
	"data": "Use deterministic data tools only; do not infer or invent account or market state.",
	"scoring": "Use deterministic scoring tools only; do not calculate or alter scores.",
	"risk": "Use deterministic risk tools only; never alter limits, approve trades, or place orders.",
	"news": "Interpret supplied M4 headlines/sentiment and compact context; provide advisory context only.",
	"bull": "Give the strongest evidence-based bullish case in at most three short bullets; no executable action.",
	"bear": "Give the strongest evidence-based bearish case in at most three short bullets; no executable action.",
	"judge": "Return only a bounded numeric score adjustment and one-line reason; never decide or order.",
	"reflection": "Summarize lessons from supplied compact evidence only; never determine actions or orders.",
}
ROLE_NAMES = {
	"data": "Data", "scoring": "Scoring", "risk": "Risk", "news": "News",
	"bull": "Bull", "bear": "Bear", "judge": "Judge", "reflection": "Reflection",
}
DETERMINISTIC_ROLES = ("data", "scoring", "risk")


@dataclass(frozen=True)
class AgentDefinition:
	key: str
	role: str
	goal: str
	backstory: str
	deterministic: bool
	tools: tuple
	llm_client: object | None = None
	delegation_allowed: bool = False


def _json_scalar(value):
	if isinstance(value, str):
		return value[:160]
	if isinstance(value, float) and not math.isfinite(value):
		return None
	if isinstance(value, (int, float, bool)) or value is None:
		return value
	return None


def build_compact_context(score_row=None, regime=None, headlines=(), recent_lessons=(), config=None):
	"""Allowlist compact prompt data; raw OHLCV and arbitrary payloads are excluded."""
	loaded = load_llm_config(config)
	score_row = score_row if isinstance(score_row, Mapping) else {}
	score = {key: _json_scalar(score_row[key])
			 for key in ("symbol", "composite", "composite_score", "confidence", "data_quality")
			 if key in score_row and _json_scalar(score_row[key]) is not None}
	components = score_row.get("sub_scores", score_row.get("components"))
	if isinstance(components, Mapping):
		score["sub_scores"] = {
			str(key)[:40]: clean for key, value in list(components.items())[:8]
			if isinstance(key, str) and (clean := _json_scalar(value)) is not None
		}

	debate = loaded["debate"]
	compact_headlines = []
	for item in headlines:
		if len(compact_headlines) >= debate["max_headlines"]:
			break
		if isinstance(item, str) and item.strip():
			compact_headlines.append({"title": item.strip()[:300]})
		elif isinstance(item, Mapping) and isinstance(item.get("title"), str) \
				and item["title"].strip():
			compact_headlines.append({
				key: item[key][:300 if key == "title" else 100]
				for key in ("title", "source", "published_at")
				if isinstance(item.get(key), str)
			})
	recent_lessons = recent_lessons if isinstance(recent_lessons, (list, tuple)) else ()
	lessons = [item.strip()[:300] for item in recent_lessons[:debate["max_recent_lessons"]]
			   if isinstance(item, str) and item.strip()]
	return {
		"score_row": score,
		"regime": regime[:100] if isinstance(regime, str) else None,
		"headlines": compact_headlines,
		"recent_lessons": lessons,
	}


def format_agent_prompt(role: str, context: Mapping, config=None) -> str:
	if role not in ADVISORY_ROLES:
		raise ValueError("Deterministic Data/Scoring/Risk roles do not receive LLM prompts.")
	loaded = load_llm_config(config)
	if role in ("bull", "bear"):
		output_rule = f"Return no more than {loaded['debate']['max_bullets']} short bullets."
	elif role == "judge":
		cap = loaded["llm_influence_cap"]
		output_rule = ("Return JSON only: {\"adjustment\": number, \"reason\": \"one short line\"}. "
					   f"The adjustment must be within [-{cap}, +{cap}]. Do not return a trade action.")
	else:
		output_rule = "Return advisory context only; never return a trade action or order."
	context = context if isinstance(context, Mapping) else {}
	compact = build_compact_context(
		context.get("score_row"), context.get("regime"), context.get("headlines", ()),
		context.get("recent_lessons", ()), loaded)
	return (f"Role: {ROLE_NAMES[role]}\n{ROLE_GUIDANCE[role]}\n{output_rule}\n"
			f"Compact context JSON:\n{json.dumps(compact, ensure_ascii=True, separators=(',', ':'))}")


def create_agents(tools_by_role=None, config=None, client_factory=None,
				  enable_advisory_llm: bool = False) -> dict[str, AgentDefinition]:
	"""Create role definitions; deterministic roles never receive an LLM client."""
	loaded = load_llm_config(config)
	tools_by_role = tools_by_role if isinstance(tools_by_role, Mapping) else {}
	if client_factory is not None:
		enable_advisory_llm = True
	if enable_advisory_llm and client_factory is None:
		client_factory = create_llm_client

	definitions = {}
	for key, role in ROLE_NAMES.items():
		client = client_factory(key, config=loaded) \
			if enable_advisory_llm and key in ADVISORY_ROLES else None
		role_tools = tools_by_role.get(key, ())
		if not isinstance(role_tools, (list, tuple)):
			role_tools = (role_tools,) if role_tools else ()
		definitions[key] = AgentDefinition(
			key=key, role=role, goal=ROLE_GUIDANCE[key], backstory=ROLE_GUIDANCE[key],
			deterministic=key in DETERMINISTIC_ROLES, tools=tuple(role_tools),
			llm_client=client, delegation_allowed=False,
		)
	return definitions


def materialize_advisory_agent(definition: AgentDefinition, agent_factory=None):
	"""Build only advisory CrewAI agents using an injectable factory."""
	if definition.deterministic or definition.key not in ADVISORY_ROLES:
		raise ValueError("Only advisory roles can be materialized as LLM agents.")
	if definition.llm_client is None:
		raise ValueError("Advisory LLM client is unavailable or not enabled.")
	if agent_factory is None:
		try:
			from crewai import Agent
		except ImportError as exc:
			raise RuntimeError("CrewAI is required to materialize advisory agents.") from exc
		agent_factory = Agent
	return agent_factory(
		role=definition.role, goal=definition.goal, backstory=definition.backstory,
		tools=list(definition.tools), llm=definition.llm_client.framework_client,
		allow_delegation=False, verbose=False,
	)


def validate_debate_bullets(response, maximum: int = 3) -> list[str]:
	text = response if isinstance(response, str) else getattr(response, "raw", None)
	if not isinstance(text, str) or isinstance(maximum, bool) \
			or not isinstance(maximum, int) or not 0 <= maximum <= 3:
		raise ValueError("Debate output must be text and maximum must be between zero and three.")
	bullets = []
	for line in text.splitlines():
		line = line.strip()
		if not line:
			continue
		if line.startswith(("- ", "* ")):
			bullet = line[2:].strip()
		elif len(line) > 2 and line[0].isdigit() and line[1:3] == ". ":
			bullet = line[3:].strip()
		else:
			raise ValueError("Debate output must contain only bullet lines.")
		if not bullet or len(bullet) > 240:
			raise ValueError("Debate bullets must be short and non-empty.")
		bullets.append(bullet)
	if len(bullets) > maximum:
		raise ValueError("Debate output exceeds the configured bullet limit.")
	return bullets


def validate_judge_output(response, influence_cap: float) -> dict:
	if isinstance(response, Mapping):
		value = dict(response)
	else:
		text = response if isinstance(response, str) else getattr(response, "raw", None)
		if not isinstance(text, str):
			raise ValueError("Judge output must be JSON text or an object.")
		stripped = text.strip()
		if stripped.startswith("```") and stripped.endswith("```"):
			text = stripped[3:-3].strip()
			if text.startswith("json"):
				text = text[4:].strip()
		try:
			value = json.loads(text)
		except json.JSONDecodeError as exc:
			raise ValueError("Judge output is not valid JSON.") from exc
	if not isinstance(value, dict) or set(value) != {"adjustment", "reason"}:
		raise ValueError("Judge output must contain exactly adjustment and reason.")
	adjustment, reason = value["adjustment"], value["reason"]
	if (isinstance(adjustment, bool) or not isinstance(adjustment, (int, float))
			or not math.isfinite(adjustment) or not -influence_cap <= adjustment <= influence_cap):
		raise ValueError("Judge adjustment is invalid or exceeds the configured cap.")
	if (not isinstance(reason, str) or not reason.strip() or "\n" in reason
			or "\r" in reason or len(reason.strip()) > 160):
		raise ValueError("Judge reason must be one short line.")
	return {"adjustment": float(adjustment), "reason": reason.strip()}


def get_judge_adjustment(generate, prompt: str, config=None) -> dict:
	loaded = load_llm_config(config)
	for attempt in range(loaded["debate"]["judge_validation_retries"] + 1):
		try:
			response = generate(prompt if attempt == 0 else
								prompt + "\nRetry once: return only valid bounded JSON.")
			return validate_judge_output(response, loaded["llm_influence_cap"])
		except Exception:
			pass
	return {"adjustment": 0.0, "reason": "Invalid or unavailable Judge output; no adjustment."}
