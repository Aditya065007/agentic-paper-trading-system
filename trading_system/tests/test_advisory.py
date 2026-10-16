"""Offline tests for the M8<->M7 advisory bridge (no network, no real LLM)."""

import asyncio
import copy
import io
import sys
import types
import unittest
from contextlib import redirect_stdout
from unittest import mock

from trading_system.crew import advisory
from trading_system.crew import trading_crew_multiStocks as orch
from trading_system.crew.llm import LLMConfigurationError, load_llm_config
from trading_system.servers.tools import scoring


class StubClient:
	def __init__(self, role, model):
		self.role = role
		self.model = model
		self.framework_client = object()


def stub_client_factory(role, config=None):
	return StubClient(role, "groq/" + config["model_assignments"][role])


def stub_agent_factory(**kwargs):
	return types.SimpleNamespace(**kwargs)


BULL = "- momentum is strong\n- volume confirms"
BEAR = "- valuation is stretched"
JUDGE_UP = '{"adjustment": 0.05, "reason": "bull case slightly stronger"}'
JUDGE_DOWN = '{"adjustment": -0.05, "reason": "bear case slightly stronger"}'


def make_runner(judge=JUDGE_UP, bull=BULL, bear=BEAR, calls=None):
	def runner(agent, prompt, expected):
		if calls is not None:
			calls.append((agent.role, prompt))
		reply = {"Bull": bull, "Bear": bear, "Judge": judge}[agent.role]
		if isinstance(reply, Exception):
			raise reply
		return reply
	return runner


def row(symbol, composite, triage, decision, held=False, regime="trending"):
	return {"symbol": symbol, "composite": composite, "triage": triage, "decision": decision,
			"reason": "deterministic", "held": held, "regime": regime,
			"sub_scores": {"trend": 0.3, "momentum": None}, "latest_close": 100.0}


def run_debate(decisions, timeframe="1Day", band="spec", runner=None):
	out = io.StringIO()
	with redirect_stdout(out):
		outcomes = advisory.run_advisory_debate(
			decisions, timeframe, band, client_factory=stub_client_factory,
			agent_factory=stub_agent_factory, crew_runner=runner or make_runner())
	return outcomes, out.getvalue()


class BandTests(unittest.TestCase):
	def test_spec_band_is_ten_points_on_both_sides_of_each_threshold(self):
		inside = [0.40006, 0.30, 0.50, -0.30, -0.50, 0.36]
		outside = [0.29, 0.51, 0.05, -0.29, -0.51, 0.0]
		for composite in inside:
			self.assertTrue(advisory.is_debate_candidate({"composite": composite}, "1Day"), composite)
		for composite in outside:
			self.assertFalse(advisory.is_debate_candidate({"composite": composite}, "1Day"), composite)

	def test_hourly_band_uses_the_hourly_threshold(self):
		self.assertTrue(advisory.is_debate_candidate({"composite": 0.55}, "1Hour"))
		self.assertFalse(advisory.is_debate_candidate({"composite": 0.30}, "1Hour"))

	def test_borderline_mode_only_takes_borderline_rows(self):
		self.assertTrue(advisory.is_debate_candidate(
			{"composite": 0.36, "triage": "borderline"}, "1Day", "borderline"))
		self.assertFalse(advisory.is_debate_candidate(
			{"composite": 0.40006, "triage": "clear_buy"}, "1Day", "borderline"))

	def test_invalid_rows_are_never_candidates(self):
		for bad in (None, {}, {"composite": None}, {"composite": float("nan")},
					{"composite": True}, {"composite": "0.4"}):
			self.assertFalse(advisory.is_debate_candidate(bad, "1Day"))
		self.assertFalse(advisory.is_debate_candidate({"composite": 0.4}, "15Min"))
		with self.assertRaises(ValueError):
			advisory.is_debate_candidate({"composite": 0.4}, "1Day", "everything")

	def test_thresholds_match_the_scorer(self):
		self.assertEqual(advisory.TRIAGE_THRESHOLDS, scoring.TRIAGE_THRESHOLDS)
		self.assertEqual(advisory.BAND_BUFFER, scoring.BORDERLINE_BUFFER)

	def test_recheck_is_long_only(self):
		self.assertEqual(advisory.recheck_decision(0.40, "1Day", False), "BUY")
		self.assertEqual(advisory.recheck_decision(0.39, "1Day", False), "HOLD")
		self.assertEqual(advisory.recheck_decision(-0.41, "1Day", False), "HOLD")
		self.assertEqual(advisory.recheck_decision(-0.41, "1Day", True), "SELL")


class ConfigTests(unittest.TestCase):
	def test_verified_config_loads_with_free_tier_models(self):
		config = load_llm_config(advisory.ADVISORY_CONFIG_PATH)
		self.assertTrue(config["groq_integration_verified"])
		self.assertEqual(config["model_assignments"]["judge"], "openai/gpt-oss-120b")
		for role in ("news", "bull", "bear", "reflection"):
			self.assertEqual(config["model_assignments"][role], "openai/gpt-oss-20b")
		self.assertEqual(config["max_output_tokens"], 1024)
		self.assertLessEqual(config["llm_influence_cap"], 0.15)

	def test_original_unverified_config_still_refuses(self):
		rows = [row("AAPL", 0.40006, "clear_buy", "BUY")]
		with self.assertRaises(LLMConfigurationError), redirect_stdout(io.StringIO()):
			advisory.run_advisory_debate(
				rows, "1Day", "spec", config=advisory.ADVISORY_CONFIG_PATH.with_name("llm.yaml"))
		self.assertEqual(rows[0]["decision"], "BUY")


class DebateTests(unittest.TestCase):
	def test_borderline_hold_can_become_buy_after_recheck(self):
		rows = [row("AAA", 0.36, "borderline", "HOLD")]
		outcomes, text = run_debate(rows)
		self.assertEqual(rows[0]["decision"], "BUY")
		self.assertAlmostEqual(rows[0]["adjusted_composite"], 0.41)
		self.assertEqual(rows[0]["composite"], 0.36)  # deterministic score is preserved
		self.assertEqual(outcomes[0]["decision_before"], "HOLD")
		self.assertEqual(outcomes[0]["decision_after"], "BUY")
		self.assertEqual(outcomes[0]["status"], "ok")
		self.assertIn("thresholds rechecked in code", rows[0]["reason"])
		self.assertIn("Bull", text)
		self.assertIn("Judge", text)

	def test_marginal_clear_buy_can_be_downgraded(self):
		rows = [row("AAPL", 0.40006, "clear_buy", "BUY")]
		outcomes, _ = run_debate(rows, runner=make_runner(judge=JUDGE_DOWN))
		self.assertEqual(rows[0]["decision"], "HOLD")
		self.assertEqual(outcomes[0]["decision_after"], "HOLD")

	def test_clear_buy_stays_buy_when_still_above_threshold(self):
		rows = [row("MSFT", 0.45, "clear_buy", "BUY")]
		run_debate(rows, runner=make_runner(judge=JUDGE_DOWN))
		self.assertEqual(rows[0]["decision"], "BUY")

	def test_rows_outside_the_band_are_not_touched_or_debated(self):
		calls = []
		rows = [row("NVDA", 0.05, "hold", "HOLD"), row("AAPL", 0.36, "borderline", "HOLD")]
		before = copy.deepcopy(rows[0])
		run_debate(rows, runner=make_runner(calls=calls))
		self.assertEqual(rows[0], before)
		self.assertEqual(sorted({role for role, _ in calls}), ["Bear", "Bull", "Judge"])
		self.assertEqual(len(calls), 3)

	def test_nothing_in_band_means_no_agents_are_built(self):
		factory = mock.Mock()
		with redirect_stdout(io.StringIO()):
			outcomes = advisory.run_advisory_debate(
				[row("NVDA", 0.05, "hold", "HOLD")], "1Day", "spec",
				client_factory=factory, agent_factory=stub_agent_factory)
		self.assertEqual(outcomes, [])
		factory.assert_not_called()

	def test_held_symbol_can_be_sold_after_negative_adjustment(self):
		rows = [row("XOM", -0.36, "borderline", "HOLD", held=True)]
		run_debate(rows, runner=make_runner(judge=JUDGE_DOWN))
		self.assertEqual(rows[0]["decision"], "SELL")

	def test_unheld_symbol_is_never_sold(self):
		rows = [row("XOM", -0.36, "borderline", "HOLD", held=False)]
		run_debate(rows, runner=make_runner(judge=JUDGE_DOWN))
		self.assertEqual(rows[0]["decision"], "HOLD")

	def test_adjustment_above_cap_falls_back_to_no_change(self):
		rows = [row("AAA", 0.36, "borderline", "HOLD")]
		bad = '{"adjustment": 0.9, "reason": "all in"}'
		outcomes, _ = run_debate(rows, runner=make_runner(judge=bad))
		self.assertEqual(rows[0]["decision"], "HOLD")
		self.assertEqual(rows[0]["reason"], "deterministic")
		self.assertEqual(outcomes[0]["status"], "fallback")
		self.assertEqual(outcomes[0]["adjustment"], 0.0)
		self.assertEqual(outcomes[0]["decision_after"], outcomes[0]["decision_before"])
		self.assertTrue(outcomes[0]["warnings"])

	def test_judge_that_also_returns_an_action_is_rejected(self):
		rows = [row("AAA", 0.36, "borderline", "HOLD")]
		bad = '{"adjustment": 0.05, "reason": "ok", "action": "BUY"}'
		outcomes, _ = run_debate(rows, runner=make_runner(judge=bad))
		self.assertEqual(rows[0]["decision"], "HOLD")
		self.assertEqual(outcomes[0]["status"], "fallback")

	def test_judge_gets_one_retry_then_fails_closed(self):
		calls = []
		rows = [row("AAA", 0.36, "borderline", "HOLD")]
		run_debate(rows, runner=make_runner(judge="not json", calls=calls))
		self.assertEqual([role for role, _ in calls].count("Judge"), 2)
		self.assertEqual(rows[0]["decision"], "HOLD")

	def test_provider_error_never_changes_the_decision(self):
		rows = [row("AAPL", 0.40006, "clear_buy", "BUY")]
		outcomes, _ = run_debate(rows, runner=make_runner(judge=RuntimeError("429 rate limited")))
		self.assertEqual(rows[0]["decision"], "BUY")
		self.assertEqual(rows[0]["reason"], "deterministic")
		self.assertEqual(outcomes[0]["decision_after"], "BUY")
		self.assertEqual(outcomes[0]["status"], "fallback")

	def test_a_missing_side_means_no_adjustment_and_no_decision_change(self):
		for side in ("bull", "bear"):
			rows = [row("AAA", 0.36, "borderline", "HOLD")]
			outcomes, text = run_debate(rows, runner=make_runner(**{side: RuntimeError("boom")}))
			self.assertEqual(outcomes[0][side], [])
			self.assertEqual(outcomes[0]["status"], "fallback")
			self.assertEqual(outcomes[0]["adjustment"], 0.0)
			self.assertEqual(rows[0]["decision"], "HOLD")
			self.assertEqual(rows[0]["reason"], "deterministic")
			self.assertTrue(any(f"{side} unavailable" in w for w in outcomes[0]["warnings"]))
			self.assertTrue(any("debate incomplete" in w for w in outcomes[0]["warnings"]))
			self.assertIn("debate incomplete", text)

	def test_an_incomplete_debate_cannot_push_a_borderline_row_over_a_threshold(self):
		rows = [row("AAA", 0.36, "borderline", "HOLD")]
		run_debate(rows, runner=make_runner(bear=RuntimeError("boom"), judge=JUDGE_UP))
		self.assertEqual(rows[0]["decision"], "HOLD")  # JUDGE_UP alone would have made it BUY

	def test_bullets_over_the_limit_are_rejected(self):
		rows = [row("AAA", 0.36, "borderline", "HOLD")]
		outcomes, _ = run_debate(rows, runner=make_runner(bull="- a\n- b\n- c\n- d"))
		self.assertEqual(outcomes[0]["bull"], [])

	def test_prompts_exclude_raw_data_and_carry_the_debate(self):
		calls = []
		rows = [row("AAA", 0.36, "borderline", "HOLD")]
		run_debate(rows, runner=make_runner(calls=calls))
		prompts = dict(calls)
		self.assertIn("Bull case:", prompts["Judge"])
		self.assertIn("momentum is strong", prompts["Judge"])
		self.assertIn("Bear case:", prompts["Judge"])
		self.assertIn("adjustment", prompts["Judge"])
		self.assertNotIn("latest_close", prompts["Bull"])

	def test_models_are_recorded_from_the_clients(self):
		rows = [row("AAA", 0.36, "borderline", "HOLD")]
		outcomes, _ = run_debate(rows)
		self.assertEqual(outcomes[0]["models"]["judge"], "groq/openai/gpt-oss-120b")
		self.assertEqual(outcomes[0]["models"]["bull"], "groq/openai/gpt-oss-20b")

	def test_agents_are_materialized_without_delegation(self):
		seen = []

		def factory(**kwargs):
			seen.append(kwargs)
			return types.SimpleNamespace(**kwargs)

		with redirect_stdout(io.StringIO()):
			advisory.run_advisory_debate(
				[row("AAA", 0.36, "borderline", "HOLD")], "1Day", "spec",
				client_factory=stub_client_factory, agent_factory=factory,
				crew_runner=make_runner())
		self.assertEqual(sorted(item["role"] for item in seen), ["Bear", "Bull", "Judge"])
		self.assertTrue(all(item["allow_delegation"] is False for item in seen))

	def test_error_text_is_redacted_and_bounded(self):
		with mock.patch.dict("os.environ", {"GROQ_API_KEY": "gsk_secretvalue"}):
			text = advisory._redact("auth failed for gsk_secretvalue " + "x" * 500)
		self.assertNotIn("gsk_secretvalue", text)
		self.assertLessEqual(len(text), 200)

	def test_default_runner_uses_a_real_crewai_crew(self):
		recorded = {}

		class Task:
			def __init__(self, **kwargs):
				recorded["task"] = kwargs

		class Crew:
			def __init__(self, **kwargs):
				recorded["crew"] = kwargs

			def kickoff(self):
				return types.SimpleNamespace(raw="done")

		fake = types.ModuleType("crewai")
		fake.Task, fake.Crew = Task, Crew
		fake.Process = types.SimpleNamespace(sequential="sequential")
		with mock.patch.dict(sys.modules, {"crewai": fake}):
			result = advisory.default_crew_runner("AGENT", "prompt", "expected")
		self.assertEqual(result.raw, "done")
		self.assertEqual(recorded["task"]["agent"], "AGENT")
		self.assertEqual(recorded["crew"]["tasks"][0].__class__, Task)
		self.assertEqual(recorded["crew"]["process"], "sequential")


class GroqCompatTests(unittest.TestCase):
	"""CrewAI flags messages with cache_breakpoint; Groq rejects it (found on the first live run)."""

	@staticmethod
	def fake_litellm():
		module = types.SimpleNamespace(calls=[])

		def completion(*args, **kwargs):
			module.calls.append(("sync", kwargs))
			return "sync-ok"

		async def acompletion(*args, **kwargs):
			module.calls.append(("async", kwargs))
			return "async-ok"

		module.completion, module.acompletion = completion, acompletion
		return module

	def test_only_the_flag_is_removed_and_inputs_are_not_mutated(self):
		original = [{"role": "system", "content": "s", "cache_breakpoint": True},
					{"role": "user", "content": "u"}]
		cleaned = advisory.strip_unsupported_message_keys(original)
		self.assertEqual(cleaned, [{"role": "system", "content": "s"},
								   {"role": "user", "content": "u"}])
		self.assertIn("cache_breakpoint", original[0])
		self.assertEqual(advisory.strip_unsupported_message_keys("not a list"), "not a list")

	def test_groq_calls_are_cleaned_sync_and_async_but_other_providers_are_not(self):
		module = self.fake_litellm()
		self.assertTrue(advisory.install_groq_message_compat(module))
		flagged = [{"role": "system", "content": "s", "cache_breakpoint": True}]
		self.assertEqual(module.completion(model="groq/openai/gpt-oss-20b", messages=flagged,
										   temperature=0), "sync-ok")
		self.assertEqual(asyncio.run(module.acompletion(model="groq/openai/gpt-oss-120b",
														 messages=flagged)), "async-ok")
		module.completion(model="openai/gpt-4o", messages=flagged)
		sent = [kwargs for _, kwargs in module.calls]
		self.assertNotIn("cache_breakpoint", sent[0]["messages"][0])
		self.assertEqual(sent[0]["temperature"], 0)
		self.assertNotIn("cache_breakpoint", sent[1]["messages"][0])
		self.assertIn("cache_breakpoint", sent[2]["messages"][0])

	def test_install_is_idempotent(self):
		module = self.fake_litellm()
		advisory.install_groq_message_compat(module)
		first = (module.completion, module.acompletion)
		advisory.install_groq_message_compat(module)
		self.assertEqual((module.completion, module.acompletion), first)

	def test_missing_litellm_is_not_an_error(self):
		with mock.patch.dict(sys.modules, {"litellm": None}):
			self.assertFalse(advisory.install_groq_message_compat())


class FailureVisibilityTests(unittest.TestCase):
	def test_console_shows_why_the_judge_failed(self):
		rows = [row("AAA", 0.36, "borderline", "HOLD")]
		outcomes, text = run_debate(rows, runner=make_runner(judge=RuntimeError("429 rate limited")))
		self.assertIn("! AAA judge unavailable or invalid (429 rate limited)", text)
		self.assertIn("429 rate limited", outcomes[0]["warnings"][0])

	def test_console_shows_why_an_agent_failed(self):
		rows = [row("AAA", 0.36, "borderline", "HOLD")]
		_, text = run_debate(rows, runner=make_runner(bull=RuntimeError("boom")))
		self.assertIn("! AAA bull unavailable: boom", text)

	def test_invalid_judge_output_is_labelled_as_such(self):
		rows = [row("AAA", 0.36, "borderline", "HOLD")]
		outcomes, _ = run_debate(rows, runner=make_runner(judge="not json"))
		self.assertIn("output failed validation", outcomes[0]["warnings"][0])


def fake_crewai(outputs, fail_at=None):
	"""Stand-in for the crewai module that records how the crew is built."""
	created = {"tasks": [], "crews": []}

	class Task:
		def __init__(self, **kwargs):
			self.kw = kwargs
			self.output = None
			created["tasks"].append(self)

	class Crew:
		def __init__(self, **kwargs):
			self.kw = kwargs
			created["crews"].append(self)

		def kickoff(self):
			for task in self.kw["tasks"]:
				role = task.kw["agent"].role.lower()
				if role == fail_at:
					raise RuntimeError("provider exploded key=gsk_hidden")
				task.output = types.SimpleNamespace(raw=outputs[role])

	module = types.ModuleType("crewai")
	module.Task, module.Crew = Task, Crew
	module.Process = types.SimpleNamespace(sequential="sequential")
	return module, created


def agents_for(*roles):
	return {role: types.SimpleNamespace(role=role.capitalize()) for role in roles}


def prompts_for(*roles):
	return {role: f"{role} prompt" for role in roles}


OUTPUTS = {"news": "- n", "bull": BULL, "bear": BEAR, "judge": JUDGE_UP}


class MultiAgentCrewTests(unittest.TestCase):
	def run_default(self, roles, outputs=OUTPUTS, fail_at=None):
		module, created = fake_crewai(outputs, fail_at)
		with mock.patch.dict(sys.modules, {"crewai": module}):
			result = advisory.default_debate_runner(
				agents_for(*roles), prompts_for(*roles), advisory.EXPECTED)
		return result, created

	def test_one_crew_with_three_sequential_agent_tasks(self):
		result, created = self.run_default(("bull", "bear", "judge"))
		self.assertEqual(len(created["crews"]), 1)
		crew = created["crews"][0].kw
		self.assertEqual(crew["process"], "sequential")
		self.assertEqual([a.role for a in crew["agents"]], ["Bull", "Bear", "Judge"])
		self.assertEqual([t.kw["agent"].role for t in crew["tasks"]], ["Bull", "Bear", "Judge"])
		self.assertEqual(result["outputs"]["judge"], JUDGE_UP)
		self.assertEqual(result["errors"], {})

	def test_bear_is_independent_and_the_judge_gets_both_cases(self):
		_, created = self.run_default(("bull", "bear", "judge"))
		bull, bear, judge = created["tasks"]
		self.assertEqual(bull.kw["context"], [])
		self.assertEqual(bear.kw["context"], [])  # explicit: CrewAI would otherwise chain Bull -> Bear
		self.assertEqual(judge.kw["context"], [bull, bear])

	def test_news_agent_runs_first_and_feeds_everyone_when_present(self):
		_, created = self.run_default(("news", "bull", "bear", "judge"))
		news, bull, bear, judge = created["tasks"]
		self.assertEqual(news.kw["agent"].role, "News")
		self.assertEqual(bull.kw["context"], [news])
		self.assertEqual(bear.kw["context"], [news])
		self.assertEqual(judge.kw["context"], [news, bull, bear])

	def test_outputs_of_finished_tasks_survive_a_crew_failure(self):
		result, _ = self.run_default(("bull", "bear", "judge"), fail_at="bear")
		self.assertEqual(result["outputs"], {"bull": BULL})
		self.assertIn("crew", result["errors"])

	def test_crew_error_text_is_redacted(self):
		with mock.patch.dict("os.environ", {"GROQ_API_KEY": "gsk_hidden"}):
			result, _ = self.run_default(("bull", "bear", "judge"), fail_at="bull")
		self.assertNotIn("gsk_hidden", result["errors"]["crew"])

	def test_blank_task_output_counts_as_missing(self):
		result, _ = self.run_default(("bull", "bear", "judge"),
									 outputs={"bull": "  ", "bear": BEAR, "judge": JUDGE_UP})
		self.assertNotIn("bull", result["outputs"])


class DebateStructureTests(unittest.TestCase):
	def debate(self, rows, fake_debate, **kwargs):
		out = io.StringIO()
		with redirect_stdout(out):
			outcomes = advisory.run_advisory_debate(
				rows, "1Day", "spec", client_factory=stub_client_factory,
				agent_factory=stub_agent_factory, debate_runner=fake_debate, **kwargs)
		return outcomes, out.getvalue()

	def test_the_default_path_uses_one_multi_agent_crew_per_symbol(self):
		seen = []

		def fake_default(agents, prompts, expected):
			seen.append(sorted(agents))
			return {"outputs": {"bull": BULL, "bear": BEAR, "judge": JUDGE_UP}, "errors": {}}

		rows = [row("AAA", 0.36, "borderline", "HOLD"), row("BBB", 0.44, "clear_buy", "BUY")]
		with mock.patch.object(advisory, "default_debate_runner", fake_default), \
				redirect_stdout(io.StringIO()):
			advisory.run_advisory_debate(rows, "1Day", "spec", client_factory=stub_client_factory,
										 agent_factory=stub_agent_factory)
		self.assertEqual(seen, [["bear", "bull", "judge"]] * 2)

	def test_crew_level_failure_names_the_reason_and_changes_nothing(self):
		def failing(agents, prompts, expected):
			return {"outputs": {}, "errors": {"crew": "litellm.BadRequestError: nope"}}

		rows = [row("AAA", 0.36, "borderline", "HOLD")]
		outcomes, text = self.debate(rows, failing, crew_runner=make_runner(judge=RuntimeError("x")))
		self.assertEqual(rows[0]["decision"], "HOLD")
		self.assertEqual(outcomes[0]["status"], "fallback")
		self.assertIn("bull unavailable: litellm.BadRequestError: nope", text)

	def test_invalid_crew_judge_output_gets_one_retry_through_the_single_runner(self):
		retried = []

		def crew(agents, prompts, expected):
			return {"outputs": {"bull": BULL, "bear": BEAR, "judge": "not json"}, "errors": {}}

		def single(agent, prompt, expected):
			retried.append(prompt)
			return JUDGE_UP

		rows = [row("AAA", 0.36, "borderline", "HOLD")]
		outcomes, _ = self.debate(rows, crew, crew_runner=single)
		self.assertEqual(len(retried), 1)
		self.assertIn("Retry once", retried[0])
		self.assertIn("momentum is strong", retried[0])
		self.assertEqual(outcomes[0]["status"], "ok")
		self.assertEqual(rows[0]["decision"], "BUY")

	def test_news_agent_joins_the_crew_only_when_headlines_exist(self):
		seen = []

		def crew(agents, prompts, expected):
			seen.append(sorted(agents))
			return {"outputs": {"bull": BULL, "bear": BEAR, "judge": JUDGE_UP}, "errors": {}}

		plain = row("AAA", 0.36, "borderline", "HOLD")
		with_news = dict(row("BBB", 0.36, "borderline", "HOLD"), headlines=["Apple beats estimates"])
		outcomes, text = self.debate([plain, with_news], crew)
		self.assertEqual(seen, [["bear", "bull", "judge"], ["bear", "bull", "judge", "news"]])
		self.assertEqual(outcomes[1]["crew"], ["News", "Bull", "Bear", "Judge"])
		self.assertNotIn("News[LLM: idle", text)

	def test_roster_lists_all_eight_named_agents(self):
		rows = [row("AAA", 0.36, "borderline", "HOLD")]
		_, text = run_debate(rows)
		for name in ("Data[code]", "Scoring[code]", "Risk[code]", "News[LLM: idle, no headlines]",
					 "Bull[groq/openai/gpt-oss-20b]", "Bear[groq/openai/gpt-oss-20b]",
					 "Judge[groq/openai/gpt-oss-120b]", "Reflection[LLM: not run]"):
			self.assertIn(name, text)

	def test_every_prompt_carries_the_grounding_rule(self):
		calls = []
		rows = [row("AAA", 0.36, "borderline", "HOLD")]
		run_debate(rows, runner=make_runner(calls=calls))
		self.assertEqual(len(calls), 3)
		for _, prompt in calls:
			self.assertIn(advisory.GROUNDING_RULE, prompt)

	def test_tracing_is_off_by_default_but_the_persons_choice_wins(self):
		rows = [row("AAA", 0.36, "borderline", "HOLD")]
		with mock.patch.dict("os.environ", {}, clear=False) as env:
			env.pop("CREWAI_TRACING_ENABLED", None)
			run_debate(rows)
			self.assertEqual(env["CREWAI_TRACING_ENABLED"], "false")
		with mock.patch.dict("os.environ", {"CREWAI_TRACING_ENABLED": "true"}):
			run_debate([row("AAA", 0.36, "borderline", "HOLD")])
			self.assertEqual(__import__("os").environ["CREWAI_TRACING_ENABLED"], "true")


M2_ROWS = [
	{"symbol": "AAPL", "score": {"composite": 0.40006, "triage": "clear_buy", "regime": "ranging",
								  "sub_scores": {}}, "indicators": {"context": {"last_close": 341.33}}},
	{"symbol": "NVDA", "score": {"composite": 0.05, "triage": "hold", "regime": "ranging",
								  "sub_scores": {}}, "indicators": {"context": {"last_close": 228.5}}},
]


class FakeMCP:
	def __init__(self, *args, **kwargs):
		pass

	async def __aenter__(self):
		return self

	async def __aexit__(self, *exc):
		return False

	async def list_tools(self):
		return ["fetch_ohlcv_batch", "calculate_m2_batch"]

	async def call_tool(self, name, arguments):
		if name == "fetch_ohlcv_batch":
			return {"status": "ok", "results": [], "missing": []}
		return {"status": "ok", "results": copy.deepcopy(M2_ROWS), "missing": []}


async def fake_news(mcp, symbols, available_tools):
	return {symbol: None for symbol in symbols}, []


class OrchestratorWiringTests(unittest.TestCase):
	def run_strategy(self, **kwargs):
		payloads = []

		def capture(payload, *args, **kw):
			payloads.append(payload)
			return ("report.md", "journal.md")

		patches = [
			mock.patch.object(orch, "DirectMCPClient", FakeMCP),
			mock.patch.object(orch, "get_holdings", lambda: []),
			mock.patch.object(orch, "fetch_news_scores", fake_news),
			mock.patch.object(orch, "apply_m1_price_fallback", lambda **kw: None),
			mock.patch.object(orch, "print_report", lambda **kw: None),
			mock.patch.object(orch, "_write_m9_artifacts", capture),
		]
		out = io.StringIO()
		for patch in patches:
			patch.start()
		try:
			with redirect_stdout(out):
				result = asyncio.run(orch.run_strategy(["AAPL", "NVDA"], 180, "1Day", **kwargs))
		finally:
			for patch in patches:
				patch.stop()
		return result, payloads, out.getvalue()

	def test_default_run_never_touches_advisory(self):
		with mock.patch.object(advisory, "run_advisory_debate",
							   side_effect=AssertionError("must not run")):
			result, payloads, text = self.run_strategy()
		self.assertNotIn("advisory", result)
		self.assertNotIn("advisory", payloads[0])
		self.assertNotIn("LLM advisory", text)
		self.assertEqual([d["decision"] for d in result["decisions"]], ["BUY", "HOLD"])

	def test_advisory_run_applies_debate_reports_and_journals(self):
		def fake_debate(decisions, timeframe, band_mode):
			decisions[0]["decision"] = "HOLD"
			return [{"symbol": "AAPL", "band_mode": band_mode, "composite": 0.40006,
					 "adjustment": -0.05, "adjusted_composite": 0.35006,
					 "decision_before": "BUY", "decision_after": "HOLD", "bull": ["b"],
					 "bear": ["c"], "reason": "r", "status": "ok",
					 "models": {"bull": "m1", "bear": "m1", "judge": "m2"}, "warnings": []}]

		memory = mock.Mock(return_value="entry.md")
		with mock.patch.object(advisory, "run_advisory_debate", fake_debate), \
				mock.patch("trading_system.crew.memory.record_memory", memory):
			result, payloads, text = self.run_strategy(advisory_band="spec")
		self.assertIn("LLM advisory: ENABLED", text)
		self.assertEqual(result["decisions"][0]["decision"], "HOLD")
		self.assertEqual(result["plan"][0]["decision"], "HOLD")  # sizing follows the final decision
		self.assertEqual(payloads[0]["advisory"][0]["decision_after"], "HOLD")
		self.assertFalse(result["degraded"])
		agents = [call.args[0]["agent"] for call in memory.call_args_list]
		self.assertEqual(agents, ["M7 Bull AAPL", "M7 Bear AAPL", "M7 Judge AAPL"])
		self.assertEqual(len(result["advisory_journal_paths"]), 3)
		self.assertEqual(memory.call_args_list[0].args[0]["symbols"], ["AAPL"])

	def test_advisory_failure_keeps_every_decision_and_marks_degraded(self):
		with mock.patch.object(advisory, "run_advisory_debate",
							   side_effect=LLMConfigurationError("GROQ_API_KEY is not configured")):
			result, payloads, text = self.run_strategy(advisory_band="spec")
		self.assertEqual(result["status"], "ok")
		self.assertEqual([d["decision"] for d in result["decisions"]], ["BUY", "HOLD"])
		self.assertTrue(result["degraded"])
		self.assertTrue(any("M7 advisory unavailable" in w for w in result["advisory_warnings"]))
		self.assertIn("deterministic decisions unchanged", text)
		self.assertTrue(any("M7 advisory unavailable" in w for w in payloads[0]["warnings"]))

	def test_failure_text_never_contains_the_key(self):
		error = RuntimeError("bad request key=gsk_topsecret")
		with mock.patch.dict("os.environ", {"GROQ_API_KEY": "gsk_topsecret"}), \
				mock.patch.object(advisory, "run_advisory_debate", side_effect=error):
			result, payloads, text = self.run_strategy(advisory_band="spec")
		self.assertNotIn("gsk_topsecret", text)
		self.assertNotIn("gsk_topsecret", " ".join(result["advisory_warnings"]))

	def test_nothing_in_band_is_not_degraded(self):
		with mock.patch.object(advisory, "run_advisory_debate", return_value=[]):
			result, _, _ = self.run_strategy(advisory_band="spec")
		self.assertEqual(result["advisory"], [])
		self.assertFalse(result["degraded"])


class CliTests(unittest.TestCase):
	def parse(self, *argv):
		with mock.patch.object(sys, "argv", ["prog", *argv]):
			return orch.parse_args()

	def test_defaults_are_unchanged(self):
		args = self.parse("AAPL", "MSFT")
		self.assertFalse(args.advisory)
		self.assertEqual(args.debate_band, "spec")

	def test_flags_parse(self):
		args = self.parse("AAPL", "--advisory", "--debate-band", "borderline")
		self.assertTrue(args.advisory)
		self.assertEqual(args.debate_band, "borderline")

	def test_async_main_passes_advisory_only_when_requested(self):
		for argv, expected in ((("AAPL",), {}), (("AAPL", "--advisory"), {"advisory_band": "spec"})):
			run = mock.AsyncMock(return_value={"status": "ok"})
			with mock.patch.object(sys, "argv", ["prog", *argv]), \
					mock.patch.object(orch, "run_strategy", run):
				code = asyncio.run(orch.async_main())
			self.assertEqual(code, 0)
			self.assertEqual(run.call_args.kwargs.get("advisory_band"), expected.get("advisory_band"))
			self.assertEqual("advisory_band" in run.call_args.kwargs, bool(expected))


if __name__ == "__main__":
	unittest.main()
