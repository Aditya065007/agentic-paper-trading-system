"""Central Groq/CrewAI configuration with offline-testable construction."""

import math
import os
import time
from collections.abc import Callable, Mapping
from pathlib import Path


CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "llm.yaml"
ADVISORY_ROLES = ("news", "bull", "bear", "judge", "reflection")


class LLMConfigurationError(ValueError):
	"""Raised for invalid or unverified LLM configuration."""


def load_llm_config(source=None) -> dict:
	"""Read and validate the central YAML config or a supplied test mapping."""
	if source is None:
		source = CONFIG_PATH
	if isinstance(source, Mapping):
		config = dict(source)
	else:
		try:
			import yaml
		except ImportError as exc:
			raise LLMConfigurationError("PyYAML is required to read config/llm.yaml.") from exc
		try:
			with Path(source).open("r", encoding="utf-8") as stream:
				config = yaml.safe_load(stream)
		except OSError as exc:
			raise LLMConfigurationError("LLM configuration could not be read.") from exc
		except yaml.YAMLError as exc:
			raise LLMConfigurationError("LLM configuration contains invalid YAML.") from exc
	_validate_config(config)
	return config


def _validate_config(config) -> None:
	if not isinstance(config, Mapping) or config.get("provider") != "groq":
		raise LLMConfigurationError("provider must be 'groq'.")
	if config.get("api_key_env") != "GROQ_API_KEY":
		raise LLMConfigurationError("api_key_env must be GROQ_API_KEY.")
	if not isinstance(config.get("groq_integration_verified"), bool):
		raise LLMConfigurationError("groq_integration_verified must be an explicit boolean.")

	assignments = config.get("model_assignments")
	if not isinstance(assignments, Mapping) or any(
			role not in assignments for role in ADVISORY_ROLES):
		raise LLMConfigurationError("Model assignments are required for advisory roles.")
	if any(not isinstance(assignments[role], str) or not assignments[role].strip()
		   for role in ADVISORY_ROLES):
		raise LLMConfigurationError("Each advisory role requires a model ID or [VERIFY] placeholder.")

	temperature = config.get("temperature")
	if isinstance(temperature, bool) or not isinstance(temperature, (int, float)) \
			or not math.isfinite(temperature):
		raise LLMConfigurationError("temperature must be a finite number.")
	token_limit = config.get("max_output_tokens")
	if token_limit is not None and (
			isinstance(token_limit, bool) or not isinstance(token_limit, int) or token_limit < 1):
		raise LLMConfigurationError("max_output_tokens must be a positive integer or null.")

	retry = config.get("retry")
	if not isinstance(retry, Mapping):
		raise LLMConfigurationError("retry configuration is required.")
	attempts = retry.get("max_attempts")
	initial = retry.get("initial_backoff_seconds")
	maximum = retry.get("max_backoff_seconds")
	if isinstance(attempts, bool) or not isinstance(attempts, int) or attempts < 1:
		raise LLMConfigurationError("retry.max_attempts must be a positive integer.")
	if any(isinstance(value, bool) or not isinstance(value, (int, float))
		   or not math.isfinite(value) or value < 0 for value in (initial, maximum)):
		raise LLMConfigurationError("Retry backoff values must be finite and non-negative.")
	if initial > maximum:
		raise LLMConfigurationError("Initial retry backoff cannot exceed its maximum.")

	debate = config.get("debate")
	if not isinstance(debate, Mapping):
		raise LLMConfigurationError("debate configuration is required.")
	for field in ("max_bullets", "max_headlines", "max_recent_lessons"):
		value = debate.get(field)
		if isinstance(value, bool) or not isinstance(value, int) or value < 0:
			raise LLMConfigurationError(f"debate.{field} must be a non-negative integer.")
	if debate["max_bullets"] > 3 or debate["max_headlines"] > 3 \
			or debate["max_recent_lessons"] > 3:
		raise LLMConfigurationError("Debate bullets and prompt context are limited to three items.")
	if isinstance(debate.get("judge_validation_retries"), bool) \
			or debate.get("judge_validation_retries") != 1:
		raise LLMConfigurationError("Judge validation must allow exactly one retry.")

	cap = config.get("llm_influence_cap")
	if isinstance(cap, bool) or not isinstance(cap, (int, float)) \
			or not math.isfinite(cap) or not 0 <= cap <= 0.15:
		raise LLMConfigurationError("llm_influence_cap must be between 0 and 0.15.")


def _verified_model(config: Mapping, role: str) -> str:
	model = config["model_assignments"].get(role)
	if not isinstance(model, str) or not model.strip() or "[VERIFY" in model.upper():
		raise LLMConfigurationError(f"Model for '{role}' is unverified; check config/llm.yaml.")
	if config.get("max_output_tokens") is None:
		raise LLMConfigurationError("max_output_tokens is unset; verify the selected model limit.")
	return model.strip()


def _framework_llm(model, api_key, temperature, max_output_tokens, factory=None):
	if factory is None:
		try:
			from crewai import LLM
		except ImportError as exc:
			raise LLMConfigurationError("CrewAI is required to construct a Groq client.") from exc
		factory = LLM
	return factory(model=f"groq/{model}", api_key=api_key,
				   temperature=temperature, max_tokens=max_output_tokens)


class LLMClient:
	"""Advisory call wrapper with bounded retries and code-only fallback."""

	def __init__(self, role: str, model: str, config: Mapping, framework_client,
				 sleep: Callable[[float], None] = time.sleep):
		self.role = role
		self.model = model
		self.config = config
		self._framework_client = framework_client
		self._sleep = sleep

	@property
	def framework_client(self):
		return self._framework_client

	@staticmethod
	def _response_text(response):
		if isinstance(response, str):
			return response
		if isinstance(response, Mapping):
			text = response.get("text", response.get("content"))
			return text if isinstance(text, str) else None
		text = getattr(response, "content", None)
		return text if isinstance(text, str) else None

	@staticmethod
	def _usage(response):
		usage = response.get("usage") if isinstance(response, Mapping) \
			else getattr(response, "usage", None)
		if isinstance(usage, Mapping):
			return {key: value for key, value in usage.items()
					if isinstance(value, (str, int, float, bool)) or value is None}
		if usage is not None:
			fields = ("prompt_tokens", "completion_tokens", "total_tokens")
			data = {key: getattr(usage, key) for key in fields
					if isinstance(getattr(usage, key, None), (int, float))}
			return data or None
		return None

	def call_advisory(self, prompt: str, fallback: Callable[[], object] | None = None) -> dict:
		retry = self.config["retry"]
		delay = retry["initial_backoff_seconds"]
		for attempt in range(retry["max_attempts"]):
			try:
				response = self._framework_client.call(prompt)
				text = self._response_text(response)
				if text is None:
					raise ValueError("LLM returned no text.")
				return {"text": text, "model": self.model, "usage": self._usage(response),
						"warnings": [], "used_fallback": False}
			except Exception:
				if attempt + 1 < retry["max_attempts"]:
					self._sleep(min(delay, retry["max_backoff_seconds"]))
					delay = min(delay * 2, retry["max_backoff_seconds"])

		value = None
		used_fallback = fallback is not None
		if fallback is not None:
			try:
				value = fallback()
			except Exception:
				used_fallback = False
		return {"text": value, "model": self.model, "usage": None,
				"warnings": ["LLM call failed; continue with deterministic code-only behavior."],
				"used_fallback": used_fallback}


def create_llm_client(role: str, config=None, environ=None, llm_factory=None,
					  sleep: Callable[[float], None] = time.sleep) -> LLMClient:
	"""Construct a role-specific client only after key/model/provider checks pass."""
	if role not in ADVISORY_ROLES:
		raise LLMConfigurationError(f"Role '{role}' is deterministic and cannot have an LLM client.")
	loaded = load_llm_config(config)
	environment = os.environ if environ is None else environ
	api_key = environment.get(loaded["api_key_env"], "").strip()
	if not api_key:
		raise LLMConfigurationError("GROQ_API_KEY is not configured in the environment.")
	if not loaded["groq_integration_verified"]:
		raise LLMConfigurationError(
			"[VERIFY] Confirm CrewAI/Groq model, key, call, and usage behavior before enabling.")
	model = _verified_model(loaded, role)
	framework_client = _framework_llm(
		model, api_key, loaded["temperature"], loaded["max_output_tokens"], llm_factory)
	return LLMClient(role, f"groq/{model}", loaded, framework_client, sleep=sleep)
