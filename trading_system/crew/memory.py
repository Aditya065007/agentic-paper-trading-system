"""Offline filesystem journal for cycle, advisory, and experiment records."""

import json
import math
import os
import re
from collections.abc import Mapping
from datetime import date, datetime, timezone
from pathlib import Path


MEMORY_DIR = Path(__file__).resolve().parents[1] / "memory"
_SECRET_KEYS = ("api_key", "apikey", "secret", "password", "credential", "access_token",
				"authorization", "bearer_token")
_SECRET_ASSIGNMENT = re.compile(
	r"(?i)(\b[\w.-]*(?:api[_ -]?key|secret|password|access[_ -]?token|authorization|credential)\b\s*[:=]\s*)"
	r"([^\s,;]+)"
)
_BEARER_TOKEN = re.compile(r"(?i)\b(?:authorization\s*[:=]\s*)?Bearer\s+[A-Za-z0-9._~+/=-]+")
_SAFE_COMPONENT = re.compile(r"[^A-Za-z0-9._-]+")


def _known_secrets() -> tuple[str, ...]:
	names = ("NEWSAPI_KEY", "GROQ_API_KEY", "ALPACA_API_KEY", "ALPACA_SECRET_KEY",
			 "ALPACA_API_SECRET", "API_KEY", "ACCESS_TOKEN")
	return tuple(sorted({os.environ[name] for name in names
						 if os.environ.get(name)}, key=len, reverse=True))


def redact_text(value: str) -> str:
	"""Remove common credential assignments and known environment secret values."""
	safe = _BEARER_TOKEN.sub("Bearer [REDACTED]", value)
	safe = _SECRET_ASSIGNMENT.sub(r"\1[REDACTED]", safe)
	for secret in _known_secrets():
		safe = safe.replace(secret, "[REDACTED]")
	return safe


def sanitize(value, key=None):
	"""Convert arbitrary partial input to JSON-safe values and redact secrets."""
	if key is not None and any(fragment in str(key).casefold() for fragment in _SECRET_KEYS):
		return "[REDACTED]"
	if isinstance(value, Mapping):
		return {str(name): sanitize(item, name) for name, item in value.items()}
	if isinstance(value, (list, tuple, set)):
		return [sanitize(item) for item in value]
	if isinstance(value, str):
		return redact_text(value)
	if isinstance(value, (datetime, date)):
		return value.isoformat()
	if isinstance(value, float) and not math.isfinite(value):
		return None
	if value is None or isinstance(value, (bool, int, float)):
		return value
	return redact_text(str(value))


def safe_component(value, fallback="record", maximum=64) -> str:
	"""Make one bounded filename component without path separators or traversal."""
	text = redact_text(str(value or "")).strip()
	text = _SAFE_COMPONENT.sub("_", text).strip("._-")
	if not text or text in {".", ".."}:
		return fallback
	return text[:maximum]


def safe_inline(value) -> str:
	"""Render sanitized Markdown metadata on one line."""
	clean = sanitize(value)
	if isinstance(clean, (dict, list)):
		text = json.dumps(clean, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
	else:
		text = str(clean)
	return redact_text(text).replace("\r", " ").replace("\n", " ")


def _timestamp(value=None) -> str:
	if isinstance(value, datetime):
		current = value
	elif isinstance(value, str) and value.strip():
		return redact_text(value.strip())
	else:
		current = datetime.now(timezone.utc)
	if current.tzinfo is None:
		current = current.replace(tzinfo=timezone.utc)
	return current.isoformat()


def _filename_timestamp(value: str) -> str:
	cleaned = re.sub(r"[^0-9TZ]", "", value.upper())
	return cleaned[:32] or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _json_block(value) -> str:
	safe = sanitize(value)
	return json.dumps(safe, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)


def _render_entry(entry: Mapping, timestamp: str) -> str:
	safe = sanitize(entry)
	title = safe_component(safe.get("title") or safe.get("entry_type") or "Journal entry", maximum=100)
	lines = [f"# {title}", "", f"- Timestamp: {timestamp}"]
	for key, label in (("run_id", "Run ID"), ("cycle_id", "Cycle ID"), ("symbol", "Symbol"),
					   ("timeframe", "Timeframe"), ("agent", "Agent/role"),
					   ("status", "Status"), ("outcome", "Outcome")):
		value = safe.get(key)
		if value is not None:
			lines.append(f"- {label}: {safe_inline(value)}")
	lines.append("")
	sections = (
		("Decision / advisory", ("decision", "advisory", "output")),
		("Deterministic score / context", ("score", "score_row", "deterministic_context", "context")),
		("News / sentiment", ("news", "sentiment", "news_context")),
		("Risk / sizing context", ("risk", "risk_context", "sizing_context")),
		("Result", ("result", "final_outcome")),
		("Experiment", ("what_was_tested", "configuration", "tested")),
		("Warnings", ("warnings", "warning")),
		("Errors", ("errors", "error")),
	)
	consumed = {"title", "entry_type", "timestamp", "run_id", "cycle_id", "symbol",
				"timeframe", "agent", "status", "outcome"}
	for heading, keys in sections:
		selected = {key: safe[key] for key in keys if safe.get(key) is not None}
		if selected:
			lines.extend((f"## {heading}", "", "```json", _json_block(selected), "```", ""))
			consumed.update(selected)
	extra = {key: value for key, value in safe.items() if key not in consumed and value is not None}
	if extra:
		lines.extend(("## Additional context", "", "```json", _json_block(extra), "```", ""))
	return "\n".join(lines).rstrip() + "\n"


def _write_markdown(entry: Mapping, directory, prefix: str) -> Path:
	safe = sanitize(entry)
	timestamp = _timestamp(safe.get("timestamp"))
	run_id = safe_component(safe.get("run_id") or safe.get("cycle_id"), "run", 48)
	symbol = safe_component(safe.get("symbol"), "multi", 32)
	role = safe_component(safe.get("agent") or safe.get("entry_type"), prefix, 32)
	filename = f"{_filename_timestamp(timestamp)}_{run_id}_{symbol}_{role}.md"
	target_dir = Path(directory) if directory is not None else MEMORY_DIR
	target_dir.mkdir(parents=True, exist_ok=True)
	content = _render_entry(safe, timestamp)
	destination = target_dir / filename
	suffix = 1
	while True:
		candidate = destination if suffix == 1 else destination.with_name(
			f"{destination.stem}_{suffix}{destination.suffix}")
		try:
			with candidate.open("x", encoding="utf-8", newline="\n") as stream:
				stream.write(content)
			return candidate
		except FileExistsError:
			suffix += 1


def record_memory(entry, memory_dir=None) -> Path:
	"""Write a tolerant, sanitized journal entry as a timestamped Markdown file."""
	normalized = dict(entry) if isinstance(entry, Mapping) else {"details": entry}
	return _write_markdown(normalized, memory_dir, "journal")


def record_experiment(experiment, memory_dir=None) -> Path:
	"""Write a supplied experiment success/failure record without inventing results."""
	normalized = dict(experiment) if isinstance(experiment, Mapping) else {"result": experiment}
	normalized.setdefault("entry_type", "experiment")
	normalized.setdefault("title", "Experiment record")
	return _write_markdown(normalized, memory_dir, "experiment")
