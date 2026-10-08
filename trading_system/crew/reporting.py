"""Offline Markdown reports for supplied cycle and experiment results."""

import json
from collections.abc import Mapping
from pathlib import Path

from .memory import _filename_timestamp, _timestamp, safe_component, safe_inline, sanitize


REPORTS_DIR = Path(__file__).resolve().parents[1] / "run_reports"


def _json_block(value) -> str:
	return json.dumps(sanitize(value), ensure_ascii=False, sort_keys=True,
					  indent=2, allow_nan=False)


def render_run_report(report) -> str:
	"""Render a sanitized, human-readable Markdown report from partial input."""
	safe = sanitize(report if isinstance(report, Mapping) else {"result": report})
	timestamp = _timestamp(safe.get("timestamp") or safe.get("as_of"))
	symbols = safe.get("symbols")
	lines = ["# Paper-Trading Run Report", "", f"- Timestamp: {timestamp}"]
	for key, label in (("run_id", "Run ID"), ("timeframe", "Timeframe"),
					   ("profile", "Profile"), ("sleeve", "Sleeve"), ("status", "Status"),
					   ("outcome", "Outcome")):
		value = safe.get(key)
		if value is not None:
			lines.append(f"- {label}: {safe_inline(value)}")
	if symbols is not None:
		symbol_text = ", ".join(safe_inline(item) for item in symbols) \
			if isinstance(symbols, (list, tuple)) else safe_inline(symbols)
		lines.append(f"- Symbols: {symbol_text}")

	sections = (
		("Deterministic signal / decision context", ("score_results", "triage_results", "decisions",
													   "deterministic_context", "signals")),
		("LLM advisory context", ("advisory_results", "advisories", "llm_context")),
		("Risk / Guardian context", ("risk_results", "risk_context", "guardian")),
		("Action / result", ("actions", "action", "results", "result", "final_outcome")),
		("Warnings", ("warnings", "warning")),
		("Errors / degraded state", ("errors", "error", "failed_stages", "blocked_stages",
									  "degraded")),
	)
	consumed = {"timestamp", "as_of", "run_id", "timeframe", "profile", "sleeve", "status",
				"outcome", "symbols"}
	for heading, fields in sections:
		selected = {field: safe[field] for field in fields if safe.get(field) is not None}
		if selected:
			lines.extend(("", f"## {heading}", "", "```json", _json_block(selected), "```"))
			consumed.update(selected)

	remaining = {key: value for key, value in safe.items()
				 if key not in consumed and value is not None}
	if remaining:
		lines.extend(("", "## Additional supplied context", "", "```json",
					  _json_block(remaining), "```"))
	return "\n".join(lines).rstrip() + "\n"


def _write_report(report, reports_dir=None, prefix="run") -> Path:
	safe = sanitize(report if isinstance(report, Mapping) else {"result": report})
	timestamp = _timestamp(safe.get("timestamp") or safe.get("as_of"))
	run_id = safe_component(safe.get("run_id") or safe.get("experiment_id"), "run", 48)
	filename = f"{_filename_timestamp(timestamp)}_{run_id}_{prefix}.md"
	directory = Path(reports_dir) if reports_dir is not None else REPORTS_DIR
	directory.mkdir(parents=True, exist_ok=True)
	content = render_run_report(safe)
	destination = directory / filename
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


def record_run_report(report, reports_dir=None) -> Path:
	"""Write one timestamped run report without making decisions or side effects."""
	return _write_report(report, reports_dir, "run")


def record_experiment_report(experiment, reports_dir=None) -> Path:
	"""Write a supplied success/failure experiment record; no result is inferred."""
	entry = dict(experiment) if isinstance(experiment, Mapping) else {"result": experiment}
	entry.setdefault("title", "Experiment record")
	entry.setdefault("status", entry.get("outcome", "unspecified"))
	return _write_report(entry, reports_dir, "experiment")
