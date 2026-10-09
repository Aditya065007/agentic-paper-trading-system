"""Offline-first viewer for local M9 run reports and journal entries."""

import re
from datetime import datetime
from pathlib import Path

from trading_system.crew.memory import redact_text


PROJECT_DIR = Path(__file__).resolve().parents[1]
REPORTS_DIR = PROJECT_DIR / "run_reports"
MEMORY_DIR = PROJECT_DIR / "memory"
ARTIFACT_DIRS = {"reports": REPORTS_DIR, "journal": MEMORY_DIR}
MAX_ARTIFACT_BYTES = 2_000_000


def _safe_root(directory) -> Path | None:
	try:
		configured = Path(directory)
		if configured.is_symlink():
			return None
		root = configured.resolve(strict=False)
		root.mkdir(parents=True, exist_ok=True)
		return root
	except (OSError, RuntimeError, TypeError, ValueError):
		return None


def discover_markdown(directory) -> list[Path]:
	"""List direct-child Markdown artifacts newest first, skipping unsafe entries."""
	root = _safe_root(directory)
	if root is None:
		return []
	try:
		paths = []
		for path in root.iterdir():
			try:
				if path.suffix.lower() != ".md" or path.is_symlink() or not path.is_file():
					continue
				resolved = path.resolve(strict=True)
				if resolved.parent != root or resolved.stat().st_size > MAX_ARTIFACT_BYTES:
					continue
				paths.append(resolved)
			except (OSError, RuntimeError, ValueError):
				continue
		return sorted(paths, key=lambda path: (path.stat().st_mtime_ns, path.name), reverse=True)
	except (OSError, RuntimeError, ValueError):
		return []


def load_markdown_artifact(directory, selection) -> tuple[str | None, str | None]:
	"""Read one direct-child Markdown artifact contained within the approved root."""
	root = _safe_root(directory)
	if root is None or not isinstance(selection, (str, Path)):
		return None, "Artifact directory or selection is invalid."
	try:
		candidate = Path(selection)
		if candidate.is_absolute():
			if candidate.is_symlink():
				return None, "Symbolic-link artifacts are not displayed."
			resolved = candidate.resolve(strict=True)
		else:
			if candidate.name != str(candidate) or candidate.name in {".", ".."}:
				return None, "Unsafe artifact path rejected."
			joined = root / candidate.name
			if joined.is_symlink():
				return None, "Symbolic-link artifacts are not displayed."
			resolved = joined.resolve(strict=True)
		if resolved.parent != root or resolved.suffix.lower() != ".md" or not resolved.is_file():
			return None, "Selected file is outside the permitted Markdown directory."
		if resolved.stat().st_size > MAX_ARTIFACT_BYTES:
			return None, "Selected artifact exceeds the display size limit."
		return redact_text(resolved.read_text(encoding="utf-8")), None
	except FileNotFoundError:
		return None, "Selected artifact is no longer available."
	except (OSError, UnicodeError, RuntimeError, ValueError):
		return None, "Selected artifact could not be read."


def artifact_summary(markdown: str) -> dict:
	"""Extract only supplied top-level report metadata for overview display."""
	summary = {"title": None, "timestamp": None, "run_id": None,
			   "status": None, "outcome": None, "warnings": [], "errors": []}
	if not isinstance(markdown, str):
		summary["warnings"].append("Artifact contents are malformed.")
		return summary
	for line in markdown.splitlines():
		stripped = line.strip()
		if stripped.startswith("# ") and summary["title"] is None:
			summary["title"] = stripped[2:300]
		match = re.match(r"^-\s+(Timestamp|Run ID|Cycle ID|Status|Outcome):\s*(.*)$", stripped)
		if match:
			key = {"Run ID": "run_id", "Cycle ID": "run_id"}.get(
				match.group(1), match.group(1).lower().replace(" ", "_"))
			summary[key] = match.group(2)[:300]

	for field, heading in (("warnings", "## Warnings"), ("errors", "## Errors"),
						   ("errors", "## Errors / degraded state")):
		active = False
		for line in markdown.splitlines():
			if line.startswith("## "):
				active = line.strip() == heading
				continue
			if active and line.strip() and not line.startswith("```"):
				item = line.strip()
				if item.startswith(("- ", "* ")):
					item = item[2:].strip()
				summary[field].append(item[:500])
	return summary


def _display_timestamp(value):
	if not value:
		return "Timestamp unavailable"
	try:
		return datetime.fromisoformat(value.replace("Z", "+00:00")).strftime("%Y-%m-%d %H:%M UTC")
	except (TypeError, ValueError):
		return str(value)[:100]


def main(st_module=None):
	"""Render the local artifact viewer. Streamlit is imported only at runtime."""
	if st_module is None:
		try:
			import streamlit as st_module
		except ImportError as exc:
			raise RuntimeError("Install the streamlit dependency to run the M10 UI.") from exc

	st_module.set_page_config(page_title="Paper Trading | Local Reports", layout="wide")
	st_module.title("Paper Trading System")
	st_module.caption("Offline artifact viewer · no trading, broker, news, or LLM calls")

	report_paths = discover_markdown(REPORTS_DIR)
	journal_paths = discover_markdown(MEMORY_DIR)
	latest_summary = None
	latest_error = None
	if report_paths:
		latest_content, latest_error = load_markdown_artifact(REPORTS_DIR, report_paths[0])
		if latest_content is not None:
			latest_summary = artifact_summary(latest_content)

	metric_columns = st_module.columns(3)
	metric_columns[0].metric("System mode", "Offline")
	metric_columns[1].metric("Run reports", len(report_paths))
	metric_columns[2].metric("Journal entries", len(journal_paths))
	if latest_summary:
		st_module.caption(f"Latest report: {_display_timestamp(latest_summary['timestamp'])}")
	elif latest_error:
		st_module.warning(latest_error)

	page = st_module.sidebar.radio("View", ("Overview", "Run reports", "Journal", "System info"))
	if page == "Overview":
		st_module.subheader("Latest local activity")
		if not report_paths:
			st_module.info("No run reports available yet.")
		elif latest_summary:
			st_module.write({key: value for key, value in latest_summary.items()
							 if value not in (None, [], "")})
		if not journal_paths:
			st_module.info("No journal entries available yet.")
	elif page in ("Run reports", "Journal"):
		directory = REPORTS_DIR if page == "Run reports" else MEMORY_DIR
		paths = report_paths if page == "Run reports" else journal_paths
		empty_message = "No run reports available yet." if page == "Run reports" \
			else "No journal entries available yet."
		st_module.subheader(page)
		if not paths:
			st_module.info(empty_message)
			return
		selected = st_module.selectbox(
			"Select a file", paths, format_func=lambda path: path.name)
		content, error = load_markdown_artifact(directory, selected)
		if error:
			st_module.warning(error)
		elif content is not None:
			summary = artifact_summary(content)
			if summary["status"] or summary["outcome"]:
				st_module.caption(" · ".join(value for value in
											 (summary["status"], summary["outcome"]) if value))
			st_module.markdown(content, unsafe_allow_html=False)
	else:
		st_module.subheader("System information")
		st_module.write({
			"mode": "offline artifact display",
			"M8 orchestration": "not connected in this branch",
			"reports directory": str(REPORTS_DIR),
			"journal directory": str(MEMORY_DIR),
			"report count": len(report_paths),
			"journal count": len(journal_paths),
		})


if __name__ == "__main__":
	main()
