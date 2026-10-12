import os
import re
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from trading_system.crew.memory import record_experiment, record_memory, safe_component
from trading_system.crew.reporting import (
    record_experiment_report,
    record_run_report,
    render_run_report,
)


class MemoryReportingTests(unittest.TestCase):
    def test_memory_entry_is_timestamped_markdown(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = record_memory({
                "run_id": "run-42",
                "symbol": "AAPL",
                "timeframe": "1Day",
                "agent": "Judge",
                "advisory": {"adjustment": 0.1, "reason": "Context only"},
            }, temporary)
            content = path.read_text(encoding="utf-8")
        self.assertEqual(path.suffix, ".md")
        self.assertIn("# Journal", content)
        self.assertIn("Timestamp:", content)
        self.assertIn("run-42", content)
        self.assertIn("AAPL", content)
        self.assertIn("Context only", content)

    def test_safe_filename_components_remove_path_traversal(self):
        component = safe_component("../../outside\\folder:api/key", fallback="safe")
        self.assertNotIn("/", component)
        self.assertNotIn("\\", component)
        self.assertNotIn("..", component)
        self.assertRegex(component, r"^[A-Za-z0-9._-]+$")

    def test_partial_and_malformed_entries_are_recorded_safely(self):
        with tempfile.TemporaryDirectory() as temporary:
            partial = record_memory({"warning": "one field only"}, temporary)
            malformed = record_memory(None, temporary)
            partial_text = partial.read_text(encoding="utf-8")
            malformed_text = malformed.read_text(encoding="utf-8")
        self.assertIn("one field only", partial_text)
        self.assertIn("# Journal_entry", malformed_text)
        self.assertIn("Timestamp:", malformed_text)

    def test_directory_is_created_and_duplicate_names_do_not_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "nested" / "journal"
            entry = {"timestamp": "2026-10-09T10:11:12+00:00", "run_id": "r1",
                     "symbol": "AAPL", "agent": "Data", "result": "recorded"}
            first = record_memory(entry, target)
            second = record_memory(entry, target)
            self.assertTrue(target.is_dir())
            self.assertNotEqual(first, second)
            self.assertIn("_2.md", second.name)
            self.assertEqual(first.read_text(encoding="utf-8"), second.read_text(encoding="utf-8"))

    def test_experiment_success_and_failure_are_preserved(self):
        with tempfile.TemporaryDirectory() as temporary:
            success = record_experiment({
                "experiment_id": "exp-ok", "timestamp": "2026-10-09T00:00:00Z",
                "status": "success", "what_was_tested": "offline parse",
                "result": {"passed": True},
            }, temporary)
            failure = record_experiment({
                "experiment_id": "exp-fail", "timestamp": "2026-10-09T00:01:00Z",
                "status": "failure", "what_was_tested": "mock timeout",
                "errors": ["timeout"],
            }, temporary)
            success_text = success.read_text(encoding="utf-8")
            failure_text = failure.read_text(encoding="utf-8")
        self.assertIn("success", success_text)
        self.assertIn("offline parse", success_text)
        self.assertIn("failure", failure_text)
        self.assertIn("timeout", failure_text)

    def test_run_report_contains_supplied_decision_advisory_risk_and_warnings(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = record_run_report({
                "run_id": "cycle-7", "timestamp": "2026-10-09T10:00:00Z",
                "symbols": ["AAPL"], "timeframe": "1Day", "status": "complete",
                "decisions": [{"symbol": "AAPL", "decision": "HOLD"}],
                "advisory_results": [{"symbol": "AAPL", "summary": "Context only"}],
                "risk_context": {"state": "not-run"},
                "actions": [], "warnings": ["sample warning"], "degraded": False,
            }, temporary)
            markdown = path.read_text(encoding="utf-8")
            self.assertTrue(path.exists())
        self.assertIn("Paper-Trading Run Report", markdown)
        self.assertIn("AAPL", markdown)
        self.assertIn("Context only", markdown)
        self.assertIn("not-run", markdown)
        self.assertIn("sample warning", markdown)

    def test_experiment_report_records_failure_without_inventing_result(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = record_experiment_report({
                "experiment_id": "exp-9", "timestamp": "2026-10-09T11:00:00Z",
                "status": "failure", "what_was_tested": "mock import",
                "errors": ["dependency unavailable"],
            }, temporary)
            markdown = path.read_text(encoding="utf-8")
        self.assertIn("Experiment record", markdown)
        self.assertIn("failure", markdown)
        self.assertIn("dependency unavailable", markdown)
        self.assertNotIn("success", markdown)

    def test_malformed_report_input_renders_safely(self):
        markdown = render_run_report(["unexpected", "sequence"])
        self.assertIn("unexpected", markdown)

    def test_secret_fields_known_env_secrets_and_assignments_are_redacted(self):
        secret = "m9-test-secret-value"
        with tempfile.TemporaryDirectory() as temporary, \
                patch.dict(os.environ, {"GROQ_API_KEY": secret}):
            path = record_run_report({
                "run_id": "secret-check", "timestamp": "2026-10-09T12:00:00Z",
                "context": {"api_key": secret, "note": f"literal={secret}"},
                "errors": ["NEWSAPI_KEY=another-secret", "Authorization: Bearer abc.def.token"],
            }, temporary)
            markdown = path.read_text(encoding="utf-8")
        self.assertNotIn(secret, markdown)
        self.assertNotIn("another-secret", markdown)
        self.assertNotIn("abc.def.token", markdown)
        self.assertGreaterEqual(markdown.count("[REDACTED]"), 3)

    def test_rendering_is_deterministic_for_same_supplied_report(self):
        data = {"run_id": "stable", "timestamp": "2026-10-09T12:00:00Z",
                "warnings": ["same warning"], "score": {"composite": 0.2}}
        self.assertEqual(render_run_report(data), render_run_report(data))

    def test_multiline_metadata_cannot_inject_markdown_sections(self):
        markdown = render_run_report({
            "run_id": "cycle-1\n## Injected heading",
            "status": "complete\n## False section",
        })
        self.assertNotIn("\n## Injected heading", markdown)
        self.assertNotIn("\n## False section", markdown)


if __name__ == "__main__":
    unittest.main()
