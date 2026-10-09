import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

REPOSITORY_DIR = Path(__file__).resolve().parents[2]
if str(REPOSITORY_DIR) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_DIR))

from trading_system.ui import app


class M10UITests(unittest.TestCase):
    def test_module_imports_without_streamlit(self):
        self.assertTrue(callable(app.main))

    def test_missing_directories_return_empty_artifact_lists(self):
        with tempfile.TemporaryDirectory() as temporary:
            reports = Path(temporary) / "not-created-reports"
            memory = Path(temporary) / "not-created-memory"
            self.assertEqual(app.discover_markdown(reports), [])
            self.assertEqual(app.discover_markdown(memory), [])
            self.assertTrue(reports.is_dir())
            self.assertTrue(memory.is_dir())

    def test_report_discovery_orders_newest_first(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            old = root / "old.md"
            new = root / "new.md"
            old.write_text("# Old", encoding="utf-8")
            new.write_text("# New", encoding="utf-8")
            os.utime(old, (100, 100))
            os.utime(new, (200, 200))
            self.assertEqual([path.name for path in app.discover_markdown(root)], ["new.md", "old.md"])

    def test_markdown_loading_and_report_metadata(self):
        content = (
            "# Cycle report\n\n- Timestamp: 2026-10-09T12:30:00Z\n"
            "- Run ID: run-7\n- Status: complete\n- Outcome: recorded\n"
            "\n## Warnings\n\n- fixture warning\n"
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "cycle.md"
            path.write_text(content, encoding="utf-8")
            loaded, error = app.load_markdown_artifact(temporary, path.name)
        self.assertIsNone(error)
        self.assertEqual(loaded, content)
        summary = app.artifact_summary(loaded)
        self.assertEqual(summary["title"], "Cycle report")
        self.assertEqual(summary["run_id"], "run-7")
        self.assertEqual(summary["status"], "complete")
        self.assertIn("fixture warning", summary["warnings"])

    def test_journal_discovery_and_loading(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "journal.md"
            path.write_text("# Journal\n\nEntry body", encoding="utf-8")
            entries = app.discover_markdown(temporary)
            loaded, error = app.load_markdown_artifact(temporary, entries[0])
        self.assertIsNone(error)
        self.assertIn("Entry body", loaded)

    def test_path_traversal_and_outside_absolute_paths_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary, tempfile.TemporaryDirectory() as outside:
            secret = Path(outside) / "secret.md"
            secret.write_text("do not read", encoding="utf-8")
            text, error = app.load_markdown_artifact(temporary, "../../secret.md")
            absolute_text, absolute_error = app.load_markdown_artifact(temporary, secret)
        self.assertIsNone(text)
        self.assertTrue(error)
        self.assertIsNone(absolute_text)
        self.assertTrue(absolute_error)

    def test_symlink_artifacts_are_not_loaded(self):
        with tempfile.TemporaryDirectory() as temporary, tempfile.TemporaryDirectory() as outside:
            root = Path(temporary)
            external = Path(outside) / "external.md"
            external.write_text("external data", encoding="utf-8")
            link = root / "linked.md"
            try:
                link.symlink_to(external)
            except (OSError, NotImplementedError):
                self.skipTest("symlinks unavailable on this filesystem")
            self.assertEqual(app.discover_markdown(root), [])
            content, error = app.load_markdown_artifact(root, "linked.md")
        self.assertIsNone(content)
        self.assertTrue(error)

    def test_malformed_and_unreadable_files_are_handled(self):
        with tempfile.TemporaryDirectory() as temporary:
            bad = Path(temporary) / "bad.md"
            bad.write_bytes(b"\xff\xfe")
            content, error = app.load_markdown_artifact(temporary, bad.name)
            missing_content, missing_error = app.load_markdown_artifact(temporary, "gone.md")
        self.assertIsNone(content)
        self.assertTrue(error)
        self.assertIsNone(missing_content)
        self.assertTrue(missing_error)

    def test_empty_overview_shows_explicit_empty_states(self):
        fake_streamlit = Mock()
        fake_streamlit.sidebar.radio.return_value = "Overview"
        fake_streamlit.columns.return_value = [Mock(), Mock(), Mock()]
        with tempfile.TemporaryDirectory() as temporary:
            with patch.object(app, "REPORTS_DIR", Path(temporary) / "reports"), \
                    patch.object(app, "MEMORY_DIR", Path(temporary) / "memory"):
                app.main(fake_streamlit)
        messages = [call.args[0] for call in fake_streamlit.info.call_args_list]
        self.assertIn("No run reports available yet.", messages)
        self.assertIn("No journal entries available yet.", messages)

    def test_secret_values_are_redacted_before_display(self):
        secret = "ui-test-secret-value"
        with tempfile.TemporaryDirectory() as temporary, \
                patch.dict(os.environ, {"GROQ_API_KEY": secret}):
            path = Path(temporary) / "report.md"
            path.write_text(f"# Report\n\nGROQ_API_KEY={secret}\n", encoding="utf-8")
            content, error = app.load_markdown_artifact(temporary, path.name)
        self.assertIsNone(error)
        self.assertNotIn(secret, content)
        self.assertIn("[REDACTED]", content)


if __name__ == "__main__":
    unittest.main()
