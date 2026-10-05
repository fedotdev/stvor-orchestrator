"""Policy worker keys are recommendations; resolve names, don't claim launch."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from routing_eval.score import bind_worker_roles


class TestRoleBindings(unittest.TestCase):
    def test_current_workflow_role_names_resolve_to_agent_filenames(self):
        bindings, gaps = bind_worker_roles(("scout", "researcher", "test_analyst", "implementer", "deep_reviewer"))
        self.assertEqual(
            bindings,
            {
                "scout": "scout",
                "researcher": "researcher",
                "test_analyst": "test-analyst",
                "implementer": "implementer",
                "deep_reviewer": "reviewer",
            },
        )
        self.assertEqual(gaps, [])

    def test_missing_loaded_agent_is_reported_as_gap(self):
        with tempfile.TemporaryDirectory() as root:
            Path(root, "scout.md").write_text("---\nmode: subagent\n---\n", encoding="utf-8")
            bindings, gaps = bind_worker_roles(("scout", "researcher"), Path(root))
        self.assertEqual(bindings["researcher"], "researcher")
        self.assertEqual(gaps, ["researcher"])

    def test_unknown_role_is_not_silently_mapped(self):
        bindings, gaps = bind_worker_roles(("future_role",), Path("missing-agent-dir"))
        self.assertIsNone(bindings["future_role"])
        self.assertEqual(gaps, ["future_role"])


if __name__ == "__main__":
    unittest.main()
