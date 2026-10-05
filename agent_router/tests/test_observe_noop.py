"""Observe mode must not change behaviour, and mode gating must hold.

The router never executes anything; "behaviour" here means the applied_route the
caller is told to act on. In observe it is always `unchanged`.
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from agent_router import cli
from agent_router.config import RouterConfig
from agent_router.tests.fixtures import answers as make_answers

TASK = "Найди файл, где считается интервал попутного следования. Ничего не меняй."


class _StubClient:
    """Stands in for the network without mocking urllib."""

    def __init__(self, config=None):
        self.config = config
        self.calls = 0

    def evaluate(self, state, questions):
        from agent_router.client import JevResult

        self.calls += 1
        return JevResult(
            model="jev-1.13.0",
            answers=make_answers(intent="repo_search", intent_conf=1.0, scope="trivial", scope_conf=1.0),
            usage={"input_tokens": 400, "output_tokens": 80},
            latency_ms=12,
            attempts=1,
            request_bytes=512,
            raw={},
        )


class _FailingClient:
    def __init__(self, config=None):
        self.config = config

    def evaluate(self, state, questions):
        from agent_router.client import JevTimeout

        raise JevTimeout("mock timeout")


class ModeTests(unittest.TestCase):
    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self.root = self._dir.name
        self._orig_client = cli.client_mod.JevClient
        self.addCleanup(self._dir.cleanup)
        self.addCleanup(lambda: setattr(cli.client_mod, "JevClient", self._orig_client))

    def _artifacts(self) -> list[Path]:
        directory = Path(self.root) / ".agent" / "artifacts" / "routing"
        return sorted(directory.glob("*.json"))

    def test_observe_never_changes_behaviour(self):
        cli.client_mod.JevClient = _StubClient
        payload = cli.route_once(TASK, self.root, mode="observe")
        self.assertEqual(payload["applied_route"], "unchanged")
        self.assertEqual(payload["mode"], "observe")
        # The plan is still computed and recorded, just not applied.
        self.assertIn(
            payload["plan"]["route"],
            ("fast_readonly", "orchestrated_readonly", "orchestrated_write", "manual_orchestrator", "clarification_required"),
        )
        self.assertEqual(len(self._artifacts()), 1)

    def test_observe_artifact_records_plan_and_thresholds(self):
        cli.client_mod.JevClient = _StubClient
        payload = cli.route_once(TASK, self.root, mode="observe")
        stored = json.loads(Path(payload["artifact"]).read_text(encoding="utf-8"))
        for key in ("run_id", "timestamp", "mode", "applied_route", "task", "jev", "thresholds", "plan", "errors"):
            self.assertIn(key, stored)
        self.assertIn("intent_confidence", stored["thresholds"])
        self.assertEqual(stored["jev"]["model"], "jev-1.13.0")

    def test_off_does_not_call_jev_and_records_nothing_applied(self):
        stub = _StubClient()
        cli.client_mod.JevClient = lambda config=None: stub
        payload = cli.route_once(TASK, self.root, mode="off")
        self.assertEqual(stub.calls, 0)
        self.assertEqual(payload["applied_route"], "jev_not_called")
        self.assertEqual(len(self._artifacts()), 1)

    def test_assist_waits_for_the_user(self):
        cli.client_mod.JevClient = _StubClient
        payload = cli.route_once(TASK, self.root, mode="assist")
        self.assertEqual(payload["applied_route"], "pending_confirmation")

    def test_enforce_applies_only_fast_readonly(self):
        cli.client_mod.JevClient = _StubClient
        payload = cli.route_once(TASK, self.root, mode="enforce")
        self.assertEqual(payload["plan"]["route"], "fast_readonly")
        self.assertEqual(payload["applied_route"], "fast_readonly")
        self.assertFalse(payload["behavior_changed"] is None)

    def test_enforce_never_applies_a_write_route(self):
        class WriteClient(_StubClient):
            def evaluate(self, state, questions):
                from agent_router.client import JevResult

                return JevResult(
                    model="jev-1.13.0",
                    answers=make_answers(intent="bug_fix", scope="local", scope_conf=0.9, write=0.95, tests=0.9),
                    usage={"input_tokens": 1, "output_tokens": 1},
                    latency_ms=5,
                    attempts=1,
                    request_bytes=1,
                    raw={},
                )

        cli.client_mod.JevClient = WriteClient
        payload = cli.route_once("Добавь флаг в scripts/migrate.py", self.root, mode="enforce")
        self.assertEqual(payload["plan"]["route"], "orchestrated_write")
        self.assertEqual(payload["applied_route"], "pending_confirmation")

    def test_behavior_changed_tracks_auto_routes_only(self):
        # behavior_changed used to carry its own hardcoded route list, which
        # drifted from AUTO_ROUTES. It is now derived from that single source,
        # so the only contract worth pinning is: true exactly when enforce
        # actually applied an auto route.
        cli.client_mod.JevClient = _StubClient
        applied = cli.route_once(TASK, self.root, mode="enforce")
        self.assertEqual(applied["applied_route"], "fast_readonly")
        self.assertTrue(applied["behavior_changed"])

        # A write route waits for a human, so nothing was changed.
        class WriteClient(_StubClient):
            def evaluate(self, state, questions):
                from agent_router.client import JevResult

                return JevResult(
                    model="jev-1.13.0",
                    answers=make_answers(intent="bug_fix", scope="local", scope_conf=0.9, write=0.95, tests=0.9),
                    usage={"input_tokens": 1, "output_tokens": 1},
                    latency_ms=5,
                    attempts=1,
                    request_bytes=1,
                    raw={},
                )

        cli.client_mod.JevClient = WriteClient
        pending = cli.route_once("Добавь флаг в scripts/migrate.py", self.root, mode="enforce")
        self.assertEqual(pending["applied_route"], "pending_confirmation")
        self.assertFalse(pending["behavior_changed"])

        # In observe the route is recorded but never acted on.
        cli.client_mod.JevClient = _StubClient
        observed = cli.route_once(TASK, self.root, mode="observe")
        self.assertFalse(observed["behavior_changed"])

    def test_jev_failure_falls_back_and_still_writes_artifact(self):
        cli.client_mod.JevClient = _FailingClient
        payload = cli.route_once(TASK, self.root, mode="enforce")
        self.assertEqual(payload["plan"]["route"], "manual_orchestrator")
        self.assertFalse(payload["plan"]["run_implementer"])
        self.assertEqual(payload["applied_route"], "pending_confirmation")
        self.assertTrue(payload["errors"])
        self.assertEqual(len(self._artifacts()), 1)

    def test_empty_task_is_handled_without_network(self):
        stub = _StubClient()
        cli.client_mod.JevClient = lambda config=None: stub
        payload = cli.route_once("   ", self.root, mode="observe")
        self.assertEqual(stub.calls, 0)
        self.assertEqual(payload["plan"]["route"], "manual_orchestrator")

    def test_max_parallel_default_is_two(self):
        # Guard against exceeding the OmniRoute heavy-request admission limit.
        self.assertEqual(RouterConfig().max_parallel, 2)
        os.environ["ROUTER_MAX_PARALLEL"] = "4"
        try:
            self.assertEqual(RouterConfig().max_parallel, 4)
        finally:
            del os.environ["ROUTER_MAX_PARALLEL"]


if __name__ == "__main__":
    unittest.main()
