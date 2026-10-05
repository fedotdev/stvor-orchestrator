"""Schema-v2 routing artifacts separate recommendation from execution."""

from __future__ import annotations

import json
import os
import unittest

from agent_router.artifact import SCHEMA_VERSION, build_artifact
from agent_router.config import RouterConfig
from agent_router.policy import plan_from_answers
from agent_router.tests.fixtures import answers


class TestArtifactV2(unittest.TestCase):
    def test_layers_distinguish_jev_policy_recommendation_and_execution(self):
        plan = plan_from_answers(
            answers(repo_scan=0.95, external=0.05, write=0.05),
            state="Сверь код с документацией API",
        )
        payload = build_artifact(
            run_id="r-v2",
            task="Сверь код с документацией API",
            plan=plan,
            mode="observe",
            applied_route="unchanged",
            config=RouterConfig(),
            answers=answers(repo_scan=0.95, external=0.05, write=0.05),
        )
        self.assertEqual(payload["schema_version"], 2)
        layers = payload["layers"]
        self.assertIn("needs_external_research", layers["jev_answers"])
        self.assertTrue(layers["policy_derived"]["external_comparison_hint"])
        self.assertIn("researcher", layers["recommended_roles"])
        self.assertFalse(layers["execution_plan"]["attempted"])
        self.assertEqual(layers["execution_plan"]["status"], "blocked")
        self.assertEqual(layers["checks"], {"planned": [], "executed": [], "blocked": []})
        self.assertEqual(payload["task_state"]["status"], "completed")
        self.assertEqual(layers["acceptance"]["status"], "pending")
        self.assertFalse(payload["behavior_changed"])

    def test_secrets_are_scrubbed_across_jev_answers_plan_and_errors(self):
        secret = "super-secret-value-123456"
        old = os.environ.get("ROUTER_TEST_TOKEN")
        os.environ["ROUTER_TEST_TOKEN"] = secret
        try:
            raw_answers = answers()
            raw_answers["debug"] = {"echo": secret}
            plan = plan_from_answers(answers())
            # A Jev answer can echo task content; artifact scrub must cover the
            # raw answer as well as the task and error strings.
            payload = build_artifact(
                run_id="r-secret",
                task=f"task includes {secret}",
                plan=plan,
                mode="observe",
                applied_route="unchanged",
                config=RouterConfig(api_key_env="ROUTER_TEST_NONEXISTENT"),
                answers=raw_answers,
                errors=[f"failed on {secret}"],
            )
        finally:
            if old is None:
                os.environ.pop("ROUTER_TEST_TOKEN", None)
            else:
                os.environ["ROUTER_TEST_TOKEN"] = old
        encoded = json.dumps(payload)
        self.assertNotIn(secret, encoded)
        self.assertEqual(payload["key_present"], False)

    def test_unknown_runner_is_not_claimed_as_executed(self):
        plan = plan_from_answers(answers())
        payload = build_artifact(
            run_id="r-checks",
            task="inspect tests",
            plan=plan,
            mode="observe",
            applied_route="unchanged",
            config=RouterConfig(),
        )
        checks = payload["layers"]["checks"]
        self.assertFalse(checks["executed"])
        self.assertFalse(checks["planned"])
        self.assertEqual(payload["layers"]["execution_plan"]["status"], "blocked")


if __name__ == "__main__":
    unittest.main()
