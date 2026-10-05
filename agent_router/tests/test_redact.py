"""Secrets must not reach artifacts, stdout or Jev's state field."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from agent_router import redact
from agent_router.artifact import build_artifact, write_artifact
from agent_router.config import RouterConfig
from agent_router.policy import plan_for_unavailable

FAKE_KEY = "apikey_221bb10be8479fd404db1656ee8b698e348_ad06f326ba55fa85ce119153f25082149f668f39e63abe0596d76775bd62c16e"

DIRTY_TASK = f"""Обнови конфиг, у меня TYPESAFE_API_KEY={FAKE_KEY}
и в .env есть OPENAI_API_KEY=sk-proj-AAAABBBBCCCCDDDDEEEEFFFFGGGGHHHH
Authorization: Bearer eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NSJ9.dBjftJeZ4CVPmB92K27uhbUJU1p1r_wW1gFWFOEjXk
DATABASE_URL=postgres://admin:hunter2@db.internal:5432/prod
-----BEGIN RSA PRIVATE KEY-----
MIIEowIBAAKCAQEA0Z3VS5JJcds3xfn/ygWyF0qJRRUaWmuvEXAMPLEKEYmaterial
-----END RSA PRIVATE KEY-----"""


class TestScrub(unittest.TestCase):
    def test_typesafe_key_removed(self):
        self.assertNotIn("221bb10be", redact.scrub(FAKE_KEY))

    def test_provider_key_removed(self):
        self.assertNotIn("sk-proj-AAAABBBB", redact.scrub("OPENAI_API_KEY=sk-proj-AAAABBBBCCCCDDDDEEEEFFFFGGGGHHHH"))

    def test_bearer_token_removed(self):
        out = redact.scrub("Authorization: Bearer eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NSJ9.dBjftJeZ4CVPmB92K27uhbUJU1p1r_wW1gFWFOEjXk")
        self.assertNotIn("eyJhbGciOiJIUzI1NiJ9", out)

    def test_url_credentials_removed(self):
        out = redact.scrub("DATABASE_URL=postgres://admin:hunter2@db.internal:5432/prod")
        self.assertNotIn("hunter2", out)
        self.assertIn("db.internal", out)

    def test_private_key_block_removed(self):
        out = redact.scrub(DIRTY_TASK)
        self.assertNotIn("MIIEowIBAAKCAQEA", out)

    def test_env_value_of_current_process_removed(self):
        os.environ["ROUTER_TEST_TOKEN"] = "super-secret-value-123456"
        try:
            self.assertNotIn("super-secret-value-123456", redact.scrub("token is super-secret-value-123456"))
        finally:
            del os.environ["ROUTER_TEST_TOKEN"]

    def test_non_secret_text_survives(self):
        text = "Найди обработчик ошибок в src/api и почини его"
        self.assertEqual(redact.scrub(text), text)

    def test_empty_input(self):
        self.assertEqual(redact.scrub(None), "")
        self.assertEqual(redact.scrub(""), "")

    def test_safe_env_summary_reports_presence_only(self):
        summary = redact.safe_env_summary(("TYPESAFE_API_KEY", "NOT_SET_ANYWHERE"))
        self.assertIn("TYPESAFE_API_KEY", summary)
        self.assertIn("NOT_SET_ANYWHERE", summary)
        self.assertNotIn(FAKE_KEY, json.dumps(summary))


class TestArtifact(unittest.TestCase):
    def test_artifact_scrubs_task_and_records_no_key(self):
        with tempfile.TemporaryDirectory() as root:
            payload = build_artifact(
                run_id="r1",
                task=DIRTY_TASK,
                plan=plan_for_unavailable("mock"),
                mode="observe",
                applied_route="unchanged",
                config=RouterConfig(),
            )
            path = write_artifact(root, "r1", payload, RouterConfig().artifact_dir)
            stored = path.read_text(encoding="utf-8")

        for leak in ("221bb10be", "sk-proj-AAAABBBB", "hunter2", "MIIEowIBAAKCAQEA", "eyJhbGciOiJIUzI1NiJ9"):
            self.assertNotIn(leak, stored)
        self.assertIn("[REDACTED]", stored)

    def test_key_present_is_a_bool_not_a_value(self):
        os.environ["TYPESAFE_API_KEY"] = FAKE_KEY
        try:
            payload = build_artifact(
                run_id="r2",
                task="hi",
                plan=plan_for_unavailable("mock"),
                mode="observe",
                applied_route="unchanged",
                config=RouterConfig(),
            )
        finally:
            del os.environ["TYPESAFE_API_KEY"]
        self.assertIs(payload["key_present"], True)
        self.assertNotIn(FAKE_KEY, json.dumps(payload))


if __name__ == "__main__":
    unittest.main()
