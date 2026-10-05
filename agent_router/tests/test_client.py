"""Client tests with injected opener and sleep: no network, no real waiting."""

from __future__ import annotations

import io
import json
import unittest
import urllib.error
import urllib.request

from agent_router.client import (
    JevAuthError,
    JevClient,
    JevConfigError,
    JevOverloaded,
    JevProtocolError,
    JevRateLimited,
    JevServerError,
    JevTimeout,
)
from agent_router.config import RouterConfig

from .fixtures import answers

GOOD_BODY = {
    "model": "jev-1.13.0",
    "answers": answers(),
    "usage": {"input_tokens": 512, "output_tokens": 96},
}

QUESTIONS = {"intent": {"type": "choice", "instructions": "?", "criteria": {"a": "A"}}}


class _Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


def http_error(code: int, payload: dict | None = None, headers: dict | None = None) -> urllib.error.HTTPError:
    body = json.dumps(payload or {"error": "boom"}).encode()
    return urllib.error.HTTPError(
        url="https://api.typesafe.ai/v1/systemone",
        code=code,
        msg="mock",
        hdrs=headers or {},
        fp=io.BytesIO(body),
    )


class Recorder:
    def __init__(self, behaviours: list) -> None:
        self.behaviours = list(behaviours)
        self.calls = 0
        self.slept: list[float] = []

    def open(self, request: urllib.request.Request, timeout: float):
        self.calls += 1
        behaviour = self.behaviours.pop(0)
        if isinstance(behaviour, Exception):
            raise behaviour
        return _Response(json.dumps(behaviour).encode())

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)


class TestHappyPath(unittest.TestCase):
    def setUp(self):
        self.config = RouterConfig(max_retries=2, backoff_base_seconds=0.1)
        import os

        os.environ.setdefault("TYPESAFE_API_KEY", "test-key-not-real-000000")

    def test_parses_answers_usage_and_latency(self):
        recorder = Recorder([GOOD_BODY])
        client = JevClient(config=self.config, opener=recorder.open, sleep=recorder.sleep)
        result = client.evaluate("find the file", QUESTIONS)
        self.assertEqual(result.model, "jev-1.13.0")
        self.assertEqual(result.usage["input_tokens"], 512)
        self.assertEqual(result.attempts, 1)
        self.assertEqual(recorder.calls, 1)
        self.assertIn("intent", result.answers)

    def test_missing_api_key_raises_config_error(self):
        import os

        saved = os.environ.pop("TYPESAFE_API_KEY", None)
        try:
            client = JevClient(config=self.config, opener=Recorder([GOOD_BODY]).open)
            with self.assertRaises(JevConfigError):
                client.evaluate("x", QUESTIONS)
        finally:
            if saved is not None:
                os.environ["TYPESAFE_API_KEY"] = saved


class TestErrors(unittest.TestCase):
    def setUp(self):
        import os

        os.environ.setdefault("TYPESAFE_API_KEY", "test-key-not-real-000000")
        self.config = RouterConfig(max_retries=2, backoff_base_seconds=0.1, max_backoff_seconds=1.0)

    def _client(self, behaviours):
        recorder = Recorder(behaviours)
        return JevClient(config=self.config, opener=recorder.open, sleep=recorder.sleep), recorder

    def test_401_is_not_retried(self):
        client, recorder = self._client([http_error(401)])
        with self.assertRaises(JevAuthError):
            client.evaluate("x", QUESTIONS)
        self.assertEqual(recorder.calls, 1)

    def test_422_is_not_retried(self):
        client, recorder = self._client([http_error(422, {"field": "questions.risk.criteria"})])
        with self.assertRaises(Exception) as ctx:
            client.evaluate("x", QUESTIONS)
        self.assertIn("422", str(ctx.exception))
        self.assertEqual(recorder.calls, 1)

    def test_429_retries_then_raises_rate_limited(self):
        client, recorder = self._client([http_error(429), http_error(429), http_error(429)])
        with self.assertRaises(JevRateLimited):
            client.evaluate("x", QUESTIONS)
        self.assertEqual(recorder.calls, 3)  # initial + 2 retries
        self.assertEqual(len(recorder.slept), 2)

    def test_429_then_success_succeeds(self):
        client, recorder = self._client([http_error(429), GOOD_BODY])
        result = client.evaluate("x", QUESTIONS)
        self.assertEqual(result.attempts, 2)

    def test_retry_after_header_is_honoured_but_capped(self):
        recorder = Recorder([http_error(429, None, {"retry-after": "99"}), GOOD_BODY])
        config = RouterConfig(max_retries=2, backoff_base_seconds=0.1, max_backoff_seconds=1.0)
        client = JevClient(config=config, opener=recorder.open, sleep=recorder.sleep)
        client.evaluate("x", QUESTIONS)
        self.assertEqual(recorder.slept, [1.0])  # capped at max_backoff_seconds

    def test_529_retries_then_raises_overloaded(self):
        client, recorder = self._client([http_error(529), http_error(529), http_error(529)])
        with self.assertRaises(JevOverloaded):
            client.evaluate("x", QUESTIONS)
        self.assertEqual(recorder.calls, 3)

    def test_500_retries_then_raises_server_error(self):
        client, recorder = self._client([http_error(500), http_error(503), http_error(500)])
        with self.assertRaises(JevServerError):
            client.evaluate("x", QUESTIONS)
        self.assertEqual(recorder.calls, 3)

    def test_timeout_retries_then_raises_timeout(self):
        client, recorder = self._client([TimeoutError("timed out"), TimeoutError("timed out"), TimeoutError("timed out")])
        with self.assertRaises(JevTimeout):
            client.evaluate("x", QUESTIONS)
        self.assertEqual(recorder.calls, 3)

    def test_connection_refused_retries_then_raises(self):
        errors = [urllib.error.URLError(ConnectionRefusedError("refused")) for _ in range(3)]
        client, recorder = self._client(errors)
        with self.assertRaises(Exception) as ctx:
            client.evaluate("x", QUESTIONS)
        self.assertNotIsInstance(ctx.exception, (JevAuthError, JevRateLimited))
        self.assertEqual(recorder.calls, 3)

    def test_unclassified_transport_failure_is_not_retried(self):
        error = urllib.error.URLError(OSError("name resolution failed"))
        client, recorder = self._client([error])
        with self.assertRaises(Exception):
            client.evaluate("x", QUESTIONS)
        self.assertEqual(recorder.calls, 1)
        self.assertEqual(recorder.slept, [])

    def test_invalid_json_body_raises_protocol_error(self):
        class Raw(io.BytesIO):
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

        client = JevClient(config=self.config, opener=lambda request, timeout: Raw(b"not json at all"), sleep=lambda s: None)
        with self.assertRaises(JevProtocolError):
            client.evaluate("x", QUESTIONS)

    def test_json_without_answers_raises_protocol_error(self):
        client, _ = self._client([{"model": "jev-1.13.0", "usage": {}}])
        with self.assertRaises(JevProtocolError):
            client.evaluate("x", QUESTIONS)

    def test_answers_wrong_shape_raises_protocol_error(self):
        client, _ = self._client([{"answers": []}])
        with self.assertRaises(JevProtocolError):
            client.evaluate("x", QUESTIONS)


if __name__ == "__main__":
    unittest.main()
