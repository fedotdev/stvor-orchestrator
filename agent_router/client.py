"""HTTP client for the TypeSafe System One evaluation endpoint.

Documented contract (docs.typesafe.ai/api):
    POST https://api.typesafe.ai/v1/systemone
    Authorization: Bearer <API_KEY>
    Content-Type: application/json
    body: {state, model, questions}

Documented error statuses: 401 missing/invalid key, 422 body validation,
429 rate limited, 529 overloaded. 429 and 529 are retryable with exponential
backoff; the client honours `retry-after` when present.

Only stdlib is used, so the router adds no dependency to the agent config.
"""

from __future__ import annotations

import json
import random
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable

from .config import RouterConfig


class JevError(Exception):
    """Base class for every failure surfaced by the client."""


class JevConfigError(JevError):
    """No API key configured."""


class JevAuthError(JevError):
    """401: the key is missing or invalid. Retrying will not help."""


class JevValidationError(JevError):
    """422: the request body failed server-side validation."""

    def __init__(self, message: str, detail: Any = None) -> None:
        super().__init__(message)
        self.detail = detail


class JevRateLimited(JevError):
    """429: rate limited after exhausting the retry budget."""


class JevOverloaded(JevError):
    """529: TypeSafe overloaded after exhausting the retry budget."""


class JevServerError(JevError):
    """5xx after exhausting the retry budget."""


class JevTimeout(JevError):
    """The request exceeded the configured timeout."""


class JevProtocolError(JevError):
    """HTTP 200 but the payload is not a usable System One response."""

    def __init__(self, message: str, detail: Any = None) -> None:
        super().__init__(message)
        self.detail = detail


@dataclass
class JevResult:
    model: str
    answers: dict[str, dict]
    usage: dict[str, int]
    latency_ms: int
    attempts: int
    request_bytes: int
    raw: dict = field(default_factory=dict)


class JevClient:
    def __init__(
        self,
        config: RouterConfig | None = None,
        opener: Callable[[urllib.request.Request, float], Any] | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.config = config or RouterConfig()
        # Injection point for tests: avoids patching urllib globally.
        self._open = opener or _urllib_open
        self._sleep = sleep

    def evaluate(self, state: str, questions: dict[str, dict]) -> JevResult:
        key = self.config.api_key()
        if not key:
            raise JevConfigError(f"{self.config.api_key_env} is not set")

        body = json.dumps({"state": state, "model": self.config.model, "questions": questions}).encode("utf-8")
        last_error: Exception | None = None

        for attempt in range(1, self.config.max_retries + 2):
            request = urllib.request.Request(
                self.config.endpoint,
                data=body,
                method="POST",
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            )
            started = time.monotonic()
            try:
                with self._open(request, self.config.timeout_seconds) as response:
                    raw = response.read().decode("utf-8")
                try:
                    payload = json.loads(raw)
                except ValueError as exc:
                    raise JevProtocolError("Jev returned a body that is not JSON", raw[:200]) from exc
                latency_ms = int((time.monotonic() - started) * 1000)
                return self._parse(payload, latency_ms, attempt, len(body))
            except urllib.error.HTTPError as exc:  # noqa: PERF203 - retry loop is the point
                detail = _read_json(exc)
                last_error = _classify_http(exc.code, detail)
                if not _is_retryable(exc.code) or attempt > self.config.max_retries:
                    raise last_error from exc
                self._sleep(self._backoff(attempt, exc.headers.get("retry-after")))
            except TimeoutError as exc:
                last_error = JevTimeout(f"Jev did not answer within {self.config.timeout_seconds}s")
                if attempt > self.config.max_retries:
                    raise last_error from exc
                self._sleep(self._backoff(attempt, None))
            except (urllib.error.URLError, OSError) as exc:
                reason = getattr(exc, "reason", exc)
                if isinstance(reason, TimeoutError):
                    last_error = JevTimeout(f"Jev did not answer within {self.config.timeout_seconds}s")
                    retryable = True
                elif isinstance(reason, ConnectionRefusedError):
                    last_error = JevError(f"Temporary connection failure calling Jev: {reason}")
                    retryable = True
                else:
                    # DNS resolution, TLS/certificate and other unclassified
                    # transport failures are not blindly retried.
                    last_error = JevError(f"Non-retryable network failure calling Jev: {reason}")
                    retryable = False
                if not retryable or attempt > self.config.max_retries:
                    raise last_error from exc
                self._sleep(self._backoff(attempt, None))

        raise last_error or JevError("Jev evaluation failed")

    # -- internals ---------------------------------------------------------

    def _parse(self, payload: Any, latency_ms: int, attempts: int, request_bytes: int) -> JevResult:
        if not isinstance(payload, dict):
            raise JevProtocolError("Response is not a JSON object", payload)
        answers = payload.get("answers")
        if not isinstance(answers, dict) or not answers:
            raise JevProtocolError("Response has no answers map", payload)
        usage = payload.get("usage") if isinstance(payload.get("usage"), dict) else {}
        return JevResult(
            model=str(payload.get("model", self.config.model)),
            answers=answers,
            usage={"input_tokens": int(usage.get("input_tokens", 0)), "output_tokens": int(usage.get("output_tokens", 0))},
            latency_ms=latency_ms,
            attempts=attempts,
            request_bytes=request_bytes,
            raw=payload,
        )

    def _backoff(self, attempt: int, retry_after: str | None) -> float:
        # retry-after may be seconds; the documented 429/529 behaviour is "retry
        # after a short delay", so an absurd value is ignored rather than slept.
        if retry_after:
            try:
                return min(float(retry_after), self.config.max_backoff_seconds)
            except ValueError:
                pass
        delay = min(self.config.backoff_base_seconds * (2 ** (attempt - 1)), self.config.max_backoff_seconds)
        return delay * (0.5 + random.random() / 2)  # jitter


def _is_retryable(status: int) -> bool:
    return status == 429 or status == 529 or 500 <= status < 600


def _classify_http(status: int, detail: Any) -> JevError:
    if status == 401:
        return JevAuthError("Jev rejected the API key (401)")
    if status == 422:
        return JevValidationError("Jev rejected the request body (422)", detail)
    if status == 429:
        return JevRateLimited("Jev rate limit exhausted (429)")
    if status == 529:
        return JevOverloaded("Jev overloaded after retries (529)")
    return JevServerError(f"Jev server error ({status})", detail)


def _read_json(exc: urllib.error.HTTPError) -> Any:
    try:
        return json.loads(exc.read().decode("utf-8", "replace"))
    except Exception:  # noqa: BLE001 - diagnostics only
        return None


def _urllib_open(request: urllib.request.Request, timeout: float):
    # socket-level timeouts surface as TimeoutError/OSError and are classified in evaluate().
    return urllib.request.urlopen(request, timeout=timeout)
