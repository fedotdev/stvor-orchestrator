"""Secret scrubbing for logs and artifacts.

Repository content and task text are untrusted input. Anything that looks like a
credential is replaced before the task text reaches a log line or a routing
artifact, including the live API key value held in the environment.
"""

from __future__ import annotations

import os
import re

PLACEHOLDER = "[REDACTED]"

# Ordered: specific shapes first so a generic rule cannot mangle a longer match.
_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("private_key", re.compile(r"-----BEGIN[A-Z ]*PRIVATE KEY-----.*?-----END[A-Z ]*PRIVATE KEY-----", re.S)),
    ("authorization", re.compile(r"(?i)\b(bearer|basic)\s+[A-Za-z0-9._\-]{12,}")),
    ("env_assignment", re.compile(r"(?im)^([A-Z0-9_]*(?:KEY|TOKEN|SECRET|PASSWORD|PASSWD|PWD|CREDENTIAL)[A-Z0-9_]*)\s*=\s*\S+")),
    ("inline_assignment", re.compile(r"(?i)\b(api[_-]?key|apikey|token|secret|password|passwd|pwd|access[_-]?key)\b\s*[:=]\s*[\"']?[^\s\"',}]{6,}[\"']?")),
    ("provider_key", re.compile(r"\b(?:sk|pk|rk)-[A-Za-z0-9_\-]{12,}\b")),
    ("known_prefixes", re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{16,}\b")),
    ("aws_access_key", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")),
    ("typesafe_key", re.compile(r"\bapikey_[A-Za-z0-9_]{16,}\b")),
    ("basic_auth_url", re.compile(r"(?i)\b([a-z][a-z0-9+.\-]*://)[^/\s:@]+:[^/\s@]+@")),
    ("jwt", re.compile(r"\beyJ[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}\b")),
)

# Environment variable names whose *values* must never be echoed.
_SECRET_NAME_RE = re.compile(r"(?i)(KEY|TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIAL|API_KEY)")


def _env_secret_values() -> list[str]:
    values: list[str] = []
    for name, value in os.environ.items():
        if not value or len(value) < 12 or not _SECRET_NAME_RE.search(name):
            continue
        values.append(value)
    # Longest first so a short value cannot shadow a longer overlapping one.
    return sorted(set(values), key=len, reverse=True)


def scrub(text: str | None, extra_secrets: tuple[str, ...] = ()) -> str:
    """Return `text` with credential-shaped substrings replaced."""
    if not text:
        return ""
    out = text
    # Live environment values first: a secret pasted into a task is caught even
    # when its shape matches no pattern.
    candidates = set(_env_secret_values()) | {s for s in extra_secrets if s and len(s) >= 12}
    for secret in sorted(candidates, key=len, reverse=True):
        out = out.replace(secret, PLACEHOLDER)
    for _label, pattern in _PATTERNS:
        out = pattern.sub(PLACEHOLDER, out)
    return out


def safe_env_summary(names: tuple[str, ...]) -> dict[str, bool]:
    """Report only whether named env vars are set, never their values."""
    return {name: bool(os.environ.get(name)) for name in names}
