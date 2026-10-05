"""Shared fixtures: build Jev-shaped answers without touching the network."""

from __future__ import annotations

import copy
from typing import Any


def answers(
    *,
    intent: str = "repo_search",
    intent_conf: float = 1.0,
    scope: str = "trivial",
    scope_conf: float = 1.0,
    repo_scan: float = 0.9,
    external: float = 0.05,
    write: float = 0.05,
    tests: float = 0.05,
    clarification: float = 0.0,
    sensitive: float = 0.0,
    cross_source: float = 0.0,
    risk_probs: dict[str, float] | None = None,
) -> dict[str, Any]:
    probs = risk_probs if risk_probs is not None else {"0": 1.0, "1": 0.0, "2": 0.0}
    intent_probs = {intent: 0.9, "other": 0.1}
    scope_probs = {scope: 0.9, "unknown": 0.1}
    return {
        "intent": {"type": "choice", "choice": intent, "confidence": intent_conf, "probabilities": intent_probs},
        "scope": {"type": "choice", "choice": scope, "confidence": scope_conf, "probabilities": scope_probs},
        "needs_repo_scan": {"type": "noul", "noul": repo_scan},
        "needs_external_research": {"type": "noul", "noul": external},
        "needs_write": {"type": "noul", "noul": write},
        "needs_tests": {"type": "noul", "noul": tests},
        "requires_clarification": {"type": "noul", "noul": clarification},
        "sensitive_access": {"type": "noul", "noul": sensitive},
        "needs_cross_source_comparison": {"type": "noul", "noul": cross_source},
        "risk": {
            "type": "score",
            "score": 0.0,
            "confidence": 0.99,
            "legend": {"0": "read-only", "1": "local", "2": "shared"},
            "probabilities": copy.deepcopy(probs),
        },
    }
