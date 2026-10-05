"""Routing artifacts: one auditable JSON file per routed request.

Written to `<project_root>/.agent/artifacts/routing/<run_id>.json` so a run can
be re-checked after the fact without re-querying Jev. Task text is scrubbed
before it is stored; the API key never reaches this module.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import redact
from .config import AUTO_ROUTES
from .policy import RoutingPlan

SCHEMA_VERSION = 2


def _scrub_json(value: Any) -> Any:
    """Scrub every stored string, including model echoes and error details."""
    if isinstance(value, str):
        return redact.scrub(value)
    if isinstance(value, dict):
        return {str(redact.scrub(str(key))): _scrub_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_scrub_json(item) for item in value]
    if isinstance(value, tuple):
        return [_scrub_json(item) for item in value]
    return value


def make_run_id(task: str, now: datetime | None = None) -> str:
    stamp = (now or datetime.now(timezone.utc)).strftime("%Y%m%dT%H%M%S")
    digest = hashlib.sha256(task.encode("utf-8")).hexdigest()[:8]
    return f"{stamp}-{digest}"


def build_artifact(
    *,
    run_id: str,
    task: str,
    plan: RoutingPlan,
    mode: str,
    applied_route: str,
    config: Any,
    model: str | None = None,
    answers: dict | None = None,
    latency_ms: int | None = None,
    attempts: int | None = None,
    errors: list[str] | None = None,
    started_at: datetime | None = None,
) -> dict:
    """Assemble the artifact payload. `task` is scrubbed, never stored raw."""
    started = started_at or datetime.now(timezone.utc)
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "timestamp": started.isoformat(),
        "mode": mode,
        "requested_route": plan.route,
        "applied_route": applied_route,
        "behavior_changed": mode == "enforce" and applied_route in AUTO_ROUTES,
        "task": redact.scrub(task),
        "jev": {
            "model": redact.scrub(model),
            "answers": _scrub_json(answers or {}),
            "latency_ms": latency_ms,
            "attempts": attempts,
        },
        "thresholds": {
            "intent_confidence": config.intent_confidence,
            "noul_threshold": config.noul_threshold,
            "size_escalate_confidence": config.size_escalate_confidence,
            "high_risk_probability": config.high_risk_probability,
            "max_parallel": config.max_parallel,
        },
        "layers": {
            "jev_answers": _scrub_json(answers or {}),
            "policy_derived": _scrub_json(plan.jev_snapshot.get("facts", {}).get("policy_derived", {})),
            "policy_guard_flags": _scrub_json(plan.jev_snapshot.get("guard_flags", [])),
            "recommended_roles": list(plan.workers),
            "routing_reason": redact.scrub(plan.reason),
            "execution_plan": {
                "status": "blocked",
                "executor": None,
                "attempted": False,
                "reason": "routing-only harness; no executor configured",
            },
            "checks": {"planned": [], "executed": [], "blocked": []},
            "acceptance": {"status": "pending", "reviewer_verdict": None},
        },
        "task_state": {
            "task_id": run_id,
            "parent_task_id": None,
            "role": "router",
            "agent_id": None,
            "status": "completed",
            "dependencies": [],
            "allowed_change_scope": None,
            "input_artifacts": [],
            "output_artifacts": [],
        },
        "plan": _scrub_json(plan.to_dict()),
        "errors": _scrub_json(errors or []),
        "key_present": bool(config.api_key()),
    }


def write_artifact(project_root: str | Path, run_id: str, payload: dict, artifact_dir: str | Path) -> Path:
    directory = Path(project_root) / artifact_dir
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{run_id}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
