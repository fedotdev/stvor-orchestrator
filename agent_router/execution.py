"""Execution-state vocabulary and failure classification; no scheduler or retries.

The router currently emits plans but does not launch workers. This module keeps
status transitions and retryability explicit for a future real executor without
pretending that execution has happened here.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class TaskStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"
    CANCELLED = "cancelled"


class FailureKind(StrEnum):
    RETRYABLE_TRANSPORT = "retryable_transport"
    RETRYABLE_RATE_LIMIT = "retryable_rate_limit"
    PERMISSION_DENIED = "permission_denied"
    INVALID_CONFIGURATION = "invalid_configuration"
    MISSING_TOOL = "missing_tool"
    INVALID_CONTRACT = "invalid_contract"
    MISSING_INPUT = "missing_input"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class Failure:
    kind: FailureKind
    message: str
    retryable: bool


def classify_failure(kind: FailureKind, message: str) -> Failure:
    """Classify one failure; retry ownership belongs to the caller."""
    retryable = kind in {
        FailureKind.RETRYABLE_TRANSPORT,
        FailureKind.RETRYABLE_RATE_LIMIT,
    }
    return Failure(kind=kind, message=message, retryable=retryable)


@dataclass(frozen=True)
class AcceptanceEvidence:
    required_workers: tuple[str, ...] = ()
    worker_statuses: dict[str, TaskStatus] | None = None
    required_sources_present: bool = True
    reviewer_verdict: str | None = None
    unresolved_blocker_or_high: bool = False
    required_checks: tuple[str, ...] = ()
    executed_checks: tuple[str, ...] = ()
    result_covers_request: bool = True


def acceptance_blockers(evidence: AcceptanceEvidence) -> list[str]:
    """Return blockers; empty means only that these supplied gates are met."""
    statuses = evidence.worker_statuses or {}
    blockers = [
        f"worker_not_completed:{role}"
        for role in evidence.required_workers
        if statuses.get(role) != TaskStatus.COMPLETED
    ]
    if not evidence.required_sources_present:
        blockers.append("required_source_missing")
    if evidence.reviewer_verdict == "reject":
        blockers.append("reviewer_rejected")
    elif evidence.reviewer_verdict not in (None, "accept", "accept_with_followups"):
        blockers.append("reviewer_verdict_invalid")
    if evidence.unresolved_blocker_or_high:
        blockers.append("unresolved_blocker_or_high")
    missing_checks = set(evidence.required_checks) - set(evidence.executed_checks)
    blockers.extend(f"required_check_not_executed:{name}" for name in sorted(missing_checks))
    if not evidence.result_covers_request:
        blockers.append("result_does_not_cover_request")
    return blockers


def transition_status(current: TaskStatus, target: TaskStatus) -> TaskStatus:
    """Validate a small terminal-state machine; no implicit retry/restart."""
    allowed = {
        TaskStatus.PENDING: {TaskStatus.RUNNING, TaskStatus.BLOCKED, TaskStatus.CANCELLED},
        TaskStatus.RUNNING: {
            TaskStatus.COMPLETED,
            TaskStatus.FAILED,
            TaskStatus.BLOCKED,
            TaskStatus.CANCELLED,
        },
        TaskStatus.COMPLETED: set(),
        TaskStatus.FAILED: set(),
        TaskStatus.BLOCKED: set(),
        TaskStatus.CANCELLED: set(),
    }
    if target not in allowed[current]:
        raise ValueError(f"invalid task status transition: {current.value} -> {target.value}")
    return target
