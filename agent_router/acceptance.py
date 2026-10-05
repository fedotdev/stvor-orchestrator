"""Acceptance domain: three separate state machines and a digest-bound decision.

Deliberately not wired into the router yet. The router still emits plans with
``status="blocked"`` and nothing here launches workers. This module exists so a
future scheduler cannot collapse three different facts into one status:

  * ``TaskExecutionStatus.SUCCEEDED`` - a worker finished an attempt.
  * ``CheckStatus.PASSED``           - a command exited 0 on some digest.
  * ``RunAcceptanceStatus.ACCEPTED`` - the run as a whole is done.

Three invariants the rest of the system may rely on:

  * A legacy ``VERDICT:`` line is never read by the acceptance policy. It is
    kept only as a compatibility projection, so the old reviewer prompt cannot
    become a second source of truth.
  * Evidence is bound to a change-set digest. If the workspace moved after the
    evidence was produced, the evidence is STALE and acceptance is refused.
  * Acceptance is a pure function of stored records. It never touches the
    filesystem, runs a command, or repairs code.

ponytail: the criteria/evidence graph is deliberately shallow here (one
criterion label per check). Full fan-out graphs, worktree isolation and restart
recovery are later increments; this contract is shaped so they can be added
without changing ``AcceptanceEngine``.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, replace
from enum import StrEnum
from pathlib import Path
from typing import Iterable, Protocol

# -- state machines ----------------------------------------------------------
# These are three independent machines. Collapsing them is the bug this module
# exists to prevent, so they are separate types and never aliased to each other.


class TaskExecutionStatus(StrEnum):
    """Lifecycle of one worker attempt. Says nothing about checks or acceptance."""

    QUEUED = "queued"
    DISPATCHING = "dispatching"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    TIMED_OUT = "timed_out"
    CANCEL_REQUESTED = "cancel_requested"
    CANCELLED = "cancelled"
    UNKNOWN = "unknown"


#: Legal edges for one attempt. CANCEL_REQUESTED may still land on SUCCEEDED:
#: the worker can finish before the host confirms the abort.
_TASK_TRANSITIONS: dict[TaskExecutionStatus, frozenset[TaskExecutionStatus]] = {
    TaskExecutionStatus.QUEUED: frozenset({TaskExecutionStatus.DISPATCHING, TaskExecutionStatus.CANCELLED}),
    TaskExecutionStatus.DISPATCHING: frozenset(
        {TaskExecutionStatus.RUNNING, TaskExecutionStatus.FAILED, TaskExecutionStatus.CANCELLED}
    ),
    TaskExecutionStatus.RUNNING: frozenset(
        {
            TaskExecutionStatus.SUCCEEDED,
            TaskExecutionStatus.FAILED,
            TaskExecutionStatus.TIMED_OUT,
            TaskExecutionStatus.CANCEL_REQUESTED,
            TaskExecutionStatus.UNKNOWN,
        }
    ),
    TaskExecutionStatus.CANCEL_REQUESTED: frozenset(
        {
            TaskExecutionStatus.CANCELLED,
            TaskExecutionStatus.SUCCEEDED,
            TaskExecutionStatus.FAILED,
            TaskExecutionStatus.UNKNOWN,
        }
    ),
    TaskExecutionStatus.SUCCEEDED: frozenset(),
    TaskExecutionStatus.FAILED: frozenset(),
    TaskExecutionStatus.TIMED_OUT: frozenset(),
    TaskExecutionStatus.CANCELLED: frozenset(),
    TaskExecutionStatus.UNKNOWN: frozenset(),
}

#: An attempt in one of these states has no confirmed outcome. Acceptance is
#: refused while any is present, and UNKNOWN outranks every other run status.
_INDETERMINATE_ATTEMPT = frozenset(
    {TaskExecutionStatus.UNKNOWN, TaskExecutionStatus.CANCEL_REQUESTED}
)
#: Still legitimately in flight.
_IN_FLIGHT_ATTEMPT = frozenset(
    {
        TaskExecutionStatus.QUEUED,
        TaskExecutionStatus.DISPATCHING,
        TaskExecutionStatus.RUNNING,
    }
)


class CheckStatus(StrEnum):
    """Lifecycle of one verification command. STALE is not in the original
    vocabulary: it is assigned by comparison against the run digest, never by
    the runner itself."""

    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    ERROR = "error"
    SKIPPED = "skipped"
    STALE = "stale"


class RunAcceptanceStatus(StrEnum):
    """Outcome of the run as a whole. Never set implicitly by a worker status."""

    CREATED = "created"
    RUNNING = "running"
    AWAITING_APPROVAL = "awaiting_approval"
    VERIFYING = "verifying"
    REVISE = "revise"
    ACCEPTED = "accepted"
    BLOCKED = "blocked"
    UNKNOWN = "unknown"
    CANCELLED = "cancelled"


class FindingSeverity(StrEnum):
    BLOCKER = "blocker"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


#: Only these severities can stop acceptance while they remain unresolved.
_BLOCKING_SEVERITIES = frozenset({FindingSeverity.BLOCKER, FindingSeverity.HIGH})


class ReviewStatus(StrEnum):
    VALID = "valid"
    INVALID = "invalid"


# -- blocker codes ----------------------------------------------------------
# Stable strings: they end up in the ledger and in human reports.


class BlockerCode(StrEnum):
    TASK_NOT_SUCCEEDED = "REQUIRED_TASK_NOT_SUCCEEDED"
    TASK_OUTCOME_INDETERMINATE = "TASK_OUTCOME_INDETERMINATE"
    REQUIRED_TASK_MISSING = "REQUIRED_TASK_MISSING"
    REQUIRED_CHECK_MISSING = "REQUIRED_CHECK_MISSING"
    REQUIRED_CHECK_NOT_PASSED = "REQUIRED_CHECK_NOT_PASSED"
    REQUIRED_CHECK_STALE = "REQUIRED_CHECK_STALE"
    REQUIRED_REVIEW_MISSING = "REQUIRED_REVIEW_MISSING"
    REQUIRED_REVIEW_EVIDENCE_MISSING = "REQUIRED_REVIEW_EVIDENCE_MISSING"
    REQUIRED_REVIEW_STALE = "REQUIRED_REVIEW_STALE"
    UNRESOLVED_BLOCKING_FINDING = "UNRESOLVED_BLOCKING_FINDING"
    APPROVAL_MISSING = "APPROVAL_MISSING"
    APPROVAL_STALE = "APPROVAL_STALE"
    APPROVAL_DENIED = "APPROVAL_DENIED"
    PATH_OUTSIDE_SCOPE = "PATH_OUTSIDE_SCOPE"
    SOURCE_MISSING = "SOURCE_MISSING"
    RESULT_DOES_NOT_COVER_REQUEST = "RESULT_DOES_NOT_COVER_REQUEST"


@dataclass(frozen=True)
class Blocker:
    code: BlockerCode
    detail: str = ""
    evidence_id: str | None = None


# -- change-set digest ------------------------------------------------------


@dataclass(frozen=True)
class FileEntry:
    path: str
    status: str
    content_sha256: str


def content_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_entry(root: Path, rel_path: str, status: str = "modified") -> FileEntry:
    """Hash one file under ``root``.

    ``status`` is passed in rather than inferred from git, so untracked files
    reach the manifest: a plain ``git diff`` would miss them entirely.
    """
    digest = content_sha256(Path(root, rel_path).read_bytes())
    return FileEntry(path=rel_path.replace("\\", "/"), status=status, content_sha256=digest)


def build_manifest(base_revision: str | None, entries: Iterable[FileEntry]) -> dict:
    """Canonical manifest: paths sorted, hashes explicit, no timestamps."""
    return {
        "base_revision": base_revision,
        "files": [
            {"path": e.path, "status": e.status, "content_sha256": e.content_sha256}
            for e in sorted(entries, key=lambda e: e.path)
        ],
    }


def change_set_digest(manifest: dict) -> str:
    """SHA-256 over the canonical serialisation. Stable across key order and
    insertion order of ``files``."""
    payload = {
        "base_revision": manifest.get("base_revision"),
        "files": sorted(
            manifest.get("files", []),
            key=lambda f: f["path"],
        ),
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return "sha256:" + hashlib.sha256(blob.encode("utf-8")).hexdigest()


# -- evidence ---------------------------------------------------------------


@dataclass(frozen=True)
class TaskAttempt:
    """One attempt at one task. ``required`` means acceptance waits for it."""

    task_id: str
    role: str = ""
    status: TaskExecutionStatus = TaskExecutionStatus.QUEUED
    attempt: int = 1
    required: bool = True
    idempotency_key: str = ""


@dataclass(frozen=True)
class CheckResult:
    """Result of one verification command, bound to the digest it observed."""

    evidence_id: str
    subject_digest: str
    status: CheckStatus
    producer: str = "test_runner"
    created_at: str = ""
    command: str = ""
    exit_code: int | None = None
    profile: str = ""
    criterion: str | None = None


@dataclass(frozen=True)
class Finding:
    """A review finding. Every field is required so a review cannot degrade into
    an unstructured verdict line."""

    finding_id: str
    severity: FindingSeverity
    summary: str
    evidence: str = ""
    impact: str = ""
    required_fix: str = ""
    verification: str = ""
    resolved: bool = False


@dataclass(frozen=True)
class ReviewEvidence:
    """Reviewer output as evidence.

    ``legacy_verdict`` is stored for compatibility with the current reviewer
    prompt and is deliberately *not* consulted by :func:`evaluate_acceptance`.
    A review with no structured findings is INVALID, whatever the verdict says.
    """

    evidence_id: str
    subject_digest: str
    status: ReviewStatus = ReviewStatus.VALID
    producer: str = "reviewer"
    created_at: str = ""
    findings: tuple[Finding, ...] = ()
    legacy_verdict: str | None = None

    def unresolved(self) -> tuple[Finding, ...]:
        return tuple(f for f in self.findings if not f.resolved)


@dataclass(frozen=True)
class ApprovalRecord:
    approval_id: str
    granted: bool
    subject_digest: str


@dataclass(frozen=True)
class AcceptanceDecision:
    """The only shape acceptance returns. Not a bool, not a verdict string."""

    status: RunAcceptanceStatus
    current_digest: str
    blockers: tuple[Blocker, ...] = ()
    criteria_coverage: dict[str, tuple[str, ...]] = field(default_factory=dict)
    accepted_evidence: tuple[str, ...] = ()

    @property
    def accepted(self) -> bool:
        return self.status is RunAcceptanceStatus.ACCEPTED

    def blocker_codes(self) -> tuple[str, ...]:
        return tuple(b.code.value for b in self.blockers)


@dataclass(frozen=True)
class AcceptanceRequest:
    """Everything acceptance is allowed to look at. No filesystem access."""

    current_digest: str
    attempts: tuple[TaskAttempt, ...] = ()
    required_checks: tuple[str, ...] = ()
    checks: tuple[CheckResult, ...] = ()
    review: ReviewEvidence | None = None
    review_required: bool = True
    approval_required: bool = False
    approval: ApprovalRecord | None = None
    scope_paths: tuple[str, ...] | None = None
    forbidden_paths: tuple[str, ...] = ()
    changed_paths: tuple[str, ...] = ()
    required_sources_present: bool = True
    result_covers_request: bool = True


# -- the engine -------------------------------------------------------------


def _effective_check_status(check: CheckResult, current_digest: str) -> CheckStatus:
    """A check that passed on an older digest is not evidence about this one."""
    if check.subject_digest != current_digest:
        return CheckStatus.STALE
    return check.status


def _criteria_coverage(checks: Iterable[CheckResult], current_digest: str) -> dict[str, tuple[str, ...]]:
    coverage: dict[str, list[str]] = {}
    for check in checks:
        if check.criterion and _effective_check_status(check, current_digest) is CheckStatus.PASSED:
            coverage.setdefault(check.criterion, []).append(check.evidence_id)
    return {k: tuple(sorted(v)) for k, v in coverage.items()}


def evaluate_acceptance(request: AcceptanceRequest) -> AcceptanceDecision:
    """Pure function. Returns a decision; never mutates and never repairs.

    Order of precedence is deliberate and is what the tests pin:

    1. indeterminate attempt  -> UNKNOWN (we cannot know if a side effect ran)
    2. attempts in flight     -> RUNNING (not a failure, just not finished)
    3. approval pending       -> AWAITING_APPROVAL
    4. everything else        -> BLOCKED, or REVISE when the only thing left is
                                 a fixable review finding
    """
    digest = request.current_digest
    blockers: list[Blocker] = []

    required = [a for a in request.attempts if a.required]

    # -- 1. indeterminate outcome outranks everything ------------------------
    for attempt in request.attempts:
        if attempt.status in _INDETERMINATE_ATTEMPT:
            return AcceptanceDecision(
                status=RunAcceptanceStatus.UNKNOWN,
                current_digest=digest,
                blockers=(
                    Blocker(
                        BlockerCode.TASK_OUTCOME_INDETERMINATE,
                        detail=f"{attempt.task_id} is {attempt.status.value}",
                        evidence_id=None,
                    ),
                ),
            )

    # -- 2. still in flight -------------------------------------------------
    in_flight = [a for a in request.attempts if a.status in _IN_FLIGHT_ATTEMPT]
    if in_flight:
        detail = "; ".join(f"{a.task_id} is {a.status.value}" for a in in_flight)
        return AcceptanceDecision(
            status=RunAcceptanceStatus.RUNNING,
            current_digest=digest,
            blockers=(Blocker(BlockerCode.TASK_NOT_SUCCEEDED, detail),),
        )

    # -- 3. approval --------------------------------------------------------
    if request.approval_required:
        if request.approval is None:
            return AcceptanceDecision(
                status=RunAcceptanceStatus.AWAITING_APPROVAL,
                current_digest=digest,
                blockers=(Blocker(BlockerCode.APPROVAL_MISSING, "high-risk run requires an approval"),),
            )
        if not request.approval.granted:
            return AcceptanceDecision(
                status=RunAcceptanceStatus.BLOCKED,
                current_digest=digest,
                blockers=(
                    Blocker(BlockerCode.APPROVAL_DENIED, "", request.approval.approval_id),
                ),
            )
        if request.approval.subject_digest != digest:
            return AcceptanceDecision(
                status=RunAcceptanceStatus.BLOCKED,
                current_digest=digest,
                blockers=(
                    Blocker(
                        BlockerCode.APPROVAL_STALE,
                        f"approved {request.approval.subject_digest}, current {digest}",
                        request.approval.approval_id,
                    ),
                ),
            )

    # -- 4. remaining gates -------------------------------------------------
    if not request.required_sources_present:
        blockers.append(Blocker(BlockerCode.SOURCE_MISSING, "a required source was not available"))
    if not request.result_covers_request:
        blockers.append(
            Blocker(BlockerCode.RESULT_DOES_NOT_COVER_REQUEST, "result does not cover the request")
        )

    for attempt in required:
        if attempt.status is not TaskExecutionStatus.SUCCEEDED:
            blockers.append(
                Blocker(
                    BlockerCode.TASK_NOT_SUCCEEDED,
                    f"{attempt.task_id} is {attempt.status.value}",
                )
            )

    checks_by_id = {c.evidence_id: c for c in request.checks}
    for name in request.required_checks:
        check = checks_by_id.get(name)
        if check is None:
            blockers.append(Blocker(BlockerCode.REQUIRED_CHECK_MISSING, name, name))
            continue
        effective = _effective_check_status(check, digest)
        if effective is CheckStatus.STALE:
            blockers.append(
                Blocker(
                    BlockerCode.REQUIRED_CHECK_STALE,
                    f"{name} observed {check.subject_digest}, current {digest}",
                    name,
                )
            )
        elif effective is not CheckStatus.PASSED:
            blockers.append(
                Blocker(BlockerCode.REQUIRED_CHECK_NOT_PASSED, f"{name} is {effective.value}", name)
            )

    review = request.review
    if request.review_required:
        if review is None:
            blockers.append(
                Blocker(BlockerCode.REQUIRED_REVIEW_MISSING, "a review is required for this run")
            )
        elif review.status is not ReviewStatus.VALID or not review.findings:
            # A legacy `VERDICT:` line without findings lands here. It does not
            # reject the run directly; it makes the mandatory review unusable,
            # which blocks acceptance by a different route.
            blockers.append(
                Blocker(
                    BlockerCode.REQUIRED_REVIEW_EVIDENCE_MISSING,
                    f"review {review.evidence_id} carries no structured findings",
                    review.evidence_id,
                )
            )
        elif review.subject_digest != digest:
            blockers.append(
                Blocker(
                    BlockerCode.REQUIRED_REVIEW_STALE,
                    f"review observed {review.subject_digest}, current {digest}",
                    review.evidence_id,
                )
            )
        else:
            for finding in review.unresolved():
                if finding.severity in _BLOCKING_SEVERITIES:
                    blockers.append(
                        Blocker(
                            BlockerCode.UNRESOLVED_BLOCKING_FINDING,
                            f"{finding.finding_id} ({finding.severity.value}): {finding.summary}",
                            review.evidence_id,
                        )
                    )

    for path in _paths_outside_scope(request):
        blockers.append(Blocker(BlockerCode.PATH_OUTSIDE_SCOPE, path))

    coverage = _criteria_coverage(request.checks, digest)
    if blockers:
        only_review_findings = all(
            b.code is BlockerCode.UNRESOLVED_BLOCKING_FINDING for b in blockers
        )
        status = RunAcceptanceStatus.REVISE if only_review_findings else RunAcceptanceStatus.BLOCKED
        return AcceptanceDecision(
            status=status,
            current_digest=digest,
            blockers=tuple(blockers),
            criteria_coverage=coverage,
            accepted_evidence=(),
        )

    accepted = tuple(
        sorted(
            [c.evidence_id for c in request.checks if c.subject_digest == digest]
            + ([review.evidence_id] if review is not None and review.subject_digest == digest else [])
        )
    )
    return AcceptanceDecision(
        status=RunAcceptanceStatus.ACCEPTED,
        current_digest=digest,
        blockers=(),
        criteria_coverage=coverage,
        accepted_evidence=accepted,
    )


def _paths_outside_scope(request: AcceptanceRequest) -> tuple[str, ...]:
    forbidden = {p.replace("\\", "/") for p in request.forbidden_paths}
    if request.scope_paths is None:
        return tuple(sorted(p for p in request.changed_paths if p.replace("\\", "/") in forbidden))
    allowed = {p.replace("\\", "/") for p in request.scope_paths}
    return tuple(
        sorted(
            p
            for p in request.changed_paths
            if p.replace("\\", "/") not in allowed and p.replace("\\", "/") not in forbidden
        )
    )


# -- adapter for the previous six-flag contract ----------------------------


def request_from_legacy_flags(
    *,
    current_digest: str,
    required_workers: Iterable[str] = (),
    worker_statuses: dict[str, str] | None = None,
    required_sources_present: bool = True,
    reviewer_verdict: str | None = None,
    unresolved_blocker_or_high: bool = False,
    required_checks: Iterable[str] = (),
    executed_checks: Iterable[str] = (),
    result_covers_request: bool = True,
    review_required: bool = True,
) -> AcceptanceRequest:
    """Bridge the pre-increment ``acceptance_blockers`` flags into this contract.

    ``reviewer_verdict`` is accepted for call compatibility and then discarded:
    the old contract used it as a gate, and keeping that behaviour is exactly the
    second source of truth this increment removes. A caller that has only a
    verdict produces a run with no review evidence, which blocks on
    ``REQUIRED_REVIEW_MISSING`` / ``REQUIRED_REVIEW_EVIDENCE_MISSING``.
    """
    statuses = {
        role: TaskExecutionStatus(value)
        for role, value in (worker_statuses or {}).items()
        if value in {s.value for s in TaskExecutionStatus}
    }
    attempts = tuple(
        TaskAttempt(task_id=f"legacy:{role}", role=role, status=statuses.get(role, TaskExecutionStatus.QUEUED))
        for role in required_workers
    )
    executed = set(executed_checks)
    checks = tuple(
        CheckResult(
            evidence_id=name,
            subject_digest=current_digest,
            status=CheckStatus.PASSED if name in executed else CheckStatus.PENDING,
        )
        for name in required_checks
    )
    # A verdict with no findings cannot become valid evidence, so the legacy
    # path never produces a review object at all.
    review = None
    if unresolved_blocker_or_high:
        review = replace(
            review or ReviewEvidence(evidence_id="legacy:review", subject_digest=current_digest),
            status=ReviewStatus.VALID,
            findings=(
                Finding(
                    finding_id="legacy:finding",
                    severity=FindingSeverity.HIGH,
                    summary="unresolved blocker/high reported by legacy flags",
                    resolved=False,
                ),
            ),
        )
        review_required = True
    _ = reviewer_verdict  # deliberately unused
    return AcceptanceRequest(
        current_digest=current_digest,
        attempts=attempts,
        required_checks=tuple(required_checks),
        checks=checks,
        review=review,
        review_required=review_required,
        required_sources_present=required_sources_present,
        result_covers_request=result_covers_request,
    )


# -- TestRunner port --------------------------------------------------------


class TestRunner(Protocol):
    """Port for verification execution.

    Transitional: the implementer capability profile still exposes ad-hoc test
    commands as ``ask``, which yields no machine evidence and is not
    reproducible. Until named profiles exist, the production implementation
    refuses rather than pretending a check ran.
    """

    def run(self, profile: str, subject_digest: str) -> CheckResult: ...


class NotImplementedTestRunner:
    """Production placeholder until named profiles are extracted from project
    config. Returns an ERROR check, never PASSED."""

    def run(self, profile: str, subject_digest: str) -> CheckResult:
        return CheckResult(
            evidence_id=f"check:not_implemented:{profile}",
            subject_digest=subject_digest,
            status=CheckStatus.ERROR,
            producer="not_implemented",
            command=profile,
            profile=profile,
        )


class FakeTestRunner:
    """Test double: returns scripted results and records what it was asked."""

    def __init__(self, results: dict[str, CheckResult] | None = None, default: CheckStatus = CheckStatus.PASSED):
        self._results = dict(results or {})
        self._default = default
        self.calls: list[tuple[str, str]] = []

    def run(self, profile: str, subject_digest: str) -> CheckResult:
        self.calls.append((profile, subject_digest))
        if profile in self._results:
            return self._results[profile]
        return CheckResult(
            evidence_id=f"check:{profile}",
            subject_digest=subject_digest,
            status=self._default,
            command=profile,
            profile=profile,
        )


# -- minimal RunStore -------------------------------------------------------


class RunStore:
    """Authoritative state for runs: ``run.json`` plus an append-only event log.

    Markdown projections are out of scope here. ``events.jsonl`` is append-only
    and never rewritten, so a restart can replay the ledger.
    """

    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def run_dir(self, run_id: str) -> Path:
        path = self.root / run_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def save_run(self, run_id: str, record: dict) -> Path:
        target = self.run_dir(run_id) / "run.json"
        target.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        return target

    def load_run(self, run_id: str) -> dict:
        return json.loads((self.run_dir(run_id) / "run.json").read_text(encoding="utf-8"))

    def append_event(self, run_id: str, event: dict) -> Path:
        target = self.run_dir(run_id) / "events.jsonl"
        with target.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")
        return target

    def read_events(self, run_id: str) -> list[dict]:
        target = self.run_dir(run_id) / "events.jsonl"
        if not target.exists():
            return []
        return [json.loads(line) for line in target.read_text(encoding="utf-8").splitlines() if line.strip()]


def next_status(current: TaskExecutionStatus, target: TaskExecutionStatus) -> TaskExecutionStatus:
    """Validate an attempt transition. No implicit retry, no implicit recovery."""
    if target not in _TASK_TRANSITIONS[current]:
        raise ValueError(f"invalid attempt transition: {current.value} -> {target.value}")
    return target
