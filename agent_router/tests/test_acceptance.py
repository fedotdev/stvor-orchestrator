"""The twelve required scenarios, plus the digest and store contracts.

The completion criterion for this increment is that no single success signal can
accept a run: not an exited worker, not a passing test, not a reviewer's
"accept". These tests pin that, and pin the digest rule that makes old evidence
refuse to speak about a changed workspace.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from agent_router.acceptance import (
    AcceptanceDecision,
    AcceptanceRequest,
    ApprovalRecord,
    BlockerCode,
    CheckResult,
    CheckStatus,
    Finding,
    FindingSeverity,
    FileEntry,
    ReviewEvidence,
    ReviewStatus,
    RunAcceptanceStatus,
    RunStore,
    TaskAttempt,
    TaskExecutionStatus,
    build_manifest,
    change_set_digest,
    evaluate_acceptance,
    file_entry,
    next_status,
    request_from_legacy_flags,
    FakeTestRunner,
    NotImplementedTestRunner,
)

DIGEST_A = "sha256:" + "a" * 64
DIGEST_B = "sha256:" + "b" * 64


def succeeded(task_id: str = "t1", role: str = "implementer") -> TaskAttempt:
    return TaskAttempt(task_id=task_id, role=role, status=TaskExecutionStatus.SUCCEEDED)


def passed(evidence_id: str = "ev_test_01", subject_digest: str = DIGEST_A, **kw) -> CheckResult:
    return CheckResult(
        evidence_id=evidence_id,
        subject_digest=subject_digest,
        status=CheckStatus.PASSED,
        exit_code=0,
        command="pytest -q",
        **kw,
    )


def clean_review(evidence_id: str = "ev_rev_01", subject_digest: str = DIGEST_A, **kw) -> ReviewEvidence:
    findings = kw.pop("findings", (Finding("f0", FindingSeverity.LOW, "nit", resolved=True),))
    return ReviewEvidence(
        evidence_id=evidence_id,
        subject_digest=subject_digest,
        status=kw.pop("status", ReviewStatus.VALID),
        findings=findings,
        **kw,
    )


def green(**overrides) -> AcceptanceRequest:
    """A request that satisfies every gate; individual tests break one gate."""
    base = dict(
        current_digest=DIGEST_A,
        attempts=(succeeded(),),
        required_checks=("ev_test_01",),
        checks=(passed(),),
        review=clean_review(),
    )
    base.update(overrides)
    return AcceptanceRequest(**base)


class TestScenario01FailedCheckBlocks(unittest.TestCase):
    def test_succeeded_attempt_plus_failed_check_is_blocked(self):
        check = CheckResult(
            evidence_id="ev_test_01",
            subject_digest=DIGEST_A,
            status=CheckStatus.FAILED,
            exit_code=1,
        )
        decision = evaluate_acceptance(green(checks=(check,)))
        self.assertIs(decision.status, RunAcceptanceStatus.BLOCKED)
        self.assertIn(BlockerCode.REQUIRED_CHECK_NOT_PASSED.value, decision.blocker_codes())


class TestScenario02PassingChecksAreNotAcceptance(unittest.TestCase):
    def test_all_checks_passed_without_review_is_not_accepted(self):
        decision = evaluate_acceptance(green(review=None))
        self.assertIsNot(decision.status, RunAcceptanceStatus.ACCEPTED)
        self.assertIn(BlockerCode.REQUIRED_REVIEW_MISSING.value, decision.blocker_codes())

    def test_green_request_is_accepted(self):
        self.assertIs(evaluate_acceptance(green()).status, RunAcceptanceStatus.ACCEPTED)


class TestScenario03StaleCheck(unittest.TestCase):
    def test_passed_check_from_older_digest_becomes_stale(self):
        decision = evaluate_acceptance(green(checks=(passed(subject_digest=DIGEST_B),)))
        self.assertIsNot(decision.status, RunAcceptanceStatus.ACCEPTED)
        self.assertIn(BlockerCode.REQUIRED_CHECK_STALE.value, decision.blocker_codes())


class TestScenario04StaleReview(unittest.TestCase):
    def test_review_from_older_digest_becomes_stale(self):
        decision = evaluate_acceptance(green(review=clean_review(subject_digest=DIGEST_B)))
        self.assertIsNot(decision.status, RunAcceptanceStatus.ACCEPTED)
        self.assertIn(BlockerCode.REQUIRED_REVIEW_STALE.value, decision.blocker_codes())


class TestScenario05LegacyVerdictIsNotAGate(unittest.TestCase):
    def test_reject_without_findings_is_invalid_review_evidence(self):
        legacy = ReviewEvidence(
            evidence_id="ev_rev_01",
            subject_digest=DIGEST_A,
            status=ReviewStatus.INVALID,
            findings=(),
            legacy_verdict="reject",
        )
        decision = evaluate_acceptance(green(review=legacy))
        self.assertIs(decision.status, RunAcceptanceStatus.BLOCKED)
        self.assertIn(BlockerCode.REQUIRED_REVIEW_EVIDENCE_MISSING.value, decision.blocker_codes())
        # Crucially it is not rejected because it said "reject".
        self.assertNotIn(BlockerCode.UNRESOLVED_BLOCKING_FINDING.value, decision.blocker_codes())

    def test_accept_without_findings_is_also_unusable(self):
        legacy = ReviewEvidence(
            evidence_id="ev_rev_01",
            subject_digest=DIGEST_A,
            status=ReviewStatus.INVALID,
            findings=(),
            legacy_verdict="accept",
        )
        decision = evaluate_acceptance(green(review=legacy))
        self.assertIn(BlockerCode.REQUIRED_REVIEW_EVIDENCE_MISSING.value, decision.blocker_codes())

    def test_legacy_adapter_discards_the_verdict_entirely(self):
        request = request_from_legacy_flags(
            current_digest=DIGEST_A,
            required_workers=("implementer",),
            worker_statuses={"implementer": "succeeded"},
            required_checks=("ev_test_01",),
            executed_checks=("ev_test_01",),
            reviewer_verdict="accept",
        )
        decision = evaluate_acceptance(request)
        self.assertIsNot(decision.status, RunAcceptanceStatus.ACCEPTED)
        self.assertIsNone(request.review)


class TestScenario06UnresolvedFinding(unittest.TestCase):
    def test_unresolved_blocker_yields_revise(self):
        review = clean_review(
            findings=(
                Finding("f1", FindingSeverity.BLOCKER, "breaks the public API", required_fix="restore"),
            )
        )
        decision = evaluate_acceptance(green(review=review))
        self.assertIs(decision.status, RunAcceptanceStatus.REVISE)
        self.assertIn(BlockerCode.UNRESOLVED_BLOCKING_FINDING.value, decision.blocker_codes())

    def test_unresolved_high_yields_revise(self):
        review = clean_review(findings=(Finding("f1", FindingSeverity.HIGH, "races on retry"),))
        decision = evaluate_acceptance(green(review=review))
        self.assertIs(decision.status, RunAcceptanceStatus.REVISE)

    def test_review_failure_is_blocked_not_revise(self):
        review = clean_review(
            status=ReviewStatus.INVALID,
            findings=(Finding("f1", FindingSeverity.BLOCKER, "x"),),
        )
        decision = evaluate_acceptance(green(review=review))
        self.assertIs(decision.status, RunAcceptanceStatus.BLOCKED)


class TestScenario07ResolvedFinding(unittest.TestCase):
    def test_resolved_blocker_does_not_prevent_acceptance(self):
        review = clean_review(
            findings=(Finding("f1", FindingSeverity.BLOCKER, "was a bug", resolved=True),),
        )
        decision = evaluate_acceptance(green(review=review))
        self.assertIs(decision.status, RunAcceptanceStatus.ACCEPTED)
        self.assertEqual(decision.blockers, ())

    def test_unresolved_medium_does_not_prevent_acceptance(self):
        review = clean_review(findings=(Finding("f1", FindingSeverity.MEDIUM, "nit"),))
        self.assertIs(evaluate_acceptance(green(review=review)).status, RunAcceptanceStatus.ACCEPTED)


class TestScenario08Approval(unittest.TestCase):
    def test_missing_approval_awaits_approval(self):
        decision = evaluate_acceptance(green(approval_required=True, approval=None))
        self.assertIs(decision.status, RunAcceptanceStatus.AWAITING_APPROVAL)

    def test_denied_approval_blocks(self):
        approval = ApprovalRecord("ap_1", granted=False, subject_digest=DIGEST_A)
        decision = evaluate_acceptance(green(approval_required=True, approval=approval))
        self.assertIs(decision.status, RunAcceptanceStatus.BLOCKED)
        self.assertIn(BlockerCode.APPROVAL_DENIED.value, decision.blocker_codes())

    def test_approval_from_older_digest_blocks(self):
        approval = ApprovalRecord("ap_1", granted=True, subject_digest=DIGEST_B)
        decision = evaluate_acceptance(green(approval_required=True, approval=approval))
        self.assertIn(BlockerCode.APPROVAL_STALE.value, decision.blocker_codes())

    def test_granted_current_approval_accepts(self):
        approval = ApprovalRecord("ap_1", granted=True, subject_digest=DIGEST_A)
        decision = evaluate_acceptance(green(approval_required=True, approval=approval))
        self.assertIs(decision.status, RunAcceptanceStatus.ACCEPTED)


class TestScenario09UnknownAttempt(unittest.TestCase):
    def test_unknown_attempt_makes_run_unknown(self):
        attempt = TaskAttempt("t1", "implementer", TaskExecutionStatus.UNKNOWN)
        decision = evaluate_acceptance(green(attempts=(attempt,)))
        self.assertIs(decision.status, RunAcceptanceStatus.UNKNOWN)
        self.assertIn(BlockerCode.TASK_OUTCOME_INDETERMINATE.value, decision.blocker_codes())

    def test_cancel_requested_attempt_is_also_indeterminate(self):
        attempt = TaskAttempt("t1", "implementer", TaskExecutionStatus.CANCEL_REQUESTED)
        decision = evaluate_acceptance(green(attempts=(attempt,)))
        self.assertIs(decision.status, RunAcceptanceStatus.UNKNOWN)

    def test_unknown_outranks_a_pending_approval(self):
        attempt = TaskAttempt("t1", "implementer", TaskExecutionStatus.UNKNOWN)
        decision = evaluate_acceptance(green(attempts=(attempt,), approval_required=True))
        self.assertIs(decision.status, RunAcceptanceStatus.UNKNOWN)

    def test_in_flight_attempt_yields_running_not_failure(self):
        attempt = TaskAttempt("t1", "implementer", TaskExecutionStatus.RUNNING)
        self.assertIs(evaluate_acceptance(green(attempts=(attempt,))).status, RunAcceptanceStatus.RUNNING)


class TestScenario10WorkspaceMovedAfterChecks(unittest.TestCase):
    def test_same_evidence_against_a_new_digest_is_refused(self):
        accepted = evaluate_acceptance(green())
        self.assertIs(accepted.status, RunAcceptanceStatus.ACCEPTED)
        # The workspace moves; nothing about the evidence changed.
        after = evaluate_acceptance(green(current_digest=DIGEST_B))
        self.assertIsNot(after.status, RunAcceptanceStatus.ACCEPTED)
        codes = after.blocker_codes()
        self.assertIn(BlockerCode.REQUIRED_CHECK_STALE.value, codes)
        self.assertIn(BlockerCode.REQUIRED_REVIEW_STALE.value, codes)


class TestScenario11DigestCoversUntrackedFiles(unittest.TestCase):
    def test_untracked_file_changes_the_digest(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            Path(root, "src").mkdir()
            Path(root, "src", "a.py").write_text("x = 1\n", encoding="utf-8")
            tracked = file_entry(root, "src/a.py", "modified")
            before = change_set_digest(build_manifest("rev1", [tracked]))

            Path(root, "secret_leak.py").write_text("TOKEN = 'oops'\n", encoding="utf-8")
            untracked = file_entry(root, "secret_leak.py", "untracked")
            after = change_set_digest(build_manifest("rev1", [tracked, untracked]))
        self.assertNotEqual(before, after)

    def test_digest_is_order_independent(self):
        entries = [FileEntry("src/b.py", "modified", "ff"), FileEntry("src/a.py", "modified", "aa")]
        self.assertEqual(
            change_set_digest(build_manifest("rev1", entries)),
            change_set_digest(build_manifest("rev1", list(reversed(entries)))),
        )

    def test_digest_changes_with_base_revision(self):
        entry = FileEntry("src/a.py", "modified", "aa")
        self.assertNotEqual(
            change_set_digest(build_manifest("rev1", [entry])),
            change_set_digest(build_manifest("rev2", [entry])),
        )

    def test_digest_changes_with_content(self):
        self.assertNotEqual(
            change_set_digest(build_manifest("rev1", [FileEntry("p", "modified", "aa")])),
            change_set_digest(build_manifest("rev1", [FileEntry("p", "modified", "bb")])),
        )

    def test_manifest_records_untracked_status(self):
        manifest = build_manifest("rev1", [FileEntry("new.py", "untracked", "aa")])
        self.assertEqual(manifest["files"][0]["status"], "untracked")


class TestScenario12NoSingleSignalAccepts(unittest.TestCase):
    def test_succeeded_attempt_alone_does_not_accept(self):
        decision = evaluate_acceptance(
            AcceptanceRequest(current_digest=DIGEST_A, attempts=(succeeded(),))
        )
        self.assertIsNot(decision.status, RunAcceptanceStatus.ACCEPTED)

    def test_passing_check_alone_does_not_accept(self):
        decision = evaluate_acceptance(
            AcceptanceRequest(current_digest=DIGEST_A, checks=(passed(),))
        )
        self.assertIsNot(decision.status, RunAcceptanceStatus.ACCEPTED)

    def test_succeeded_attempt_with_passing_check_still_does_not_accept(self):
        decision = evaluate_acceptance(
            AcceptanceRequest(
                current_digest=DIGEST_A,
                attempts=(succeeded(),),
                required_checks=("ev_test_01",),
                checks=(passed(),),
            )
        )
        self.assertIsNot(decision.status, RunAcceptanceStatus.ACCEPTED)

    def test_success_signals_never_set_accepted_implicitly(self):
        """Belt and braces: no code path may derive ACCEPTED from a worker state."""
        for status in (
            TaskExecutionStatus.SUCCEEDED,
            TaskExecutionStatus.FAILED,
            TaskExecutionStatus.CANCELLED,
        ):
            attempt = TaskAttempt("t1", "implementer", status)
            decision = evaluate_acceptance(green(attempts=(attempt,), review=None))
            self.assertIsNot(decision.status, RunAcceptanceStatus.ACCEPTED, msg=status.value)


class TestScope(unittest.TestCase):
    def test_path_outside_scope_blocks(self):
        decision = evaluate_acceptance(green(scope_paths=("src/a.py",), changed_paths=("src/b.py",)))
        self.assertIn(BlockerCode.PATH_OUTSIDE_SCOPE.value, decision.blocker_codes())

    def test_forbidden_path_blocks_even_without_scope_list(self):
        decision = evaluate_acceptance(green(forbidden_paths=(".env",), changed_paths=(".env",)))
        self.assertIn(BlockerCode.PATH_OUTSIDE_SCOPE.value, decision.blocker_codes())

    def test_in_scope_paths_pass(self):
        decision = evaluate_acceptance(green(scope_paths=("src/a.py",), changed_paths=("src/a.py",)))
        self.assertIs(decision.status, RunAcceptanceStatus.ACCEPTED)


class TestDecisionShape(unittest.TestCase):
    def test_decision_is_structured_not_a_bool(self):
        decision = evaluate_acceptance(green(review=None))
        self.assertIsInstance(decision, AcceptanceDecision)
        self.assertIsInstance(decision.status, RunAcceptanceStatus)
        self.assertTrue(decision.blockers)
        self.assertFalse(decision.accepted)
        self.assertIsInstance(decision.criteria_coverage, dict)

    def test_accepted_decision_lists_its_evidence(self):
        decision = evaluate_acceptance(green())
        self.assertIn("ev_test_01", decision.accepted_evidence)
        self.assertIn("ev_rev_01", decision.accepted_evidence)

    def test_criteria_coverage_only_counts_current_digest(self):
        check = passed(evidence_id="ev_c1", criterion="ac_01")
        stale = passed(evidence_id="ev_c2", subject_digest=DIGEST_B, criterion="ac_02")
        decision = evaluate_acceptance(green(checks=(check, stale)))
        self.assertEqual(decision.criteria_coverage, {"ac_01": ("ev_c1",)})


class TestAttemptTransitions(unittest.TestCase):
    def test_happy_path(self):
        self.assertIs(
            next_status(
                next_status(next_status(TaskExecutionStatus.QUEUED, TaskExecutionStatus.DISPATCHING),
                            TaskExecutionStatus.RUNNING),
                TaskExecutionStatus.SUCCEEDED),
            TaskExecutionStatus.SUCCEEDED,
        )

    def test_cancel_requested_may_still_land_on_succeeded(self):
        self.assertIs(
            next_status(
                next_status(TaskExecutionStatus.RUNNING, TaskExecutionStatus.CANCEL_REQUESTED),
                TaskExecutionStatus.SUCCEEDED,
            ),
            TaskExecutionStatus.SUCCEEDED,
        )

    def test_cannot_skip_dispatching(self):
        with self.assertRaises(ValueError):
            next_status(TaskExecutionStatus.QUEUED, TaskExecutionStatus.SUCCEEDED)

    def test_terminal_states_are_final(self):
        for terminal in (TaskExecutionStatus.SUCCEEDED, TaskExecutionStatus.FAILED, TaskExecutionStatus.UNKNOWN):
            with self.assertRaises(ValueError):
                next_status(terminal, TaskExecutionStatus.RUNNING)


class TestTestRunnerPort(unittest.TestCase):
    def test_fake_runner_returns_scripted_status(self):
        runner = FakeTestRunner(default=CheckStatus.PASSED)
        result = runner.run("parser", DIGEST_A)
        self.assertIs(result.status, CheckStatus.PASSED)
        self.assertEqual(result.subject_digest, DIGEST_A)
        self.assertEqual(runner.calls, [("parser", DIGEST_A)])

    def test_fake_runner_can_fail_a_named_profile(self):
        runner = FakeTestRunner(
            {"unit": CheckResult("ev_unit", DIGEST_A, CheckStatus.FAILED, exit_code=1)},
            default=CheckStatus.PASSED,
        )
        self.assertIs(runner.run("unit", DIGEST_A).status, CheckStatus.FAILED)

    def test_production_runner_refuses_instead_of_faking_a_pass(self):
        result = NotImplementedTestRunner().run("parser", DIGEST_A)
        self.assertIs(result.status, CheckStatus.ERROR)
        self.assertIsNot(result.status, CheckStatus.PASSED)


class TestRunStore(unittest.TestCase):
    def test_run_roundtrips(self):
        with tempfile.TemporaryDirectory() as root:
            store = RunStore(Path(root))
            store.save_run("run_01", {"run_id": "run_01", "status": "created"})
            self.assertEqual(store.load_run("run_01")["run_id"], "run_01")

    def test_events_are_append_only(self):
        with tempfile.TemporaryDirectory() as root:
            store = RunStore(Path(root))
            store.append_event("run_01", {"event": "run.created", "seq": 1})
            store.append_event("run_01", {"event": "task.queued", "seq": 2})
            events = store.read_events("run_01")
        self.assertEqual([e["seq"] for e in events], [1, 2])
        self.assertEqual([e["event"] for e in events], ["run.created", "task.queued"])

    def test_reading_events_of_an_unknown_run_is_empty(self):
        with tempfile.TemporaryDirectory() as root:
            self.assertEqual(RunStore(Path(root)).read_events("nope"), [])

    def test_events_are_valid_jsonl(self):
        with tempfile.TemporaryDirectory() as root:
            store = RunStore(Path(root))
            store.append_event("run_01", {"event": "run.created"})
            raw = (Path(root) / "run_01" / "events.jsonl").read_text(encoding="utf-8")
        self.assertEqual(len(raw.strip().splitlines()), 1)
        self.assertEqual(json.loads(raw)["event"], "run.created")


if __name__ == "__main__":
    unittest.main()
