"""Execution contracts only: this module classifies, it does not schedule."""

from __future__ import annotations

import unittest

from agent_router.execution import (
    AcceptanceEvidence,
    FailureKind,
    TaskStatus,
    acceptance_blockers,
    classify_failure,
    transition_status,
)


class TestFailureClassification(unittest.TestCase):
    def test_only_confirmed_transient_failures_are_retryable(self):
        for kind in (FailureKind.RETRYABLE_TRANSPORT, FailureKind.RETRYABLE_RATE_LIMIT):
            with self.subTest(kind=kind):
                self.assertTrue(classify_failure(kind, "temporary").retryable)

    def test_permission_and_configuration_failures_are_non_retryable(self):
        for kind in (
            FailureKind.PERMISSION_DENIED,
            FailureKind.INVALID_CONFIGURATION,
            FailureKind.MISSING_TOOL,
            FailureKind.INVALID_CONTRACT,
            FailureKind.MISSING_INPUT,
            FailureKind.UNKNOWN,
        ):
            with self.subTest(kind=kind):
                self.assertFalse(classify_failure(kind, "cannot retry").retryable)


class TestAcceptanceGate(unittest.TestCase):
    def test_completed_evidence_has_no_blockers(self):
        evidence = AcceptanceEvidence(
            required_workers=("scout", "implementer", "reviewer"),
            worker_statuses={role: TaskStatus.COMPLETED for role in ("scout", "implementer", "reviewer")},
            required_sources_present=True,
            reviewer_verdict="accept",
            required_checks=("unit", "integration"),
            executed_checks=("unit", "integration"),
            result_covers_request=True,
        )
        self.assertEqual(acceptance_blockers(evidence), [])

    def test_incomplete_research_blocks_acceptance(self):
        evidence = AcceptanceEvidence(
            required_workers=("researcher",),
            worker_statuses={"researcher": TaskStatus.BLOCKED},
            required_sources_present=False,
        )
        blockers = acceptance_blockers(evidence)
        self.assertIn("worker_not_completed:researcher", blockers)
        self.assertIn("required_source_missing", blockers)

    def test_reviewer_reject_and_unresolved_high_block_acceptance(self):
        evidence = AcceptanceEvidence(
            reviewer_verdict="reject",
            unresolved_blocker_or_high=True,
            result_covers_request=False,
        )
        blockers = acceptance_blockers(evidence)
        self.assertIn("reviewer_rejected", blockers)
        self.assertIn("unresolved_blocker_or_high", blockers)
        self.assertIn("result_does_not_cover_request", blockers)

    def test_unexecuted_check_blocks_acceptance(self):
        evidence = AcceptanceEvidence(required_checks=("unit", "smoke"), executed_checks=("unit",))
        self.assertEqual(acceptance_blockers(evidence), ["required_check_not_executed:smoke"])


class TestTaskStateTransitions(unittest.TestCase):
    def test_pending_to_running_to_completed(self):
        self.assertEqual(transition_status(TaskStatus.PENDING, TaskStatus.RUNNING), TaskStatus.RUNNING)
        self.assertEqual(transition_status(TaskStatus.RUNNING, TaskStatus.COMPLETED), TaskStatus.COMPLETED)

    def test_pending_can_be_blocked_or_cancelled(self):
        self.assertEqual(transition_status(TaskStatus.PENDING, TaskStatus.BLOCKED), TaskStatus.BLOCKED)
        self.assertEqual(transition_status(TaskStatus.PENDING, TaskStatus.CANCELLED), TaskStatus.CANCELLED)

    def test_terminal_states_cannot_be_retried_implicitly(self):
        for current in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.BLOCKED, TaskStatus.CANCELLED):
            with self.subTest(current=current), self.assertRaises(ValueError):
                transition_status(current, TaskStatus.RUNNING)

    def test_pending_cannot_skip_running_to_completed(self):
        with self.assertRaises(ValueError):
            transition_status(TaskStatus.PENDING, TaskStatus.COMPLETED)


if __name__ == "__main__":
    unittest.main()
