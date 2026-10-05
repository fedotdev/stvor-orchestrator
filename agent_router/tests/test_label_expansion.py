"""The expansion may rewrite a declared route label; that must stay visible.

Route accuracy is computed from `expected_route5`, not from the label written in
the case file. Whenever those differ, the route metric can move without any
router change, so the rewrite is recorded per case and counted in the report.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from routing_eval.score import _normalize, load_cases

EVAL_DIR = Path(__file__).resolve().parents[2] / "routing_eval"


def _case(**fields) -> dict:
    """A minimal case with neutral facts, overridable per test."""
    base = {
        "id": "t",
        "language": "en",
        "task": "do something",
        "expected_intent": "feature",
        "expected_scope": "local",
        "expected_needs_write": False,
        "expected_needs_repo_scan": False,
        "expected_needs_external_research": False,
        "expected_needs_tests": False,
        "expected_risk": "low",
        "expected_sensitive_access": False,
        "expected_requires_clarification": False,
        "expected_cross_source": False,
        "expected_route": "manual_orchestrator",
    }
    base.update(fields)
    return base


class TestFactsBeatDeclaredLabel(unittest.TestCase):
    def test_stale_manual_label_is_replaced_by_write_facts(self):
        norm = _normalize(_case(expected_needs_write=True), EVAL_DIR / "synthetic.jsonl")
        self.assertEqual(norm["declared_route"], "manual_orchestrator")
        self.assertEqual(norm["expected_route5"], "orchestrated_write")
        self.assertTrue(norm["route_overridden"])
        self.assertEqual(norm["override_reason"], "expected_needs_write")

    def test_agreement_is_not_reported_as_an_override(self):
        norm = _normalize(_case(expected_needs_write=True, expected_route="orchestrated_write"), EVAL_DIR / "s.jsonl")
        self.assertEqual(norm["expected_route5"], "orchestrated_write")
        self.assertFalse(norm["route_overridden"])

    def test_read_only_two_source_facts_expand_to_orchestrated_readonly(self):
        norm = _normalize(
            _case(expected_needs_repo_scan=True, expected_needs_external_research=True),
            EVAL_DIR / "s.jsonl",
        )
        self.assertEqual(norm["expected_route5"], "orchestrated_readonly")
        self.assertEqual(norm["override_reason"], "expected_repo_scan_and_external_research")

    def test_undecidable_read_only_legacy_label_stays_none(self):
        norm = _normalize(_case(expected_route="orchestrated"), EVAL_DIR / "s.jsonl")
        self.assertIsNone(norm["expected_route5"])
        self.assertEqual(norm["override_reason"], "undecidable_read_only_legacy_label")
        # An absent route is undecidable, not a rewrite of a known label.
        self.assertFalse(norm["route_overridden"])

    def test_case_without_any_declared_label_is_not_an_override(self):
        case = _case()
        del case["expected_route"]
        norm = _normalize(case, EVAL_DIR / "s.jsonl")
        self.assertEqual(norm["declared_route"], "")
        self.assertFalse(norm["route_overridden"])


class TestClarificationWins(unittest.TestCase):
    def test_clarification_outranks_write_facts(self):
        norm = _normalize(
            _case(expected_requires_clarification=True, expected_needs_write=True),
            EVAL_DIR / "s.jsonl",
        )
        self.assertEqual(norm["expected_route5"], "clarification_required")
        self.assertEqual(norm["override_reason"], "expected_requires_clarification")
        self.assertTrue(norm["route_overridden"])

    def test_clarification_is_inferred_from_the_underspecified_category(self):
        norm = _normalize(
            _case(category="underspecified", expected_needs_write=True),
            EVAL_DIR / "s.jsonl",
        )
        self.assertEqual(norm["expected_route5"], "clarification_required")
        self.assertEqual(norm["override_reason"], "expected_requires_clarification")

    def test_sensitive_access_outranks_clarification(self):
        norm = _normalize(
            _case(expected_sensitive_access=True, expected_requires_clarification=True),
            EVAL_DIR / "s.jsonl",
        )
        self.assertEqual(norm["expected_route5"], "manual_orchestrator")
        self.assertEqual(norm["override_reason"], "expected_sensitive_access_or_high_risk")

    def test_high_risk_outranks_write_facts(self):
        norm = _normalize(_case(expected_risk="high", expected_needs_write=True), EVAL_DIR / "s.jsonl")
        self.assertEqual(norm["expected_route5"], "manual_orchestrator")
        self.assertEqual(norm["override_reason"], "expected_sensitive_access_or_high_risk")

    def test_explicit_route5_label_outranks_the_facts(self):
        norm = _normalize(
            _case(expected_needs_write=True, expected_route5="diagnostic_readonly"),
            EVAL_DIR / "s.jsonl",
        )
        self.assertEqual(norm["expected_route5"], "diagnostic_readonly")
        self.assertEqual(norm["override_reason"], "explicit_route5_label")


class TestTwinParity(unittest.TestCase):
    """x22 and c23 are the same task string; they must expand identically."""

    def _by_id(self, filename: str, cid: str) -> dict:
        for case in load_cases(EVAL_DIR / filename):
            if case["id"] == cid:
                return _normalize(case, EVAL_DIR / filename)
        self.fail(f"{cid} not found in {filename}")

    def test_x22_and_c23_expand_to_the_same_route(self):
        x22 = self._by_id("cases_extra.jsonl", "x22")
        c23 = self._by_id("cases.jsonl", "c23")
        self.assertEqual(x22["task"], c23["task"])
        self.assertEqual(x22["expected_route5"], c23["expected_route5"])
        self.assertEqual(x22["expected_route5"], "clarification_required")

    def test_every_override_is_reported_with_a_reason(self):
        for filename in ("cases.jsonl", "cases_extra.jsonl"):
            for case in load_cases(EVAL_DIR / filename):
                norm = _normalize(case, EVAL_DIR / filename)
                if norm["route_overridden"]:
                    self.assertTrue(norm["override_reason"], msg=f"{case['id']} override without reason")
                    self.assertNotEqual(norm["declared_route"], norm["expected_route5"], msg=case["id"])


class TestFrozenTestSetProvenance(unittest.TestCase):
    """The TEST set is frozen: no route label in it may change silently.

    Two kinds of rewrite exist and must not be conflated:
      explicit_route5_label  - a deliberate route5 label in the file. The
                               declared legacy string is just stale wording.
      fact-derived           - the expansion overrode the label from the case
                               facts. This is the one that can move the metric
                               with no label change anyone can see.
    """

    def _overrides(self, filename: str) -> dict[str, str]:
        out = {}
        for case in load_cases(EVAL_DIR / filename):
            norm = _normalize(case, EVAL_DIR / filename)
            if norm["route_overridden"]:
                out[norm["id"]] = norm["override_reason"]
        return out

    def test_test_set_has_no_fact_derived_rewrites(self):
        silent = {
            cid: reason
            for cid, reason in self._overrides("cases_extra.jsonl").items()
            if reason != "explicit_route5_label"
        }
        self.assertEqual(silent, {})

    def test_test_set_explicit_relabels_are_pinned(self):
        self.assertEqual(
            self._overrides("cases_extra.jsonl"),
            {
                cid: "explicit_route5_label"
                for cid in ("x06", "x07", "x08", "x09", "x16", "x17", "x18", "x20", "x21", "x22", "x23", "x26")
            },
        )

    def test_c23_expansion_is_stable(self):
        """Guards the annotation that was missing and silently lowered DEV."""
        cases = {c["id"]: c for c in load_cases(EVAL_DIR / "cases.jsonl")}
        norm = _normalize(cases["c23"], EVAL_DIR / "cases.jsonl")
        self.assertEqual(norm["expected_route5"], "clarification_required")
        self.assertEqual(norm["declared_route"], "manual_orchestrator")


if __name__ == "__main__":
    unittest.main()
