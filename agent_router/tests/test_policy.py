"""Policy table tests. No network, no clock, no filesystem.

The scenarios named after x14/x16/x23 come from the frozen TEST set: each is a
case where the previous three-route policy made a defensible-looking but wrong
call, so they are pinned here as regressions.
"""

from __future__ import annotations

import unittest

from agent_router.config import RouterConfig
from agent_router.policy import FORBIDDEN_FAST_READONLY, PolicyError, plan_for_unavailable, plan_from_answers

from .fixtures import answers

LOW_RISK = {"0": 1.0, "1": 0.0, "2": 0.0}
HIGH_RISK = {"0": 0.0, "1": 0.05, "2": 0.95}
MID_RISK = {"0": 0.1, "1": 0.85, "2": 0.05}


class TestRequiredScenarios(unittest.TestCase):
    """The seven scenarios the router must get right by construction."""

    def test_x16_two_read_only_sources_get_both_roles(self):
        # Compare an external normative source against the local implementation:
        # read-only, but one agent is not enough - it needs scout + researcher
        # and a synthesis step.
        plan = plan_from_answers(answers(intent="research", repo_scan=0.95, external=0.95, write=0.0))
        self.assertEqual(plan.route, "orchestrated_readonly")
        self.assertTrue(plan.run_scout)
        self.assertTrue(plan.run_researcher)
        self.assertFalse(plan.run_implementer)
        self.assertFalse(plan.allows_write)

    def test_x14_sensitive_access_is_manual_without_writing(self):
        # API keys and secrets in scope: strongest lock, no workers, no write.
        plan = plan_from_answers(answers(intent="repo_search", repo_scan=0.95, sensitive=0.97, write=0.1))
        self.assertEqual(plan.route, "manual_orchestrator")
        self.assertFalse(plan.allows_write)
        self.assertFalse(plan.run_implementer)
        self.assertTrue(plan.needs_user_confirmation)
        self.assertEqual(plan.workers, ())

    def test_x23_underspecified_asks_instead_of_starting_workers(self):
        # No object of analysis: ask the user, start nobody.
        plan = plan_from_answers(answers(intent="other", repo_scan=0.9, clarification=0.95))
        self.assertEqual(plan.route, "clarification_required")
        self.assertEqual(plan.workers, ())
        self.assertFalse(plan.allows_write)

    def test_plain_repo_search_is_one_scout(self):
        plan = plan_from_answers(answers(intent="repo_search", repo_scan=0.95))
        self.assertEqual(plan.route, "fast_readonly")
        self.assertEqual(plan.workers, ("scout",))

    def test_plain_web_research_is_one_researcher(self):
        plan = plan_from_answers(answers(intent="research", repo_scan=0.05, external=0.95))
        self.assertEqual(plan.route, "fast_readonly")
        self.assertEqual(plan.workers, ("researcher",))

    def test_write_bugfix_is_one_implementer(self):
        plan = plan_from_answers(answers(intent="bug_fix", scope="local", scope_conf=0.85, repo_scan=0.9, write=0.96, tests=0.5))
        self.assertEqual(plan.route, "orchestrated_write")
        self.assertEqual(plan.workers.count("implementer"), 1)
        self.assertTrue(plan.run_implementer)
        self.assertFalse(plan.needs_user_confirmation)

    def test_high_risk_write_is_manual_with_confirmation(self):
        plan = plan_from_answers(
            answers(intent="feature", scope="system", scope_conf=0.9, write=0.95, risk_probs=HIGH_RISK)
        )
        self.assertEqual(plan.route, "manual_orchestrator")
        self.assertFalse(plan.allows_write)
        self.assertTrue(plan.needs_user_confirmation)


class TestRuleOrder(unittest.TestCase):
    def test_sensitive_beats_clarification(self):
        plan = plan_from_answers(answers(sensitive=0.95, clarification=0.95, write=0.99, risk_probs=HIGH_RISK))
        self.assertEqual(plan.route, "manual_orchestrator")
        self.assertIn("sensitive access", plan.reason)

    def test_clarification_beats_risk(self):
        plan = plan_from_answers(answers(clarification=0.95, risk_probs=HIGH_RISK, write=0.99))
        self.assertEqual(plan.route, "clarification_required")
        self.assertIn("clarification", plan.reason)

    def test_risk_beats_write(self):
        plan = plan_from_answers(answers(write=0.99, risk_probs=HIGH_RISK))
        self.assertEqual(plan.route, "manual_orchestrator")
        self.assertIn("high risk", plan.reason)

    def test_write_beats_two_sources(self):
        plan = plan_from_answers(answers(repo_scan=0.95, external=0.95, write=0.95, risk_probs=MID_RISK))
        self.assertEqual(plan.route, "orchestrated_write")
        self.assertTrue(plan.run_implementer)

    def test_wide_write_scope_requires_confirmation(self):
        plan = plan_from_answers(answers(intent="feature", scope="multi_file", scope_conf=0.9, write=0.95))
        self.assertEqual(plan.route, "orchestrated_write")
        self.assertTrue(plan.run_implementer)
        self.assertTrue(plan.needs_user_confirmation)
        self.assertFalse(plan.allows_write)

    def test_uncertain_write_scope_requires_confirmation(self):
        plan = plan_from_answers(answers(intent="refactor", scope="unknown", scope_conf=0.95, write=0.9))
        self.assertEqual(plan.route, "orchestrated_write")
        self.assertTrue(plan.needs_user_confirmation)


class TestReadOnlyRouting(unittest.TestCase):
    def test_cross_source_comparison_gets_read_only_roles(self):
        plan = plan_from_answers(answers(intent="review", repo_scan=0.05, external=0.9, cross_source=0.93))
        self.assertEqual(plan.route, "orchestrated_readonly")
        self.assertFalse(plan.run_implementer)
        self.assertTrue(plan.run_researcher)

    def test_cross_source_without_detected_sources_still_gets_a_role(self):
        plan = plan_from_answers(answers(intent="review", repo_scan=0.2, external=0.2, cross_source=0.9))
        self.assertEqual(plan.route, "orchestrated_readonly")
        self.assertTrue(plan.workers)
        self.assertFalse(plan.run_implementer)

    def test_repo_scan_and_tests_uses_test_analyst(self):
        plan = plan_from_answers(answers(intent="review", repo_scan=0.95, tests=0.9, write=0.05))
        self.assertEqual(plan.route, "orchestrated_readonly")
        self.assertEqual(set(plan.workers), {"scout", "test_analyst"})

    def test_no_signal_at_all_asks_for_clarification(self):
        plan = plan_from_answers(answers(intent="other", repo_scan=0.1, external=0.1, write=0.1, tests=0.1))
        self.assertEqual(plan.route, "clarification_required")
        self.assertEqual(plan.workers, ())

    def test_orchestrated_readonly_never_allows_write(self):
        plan = plan_from_answers(answers(repo_scan=0.95, external=0.95))
        self.assertFalse(plan.run_implementer)
        self.assertFalse(plan.allows_write)


class TestFastReadonlyGuard(unittest.TestCase):
    def test_low_intent_confidence_vetoes_fast_readonly(self):
        # The threshold stays at 0.70; ambiguity must not buy the cheap route.
        plan = plan_from_answers(answers(intent="feature", intent_conf=0.4, repo_scan=0.95))
        self.assertNotEqual(plan.route, "fast_readonly")
        self.assertIn("post-condition", plan.reason)
        self.assertIn("low_intent_confidence", plan.reason)

    def test_confident_read_only_still_takes_the_cheap_route(self):
        plan = plan_from_answers(answers(intent="repo_search", intent_conf=0.95, repo_scan=0.95))
        self.assertEqual(plan.route, "fast_readonly")

    def test_every_forbidden_condition_is_reachable(self):
        # Guards against a guard flag that is declared but never set.
        self.assertIn("needs_write", FORBIDDEN_FAST_READONLY)
        self.assertIn("high_risk", FORBIDDEN_FAST_READONLY)
        self.assertIn("sensitive_access", FORBIDDEN_FAST_READONLY)
        self.assertIn("requires_clarification", FORBIDDEN_FAST_READONLY)
        self.assertIn("repo_scan_and_external_research", FORBIDDEN_FAST_READONLY)
        self.assertIn("cross_source_comparison", FORBIDDEN_FAST_READONLY)

    def test_no_input_produces_fast_readonly_with_write(self):
        # Property check across the dangerous combinations.
        for sensitive in (0.0, 0.95):
            for clarification in (0.0, 0.95):
                for risk in ({"0": 1.0, "1": 0.0, "2": 0.0}, HIGH_RISK):
                    for write in (0.0, 0.96):
                        for scan in (0.0, 0.96):
                            for research in (0.0, 0.96):
                                for cross in (0.0, 0.96):
                                    plan = plan_from_answers(
                                        answers(
                                            sensitive=sensitive,
                                            clarification=clarification,
                                            risk_probs=risk,
                                            write=write,
                                            repo_scan=scan,
                                            external=research,
                                            cross_source=cross,
                                        )
                                    )
                                    if plan.route == "fast_readonly":
                                        self.assertFalse(
                                            plan.allows_write,
                                            f"fast_readonly allowed a write: {plan.reason}",
                                        )
                                        self.assertFalse(
                                            sensitive >= 0.7 or write >= 0.7 or cross >= 0.7,
                                            f"fast_readonly under a forbidden condition: {plan.reason}",
                                        )


class TestFallback(unittest.TestCase):
    def test_unavailable_jev_falls_back_to_manual_without_write(self):
        plan = plan_for_unavailable("Jev unavailable (JevTimeout: ...)")
        self.assertEqual(plan.route, "manual_orchestrator")
        self.assertFalse(plan.allows_write)
        self.assertTrue(plan.needs_user_confirmation)


class TestBadAnswers(unittest.TestCase):
    def test_missing_new_question_raises(self):
        broken = answers()
        del broken["requires_clarification"]
        with self.assertRaises(PolicyError):
            plan_from_answers(broken)

    def test_noul_replaced_by_choice_raises(self):
        broken = answers()
        broken["sensitive_access"] = {"type": "choice", "choice": "yes", "confidence": 0.9, "probabilities": {"yes": 1.0}}
        with self.assertRaises(PolicyError):
            plan_from_answers(broken)

    def test_risk_without_high_level_probability_raises(self):
        broken = answers()
        broken["risk"]["probabilities"] = {"0": 1.0}
        with self.assertRaises(PolicyError):
            plan_from_answers(broken)

    def test_thresholds_are_configurable(self):
        # At 0.99 nothing clears the bar, so the request has no signal at all.
        strict = RouterConfig(noul_threshold=0.99)
        plan = plan_from_answers(answers(repo_scan=0.95, write=0.95), config=strict)
        self.assertEqual(plan.route, "clarification_required")
        # At the default the same answers are a write task.
        self.assertEqual(plan_from_answers(answers(repo_scan=0.95, write=0.95)).route, "orchestrated_write")


class TestPolicySignalsInRoutes(unittest.TestCase):
    """The six scenarios required for the signal and diagnostic routes.

    Each one pins a request text, so the deterministic signals are exercised
    through the same entry point the router uses.
    """

    def test_x16_like_comparison_ignores_a_underestimated_research_flag(self):
        # The model scored needs_external_research below the threshold, but the
        # text names a normative document: the hint must put a researcher on it.
        plan = plan_from_answers(
            answers(repo_scan=0.95, external=0.05, write=0.05),
            state="Сверь формулу из нормативного документа с реализацией в коде.",
        )
        self.assertEqual(plan.route, "orchestrated_readonly")
        self.assertTrue(plan.run_scout)
        self.assertTrue(plan.run_researcher)
        self.assertFalse(plan.allows_write)
        self.assertFalse(plan.run_implementer)

    def test_c35_like_short_diagnostic_gets_no_implementer(self):
        plan = plan_from_answers(answers(repo_scan=0.9, write=0.96, tests=0.9), state="Почини тесты")
        self.assertEqual(plan.route, "diagnostic_readonly")
        self.assertTrue(plan.run_scout)
        self.assertTrue(plan.run_test_analyst)
        self.assertFalse(plan.run_researcher)
        self.assertFalse(plan.allows_write)
        self.assertFalse(plan.run_implementer)
        self.assertFalse(plan.needs_user_confirmation)

    def test_gost_formula_comparison_gets_a_researcher(self):
        plan = plan_from_answers(
            answers(repo_scan=0.9, external=0.05, write=0.05),
            state="Сверь формулу по ГОСТ с кодом",
        )
        self.assertEqual(plan.route, "orchestrated_readonly")
        self.assertTrue(plan.run_researcher)

    def test_plain_repo_search_stays_cheap(self):
        plan = plan_from_answers(answers(repo_scan=0.9, write=0.0), state="Найди файл с расчётом headway")
        self.assertEqual(plan.route, "fast_readonly")
        self.assertEqual(plan.workers, ("scout",))
        self.assertFalse(plan.run_researcher)

    def test_no_object_at_all_asks_the_user(self):
        plan = plan_from_answers(answers(repo_scan=0.9, write=0.9, clarification=0.95), state="Сделай лучше")
        self.assertEqual(plan.route, "clarification_required")
        self.assertEqual(plan.workers, ())

    def test_high_risk_wins_over_the_comparison_hint(self):
        plan = plan_from_answers(
            answers(repo_scan=0.95, external=0.05, write=0.95, risk_probs=HIGH_RISK),
            state="Сверь расчёт с ГОСТ и поправь его в конфиге.",
        )
        self.assertEqual(plan.route, "manual_orchestrator")
        self.assertTrue(plan.needs_user_confirmation)
        self.assertFalse(plan.allows_write)
        self.assertFalse(plan.run_implementer)

    def test_signals_are_recorded_as_policy_derived_facts(self):
        plan = plan_from_answers(answers(), state="Сверь расчёт с ГОСТ.")
        derived = plan.jev_snapshot["facts"]["policy_derived"]
        self.assertIs(derived["external_comparison_hint"], True)
        self.assertIs(derived["diagnostic_readonly_hint"], False)
        # The hint is kept apart from the Jev answers rather than smuggled in
        # as one of them.
        self.assertNotIn("external_comparison_hint", plan.jev_snapshot["facts"].get("intent", {}))
        self.assertIn("not a Jev answer", derived["source"])

    def test_hint_cannot_survive_without_the_researcher(self):
        # external_comparison_hint is a hard veto on fast_readonly, enforced by
        # the post-condition rather than by rule order alone.
        self.assertIn("external_comparison_hint", FORBIDDEN_FAST_READONLY)
        plan = plan_from_answers(
            answers(repo_scan=0.0, external=0.0, write=0.0, intent_conf=0.4),
            state="Сверь код с документацией API.",
        )
        self.assertNotEqual(plan.route, "fast_readonly")
        self.assertTrue(plan.run_researcher)


    def test_x09_trivial_framing_does_not_defeat_the_safety_floor(self):
        # The request calls itself trivial and the model agrees: no scan, no
        # write, no risk. The text still names a braking-distance parameter.
        plan = plan_from_answers(
            answers(repo_scan=0.05, external=0.05, write=0.05, risk_probs=LOW_RISK),
            state="Поменяй коэффициент замедления в расчёте тормозного пути с 0.9 на 0.85, это же просто число.",
        )
        self.assertEqual(plan.route, "manual_orchestrator")
        self.assertTrue(plan.needs_user_confirmation)
        self.assertFalse(plan.allows_write)
        self.assertFalse(plan.run_implementer)

    def test_safety_floor_names_itself_instead_of_claiming_the_threshold(self):
        # Otherwise the artifact would record "P=0.05 >= 0.50" for a case the
        # model never scored as risky.
        plan = plan_from_answers(
            answers(repo_scan=0.05, write=0.05, risk_probs=LOW_RISK),
            state="Уменьши минимальный интервал безопасности между поездами.",
        )
        self.assertEqual(plan.route, "manual_orchestrator")
        self.assertIn("policy signal", plan.reason)
        self.assertIn("safety_critical_parameter_hint", plan.jev_snapshot["guard_flags"])

    def test_safety_floor_does_not_touch_an_ordinary_parameter_change(self):
        plan = plan_from_answers(
            answers(intent="refactor", scope="local", scope_conf=0.9, repo_scan=0.95, write=0.9),
            state="Поменяй значение таймаута в конфиге с 30 на 60 секунд.",
        )
        self.assertEqual(plan.route, "orchestrated_write")
        self.assertTrue(plan.allows_write)


if __name__ == "__main__":
    unittest.main()
