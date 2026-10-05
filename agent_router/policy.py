"""Pure routing policy: typed Jev answers in, RoutingPlan out.

This module never touches the network, the clock or the filesystem. The final
decision is made here in code; no model interprets it.

Six routes, so that "orchestrated" stops meaning "write":

    fast_readonly          one narrow read-only agent; used only when one role suffices
    diagnostic_readonly    a symptom was reported but no object named: collect evidence first
    orchestrated_readonly  several read-only roles plus synthesis; no implementer
    orchestrated_write     scout/research/test planning -> one implementer -> review/test
    manual_orchestrator    high risk, sensitive access, destructive work, or risk we cannot pin down
    clarification_required the request has no object, goal or success criterion

The rules are evaluated in a fixed order and the first match wins. That order is
the safety property: anything sensitive, unclarified or high risk is claimed
before a cheaper route can take the request.

`intent` and `scope` are recorded for auditing but do not select a route; the
rules are the whole decision. `intent` confidence still has one job: it can veto
fast_readonly, never grant it.

Three deterministic signals from signals.py (`external_comparison_hint`,
`diagnostic_readonly_hint`, `safety_critical_parameter_hint`) are computed from
the request text. They can only add roles or move a request to a read-only or
manual route; they never grant a write.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from .config import RouterConfig
from . import signals

ROUTES = (
    "fast_readonly",
    "diagnostic_readonly",
    "orchestrated_readonly",
    "orchestrated_write",
    "manual_orchestrator",
    "clarification_required",
)

# Score levels are ordered; index 2 is the highest risk level.
RISK_HIGH_INDEX = "2"

_NOUL_KEYS = (
    "needs_repo_scan",
    "needs_external_research",
    "needs_write",
    "needs_tests",
    "requires_clarification",
    "sensitive_access",
    "needs_cross_source_comparison",
)

_REQUIRED_ANSWERS = ("intent", "scope", "risk", *_NOUL_KEYS)

# Conditions that must each independently forbid fast_readonly. Enforced as a
# post-condition in _plan(), so reordering or editing a rule later cannot
# silently reintroduce a cheap-but-unsafe route.
FORBIDDEN_FAST_READONLY = (
    "needs_write",
    "high_risk",
    "sensitive_access",
    "requires_clarification",
    "repo_scan_and_external_research",
    "cross_source_comparison",
    "low_intent_confidence",
    "external_comparison_hint",
    "safety_critical_parameter_hint",
)


class PolicyError(ValueError):
    """Answers are missing or malformed, so no plan can be derived safely."""


@dataclass(frozen=True)
class RoutingPlan:
    route: str
    run_scout: bool
    run_researcher: bool
    run_test_analyst: bool
    run_implementer: bool
    run_deep_reviewer: bool
    needs_user_confirmation: bool
    reason: str
    jev_snapshot: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def allows_write(self) -> bool:
        return self.run_implementer and not self.needs_user_confirmation

    @property
    def workers(self) -> tuple[str, ...]:
        return tuple(
            name
            for flag, name in (
                (self.run_scout, "scout"),
                (self.run_researcher, "researcher"),
                (self.run_test_analyst, "test_analyst"),
                (self.run_implementer, "implementer"),
                (self.run_deep_reviewer, "deep_reviewer"),
            )
            if flag
        )


def plan_from_answers(
    answers: dict[str, Any],
    snapshot: dict | None = None,
    config: RouterConfig | None = None,
    state: str = "",
) -> RoutingPlan:
    """Derive a RoutingPlan from Jev answers. Raises PolicyError on bad input.

    `state` is the request text. It is used only by the deterministic signals in
    signals.py; nothing is inferred from it by a model, and the signals it
    produces are recorded as policy-derived facts, never as Jev answers.
    """
    config = config or RouterConfig()
    snapshot = snapshot or {}
    _validate(answers)

    def noul(name: str) -> float:
        return float(answers[name]["noul"])

    def yes(name: str) -> bool:
        return noul(name) >= config.noul_threshold

    intent_choice = str(answers["intent"]["choice"])
    intent_conf = float(answers["intent"]["confidence"])
    scope_choice = str(answers["scope"]["choice"])
    scope_conf = float(answers["scope"]["confidence"])

    write = yes("needs_write")
    scan = yes("needs_repo_scan")
    research = yes("needs_external_research")
    tests = yes("needs_tests")
    clarify = yes("requires_clarification")
    sensitive = yes("sensitive_access")
    cross_source = yes("needs_cross_source_comparison")
    risk_p = float(answers["risk"]["probabilities"].get(RISK_HIGH_INDEX, 0.0))
    model_risk_high = risk_p >= config.high_risk_probability
    risk_high = model_risk_high
    low_confidence = intent_conf < config.intent_confidence

    # -- deterministic, policy-derived signals (not Jev answers) ---------------
    ext_hint = signals.external_comparison_hint(state)
    diag_hint = signals.diagnostic_readonly_hint(state)
    safety_param_hint = signals.safety_critical_parameter_hint(state)
    # A named external source is research whether or not the model scored it.
    research = research or ext_hint
    # A change to a safety-critical parameter is high risk whether or not the
    # model scored it. The model's risk score is the weaker signal here: these
    # requests are framed as trivial on purpose.
    risk_high = risk_high or safety_param_hint

    # Recorded for every route, so a plan can be re-read after calibration.
    # The two *_hint keys are computed here from the request text by signals.py;
    # they are policy-derived facts and are deliberately not read as Jev answers.
    facts = {
        "intent": intent_choice,
        "intent_confidence": round(intent_conf, 4),
        "scope": scope_choice,
        "scope_confidence": round(scope_conf, 4),
        "policy_derived": {
            "external_comparison_hint": ext_hint,
            "diagnostic_readonly_hint": diag_hint,
            "safety_critical_parameter_hint": safety_param_hint,
            "source": "deterministic text match in agent_router.signals, not a Jev answer",
        },
    }

    # -- 1. secrets, credentials, personal data --------------------------------
    if sensitive:
        return _plan(
            "manual_orchestrator",
            f"sensitive access (P={noul('sensitive_access'):.2f} >= {config.noul_threshold:.2f}): credentials, "
            "secrets, personal data or confidential configuration in scope; writing disabled, confirmation required",
            snapshot,
            facts,
            confirmation=True,
            guard_flags={"sensitive_access": True},
        )

    # -- 2. the request cannot be routed safely without asking the user ---------
    if clarify:
        return _plan(
            "clarification_required",
            f"requires clarification (P={noul('requires_clarification'):.2f} >= {config.noul_threshold:.2f}): "
            "object, goal or success criterion is missing; no worker started",
            snapshot,
            facts,
        )

    # -- 3. high risk ----------------------------------------------------------
    if risk_high:
        if safety_param_hint and not model_risk_high:
            # Say which signal decided it, or the artifact claims a threshold
            # was crossed that the model never crossed.
            why = (
                "safety-critical parameter change (policy signal; the model's own risk score was "
                f"P(level {RISK_HIGH_INDEX})={risk_p:.2f})"
            )
        else:
            why = f"high risk (P(level {RISK_HIGH_INDEX})={risk_p:.2f} >= {config.high_risk_probability:.2f})"
        return _plan(
            "manual_orchestrator",
            f"{why}: writing disabled, confirmation required",
            snapshot,
            facts,
            test_analyst=True,
            confirmation=True,
            guard_flags={"high_risk": True, "safety_critical_parameter_hint": safety_param_hint},
        )

    # -- 3b. short "something is broken" request with no object named -----------
    # Gather evidence before anyone writes. Never reached for high risk or
    # sensitive access, because those rules already returned above.
    if diag_hint and not ext_hint:
        test_area = signals.is_diagnostic_test_area(state)
        return _plan(
            "diagnostic_readonly",
            f"short diagnostic request (policy signal, {len(state.split())} words, "
            f"no object named); scout to collect evidence"
            + (", test analyst for the failing area" if test_area else "")
            + ", no implementer until the object is known",
            snapshot,
            facts,
            scout=True,
            test_analyst=test_area,
        )

    # -- 4. a write is intended -------------------------------------------------
    if write:
        wide_or_unclear = scope_choice in ("multi_file", "system", "unknown") or scope_conf < config.size_escalate_confidence
        return _plan(
            "orchestrated_write",
            f"write task {intent_choice}/{scope_choice} (needs_write={noul('needs_write'):.2f}, "
            f"size confidence {scope_conf:.2f}): single implementer"
            + (", deep review" if wide_or_unclear or tests else "")
            + (", confirmation required before writing" if wide_or_unclear else ""),
            snapshot,
            facts,
            scout=scan,
            researcher=research,
            test_analyst=tests,
            implementer=True,
            deep_reviewer=wide_or_unclear or tests,
            confirmation=wide_or_unclear,
        )

    # -- 5. two read-only sources ----------------------------------------------
    if scan and research:
        return _plan(
            "orchestrated_readonly",
            f"read-only across two sources (scan={noul('needs_repo_scan'):.2f}, "
            f"research={noul('needs_external_research'):.2f}): scout + researcher + synthesis, no implementer",
            snapshot,
            facts,
            scout=True,
            researcher=True,
            guard_flags={"repo_scan_and_external_research": True},
        )

    # -- 6. cross-source comparison: roles follow the sources -------------------
    if cross_source:
        roles = [name for name, on in (("scout", scan), ("researcher", research)) if on] or ["scout"]
        return _plan(
            "orchestrated_readonly",
            f"cross-source comparison (P={noul('needs_cross_source_comparison'):.2f}): read-only roles "
            f"{' + '.join(roles)} + synthesis, no implementer",
            snapshot,
            facts,
            scout=scan or not research,
            researcher=research,
            guard_flags={"cross_source_comparison": True},
        )

    # -- 7. repository and test inspection, no code change ----------------------
    if scan and tests:
        return _plan(
            "orchestrated_readonly",
            f"read-only inspection of code and tests (scan={noul('needs_repo_scan'):.2f}, "
            f"tests={noul('needs_tests'):.2f}): scout + test analyst, no implementer",
            snapshot,
            facts,
            scout=True,
            test_analyst=True,
        )

    # -- 8. repository only ------------------------------------------------------
    if scan:
        return _plan(
            "fast_readonly",
            f"read-only {intent_choice}, repository only (needs_write={noul('needs_write'):.2f}): one scout, no implementer",
            snapshot,
            facts,
            scout=True,
            guard_flags={"low_intent_confidence": low_confidence},
        )

    # -- 9. external sources only -------------------------------------------------
    if research:
        if ext_hint:
            # A named external contract is never a single cheap agent: the
            # comparison needs its own read-only pass.
            return _plan(
                "orchestrated_readonly",
                f"external comparison without a repository scan "
                f"(policy signal external_comparison_hint={ext_hint}); researcher + comparison pass, no implementer",
                snapshot,
                facts,
                researcher=True,
            )
        return _plan(
            "fast_readonly",
            f"read-only {intent_choice}, external sources only (needs_write={noul('needs_write'):.2f}): "
            "one researcher, no implementer",
            snapshot,
            facts,
            researcher=True,
            guard_flags={"low_intent_confidence": low_confidence},
        )

    # -- 10. nothing to route on ---------------------------------------------------
    return _plan(
        "clarification_required",
        "no repository access, no external research and no write needed: nothing to route on, "
        "clarification required",
        snapshot,
        facts,
    )


def plan_for_unavailable(reason: str, snapshot: dict | None = None) -> RoutingPlan:
    """Rule 0: any Jev failure degrades to the safe manual route, writing disabled."""
    return RoutingPlan(
        route="manual_orchestrator",
        run_scout=False,
        run_researcher=False,
        run_test_analyst=False,
        run_implementer=False,
        run_deep_reviewer=False,
        needs_user_confirmation=True,
        reason=reason,
        jev_snapshot=snapshot or {},
    )


def _plan(
    route: str,
    reason: str,
    snapshot: dict,
    facts: dict,
    *,
    scout: bool = False,
    researcher: bool = False,
    test_analyst: bool = False,
    implementer: bool = False,
    deep_reviewer: bool = False,
    confirmation: bool = False,
    guard_flags: dict | None = None,
) -> RoutingPlan:
    """Build a plan and enforce the fast_readonly post-condition.

    `guard_flags` records which unsafe conditions held when the route was chosen.
    If the route is fast_readonly while any of them is set, the plan is
    downgraded instead of shipped. With the current rule order the only
    condition that can fire here is low intent confidence; the rest are already
    handled by earlier rules.
    """
    active = sorted(name for name, is_set in (guard_flags or {}).items() if is_set)
    blocked = [name for name in active if name in FORBIDDEN_FAST_READONLY]
    payload = {"facts": facts, "guard_flags": active}
    if route == "fast_readonly" and blocked:
        payload["blocked_by"] = blocked
        return RoutingPlan(
            route="manual_orchestrator",
            run_scout=False,
            run_researcher=False,
            run_test_analyst=False,
            run_implementer=False,
            run_deep_reviewer=False,
            needs_user_confirmation=True,
            reason=f"post-condition: fast_readonly vetoed by {blocked}; {reason}",
            jev_snapshot={**snapshot, **payload} if snapshot else payload,
        )
    return RoutingPlan(
        route=route,
        run_scout=scout,
        run_researcher=researcher,
        run_test_analyst=test_analyst,
        run_implementer=implementer,
        run_deep_reviewer=deep_reviewer,
        needs_user_confirmation=confirmation,
        reason=reason,
        jev_snapshot={**snapshot, **payload} if snapshot else payload,
    )


def _validate(answers: dict[str, Any]) -> None:
    if not isinstance(answers, dict):
        raise PolicyError("answers must be a mapping")
    missing = [key for key in _REQUIRED_ANSWERS if key not in answers]
    if missing:
        raise PolicyError(f"missing answers: {', '.join(missing)}")

    for key in ("intent", "scope"):
        answer = answers[key]
        if not isinstance(answer.get("choice"), str) or not isinstance(answer.get("probabilities"), dict):
            raise PolicyError(f"{key} answer is not a choice answer")
        if not isinstance(answer.get("confidence"), (int, float)):
            raise PolicyError(f"{key} answer has no confidence")

    for key in _NOUL_KEYS:
        if not isinstance(answers[key].get("noul"), (int, float)):
            raise PolicyError(f"{key} answer is not a noul answer")

    risk = answers["risk"]
    if not isinstance(risk.get("probabilities"), dict) or RISK_HIGH_INDEX not in risk["probabilities"]:
        raise PolicyError("risk answer has no level probabilities")
