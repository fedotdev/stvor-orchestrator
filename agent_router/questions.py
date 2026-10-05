"""The fixed, atomic question set sent to Jev in a single request.

Batching every question into one call is the documented cheap path (the
parallel-questions cookbook reports ~12x cheaper and ~10x faster than one call
per question). The set is deliberately disjoint: each question asks about a
different axis and no two options overlap.

The `scope` instruction is phrased around whether files must change. During
calibration the generic "how much of the codebase must be touched" wording
produced unstable confidence (0.39-0.43) on pure read-only tasks, which wrongly
pushed them to the manual route; the write-anchored wording returned 0.97-1.00
for the same tasks. Option labels are unchanged.
"""

from __future__ import annotations

INTENT_OPTIONS = (
    "repo_search",
    "bug_fix",
    "feature",
    "refactor",
    "review",
    "research",
    "documentation",
    "other",
)

SCOPE_OPTIONS = ("trivial", "local", "multi_file", "system", "unknown")

RISK_LEVELS = (
    "Read-only: nothing in the repository is modified",
    "Local and reversible: edits are confined to a small area and easily undone",
    "Shared or hard to reverse: changes data, schemas, secrets, dependencies, CI or auth",
)

QUESTIONS: dict[str, dict] = {
    "intent": {
        "type": "choice",
        "instructions": "What is the primary goal of the request below? Choose exactly one.",
        "criteria": {
            "repo_search": "Locate or find existing code, files, symbols or configuration without changing anything",
            "bug_fix": "Repair behaviour that is currently broken or wrong",
            "feature": "Add capability that does not exist yet",
            "refactor": "Restructure existing code without changing its behaviour",
            "review": "Assess existing code, a diff or a design without changing it",
            "research": "Investigate external sources, the web or third-party documentation",
            "documentation": "Write or update documentation, comments or README files",
            "other": "The goal does not fit any option above",
        },
    },
    # Calibrated: anchored on whether files must change at all, because the
    # read-only fast path is exactly the case this question must be certain about.
    "scope": {
        "type": "choice",
        "instructions": (
            "If the assistant must change repository files to fulfil this request, how large is that change? "
            "If no repository file needs to be modified, answer trivial."
        ),
        "criteria": {
            "trivial": "No repository file is modified; the answer comes from the request itself",
            "local": "A single file or single function changes",
            "multi_file": "Several files or modules must change together",
            "system": "Architecture, build, dependencies, configuration or cross-cutting behaviour",
            "unknown": "Not enough information to judge the size",
        },
    },
    "needs_repo_scan": {
        "type": "noul",
        "instructions": "Must the assistant inspect repository files to answer or fulfil this request?",
        "criteria": {
            "true": "Reading or searching repository content is required to proceed",
            "false": "The request can be fulfilled from the request text alone",
        },
    },
    "needs_external_research": {
        "type": "noul",
        "instructions": (
            "Does fulfilling this request require reading a source outside the repository, in order to: consult "
            "an official API reference; read the documentation of a third-party library; apply a standard, GOST "
            "or other normative document; work from an external specification; check the code against an external "
            "contract; or compare an implementation against a normative source or documentation?"
        ),
        "criteria": {
            "true": "The answer depends on material that is not in the repository: an API reference, third-party "
            "documentation, a standard or normative document, an external specification or contract, or any other "
            "outside source that has to be looked up",
            "false": "Every fact needed to finish the request is already present in the request itself or in the "
            "working repository, so no outside lookup is required",
        },
    },
    "needs_write": {
        "type": "noul",
        "instructions": "Does fulfilling this request require modifying or creating files in the repository?",
    },
    "needs_tests": {
        "type": "noul",
        "instructions": "Does this request require writing or running automated tests to be considered done?",
    },
    # Phrased as "can the assistant proceed" rather than "is a success criterion
    # written down". The stricter phrasing fired on well-specified requests whose
    # success criterion is merely implied (e.g. "show me the file and the
    # function name"), pushing them to clarification_required. The criteria below
    # state that an implied result still counts as clear.
    "requires_clarification": {
        "type": "noul",
        "instructions": (
            "Is the request too vague to act on: is the object to work on missing, or is the goal absent or "
            "self-contradictory, so that the assistant would have to ask the user what to do? A request is NOT "
            "vague merely because it does not spell out a success criterion, and NOT merely because it does not "
            "give a file path or a link. A task that names what to compare, what to change or what to look up is "
            "clear: finding the artifact itself is the assistant's job, not a reason to ask."
        ),
        "criteria": {
            "true": "The assistant cannot tell which object to work on, or what the goal is, or the goals contradict each other",
            "false": "The object and the goal are both evident from the request, even if no path, link or success criterion is given",
        },
    },
    "sensitive_access": {
        "type": "noul",
        "instructions": (
            "Does the task require searching for, revealing, using or changing API keys, tokens, passwords, "
            "credentials, personal data, secrets or confidential configuration?"
        ),
        "criteria": {
            "true": "The work touches credentials, secrets, personal data or confidential configuration",
            "false": "The work touches none of those",
        },
    },
    "needs_cross_source_comparison": {
        "type": "noul",
        "instructions": (
            "Does the task require comparing or reconciling facts from two different sources, such as the local "
            "repository against an external standard or API documentation, code against a normative source, "
            "or a diff against requirements?"
        ),
        "criteria": {
            "true": "Two or more distinct sources must be compared and the differences reported",
            "false": "A single source is enough to answer or fulfil the request",
        },
    },
    "risk": {
        "type": "score",
        "instructions": "How risky is carrying out this request in a real codebase?",
        "criteria": list(RISK_LEVELS),
    },
}

# Question ids the policy consumes, in the order they are reported.
ANSWER_IDS = tuple(QUESTIONS.keys())
