"""Deterministic text signals.

These are policy-derived, not model answers. They are computed from the request
text with fixed patterns so that the same request always yields the same signal,
and they are recorded in the artifact as facts under a key that cannot be
mistaken for a Jev answer.

Every signal is a *floor*, never a ceiling: a signal can only make the router
more cautious (more roles, a read-only route, or a question to the user). None
of them can grant a write permission.
"""

from __future__ import annotations

import re

# -- external_comparison_hint -------------------------------------------------
# The request ties the codebase to something defined outside it, so an external
# source has to be read even when the model scored needs_external_research low.

_COMPARISON = re.compile(
    r"сравн\w*"
    r"|свер\w*"                     # сверь, сверить, сверяет, сверка
    r"|сопостав\w*"
    r"|соответств\w*"
    r"|совпад\w*"                   # "совпадает ли расчёт с формулой из ГОСТ"
    r"|проверь\s+по"
    r"|провер\w*\s+против"
    r"|compare\b"
    r"|comparing\b"
    r"|verif\w*[^.]{0,40}?against\b"     # "verify the code against RFC 7231"
    r"|check\w*[^.]{0,40}?against\b"
    r"|conforms?\s+to\b"
    r"|reconcil\w+\b"
    r"|cross[- ]check\w*"
    r"|\bmatch(?:es|ing)?\b",
    re.IGNORECASE,
)

_EXTERNAL_SOURCE = re.compile(
    r"гост\b"
    r"|птр\b"                       # ПТР: rail operating regulations, a normative source
    r"|стандарт\w*"
    r"|норматив\w*"
    r"|спецификац\w*"
    r"|контракт\w*"
    r"|документаци\w*"
    r"|\breference\b"
    r"|\brfc\b"
    r"|external\s+api\b"
    r"|vendor\s+docs?\b"
    r"|third[- ]party\s+docs?\b"
    r"|\bspec\w*"
    r"|\bcontract\b"
    r"|\bstandard\b",
    re.IGNORECASE,
)

# -- diagnostic_readonly_hint -------------------------------------------------
# A short statement that something is broken, naming the area but not the object.
# "Почини тесты", "Сборка сломалась", "Почему CI красный?".

_DIAGNOSTIC_AREA = re.compile(
    r"тест\w*"
    r"|\bci\b"
    r"|сборк\w*"
    r"|\bbuild\b"
    r"|\bpipeline\b"
    r"|\bpytest\b"
    r"|\bcoverage\b",
    re.IGNORECASE,
)

# The area must be reported as broken, or addressed directly. Without this the
# signal also fires on a request that merely *mentions* the area while doing
# something else, e.g. "add a README section about running the tests" - there
# the tests are a topic, not a symptom, and the request has a real deliverable.
_DIAGNOSTIC_SYMPTOM = re.compile(
    r"пад\w*"
    r"|слом\w*"
    r"|красн\w*"
    r"|не\s+проходит"
    r"|не\s+работает"
    r"|регресс\w*"
    r"|\bfail\w*"
    r"|\bbroken\b"
    r"|\berror\w*"
    r"|\bflak\w*"
    r"|\bwhy\b",
    re.IGNORECASE,
)

_DIAGNOSTIC_FIX_VERB = re.compile(
    r"почин\w*"
    r"|исправ\w*"
    r"|отлад\w*"
    r"|\bfix\b"
    r"|\bfixing\b"
    r"|\brepair\w*"
    r"|\bdebug\b",
    re.IGNORECASE,
)

# Anything that points at a concrete object: the request is then specific enough
# to act on and does not need a diagnostic pass.
_OBJECT_MARKER = re.compile(
    r"[/\\]"                       # path separator
    r"|\b\w+\.(py|ts|js|json|ya?ml|toml|md|txt|cfg|ini|sql|sh)\b"
    r"|\b\w+\("                   # function call
    r"|\btest_\w+"
    r"|\b[A-Z][A-Za-z0-9]*Error\b"  # exception type
    r"|\bline\s+\d+"
    r"|\bстрок\w*\s+\d+"
    r"|https?://"
    r"|`[^`]+`"
    r"|['\"][^'\"]{3,}['\"]"
    r"|\b[A-Z_]{4,}\b",             # CONSTANT / ENV_VAR
    re.IGNORECASE,
)

_DESTRUCTIVE = re.compile(
    r"удал\w*"
    r"|снес\w*"
    r"|стере\w*"
    r"|очист\w*"
    r"|сброс\w*"
    r"|пересозда\w*"
    r"|миграц\w*"
    r"|\bdrop\b"
    r"|\bdelete\b"
    r"|\bremove\b"
    r"|\bwipe\b"
    r"|\btruncate\b"
    r"|\breset\b"
    r"|\bpurge\b"
    r"|\brm\s+-[rf]",
    re.IGNORECASE,
)

# A request longer than this names more than a symptom.
DIAGNOSTIC_MAX_WORDS = 10


def external_comparison_hint(state: str | None) -> bool:
    """True when the request ties the code to a source defined outside it."""
    if not state:
        return False
    return bool(_COMPARISON.search(state) and _EXTERNAL_SOURCE.search(state))


def diagnostic_readonly_hint(state: str | None) -> bool:
    """True for a short "something is broken" request with no object named.

    Requires all of: a named area, that area being reported as broken or
    addressed by a fix verb, no object marker, and brevity. False for a
    destructive request, and false for a request with no subject at all
    ("Сделай лучше"), which is a question for the user rather than a diagnosis.
    """
    if not state:
        return False
    if _DESTRUCTIVE.search(state):
        return False
    if not _DIAGNOSTIC_AREA.search(state):
        return False
    if not (_DIAGNOSTIC_SYMPTOM.search(state) or _DIAGNOSTIC_FIX_VERB.search(state)):
        return False
    if _OBJECT_MARKER.search(state):
        return False
    return len(state.split()) <= DIAGNOSTIC_MAX_WORDS


def is_destructive(state: str | None) -> bool:
    """True when the request asks for something irreversible."""
    return bool(state) and bool(_DESTRUCTIVE.search(state))


# Narrower than _DIAGNOSTIC_AREA: only the areas where a test analyst is the
# right second role, i.e. where the request names tests, CI or a build.
_TEST_BUILD_AREA = re.compile(
    r"тест\w*"
    r"|\bci\b"
    r"|\bbuild\b"
    r"|сборк\w*"
    r"|\bpipeline\b"
    r"|\bpytest\b"
    r"|\bcoverage\b"
    r"|\bflak\w*"
    r"|\bregress\w*"
    r"|\btest\w*",
    re.IGNORECASE,
)


def is_diagnostic_test_area(state: str | None) -> bool:
    """True when a diagnostic request names the tests, CI or a build."""
    return bool(state) and bool(_TEST_BUILD_AREA.search(state))


# -- safety_critical_parameter_hint ------------------------------------------
# A request to change a safety-critical railway parameter. The model's own risk
# score is not reliable here: these requests are usually framed as trivial on
# purpose ("this is just a number"), which is what the evaluation set traps
# probe. So the change verb and the parameter are read off the text instead.
#
# The change verb is what keeps this narrow. Merely *mentioning* a safety
# parameter is common and harmless - "where is the headway interval computed",
# "which tests cover the interlocking module" - and none of those are changes.

_SAFETY_CHANGE_VERB = re.compile(
    r"измени\w*"
    r"|поменя\w*"
    r"|увелич\w*"
    r"|уменьш\w*"
    r"|снизь\w*"
    r"|повысь\w*"
    r"|подним\w*"
    r"|опуст\w*"
    r"|задай\w*"
    r"|выстав\w*"
    r"|настрой\w*"
    r"|подстрой\w*"
    r"|\bset\b"
    r"|\bchange\b"
    r"|\bupdate\b"
    r"|\breduce\b"
    r"|\bincrease\b"
    r"|\bmodify\b"
    r"|\blower\b"
    r"|\braise\b",
    re.IGNORECASE,
)

# Rail-safety parameters. "speed" and "safety" are only matched next to a rail
# noun: "network speed" and "password security" are not this.
_SAFETY_PARAMETER = re.compile(
    r"тормозн\w*"
    r"|интервал\w*\s+(?:безопасн|минимальн|миним)"
    r"|минимальн\w*\s+интервал"
    r"|интервал\w*\s+попутного"
    r"|безопасн\w*\s+дистанц\w*"
    r"|дистанци\w*\s+безопасн"
    r"|коэффициент\w*\s+(?:замедлен|торможен|запас|безопасн)"
    r"|замедлени\w*"
    r"|максимальн\w*\s+скорост"
    r"|скорост\w*\s+(?:поезд|состав|движени)"
    r"|деблок\w*"
    r"|ж[её]лт\w*\s+сигнал"
    r"|красн\w*\s+сигнал"
    r"|блокировк\w*"
    r"|перегон\w*"
    r"|график\w*\s+движени\w*"
    r"|аварийн\w*"
    r"|экстренн\w*"
    r"|fail[- ]?safe"
    r"|safety\s+(?:margin|interval|limit|distance)"
    r"|safe\s+(?:interval|distance|margin)"
    r"|minimum\s+interval"
    r"|braking"
    r"|\bbrake\b"
    r"|\bheadway\b"
    r"|block\s+length"
    r"|\binterlocking\b"
    r"|\bemergency\b"
    r"|\bhazard\b"
    r"|deceleration"
    r"|speed\s+limit",
    re.IGNORECASE,
)


def safety_critical_parameter_hint(state: str | None) -> bool:
    """True when the request asks to change a safety-critical parameter.

    A floor on risk, never a grant of write access: it can only push a request
    towards a human, which is the safe direction.
    """
    if not state:
        return False
    return bool(_SAFETY_CHANGE_VERB.search(state) and _SAFETY_PARAMETER.search(state))
