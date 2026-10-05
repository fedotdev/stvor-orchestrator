"""All router thresholds and modes in one place.

Every value is overridable from the environment so calibration never requires a
code change. Defaults come from the calibration run against jev-1.13.0:
intent confidence separated cleanly (1.00 vs 0.56/0.66), the write Noul had a
huge margin (0.92-0.96 vs 0.04-0.06), and score confidence on the size Choice
was unstable for read-only work and is therefore not used as a gate.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

MODES = ("off", "observe", "assist", "enforce")

# Only fast_readonly may be applied automatically; writes and high risk always
# wait for the user, so enforce never runs the orchestrator unattended.
AUTO_ROUTES = ("fast_readonly",)


def _float_env(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _mode_env(default: str = "observe") -> str:
    raw = (os.environ.get("ROUTER_MODE") or default).strip().lower()
    return raw if raw in MODES else default


@dataclass(frozen=True)
class RouterConfig:
    # Model: pin the versioned ID so calibrated thresholds do not silently move
    # when an alias is repointed at a new release.
    model: str = field(default_factory=lambda: os.environ.get("TYPESAFE_MODEL", "jev-1.13.0"))
    api_key_env: str = "TYPESAFE_API_KEY"
    endpoint: str = "https://api.typesafe.ai/v1/systemone"

    # Thresholds
    intent_confidence: float = field(default_factory=lambda: _float_env("ROUTER_INTENT_CONFIDENCE", 0.70))
    noul_threshold: float = field(default_factory=lambda: _float_env("ROUTER_NOUL_THRESHOLD", 0.70))
    # Size confidence escalates depth but never gates: low confidence here means
    # "use the deeper orchestrator", not "ask the user".
    size_escalate_confidence: float = field(default_factory=lambda: _float_env("ROUTER_SIZE_ESCALATE_CONFIDENCE", 0.60))
    high_risk_probability: float = field(default_factory=lambda: _float_env("ROUTER_HIGH_RISK_PROB", 0.50))

    # Runtime
    mode: str = field(default_factory=_mode_env)
    max_parallel: int = field(default_factory=lambda: _int_env("ROUTER_MAX_PARALLEL", 2))
    timeout_seconds: float = field(default_factory=lambda: _float_env("ROUTER_TIMEOUT_S", 20.0))
    max_retries: int = field(default_factory=lambda: _int_env("ROUTER_MAX_RETRIES", 2))
    backoff_base_seconds: float = field(default_factory=lambda: _float_env("ROUTER_BACKOFF_BASE_S", 0.5))
    max_backoff_seconds: float = field(default_factory=lambda: _float_env("ROUTER_MAX_BACKOFF_S", 8.0))

    # Artifacts live under the working project root, not under the agent config.
    artifact_dir: str = field(
        default_factory=lambda: os.environ.get("ROUTER_ARTIFACT_DIR", os.path.join(".agent", "artifacts", "routing"))
    )

    def api_key(self) -> str | None:
        return os.environ.get(self.api_key_env) or None


DEFAULT_CONFIG = RouterConfig()
