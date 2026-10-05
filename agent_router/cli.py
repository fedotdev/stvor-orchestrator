"""Harness entry point: `python -m agent_router route ...`.

The harness classifies a request and emits a plan. It never executes the plan —
applying it stays with the caller (the opencode plugin tool, or a human), so an
observe-mode run cannot change agent behaviour.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from . import artifact as artifact_mod
from . import client as client_mod
from .config import AUTO_ROUTES, DEFAULT_CONFIG, RouterConfig
from .policy import PolicyError, plan_for_unavailable, plan_from_answers
from .questions import QUESTIONS

EXIT_OK = 0
EXIT_USAGE = 2


def _read_task(args: argparse.Namespace) -> str:
    if args.stdin:
        return sys.stdin.read()
    if args.task_file:
        return Path(args.task_file).read_text(encoding="utf-8")
    return args.task or ""


def _applied_route(mode: str, plan) -> str:
    """What the caller is allowed to do with this plan, per mode."""
    if mode == "off":
        return "jev_not_called"
    if mode == "observe":
        return "unchanged"
    if mode == "assist":
        return "pending_confirmation"
    # enforce: only the safe read-only route may be applied without a human.
    if plan.route in AUTO_ROUTES and not plan.needs_user_confirmation:
        return plan.route
    return "pending_confirmation"


def route_once(task: str, project_root: str, mode: str | None = None, config: RouterConfig | None = None) -> dict:
    config = config or DEFAULT_CONFIG
    mode = (mode or config.mode)
    if mode not in ("off", "observe", "assist", "enforce"):
        mode = "observe"

    run_id = artifact_mod.make_run_id(task)
    started = datetime.now(timezone.utc)

    if mode == "off" or not task.strip():
        plan = plan_for_unavailable(
            "Jev not called (mode=off)" if mode == "off" else "empty task text",
            {},
        )
        return _finalize(plan, None, run_id, task, mode, config, project_root, started, errors=[f"skipped: {plan.reason}"])

    try:
        result = client_mod.JevClient(config).evaluate(task, QUESTIONS)
    except client_mod.JevError as exc:
        plan = plan_for_unavailable(f"Jev unavailable ({type(exc).__name__}: {exc}); safe fallback, writing disabled", {})
        return _finalize(plan, None, run_id, task, mode, config, project_root, started, errors=[f"{type(exc).__name__}: {exc}"])
    except Exception as exc:  # noqa: BLE001 - the router must never break the agent
        plan = plan_for_unavailable(f"unexpected router error ({type(exc).__name__}: {exc}); safe fallback, writing disabled", {})
        return _finalize(plan, None, run_id, task, mode, config, project_root, started, errors=[f"{type(exc).__name__}: {exc}"])

    snapshot = {
        "model": result.model,
        "latency_ms": result.latency_ms,
        "attempts": result.attempts,
        "usage": result.usage,
        "request_bytes": result.request_bytes,
    }
    try:
        plan = plan_from_answers(result.answers, snapshot, config, state=task)
    except PolicyError as exc:
        plan = plan_for_unavailable(f"unusable Jev answers ({exc}); safe fallback, writing disabled", snapshot)
        return _finalize(plan, result, run_id, task, mode, config, project_root, started, errors=[f"PolicyError: {exc}"])
    return _finalize(plan, result, run_id, task, mode, config, project_root, started)


def _finalize(plan, result, run_id, task, mode, config, project_root, started, errors=None) -> dict:
    applied = _applied_route(mode, plan)
    payload = artifact_mod.build_artifact(
        run_id=run_id,
        task=task,
        plan=plan,
        mode=mode,
        applied_route=applied,
        config=config,
        model=getattr(result, "model", None),
        answers=getattr(result, "answers", None),
        latency_ms=getattr(result, "latency_ms", None),
        attempts=getattr(result, "attempts", None),
        errors=errors,
        started_at=started,
    )
    try:
        path = artifact_mod.write_artifact(project_root, run_id, payload, config.artifact_dir)
        payload["artifact"] = str(path)
    except OSError as exc:
        payload["artifact_error"] = f"{type(exc).__name__}: {exc}"
    return payload


def _summary(payload: dict) -> str:
    plan = payload["plan"]
    flags = [
        name
        for name, on in (
            ("scout", plan["run_scout"]),
            ("researcher", plan["run_researcher"]),
            ("test_analyst", plan["run_test_analyst"]),
            ("implementer", plan["run_implementer"]),
            ("deep_reviewer", plan["run_deep_reviewer"]),
        )
        if on
    ]
    return (
        f"route={plan['route']} applied={payload['applied_route']} mode={payload['mode']}\n"
        f"agents={','.join(flags) or 'none'} confirm={plan['needs_user_confirmation']}\n"
        f"reason={plan['reason']}\n"
        f"run_id={payload['run_id']} artifact={payload.get('artifact', 'not written')}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="agent_router", description="Jev-based request router (plan only, never executes).")
    sub = parser.add_subparsers(dest="command", required=True)
    route = sub.add_parser("route", help="classify a request and write a routing artifact")
    route.add_argument("--task", help="request text")
    route.add_argument("--task-file", help="read the request from a file")
    route.add_argument("--stdin", action="store_true", help="read the request from stdin")
    route.add_argument("--project-root", default=".", help="project root that owns .agent/artifacts (default: cwd)")
    route.add_argument("--mode", choices=("off", "observe", "assist", "enforce"), help="override ROUTER_MODE")
    route.add_argument("--json", action="store_true", help="print the full artifact JSON")
    route.add_argument("--quiet", action="store_true", help="suppress the human summary")
    args = parser.parse_args(argv)

    if args.command == "route":
        task = _read_task(args)
        if not task.strip() and not (args.task or args.task_file or args.stdin):
            print("route: no task given (use --task, --task-file or --stdin)", file=sys.stderr)
            return EXIT_USAGE
        started = time.monotonic()
        payload = route_once(task, args.project_root, args.mode)
        payload["harness_ms"] = int((time.monotonic() - started) * 1000)
        if args.json:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        elif not args.quiet:
            print(_summary(payload))
        return EXIT_OK

    return EXIT_USAGE


if __name__ == "__main__":
    raise SystemExit(main())
