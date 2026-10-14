"""The harmless subagent (scout) cannot cause side effects.

Restrictions live in the frontmatter permission block, not in prose. OpenCode V1
defaults an absent permission to "allow"
(https://dev.opencode.ai/docs/permissions: "Most permissions default to
'allow'"), so a role without an explicit deny is not harmless. These tests read
files statically and never launch an agent, call the network or touch Jev.
"""

from __future__ import annotations

import re
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
AGENT_DIR = ROOT / "agent"
PLUGIN = ROOT / "agent_router" / "plugin" / "jev-router.js"

# The role that must be side-effect free. Kept explicit so a new agent is not
# silently treated as harmless without its own denies.
HARMLESS_AGENTS = ("scout",)
# These agents' permission blocks must not change versus the base.
OTHER_AGENTS = ("orchestrator", "researcher", "reviewer", "test-analyst")

# Side-effect tools that must be an explicit deny (not merely absent, which
# defaults to allow).
MUST_DENY = ("edit", "bash", "webfetch", "websearch", "task")
# The only tools the harmless role may hold an explicit allow for.
MAY_ALLOW = {"read", "grep", "glob", "list"}

# provider/model, same shape the existing agents use (e.g. octava/gpt-5.6-luna).
MODEL_RE = re.compile(r"^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$")

CONTRACT_SECTIONS = ("## Findings", "## Not verified", "## Blocked")


def _agent_text(name: str) -> str:
    return (AGENT_DIR / f"{name}.md").read_text(encoding="utf-8")


def _frontmatter(text: str) -> str:
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        raise AssertionError("missing frontmatter")
    end = lines.index("---", 1)
    return "\n".join(lines[1:end])


def _permission_block(text: str) -> str:
    fm = _frontmatter(text).splitlines()
    start = next((i for i, line in enumerate(fm) if line.startswith("permission:")), None)
    if start is None:
        raise AssertionError("frontmatter has no permission block")
    return "\n".join(fm[start:])


def _permission_effects(text: str) -> dict[str, str]:
    """Flat tool -> effect for the harmless permission block.

    The harmless profile is deliberately flat. A nested map (like a bash command
    whitelist) would let a side-effect tool through, so a deeper indent is a
    failure rather than an entry to parse.
    """
    effects: dict[str, str] = {}
    for line in _permission_block(text).splitlines()[1:]:
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        if indent != 2 or ":" not in line:
            raise AssertionError(f"unexpected nested permission entry: {line!r}")
        key, _, value = line.strip().partition(":")
        effects[key.strip()] = value.strip()
    return effects


class HarmlessPermissions(unittest.TestCase):
    def test_harmless_denies_every_side_effect_tool(self):
        for name in HARMLESS_AGENTS:
            effects = _permission_effects(_agent_text(name))
            for tool in MUST_DENY:
                self.assertEqual(effects.get(tool), "deny", f"{name}: {tool} must be explicit deny")

    def test_harmless_allows_only_read_tools(self):
        for name in HARMLESS_AGENTS:
            effects = _permission_effects(_agent_text(name))
            allowed = {tool for tool, effect in effects.items() if effect == "allow"}
            self.assertLessEqual(allowed, MAY_ALLOW, f"{name}: unexpected allow: {sorted(allowed - MAY_ALLOW)}")

    def test_harmless_has_no_nested_bash_whitelist(self):
        # _permission_effects raises on nesting; calling it is the assertion.
        for name in HARMLESS_AGENTS:
            _permission_effects(_agent_text(name))


class OtherAgentsUnchanged(unittest.TestCase):
    """Constraint: no other agent's permission block became less strict."""

    def _base_ref(self) -> str:
        try:
            out = subprocess.run(
                ["git", "rev-list", "--max-parents=0", "HEAD"],
                cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace", check=True,
            )
        except (OSError, subprocess.CalledProcessError) as exc:
            self.skipTest(f"git base ref unavailable: {exc}")
        return out.stdout.split()[0]

    def _base_text(self, name: str, ref: str) -> str:
        try:
            out = subprocess.run(
                ["git", "show", f"{ref}:agent/{name}.md"],
                cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace", check=True,
            )
        except subprocess.CalledProcessError:
            self.skipTest(f"base copy of {name}.md missing at {ref}")
        return out.stdout

    def test_other_agents_permission_blocks_unchanged(self):
        ref = self._base_ref()
        for name in OTHER_AGENTS:
            base = _permission_block(self._base_text(name, ref))
            current = _permission_block(_agent_text(name))
            # Identical is stronger than "not less strict"; it proves no
            # loosening and no accidental edit.
            self.assertEqual(current, base, f"{name}: permission block differs from base commit {ref}")

    def test_implementer_has_intended_permissions(self):
        block = _permission_block(_agent_text("implementer"))
        expected = "\n".join((
            "permission:",
            "  edit: allow",
            "  task: deny",
            "  webfetch: deny",
            "  websearch: deny",
            "  bash:",
            '    "*": deny',
            '    "git status*": allow',
            '    "git diff*": allow',
            '    "git log*": allow',
            '    "git show*": allow',
            '    "git add *": allow',
            '    "git commit*": deny',
            '    "git push*": deny',
            '    "git reset*": deny',
            '    "git checkout*": deny',
            '    "git clean*": deny',
            '    "git rebase*": deny',
            '    "git stash*": deny',
        ))
        self.assertEqual(block, expected)


class PluginSurface(unittest.TestCase):
    """jev_route stays classification-only: one tool, no model switcher."""

    def setUp(self):
        self.source = PLUGIN.read_text(encoding="utf-8")

    def test_registers_only_jev_route(self):
        names = re.findall(r"\b(\w+)\s*:\s*tool\(", self.source)
        self.assertEqual(names, ["jev_route"], f"unexpected tool registrations: {names}")

    def test_module_exports_only_the_plugin(self):
        self.assertEqual(re.findall(r"export const (\w+)", self.source), ["plugin"])

    def test_no_model_switcher_invocation(self):
        lowered = self.source.lower()
        for bad in ("models use", "models set", "models reset"):
            self.assertNotIn(bad, lowered, f"plugin invokes {bad!r}")


class FrontmatterIntegrity(unittest.TestCase):
    def test_harmless_has_exactly_one_valid_model(self):
        for name in HARMLESS_AGENTS:
            models = [line for line in _frontmatter(_agent_text(name)).splitlines() if line.startswith("model:")]
            self.assertEqual(len(models), 1, f"{name}: expected exactly one model line, got {len(models)}")
            self.assertRegex(models[0].split(":", 1)[1].strip(), MODEL_RE)


class OutputContract(unittest.TestCase):
    def test_harmless_prompt_has_the_report_contract(self):
        for name in HARMLESS_AGENTS:
            body = _agent_text(name)
            for section in CONTRACT_SECTIONS:
                self.assertIn(section, body, f"{name}: missing {section}")


if __name__ == "__main__":
    unittest.main()
