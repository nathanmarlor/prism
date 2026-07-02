"""Proposing, rendering, applying, and reverting profile edits.

The profile is kept as structure, not free text: a short preamble plus one
guidance line per learned dimension. That keeps every edit clean (replace the
line for a dimension, or add it) and every change reversible. The rendered
markdown in profile.md is what the model actually reads.

The "gradient step is a diff" idea from the concept lives here. A fired
dimension maps to one canonical instruction; the proposal is the before/after
of a single line, which is exactly what gets shown to the user for approval.
"""
from __future__ import annotations

import time
from typing import Any


# (dimension, direction) -> the guidance line that expresses it.
_INSTRUCTIONS: dict[tuple[str, str], str] = {
    ("length", "less"): "Keep replies short and to the point. Lead with the answer; expand only when asked.",
    ("length", "more"): "Give fuller answers by default, with useful detail and context.",
    ("tone", "warmer"): "Keep a warm, relaxed tone. A little personality is welcome.",
    ("tone", "cooler"): "Keep a professional, businesslike tone.",
    ("format", "list"): "Prefer bulleted lists when they make the answer easier to scan.",
    ("format", "prose"): "Answer in prose. Avoid bulleted lists unless asked for one.",
    ("directness", "more"): "Be direct. State the answer plainly and skip unnecessary hedging.",
}


def default_profile() -> dict[str, Any]:
    return {"preamble": "Voice and formatting preferences, learned from how you actually respond:",
            "rules": {}}


def render(profile: dict[str, Any]) -> str:
    lines = [profile.get("preamble", "").strip(), ""]
    rules = profile.get("rules", {})
    if not rules:
        lines.append("(no learned preferences yet)")
    else:
        for dim in sorted(rules):
            lines.append(f"- {rules[dim]}")
    return "\n".join(lines).strip() + "\n"


def propose(fired: dict[str, Any], profile: dict[str, Any], seq: int) -> dict[str, Any] | None:
    """Build a single-line edit proposal for one fired dimension."""
    dim = fired["dimension"]
    direction = fired["direction"]
    after = _INSTRUCTIONS.get((dim, direction))
    if after is None:
        return None  # no canonical instruction for this dimension yet

    before = profile.get("rules", {}).get(dim)
    if before == after:
        return None  # profile already says this; nothing to do

    return {
        "id": f"p{seq}",
        "dimension": dim,
        "direction": direction,
        "kind": "replace" if before else "add",
        "before": before,
        "after": after,
        "score": fired.get("score"),
        "sessions": fired.get("sessions"),
        "explicit": fired.get("explicit"),
        "created_at": time.time(),
    }


def render_diff(proposal: dict[str, Any]) -> str:
    """A readable unified-style diff for one proposal."""
    out = [f"profile.md  ({proposal['kind']} · {proposal['dimension']} → {proposal['direction']})"]
    if proposal["before"]:
        out.append(f"- {proposal['before']}")
    out.append(f"+ {proposal['after']}")
    return "\n".join(out)


def apply(proposal: dict[str, Any], profile: dict[str, Any]) -> dict[str, Any]:
    """Return a new profile struct with the proposal applied."""
    new = {"preamble": profile.get("preamble", default_profile()["preamble"]),
           "rules": dict(profile.get("rules", {}))}
    new["rules"][proposal["dimension"]] = proposal["after"]
    return new


def snapshot(profile: dict[str, Any], version: int, reason: str) -> dict[str, Any]:
    return {
        "version": version,
        "profile": {"preamble": profile.get("preamble", ""),
                    "rules": dict(profile.get("rules", {}))},
        "text": render(profile),
        "applied_at": time.time(),
        "reason": reason,
    }
