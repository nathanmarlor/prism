"""Building the evaluator spec and validating the judge before trusting it.

`build_spec` turns a one-line description ("warm and concise support replies")
into a small set of target dimensions and a rubric. This is the step the Plurai
platform performs in the real product; here it is a transparent keyword mapping
so the whole thing is inspectable and offline.

`validate` is the step most projects skip and the one that decides whether the
rest is worth anything. It runs the judge over the user's own labelled pairs
and reports agreement. It also runs a cheap reward-hacking check: if the judge's
choices line up almost perfectly with "whichever response is shorter", that is a
warning sign that it is grading length rather than the thing you care about.
"""
from __future__ import annotations

import re
from typing import Any

from . import judge as judge_mod

# keyword -> (dimension, direction)
_KEYWORDS: list[tuple[str, str, str]] = [
    (r"concise|brief|short|terse|to the point|succinct", "length", "less"),
    (r"detailed|thorough|in-?depth|comprehensive|fuller", "length", "more"),
    (r"warm|friendly|casual|approachable|relaxed", "tone", "warmer"),
    (r"formal|professional|businesslike|serious", "tone", "cooler"),
    (r"bullet|list|scannable", "format", "list"),
    (r"prose|paragraph|no bullets", "format", "prose"),
    (r"direct|blunt|no hedging|plain-?spoken", "directness", "more"),
]

TRUST_THRESHOLD = 0.8       # judge must agree with human labels at least this often
LENGTH_BIAS_FLAG = 0.9      # if choices match "shorter wins" above this, warn


def build_spec(description: str) -> dict[str, Any]:
    found: dict[str, str] = {}
    for pattern, dim, direction in _KEYWORDS:
        if re.search(pattern, description, re.IGNORECASE):
            found.setdefault(dim, direction)

    if not found:
        # Seed with the two most common style axes so the loop can start.
        found = {"length": "less", "tone": "warmer"}

    dimensions = [{"name": dim, "direction": direction,
                   "description": f"prefers responses that are {direction} on {dim}"}
                  for dim, direction in found.items()]
    rubric = "A good response is: " + "; ".join(
        f"{d['direction']} on {d['name']}" for d in dimensions) + "."
    return {"description": description, "dimensions": dimensions, "rubric": rubric}


def _words(text: str) -> int:
    return len(re.findall(r"\S+", text or ""))


def validate(labels: list[dict[str, Any]], spec: dict[str, Any]) -> dict[str, Any]:
    """Run the judge over labelled pairs and report whether to trust it.

    Each label: {"prompt", "a", "b", "pick"} where pick is "a" or "b" (the
    human's preferred response).
    """
    if not labels:
        return {"trusted": False, "agreement": 0.0, "n": 0,
                "reason": "no labels provided"}

    judge = judge_mod.build_judge(spec)
    agree = 0
    shorter_alignment = 0
    for ex in labels:
        got = judge.pick(ex["prompt"], ex["a"], ex["b"])
        if got == ex["pick"]:
            agree += 1
        shorter = "a" if _words(ex["a"]) <= _words(ex["b"]) else "b"
        if got == shorter:
            shorter_alignment += 1

    n = len(labels)
    agreement = agree / n
    length_bias = shorter_alignment / n
    trusted = agreement >= TRUST_THRESHOLD

    warnings = []
    if length_bias >= LENGTH_BIAS_FLAG and not _length_is_target(spec):
        warnings.append(
            "Judge chose the shorter response almost every time, but length is "
            "not a stated target. It may be grading length rather than quality.")

    reason = ("agreement meets the trust threshold" if trusted
              else f"agreement {agreement:.0%} is below the {TRUST_THRESHOLD:.0%} threshold; "
                   "add or fix labels, or refine the description")
    return {"trusted": trusted, "agreement": round(agreement, 3),
            "length_bias": round(length_bias, 3), "n": n,
            "warnings": warnings, "reason": reason}


def _length_is_target(spec: dict[str, Any]) -> bool:
    return any(d["name"] == "length" for d in spec.get("dimensions", []))


def spec_from_pairs(pairs: list[dict[str, Any]]) -> dict[str, Any]:
    """Infer a judge spec from mined preference pairs.

    Walks the list, counts (dimension, direction) hits from the pair metadata,
    then keeps dimensions where one direction is clearly dominant. The rubric is
    a natural-language summary of the strongest signals.  If no dimensions
    survived the signal test the function falls back to a minimal spec so the
    loop can still run.
    """
    dim_votes: dict[str, dict[str, int]] = {}
    for p in pairs:
        dim = p.get("dimension", "unknown")
        dirn = p.get("direction", "unknown")
        if dim == "unknown":
            continue
        dim_votes.setdefault(dim, {"more": 0, "less": 0, "warmer": 0,
                                   "cooler": 0, "list": 0, "prose": 0,
                                   "direct": 0})
        dim_votes[dim][dirn] = dim_votes[dim].get(dirn, 0) + 1

    dimensions: list[dict[str, str]] = []
    for dim, votes in dim_votes.items():
        total = sum(votes.values())
        if total < 2:
            continue  # need at least 2 hits on the same dimension
        best_dir = max(votes, key=votes.get)  # type: ignore[arg-type]
        best_count = votes[best_dir]
        # Dominance ratio: best direction must be > 60 % of signals.
        if best_count / total < 0.6:
            continue
        dimensions.append({
            "name": dim,
            "direction": best_dir,
            "description": f"prefers responses that are {best_dir} on {dim} "
                           f"({best_count}/{total} signals)",
        })

    if not dimensions:
        # Fall back: no clear signal — nothing to optimise yet.
        dimensions = []
        rubric = "Insufficient correction signal to build a rubric."
    else:
        rubric = "A good response is: " + "; ".join(
            f"{d['direction']} on {d['name']}" for d in dimensions) + "."

    return {
        "description": "inferred from transcript corrections",
        "dimensions": dimensions,
        "rubric": rubric,
    }
