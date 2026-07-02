"""The evidence buffer and the threshold engine.

This is the learning rate of the whole system. A single interaction is almost
all noise, so nothing updates on one turn. Evidence accumulates per dimension,
decays with age, and only a dimension whose signal is strong, corroborated
across sessions, and directionally consistent is allowed to trigger an edit.

Three protections are built into the arithmetic rather than bolted on:

  Recency decay   old evidence fades (exponential half-life), so the profile
                  tracks preference drift instead of freezing on early signal.

  Session cap     any one session contributes at most SESSION_CAP toward a
                  dimension. This is the "one bad session can't win" guardrail:
                  because the cap is below the threshold, no single session can
                  cross it alone, no matter how many times the user repeats a
                  complaint. Firing requires corroboration across sessions.

  Direction margin   a dimension with conflicting signal (some users want it
                  shorter, some longer) stays put until one direction clearly
                  leads. Mixed evidence is a reason to hold, not to guess.
"""
from __future__ import annotations

import math
from typing import Any


# Tuning knobs. These are the dials a product team would expose.
HALFLIFE_DAYS = 14.0        # evidence loses half its weight every two weeks
SESSION_CAP = 1.0           # max contribution from a single session per direction
THRESHOLD = 1.5             # decayed, capped score needed to trigger an edit
DIRECTION_MARGIN = 0.5      # dominant direction must lead the runner-up by this

# Dimensions that carry no directional edit and therefore never auto-fire.
_NON_DIRECTIONAL = {"_global", "relevance"}

_SECONDS_PER_DAY = 86400.0


def _decay(weight: float, age_seconds: float) -> float:
    if age_seconds <= 0:
        return weight
    halflife_seconds = HALFLIFE_DAYS * _SECONDS_PER_DAY
    return weight * math.pow(0.5, age_seconds / halflife_seconds)


def summarize(buffer: list[dict[str, Any]], now: float) -> dict[str, Any]:
    """Collapse raw evidence into a per-dimension, per-direction picture.

    Returns { dimension: { "directions": {dir: score},
                           "sessions": {dir: n_distinct_sessions},
                           "explicit": {dir: bool},
                           "dominant": dir, "score": float, "runner_up": float } }
    """
    # dimension -> direction -> session -> capped running weight
    acc: dict[str, dict[str, dict[str, float]]] = {}
    explicit: dict[str, dict[str, bool]] = {}

    for ev in buffer:
        dim = ev["dimension"]
        direction = ev["direction"]
        session = ev["session"]
        decayed = _decay(float(ev["weight"]), now - float(ev["at"]))

        by_dir = acc.setdefault(dim, {}).setdefault(direction, {})
        # Cap the *per-session* contribution to this direction.
        by_dir[session] = min(SESSION_CAP, by_dir.get(session, 0.0) + decayed)

        if ev.get("signal") == "correction":
            explicit.setdefault(dim, {}).setdefault(direction, True)

    summary: dict[str, Any] = {}
    for dim, dirs in acc.items():
        dir_scores = {d: sum(sessions.values()) for d, sessions in dirs.items()}
        dir_sessions = {d: len(sessions) for d, sessions in dirs.items()}
        ordered = sorted(dir_scores.items(), key=lambda kv: kv[1], reverse=True)
        dominant, top = ordered[0]
        runner = ordered[1][1] if len(ordered) > 1 else 0.0
        summary[dim] = {
            "directions": dir_scores,
            "sessions": dir_sessions,
            "explicit": explicit.get(dim, {}),
            "dominant": dominant,
            "score": round(top, 3),
            "runner_up": round(runner, 3),
        }
    return summary


def crossed(summary: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the dimensions whose evidence clears every bar.

    A dimension fires only if it is directional, its dominant direction meets
    the threshold, that direction is supported by more than one session (the
    cap makes single-session firing impossible), and it clearly leads any
    competing direction.
    """
    fired: list[dict[str, Any]] = []
    for dim, s in summary.items():
        if dim in _NON_DIRECTIONAL:
            continue
        direction = s["dominant"]
        if direction in ("unknown", "hold"):
            continue
        score = s["score"]
        sessions = s["sessions"].get(direction, 0)
        leads = (score - s["runner_up"]) >= DIRECTION_MARGIN
        if score >= THRESHOLD and sessions >= 2 and leads:
            fired.append({
                "dimension": dim,
                "direction": direction,
                "score": score,
                "sessions": sessions,
                "explicit": bool(s["explicit"].get(direction)),
            })
    fired.sort(key=lambda f: f["score"], reverse=True)
    return fired


def held(summary: dict[str, Any]) -> list[dict[str, Any]]:
    """Dimensions with signal that is real but not yet actionable.

    Useful for /status: shows the user what the loop is watching but not acting
    on, and why (below threshold, mixed direction, or single-session only).
    """
    out: list[dict[str, Any]] = []
    fired_dims = {f["dimension"] for f in crossed(summary)}
    for dim, s in summary.items():
        if dim in _NON_DIRECTIONAL or dim in fired_dims:
            continue
        direction = s["dominant"]
        sessions = s["sessions"].get(direction, 0)
        if s["score"] <= 0:
            continue
        if direction == "unknown":
            reason = "direction unclear (a miss, but no stated fix)"
        elif sessions < 2:
            reason = "only one session so far"
        elif (s["score"] - s["runner_up"]) < DIRECTION_MARGIN:
            reason = "mixed signal between directions"
        else:
            reason = "below threshold"
        out.append({
            "dimension": dim, "direction": direction,
            "score": s["score"], "sessions": sessions, "reason": reason,
        })
    out.sort(key=lambda h: h["score"], reverse=True)
    return out
