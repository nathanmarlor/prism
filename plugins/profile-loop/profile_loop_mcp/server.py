"""MCP server for the Profile Loop.

Exposes the loop as tools Claude Code can call. The server holds no logic of its
own beyond orchestration: signal extraction, the threshold engine, diff
proposal, and validation all live in their own modules and are unit-tested
independently. Every tool reads and writes the local state via `store`.

Run standalone with:  python -m profile_loop_mcp.server
Claude Code starts it automatically via the plugin's .mcp.json.
"""
from __future__ import annotations

import json
import time
from typing import Any

from mcp.server.fastmcp import FastMCP

from . import store, signals, buffer, diff, evaluator, transcripts

mcp = FastMCP("profile-loop")


def _profile(state: dict[str, Any]) -> dict[str, Any]:
    return state.get("profile") or diff.default_profile()


@mcp.tool()
def pl_init(description: str = "") -> str:
    """Start or reset a loop from a one-line description of the preferred style.

    Pass something like "warm and concise support replies". Leave empty to cold
    start from sensible default dimensions. Returns the target dimensions and the
    next step (validating the judge).
    """
    state = store.load_state()
    spec = evaluator.build_spec(description or "")
    state["target"] = description or None
    state["judge"] = spec
    state["judge_validated"] = False
    state.setdefault("profile", diff.default_profile())
    store.save_state(state)
    dims = ", ".join(f"{d['name']}→{d['direction']}" for d in spec["dimensions"])
    return (f"Loop initialised. Target dimensions: {dims}\n"
            f"Rubric: {spec['rubric']}\n\n"
            "Next: validate the judge against ~20-30 of your own labelled pairs "
            "with pl_validate before the loop is allowed to act.")


@mcp.tool()
def pl_validate(labels_json: str) -> str:
    """Validate the judge against human labels before trusting it.

    labels_json is a JSON array of objects: {"prompt","a","b","pick"} where pick
    is "a" or "b" (the response you prefer). Reports agreement and whether the
    judge is now trusted. The loop refuses to apply edits until this passes.
    """
    state = store.load_state()
    if not state.get("judge"):
        return "No judge yet. Run pl_init first."
    try:
        labels = json.loads(labels_json)
    except json.JSONDecodeError as exc:
        return f"Could not parse labels_json: {exc}"

    result = evaluator.validate(labels, state["judge"])
    state["judge_validated"] = result["trusted"]
    store.save_state(state)

    lines = [f"Agreement with your labels: {result['agreement']:.0%} over {result['n']} pairs.",
             result["reason"]]
    for w in result.get("warnings", []):
        lines.append(f"Warning: {w}")
    lines.append("Judge is TRUSTED. The loop can now act." if result["trusted"]
                 else "Judge is NOT trusted yet. The loop will keep observing but not edit.")
    return "\n".join(lines)


@mcp.tool()
def pl_observe(prompt: str, response: str, followup: str = "", session: str = "default") -> str:
    """Record one live interaction as evidence.

    prompt/response are the exchange; followup is the user's next message if any
    (a correction like "shorter" is the strongest signal). session groups turns
    so no single session can dominate a decision.
    """
    state = store.load_state()
    events = signals.extract(prompt, response, followup or None, session)
    for e in events:
        state["buffer"].append(e.to_dict())
    state["counters"]["interactions"] += 1
    store.save_state(state)
    if not events:
        return "Recorded. No usable signal from this interaction."
    desc = ", ".join(f"{e.dimension}→{e.direction} ({e.signal})" for e in events)
    return f"Recorded {len(events)} signal(s): {desc}"


@mcp.tool()
def pl_status() -> str:
    """Show what the loop has learned, what it's watching, and what's ready."""
    state = store.load_state()
    now = time.time()
    summ = buffer.summarize(state["buffer"], now)
    fired = buffer.crossed(summ)
    watching = buffer.held(summ)

    out = [f"Target: {state.get('target') or '(cold start)'}",
           f"Judge trusted: {state.get('judge_validated', False)}",
           f"Interactions observed: {state['counters']['interactions']}",
           f"Profile version: {len(state.get('versions', []))}", ""]
    out.append("Ready to apply:" if fired else "Ready to apply: none")
    for f in fired:
        tag = "explicit" if f["explicit"] else "implicit"
        out.append(f"  - {f['dimension']} → {f['direction']} "
                   f"(score {f['score']}, {f['sessions']} sessions, {tag})")
    if watching:
        out.append("\nWatching (not yet actionable):")
        for h in watching:
            out.append(f"  - {h['dimension']} → {h['direction']}: {h['reason']}")
    if fired:
        out.append("\nRun pl_review to see the proposed edits.")
    return "\n".join(out)


@mcp.tool()
def pl_review() -> str:
    """Propose profile edits for every dimension that has crossed the threshold.

    Shows each as a diff and stores it for approval. Nothing is applied yet, and
    nothing is proposed unless the judge is trusted.
    """
    state = store.load_state()
    if not state.get("judge_validated"):
        return ("Judge is not trusted yet, so the loop will not propose edits. "
                "Run pl_validate with more or better labels first.")
    now = time.time()
    fired = buffer.crossed(buffer.summarize(state["buffer"], now))
    if not fired:
        return "Nothing has crossed the threshold. Keep observing."

    proposals = {}
    blocks = []
    prof = _profile(state)
    for f in fired:
        state["counters"]["proposal_seq"] += 1
        p = diff.propose(f, prof, state["counters"]["proposal_seq"])
        if p is None:
            continue
        proposals[p["id"]] = p
        blocks.append(f"[{p['id']}]\n{diff.render_diff(p)}")
    state["proposals"] = proposals
    store.save_state(state)
    if not blocks:
        return "The profile already reflects the current signal. Nothing to change."
    return ("Proposed edits (apply with pl_apply <id>, or pl_apply all):\n\n"
            + "\n\n".join(blocks))


@mcp.tool()
def pl_apply(proposal_id: str) -> str:
    """Apply a proposed edit by id, or 'all'. Versions the profile for revert."""
    state = store.load_state()
    proposals = state.get("proposals", {})
    if not proposals:
        return "No pending proposals. Run pl_review first."
    ids = list(proposals) if proposal_id.strip().lower() == "all" else [proposal_id]
    prof = _profile(state)
    applied = []
    for pid in ids:
        p = proposals.get(pid)
        if not p:
            continue
        prof = diff.apply(p, prof)
        applied.append(pid)
        version = len(state.get("versions", [])) + 1
        state.setdefault("versions", []).append(
            diff.snapshot(prof, version, f"{p['dimension']}→{p['direction']} ({pid})"))
    if not applied:
        return f"No matching proposal for '{proposal_id}'."
    state["profile"] = prof
    state["proposals"] = {k: v for k, v in proposals.items() if k not in applied}
    store.save_state(state)
    store.write_profile(diff.render(prof))
    return (f"Applied {', '.join(applied)}. Profile is now version "
            f"{len(state['versions'])}.\n\n--- profile.md ---\n{store.read_profile()}")


@mcp.tool()
def pl_revert(version: int = 0) -> str:
    """Revert to a previous profile version (0 = the one before the latest)."""
    state = store.load_state()
    versions = state.get("versions", [])
    if len(versions) < 2:
        return "Nothing to revert to yet."
    target = versions[version - 1] if version > 0 else versions[-2]
    state["profile"] = {"preamble": target["profile"]["preamble"],
                        "rules": dict(target["profile"]["rules"])}
    restore = len(versions) + 1
    state["versions"].append(diff.snapshot(state["profile"], restore,
                                            f"revert to v{target['version']}"))
    store.save_state(state)
    store.write_profile(diff.render(state["profile"]))
    return f"Reverted to v{target['version']} content.\n\n--- profile.md ---\n{store.read_profile()}"


@mcp.tool()
def pl_show_profile() -> str:
    """Show the current learned profile (the text the model reads)."""
    text = store.read_profile()
    return text if text.strip() else "No profile yet. Run pl_init."


@mcp.tool()
def pl_reset() -> str:
    """Erase the profile, buffer, and history for a clean start."""
    store.reset()
    return "Profile Loop state cleared."


@mcp.tool()
def pl_validate_from_transcripts(
    root: str | None = None,
    limit: int = 50,
) -> str:
    """Validate the judge by mining correction pairs from transcript history.

    Walks Claude Code session transcripts, extracts user→response→correction→
    improved-response sequences, infers a judge spec from the mined dimensions,
    and validates the current judge against the inferred spec.

    Pass ``root`` to override the default transcript directory
    (~/.claude/projects). ``limit`` caps the number of pairs mined.

    If the judge is trusted at the end the loop is ready to act. Otherwise the
    report points at signal quality so you know what to fix.
    """
    # 1. Mine pairs from transcripts.
    sessions = transcripts.find_sessions(root)
    if not sessions:
        rpt = (f"No transcript sessions found at {root or 'default location'}. "
               f"Ensure Claude Code has session history and "
               f"PROFILE_LOOP_TRANSCRIPTS points to the right place.")
        return rpt

    pairs = transcripts.mine_pairs(sessions, limit=limit)
    if not pairs:
        rpt = ("Mined 0 pairs. Either there were no correction sequences in the "
               "transcripts, or none matched known correction patterns. "
               "If your transcripts are in a custom location, pass the `root` "
               "argument with the path to the session directory.")
        return rpt

    dims_mined = len(set(p["dimension"] for p in pairs if p["dimension"] != "unknown"))

    # 2. Infer a judge spec from the mined pairs.
    spec = evaluator.spec_from_pairs(pairs)

    # 3. Build synthetic labels from pairs: each pair is one labelled example.
    #    The user clearly preferred response_b (it followed a correction they sent).
    synthetic_labels: list[dict[str, str]] = []
    for p in pairs:
        synthetic_labels.append({
            "prompt": p["prompt"],
            "a": p["response_a"],
            "b": p["response_b"],
            "pick": "b",  # correction response is the user-preferred one
        })

    # 4. Validate the current judge (or build a spec-only validation if no judge
    #    is set yet, which returns spec details for the user to review).
    state = store.load_state()
    if state.get("judge"):
        result = evaluator.validate(synthetic_labels, state["judge"])
    else:
        # No existing judge — return the inferred spec for the user to review.
        dim_str = (", ".join(f"{d['name']}→{d['direction']}" for d in spec["dimensions"])
                   if spec["dimensions"] else "(none)")
        lines = [
            f"Mined {len(pairs)} pairs across {dims_mined} dimension(s).",
            "",
            "Inferred spec:",
            f"  Description: {spec['description']}",
            f"  Dimensions: {dim_str}",
            f"  Rubric: {spec['rubric']}",
            "",
            "No judge is configured. Run pl_init with the rubric above to "
            "initialise a judge from this spec, then re-run this tool to "
            "validate it against the mined pairs.",
        ]
        return "\n".join(lines)

    # 5. Report.
    lines = [
        f"Mined {len(pairs)} pairs from {len(sessions)} session(s) across "
        f"{dims_mined} dimension(s).",
        f"Agreement: {result['agreement']:.0%} over {result['n']} labelled examples.",
        result["reason"],
    ]
    for w in result.get("warnings", []):
        lines.append(f"Warning: {w}")
    lines.append("Judge is TRUSTED. The loop can now act." if result["trusted"]
                 else "Judge is NOT trusted yet. Consider running pl_init with the inferred rubric above.")
    return "\n".join(lines)


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
