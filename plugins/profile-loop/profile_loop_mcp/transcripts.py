"""Mining preference pairs from Claude Code transcript histories.

Claude Code stores per-session transcripts as JSONL files. Each line is a turn
in the conversation. A *correction pair* is the mechanical sequence:

  1. User sends a prompt → assistant produces response A
  2. User sends a correction (e.g. "shorter") → assistant produces response B
  3. Response B is treated as the user-preferred correction.

``mine_pairs`` walks sessions chronologically and emits one pair per
correction. A correction is the immediately following assistant reply that
comes after a user message matching a known correction pattern.

``detect_corrections`` turns a raw correction string into (dimension, direction,
note) — reusing the same regex rules that ``signals.extract`` uses at runtime.
No new dependencies.

The schema is internal and version-dependent; all parsing is defensive. If a
line is unreadable, the session continues. If agreement comes back oddly low,
noise is the first place to look.
"""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any

from . import signals as sig_mod


# ---------------------------------------------------------------------------
# Session discovery
# ---------------------------------------------------------------------------

# Default transcript root — follows Claude Code's project structure.
DEFAULT_TRANSCRIPT_ROOT = Path.home() / ".claude" / "projects"


def find_sessions(root: str | Path | None = None) -> list[dict[str, Any]]:
    """Scan the transcript directory for session folders with .jsonl files.

    Returns a list of dicts, each with "id" and "turns_path" keys.
    Sessions are sorted by the oldest turn timestamp (defensive: falls back to
    mtime when a file is unreadable).
    """
    root = Path(root or os.environ.get("PROFILE_LOOP_TRANSCRIPTS", str(DEFAULT_TRANSCRIPT_ROOT)))
    if not root.is_dir():
        return []

    sessions: list[dict[str, Any]] = []
    for session_dir in root.iterdir():
        if not session_dir.is_dir():
            continue
        # A session folder may contain one or more JSONL files.
        jsonl_files = sorted(session_dir.glob("*.jsonl"))
        if not jsonl_files:
            continue
        # Use the first JSONL file as the transcript.
        sessions.append({
            "id": session_dir.name,
            "turns_path": str(jsonl_files[0]),
        })

    # Sort by mtime as a proxy for age.
    sessions.sort(key=lambda s: os.path.getmtime(s["turns_path"]))
    return sessions


# ---------------------------------------------------------------------------
# Turn iteration
# ---------------------------------------------------------------------------

def _is_tool_result_block(block: Any) -> bool:
    """Return True if this block is purely a tool result with no text."""
    if not isinstance(block, dict):
        return False
    kind = block.get("type", block.get("content_type", ""))
    return kind in ("tool_result", "tool_results", "result")


def _turn_text(turn: dict[str, Any]) -> str:
    """Extract readable text from a turn, skipping tool result blocks."""
    content = turn.get("content", turn.get("text", ""))
    if isinstance(content, str):
        # Could contain mixed block types. Extract only text blocks.
        if "\n" not in content and not content.strip():
            return ""
        # Heuristic: if content is raw JSON with blocks, parse it.
        if content.strip().startswith("["):
            try:
                blocks = json.loads(content)
                if isinstance(blocks, list):
                    parts = []
                    for b in blocks:
                        if isinstance(b, dict) and b.get("type") == "text":
                            parts.append(b.get("text", ""))
                    return "\n".join(parts).strip()
            except (json.JSONDecodeError, TypeError):
                pass
        return content.strip()

    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                if _is_tool_result_block(block):
                    continue
                text = block.get("text", block.get("content", ""))
                if isinstance(text, str) and text.strip():
                    parts.append(text.strip())
        return " ".join(parts)

    return str(content).strip()


def iter_turns(turns_path: str):
    """Yield turns from a JSONL file, skipping unreadable lines.

    Each item is a dict with at least a "type" (or "role") key and optionally
    "content" (str or list of blocks), "timestamp", and "session_id".
    """
    try:
        with open(turns_path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    turn = json.loads(line)
                except json.JSONDecodeError:
                    continue
                yield turn
    except (OSError, UnicodeDecodeError):
        return


# ---------------------------------------------------------------------------
# Correction detection
# ---------------------------------------------------------------------------

def detect_corrections(correction_text: str) -> list[dict[str, str]]:
    """Check ``correction_text`` against the built-in correction rules.

    Returns a list of dicts: ``{"dimension": str, "direction": str, "note": str}``.
    Reuses the same ``_CORRECTION_RULES`` that ``signals.extract`` uses at
    runtime. The list is empty when none of the patterns match.
    """
    found: list[dict[str, str]] = []
    text_lower = correction_text.lower()
    for pattern, dim, direction, note in sig_mod._CORRECTION_RULES:
        if re.search(pattern, text_lower):
            found.append({"dimension": dim, "direction": direction, "note": note})
    return found


# ---------------------------------------------------------------------------
# Pair mining
# ---------------------------------------------------------------------------

def mine_pairs(
    sessions: list[dict[str, Any]] | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Walk sessions, extract correction pairs.

    A pair is emitted when:
      1. A user turn has text content (signal: the correction),
      2. The *subsequent* assistant turn also has text content,
      3. ``detect_corrections(correction_text)`` returns ≥ 1 match.

    The pair record has:

      { "prompt": str,              # the user message before response A
        "response_a": str,           # assistant's first response
        "correction": str,           # the correction text
        "response_b": str,           # assistant's corrected response
        "session": str,
        "dimension": str,            # detected dimension (or "unknown")
        "direction": str,            # detected direction (or "unknown")
        "note": str,                 # correction note
        "at": float,                 # timestamp of the assistant turn
      }

    Pairs are returned in chronological order, limited to ``limit`` if given.
    """
    if sessions is None:
        sessions = find_sessions()

    pairs: list[dict[str, Any]] = []
    for session in sessions:
        sid = session["id"]
        turns = list(iter_turns(session["turns_path"]))

        # Build index of user → next assistant pairs.
        i = 0
        while i < len(turns):
            turn = turns[i]
            is_user = _is_user(turn)
            if not is_user:
                i += 1
                continue

            user_text = _turn_text(turn)
            if not user_text:
                i += 1
                continue

            # Scan ahead for the next assistant turn with content.
            # We need: user → assistant(text) → user(correction) → assistant(text).
            # Find the first assistant response.
            j = i + 1
            while j < len(turns) and not _is_assistant(turns[j]):
                j += 1
            if j >= len(turns):
                break

            response_a = _turn_text(turns[j])
            if not response_a:
                i = j + 1
                continue

            # Now look ahead for a correction user message.
            k = j + 1
            correction_text = None
            while k < len(turns):
                ct = _turn_text(turns[k])
                if _is_user(turns[k]) and ct:
                    correction_text = ct
                    k += 1
                    break
                k += 1

            if not correction_text:
                i = j + 1
                continue

            # Detect corrections on the correction text.
            detected = detect_corrections(correction_text)
            if not detected:
                i = j + 1
                continue

            # Get the assistant's response after the correction.
            while k < len(turns) and not _is_assistant(turns[k]):
                k += 1
            if k >= len(turns):
                i = j + 1
                continue

            response_b = _turn_text(turns[k])
            if not response_b:
                i = j + 1
                continue

            # Build the pair.
            ts = turns[j].get("timestamp", turns[j].get("at", time.time()))
            if isinstance(ts, str):
                try:
                    ts = float(ts)
                except ValueError:
                    ts = time.time()

            pair = {
                "prompt": user_text,
                "response_a": response_a,
                "correction": correction_text,
                "response_b": response_b,
                "session": sid,
                "dimension": detected[0]["dimension"],
                "direction": detected[0]["direction"],
                "note": detected[0]["note"],
                "at": ts,
            }
            pairs.append(pair)
            i = j + 1  # skip past the correction pair

    if limit is not None:
        pairs = pairs[:limit]
    return pairs


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_user(turn: dict[str, Any]) -> bool:
    role = (turn.get("role", "") or "").lower()
    ttype = (turn.get("type", "") or "").lower()
    return role == "user" or ttype in ("user", "human")


def _is_assistant(turn: dict[str, Any]) -> bool:
    role = (turn.get("role", "") or "").lower()
    ttype = (turn.get("type", "") or "").lower()
    return role == "assistant" or ttype in ("assistant", "ai", "model")
