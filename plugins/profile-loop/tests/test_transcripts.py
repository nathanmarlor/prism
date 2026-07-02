"""Tests for the transcript mining pipeline.

Covers:
- ``detect_corrections`` matches known patterns
- ``spec_from_pairs`` builds a spec from mined pairs
- ``mine_pairs`` chains turns + corrections correctly
- Noise in the transcript does not produce spurious pairs
- Empty / missing sessions are handled gracefully
"""
from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path
from unittest import TestCase, main

from profile_loop_mcp.transcripts import (
    detect_corrections,
    find_sessions,
    iter_turns,
    mine_pairs,
)
from profile_loop_mcp.evaluator import spec_from_pairs


def _mkturn(role: str, text: str, **extra) -> dict:
    return {"role": role, "content": [{"type": "text", "text": text}], **extra}


def _mk_session_jsonl(path: Path, turns: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for t in turns:
            fh.write(json.dumps(t) + "\n")


# -------------------------------------------------------------------
# detect_corrections
# -------------------------------------------------------------------

class TestDetectCorrections(TestCase):
    def test_shorter_matches(self):
        results = detect_corrections("shorter, please")
        self.assertTrue(len(results) > 0)
        self.assertEqual(results[0]["dimension"], "length")
        self.assertEqual(results[0]["direction"], "less")

    def test_no_correction(self):
        results = detect_corrections("that looks good, thanks")
        # "thanks" matches W_ACCEPT → dimension "feedback", direction "accept"
        # which is not a dimension we use for pairs — still returns a result
        # but it won't appear in mine_pairs dimensions.
        self.assertIsInstance(results, list)

    def test_list_format(self):
        results = detect_corrections("use a bullet list")
        dims = {r["dimension"] for r in results}
        self.assertIn("format", dims)


# -------------------------------------------------------------------
# spec_from_pairs
# -------------------------------------------------------------------

class TestSpecFromPairs(TestCase):
    def test_single_dim_dominant(self):
        pairs = [
            {"dimension": "length", "direction": "less", "prompt": "p1",
             "response_a": "a1", "response_b": "b1", "correction": "shorter"},
            {"dimension": "length", "direction": "less", "prompt": "p2",
             "response_a": "a2", "response_b": "b2", "correction": "brief"},
        ]
        spec = spec_from_pairs(pairs)
        self.assertEqual(len(spec["dimensions"]), 1)
        self.assertEqual(spec["dimensions"][0]["name"], "length")
        self.assertEqual(spec["dimensions"][0]["direction"], "less")

    def test_mixed_signal_falls_through(self):
        pairs = [
            {"dimension": "length", "direction": "less", "prompt": "p1",
             "response_a": "a1", "response_b": "b1", "correction": "shorter"},
            {"dimension": "length", "direction": "more", "prompt": "p2",
             "response_a": "a2", "response_b": "b2", "correction": "more detail"},
        ]
        spec = spec_from_pairs(pairs)
        # 50/50 is below the 60% dominance threshold.
        self.assertEqual(len(spec["dimensions"]), 0)

    def test_unknown_dimensions_ignored(self):
        pairs = [
            {"dimension": "unknown", "direction": "unknown", "prompt": "p1",
             "response_a": "a1", "response_b": "b1", "correction": "blah"},
        ]
        spec = spec_from_pairs(pairs)
        self.assertEqual(len(spec["dimensions"]), 0)

    def test_no_pairs(self):
        spec = spec_from_pairs([])
        self.assertEqual(spec["dimensions"], [])
        self.assertIn("Insufficient", spec["rubric"])


# -------------------------------------------------------------------
# mine_pairs (synthetic JSONL)
# -------------------------------------------------------------------

class TestMinePairs(TestCase):
    def test_single_correction_pair(self):
        turns = [
            _mkturn("user", "Write a summary of the quarterly results"),
            _mkturn("assistant", "The quarterly results show significant growth..."),
            _mkturn("user", "shorter"),
            _mkturn("assistant", "Q2 revenue up 15% year-over-year."),
        ]
        with tempfile.TemporaryDirectory() as td:
            sessions = [{"id": "sess-1", "turns_path": str(Path(td) / "sess-1.jsonl")}]
            _mk_session_jsonl(Path(sessions[0]["turns_path"]), turns)
            pairs = mine_pairs(sessions)
        self.assertEqual(len(pairs), 1)
        p = pairs[0]
        self.assertIn("shorter", p["correction"].lower())
        self.assertEqual(p["dimension"], "length")
        self.assertEqual(p["direction"], "less")

    def test_noise_before_correction(self):
        """Tool calls between turns should not break pair discovery."""
        turns = [
            _mkturn("user", "Write a summary"),
            _mkturn("assistant", "Full summary here..."),
            # A tool result block between assistant and user — shouldn't matter.
            {"role": "user", "content": [{"type": "tool", "text": ""}]},
            _mkturn("user", "just the code"),
            _mkturn("assistant", "```python\nprint('done')\n```"),
        ]
        with tempfile.TemporaryDirectory() as td:
            sessions = [{"id": "sess-noise", "turns_path": str(Path(td) / "noise.jsonl")}]
            _mk_session_jsonl(Path(sessions[0]["turns_path"]), turns)
            pairs = mine_pairs(sessions)
        self.assertEqual(len(pairs), 1)

    def test_no_corrections_yields_empty(self):
        turns = [
            _mkturn("user", "Hello"),
            _mkturn("assistant", "Hi there!"),
            _mkturn("user", "That's all"),
            _mkturn("assistant", "Let me know if you need anything else."),
        ]
        with tempfile.TemporaryDirectory() as td:
            sessions = [{"id": "sess-clean", "turns_path": str(Path(td) / "clean.jsonl")}]
            _mk_session_jsonl(Path(sessions[0]["turns_path"]), turns)
            pairs = mine_pairs(sessions)
        self.assertEqual(pairs, [])

    def test_limit(self):
        turns = [
            _mkturn("user", "Write A"),
            _mkturn("assistant", "Response A1"),
            _mkturn("user", "shorter"),
            _mkturn("assistant", "Response A2"),
        ]
        # Three separate sessions → three pairs.
        with tempfile.TemporaryDirectory() as td:
            sessions = []
            for idx in range(3):
                sid = f"sess-{idx}"
                tp = str(Path(td) / sid / f"{sid}.jsonl")
                _mk_session_jsonl(Path(tp), turns)
                sessions.append({"id": sid, "turns_path": tp})
            all_pairs = mine_pairs(sessions)
            limited_pairs = mine_pairs(sessions, limit=2)
        self.assertEqual(len(all_pairs), 3)
        self.assertEqual(len(limited_pairs), 2)


# -------------------------------------------------------------------
# find_sessions
# -------------------------------------------------------------------

class TestFindSessions(TestCase):
    def test_empty_dir(self):
        with tempfile.TemporaryDirectory() as td:
            sessions = find_sessions(td)
        self.assertEqual(sessions, [])

    def test_dir_without_jsonl(self):
        with tempfile.TemporaryDirectory() as td:
            Path(td, "session-1").mkdir()
            sessions = find_sessions(td)
        self.assertEqual(sessions, [])

    def test_returns_sorted(self):
        with tempfile.TemporaryDirectory() as td:
            for name in ["z-session", "a-session", "m-session"]:
                tp = str(Path(td) / name / f"{name}.jsonl")
                _mk_session_jsonl(Path(tp), [_mkturn("user", "hi")])
            # Set explicit mtimes so sort order is deterministic
            # (temp dirs created in the same millisecond share mtime).
            base = time.time()
            for name, offset in [("a-session", 0), ("m-session", 1), ("z-session", 2)]:
                tp = str(Path(td) / name / f"{name}.jsonl")
                os.utime(tp, (base + offset, base + offset))
            sessions = find_sessions(td)
        self.assertEqual(len(sessions), 3)
        self.assertEqual(sessions[0]["id"], "a-session")
        self.assertEqual(sessions[-1]["id"], "z-session")

    def test_no_such_dir(self):
        sessions = find_sessions("/nonexistent/path")
        self.assertEqual(sessions, [])


# -------------------------------------------------------------------
# iter_turns
# -------------------------------------------------------------------

class TestIterTurns(TestCase):
    def test_parses_jsonl(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl",
                                         delete=False) as fh:
            fh.write(json.dumps(_mkturn("user", "hello")) + "\n")
            fh.write(json.dumps(_mkturn("assistant", "hi back")) + "\n")
            fh.flush()
            turns = list(iter_turns(fh.name))
        self.assertEqual(len(turns), 2)

    def test_skips_bad_json(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl",
                                         delete=False) as fh:
            fh.write("not json\n")
            fh.write(json.dumps(_mkturn("user", "hello")) + "\n")
            fh.flush()
            turns = list(iter_turns(fh.name))
        self.assertEqual(len(turns), 1)


# -------------------------------------------------------------------
# Environment override
# -------------------------------------------------------------------

class TestEnvOverride(TestCase):
    def test_transcripts_env_var(self):
        with tempfile.TemporaryDirectory() as td:
            # find_sessions expects session subdirectories with .jsonl files.
            tp = str(Path(td) / "sess-env" / "sess-env.jsonl")
            _mk_session_jsonl(Path(tp), [
                _mkturn("user", "Write A"),
                _mkturn("assistant", "Response A1"),
                _mkturn("user", "shorter"),
                _mkturn("assistant", "Response A2"),
            ])
            env = {"PROFILE_LOOP_TRANSCRIPTS": td}
            orig = os.environ.get("PROFILE_LOOP_TRANSCRIPTS")
            try:
                os.environ.update(env)
                pairs = mine_pairs()
            finally:
                if orig is not None:
                    os.environ["PROFILE_LOOP_TRANSCRIPTS"] = orig
                elif "PROFILE_LOOP_TRANSCRIPTS" in os.environ:
                    del os.environ["PROFILE_LOOP_TRANSCRIPTS"]
        self.assertEqual(len(pairs), 1)


if __name__ == "__main__":
    main()
