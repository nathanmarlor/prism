import pytest

from profile_loop_mcp import diff, evaluator


# --- diff ---------------------------------------------------------------

def test_propose_add_then_replace():
    prof = diff.default_profile()
    fired = {"dimension": "length", "direction": "less", "score": 2.0, "sessions": 2}
    p = diff.propose(fired, prof, seq=1)
    assert p["kind"] == "add" and p["before"] is None
    prof2 = diff.apply(p, prof)
    assert "length" in prof2["rules"]

    # Now a later, opposite fire should be a replace with a real before-line.
    fired2 = {"dimension": "length", "direction": "more", "score": 2.0, "sessions": 2}
    p2 = diff.propose(fired2, prof2, seq=2)
    assert p2["kind"] == "replace" and p2["before"] is not None
    assert p2["after"] != p2["before"]


def test_propose_noop_when_already_set():
    prof = diff.default_profile()
    fired = {"dimension": "format", "direction": "list", "score": 2.0, "sessions": 2}
    p = diff.propose(fired, prof, seq=1)
    prof2 = diff.apply(p, prof)
    assert diff.propose(fired, prof2, seq=2) is None    # nothing new to say


def test_render_contains_rules():
    prof = diff.default_profile()
    prof = diff.apply(diff.propose(
        {"dimension": "tone", "direction": "warmer"}, prof, 1), prof)
    text = diff.render(prof)
    assert "warm" in text.lower()
    assert text.strip().startswith("Voice")


def test_render_diff_shape():
    prof = diff.default_profile()
    p = diff.propose({"dimension": "directness", "direction": "more"}, prof, 1)
    d = diff.render_diff(p)
    assert d.startswith("profile.md")
    assert "+ " in d


# --- evaluator ----------------------------------------------------------

def test_build_spec_from_keywords():
    spec = evaluator.build_spec("warm and concise support replies")
    names = {d["name"]: d["direction"] for d in spec["dimensions"]}
    assert names.get("length") == "less"
    assert names.get("tone") == "warmer"


def test_build_spec_seeds_when_empty():
    spec = evaluator.build_spec("just make it good")
    assert spec["dimensions"]          # never empty


def test_validate_returns_empty_when_no_labels():
    spec = evaluator.build_spec("concise replies")
    result = evaluator.validate([], spec)
    assert result["trusted"] is False
    assert result["n"] == 0
    assert "no labels" in result["reason"]
