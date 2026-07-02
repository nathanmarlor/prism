import time

from profile_loop_mcp import signals


def test_explicit_correction_shorter():
    ev = signals.extract("q", "a long answer", "can you make it shorter?", "s1")
    assert any(e.dimension == "length" and e.direction == "less"
               and e.signal == "correction" for e in ev)


def test_correction_more_detail():
    ev = signals.extract("q", "short", "please elaborate, too short", "s1")
    assert any(e.dimension == "length" and e.direction == "more" for e in ev)


def test_tone_correction():
    ev = signals.extract("q", "resp", "too formal, lighten up", "s1")
    assert any(e.dimension == "tone" and e.direction == "warmer" for e in ev)


def test_format_list_request():
    ev = signals.extract("q", "resp", "make it a list please", "s1")
    assert any(e.dimension == "format" and e.direction == "list" for e in ev)


def test_rephrase_is_weak_and_unknown():
    ev = signals.extract("q", "resp", "no, that's not what I meant", "s1")
    assert any(e.signal == "rephrase" and e.direction == "unknown" for e in ev)
    assert all(e.weight <= signals.W_REPHRASE for e in ev)


def test_accept_is_global_hold():
    ev = signals.extract("q", "resp", "perfect, thanks!", "s1")
    assert any(e.dimension == "_global" and e.direction == "hold" for e in ev)


def test_silence_only_after_long_reply():
    long_reply = "word " * 200
    ev_long = signals.extract("q", long_reply, None, "s1")
    ev_short = signals.extract("q", "brief", None, "s1")
    assert any(e.signal == "silence" for e in ev_long)
    assert ev_short == []


def test_correction_outweighs_silence():
    corr = signals.extract("q", "r", "shorter", "s1")[0]
    assert corr.weight > signals.W_SILENCE
