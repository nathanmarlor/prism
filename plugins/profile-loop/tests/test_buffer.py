import time

from profile_loop_mcp import buffer


def _ev(dim, direction, weight, session, at, signal="correction"):
    return {"dimension": dim, "direction": direction, "weight": weight,
            "signal": signal, "session": session, "at": at, "note": ""}


def test_single_session_cannot_fire():
    now = time.time()
    # One session complains many times about length.
    buf = [_ev("length", "less", 1.0, "s1", now) for _ in range(5)]
    s = buffer.summarize(buf, now)
    assert buffer.crossed(s) == []          # session cap blocks single-session firing
    assert s["length"]["score"] <= buffer.SESSION_CAP


def test_two_sessions_fire():
    now = time.time()
    buf = [_ev("length", "less", 1.0, "s1", now),
           _ev("length", "less", 1.0, "s2", now)]
    fired = buffer.crossed(buffer.summarize(buf, now))
    assert len(fired) == 1
    assert fired[0]["dimension"] == "length"
    assert fired[0]["direction"] == "less"
    assert fired[0]["sessions"] == 2


def test_mixed_direction_holds():
    now = time.time()
    buf = [_ev("length", "less", 1.0, "s1", now),
           _ev("length", "less", 1.0, "s2", now),
           _ev("length", "more", 1.0, "s3", now),
           _ev("length", "more", 1.0, "s4", now)]
    s = buffer.summarize(buf, now)
    assert buffer.crossed(s) == []          # no clear direction
    reasons = [h["reason"] for h in buffer.held(s)]
    assert any("mixed" in r for r in reasons)


def test_recency_decay_lets_new_signal_win():
    now = time.time()
    old = now - 60 * 86400            # 60 days old, heavily decayed
    buf = [_ev("length", "more", 1.0, "s1", old),
           _ev("length", "more", 1.0, "s2", old),
           _ev("length", "less", 1.0, "s3", now),
           _ev("length", "less", 1.0, "s4", now)]
    fired = buffer.crossed(buffer.summarize(buf, now))
    assert fired and fired[0]["direction"] == "less"


def test_non_directional_never_fires():
    now = time.time()
    buf = [_ev("_global", "hold", 1.0, "s1", now, signal="accept"),
           _ev("_global", "hold", 1.0, "s2", now, signal="accept"),
           _ev("relevance", "unknown", 1.0, "s3", now, signal="rephrase"),
           _ev("relevance", "unknown", 1.0, "s4", now, signal="rephrase")]
    assert buffer.crossed(buffer.summarize(buf, now)) == []


def test_held_reports_single_session():
    now = time.time()
    buf = [_ev("tone", "warmer", 1.0, "s1", now)]
    held = buffer.held(buffer.summarize(buf, now))
    assert held and "one session" in held[0]["reason"]
