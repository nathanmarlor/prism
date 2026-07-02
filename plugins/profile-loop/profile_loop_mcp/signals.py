"""Turn a single live interaction into evidence.

The design bet from the concept: corrections are the primary, near-clean
signal. A user saying "shorter" or "just give me the code" is an unambiguous
statement about the response they just got, and it is naturally contrastive.
Everything softer (a rephrase, silence, a thank-you) is weak evidence, never
treated as reward on its own.

Each interaction produces zero or more Evidence events. An event names the
dimension it bears on, a direction (which way to move), a weight (how much to
trust it), and the session it came from (so one session cannot dominate).

The classifier here is deliberately simple and explainable: keyword and shape
heuristics. A stronger LLM classifier can be dropped in behind the same
`extract` signature without touching the buffer engine. The point is that the
engine never sees raw text, only typed, attributed evidence.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, asdict
from typing import Any


# Weights are intentionally coarse. Explicit correction dominates; silence is
# barely a whisper. These are the "reward magnitudes" of the loop.
W_CORRECTION = 1.0
W_REPHRASE = 0.4
W_ACCEPT = 0.3
W_SILENCE = 0.1


@dataclass
class Evidence:
    dimension: str        # e.g. "length", "tone", "format"
    direction: str        # "less" / "more" / "warmer" / "cooler" / "list" / "prose" ...
    weight: float
    signal: str           # human-readable signal type
    session: str
    at: float
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# --- correction patterns -------------------------------------------------
# (regex, dimension, direction, note)
_CORRECTION_RULES: list[tuple[str, str, str, str]] = [
    (r"\b(shorter|too long|less wordy|be brief|briefer|concise|tl;?dr|get to the point|cut it down|too verbose)\b",
     "length", "less", "asked for shorter"),
    (r"\b(just (the )?(code|answer|number|link)|only the|no preamble|skip the (intro|explanation|pleasantries))\b",
     "length", "less", "wanted only the essential"),
    (r"\b(more detail|explain more|too short|elaborate|expand|go deeper|say more)\b",
     "length", "more", "asked for more detail"),
    (r"\b(less formal|too formal|lighten up|be more casual|relax the tone|stop being so stiff)\b",
     "tone", "warmer", "wanted warmer/looser"),
    (r"\b(more formal|too casual|be professional|less chatty|stop being so friendly)\b",
     "tone", "cooler", "wanted more formal"),
    (r"\b(use (a )?(bullet|list)|make it a list|bullet points)\b",
     "format", "list", "wanted a list"),
    (r"\b(no bullets?|not a list|in prose|write it out|paragraphs?)\b",
     "format", "prose", "wanted prose"),
    (r"\b(stop hedging|be direct|just tell me|don'?t waffle|quit qualifying)\b",
     "directness", "more", "wanted directness"),
]

_REPHRASE_HINTS = re.compile(
    r"\b(that'?s not what i (meant|asked)|no,? i meant|you misunderstood|try again|that'?s wrong|not quite)\b",
    re.IGNORECASE,
)

_ACCEPT_HINTS = re.compile(
    r"\b(thanks|thank you|perfect|great|that works|exactly|nice|got it|ship it)\b",
    re.IGNORECASE,
)


def extract(
    prompt: str,
    response: str,
    followup: str | None,
    session: str,
    now: float | None = None,
) -> list[Evidence]:
    """Classify one interaction into evidence events.

    `followup` is the user's next message, if any. Silence (no followup) is a
    weak, ambiguous signal and is only emitted when the response was unusually
    long, since "silence after a wall of text" weakly suggests disengagement.
    """
    now = now if now is not None else time.time()
    events: list[Evidence] = []
    text = (followup or "").lower()

    if followup:
        matched_correction = False
        for pattern, dim, direction, note in _CORRECTION_RULES:
            if re.search(pattern, text, re.IGNORECASE):
                events.append(Evidence(dim, direction, W_CORRECTION, "correction",
                                       session, now, note))
                matched_correction = True

        if not matched_correction and _REPHRASE_HINTS.search(followup):
            # A miss: the answer didn't land, but the user didn't say how to fix
            # it. Attribute weakly to "relevance"; direction is unknown.
            events.append(Evidence("relevance", "unknown", W_REPHRASE, "rephrase",
                                   session, now, "user signalled a miss"))

        if not matched_correction and _ACCEPT_HINTS.search(followup):
            # Positive but low-information: reinforce the current profile lightly,
            # attributed to no single dimension.
            events.append(Evidence("_global", "hold", W_ACCEPT, "accept",
                                   session, now, "user approved"))
    else:
        # No followup. Only treat as (weak) evidence when the reply was long.
        if _word_count(response) > 180:
            events.append(Evidence("length", "less", W_SILENCE, "silence",
                                   session, now, "silence after a long reply"))

    return events


def _word_count(text: str) -> int:
    return len(re.findall(r"\S+", text or ""))
