"""Tests for LocalSLMJudge decision logic without loading a real model.

A fake llama returns scripted 'A'/'B' answers so we can verify the position
debiasing and confidence handling deterministically.
"""
import re

from profile_loop_mcp import judge as judge_mod
from profile_loop_mcp import evaluator


class FakeLlama:
    """Stands in for llama_cpp.Llama. `fn(prompt) -> 'A' or 'B'`."""
    def __init__(self, fn):
        self.fn = fn

    def create_chat_completion(self, messages, **kw):
        content = self.fn(messages[0]["content"])
        return {"choices": [{"message": {"content": content}}]}


def _spec():
    return evaluator.build_spec("concise replies")


def _sections(prompt):
    """Pull the A and B response bodies back out of the judge prompt."""
    a = re.search(r"Response A:\n(.*?)\n\nResponse B:", prompt, re.DOTALL).group(1)
    b = re.search(r"Response B:\n(.*?)\n\nWhich", prompt, re.DOTALL).group(1)
    return a, b


def test_position_bias_is_detected_as_tie():
    # A model that always says "A" (pure primacy bias) must not produce a
    # confident pick: the two orders will disagree.
    judge = judge_mod.LocalSLMJudge(_spec(), _llama=FakeLlama(lambda p: "A"))
    assert judge.pick_confident("q", "resp one", "resp two") is None
    assert judge.pick("q", "resp one", "resp two") == "a"  # safe default


def test_fair_judge_survives_the_swap():
    # A model that genuinely prefers the shorter response, regardless of
    # position, should pick it confidently in both orders.
    def prefers_shorter(prompt):
        a, b = _sections(prompt)
        return "A" if len(a.split()) <= len(b.split()) else "B"

    judge = judge_mod.LocalSLMJudge(_spec(), _llama=FakeLlama(prefers_shorter))
    short, long = "tiny answer", "a much longer answer " * 20
    assert judge.pick_confident("q", short, long) == "a"
    assert judge.pick_confident("q", long, short) == "b"
    assert judge.pick("q", short, long) == "a"


