"""Shared pytest configuration for the profile-loop plugin.

Patches build_judge so that evaluate.validate() works without llama-cpp-python.
The injected judge always answers 'A' (i.e. prefers 'a') and is injected at the
evaluator module level so validate() uses it transparently.
"""
from profile_loop_mcp import judge as judge_mod


class _FakeJudge:
    """A minimal judge that always prefers 'a'.

    Uses the same interface as LocalSLMJudge: ``pick(prompt, a, b) -> 'a'|'b'``.
    validate() only calls ``judge.pick()`` — no _ask, _vote, or pick_confident
    calls are exercised by the existing tests, so this is sufficient.
    """

    def pick(self, prompt, a, b):  # noqa: ARG002
        return "a"


def pytest_configure(config):  # noqa: ARG001
    # Replace the real build_judge so evaluator.validate() doesn't import
    # llama-cpp-python at runtime during tests.  Patching on judge_mod because
    # evaluator.py imports it as  `from . import judge as judge_mod`.
    judge_mod.build_judge = lambda spec: _FakeJudge()
