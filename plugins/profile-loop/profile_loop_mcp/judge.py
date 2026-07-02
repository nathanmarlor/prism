"""The judge: the reward model of the loop.

In the full product this is a small, fine-tuned SLM-as-judge (built the way
the Plurai plugin builds evaluators). Here the interface is what matters.

The only judge is LocalSLMJudge: a small instruct model running on CPU via
llama.cpp (GGUF). Constrained output (grammar forces 'A'/'B') and position
debiasing (every pair is run in both orders) make even a sub-2B model
trustworthy enough to validate against and distil into.

The loop refuses to act on a judge it hasn't validated. An unvalidated judge
is the fastest way to quietly poison everything downstream.
"""
from __future__ import annotations

import os
import time
from typing import Any


# Grammar that forces the model to emit exactly one token: A or B. This is what
# lets a tiny model be a reliable judge; it cannot ramble or refuse.
_AB_GRAMMAR = 'root ::= "A" | "B"'

# A small, capable default that has an official GGUF build and runs comfortably
# on CPU. Override with PROFILE_LOOP_MODEL_REPO / _FILE / _PATH for anything
# else (Llama-3.2-1B, Gemma-3-1B, Qwen3.5, or your own distilled judge).
_DEFAULT_REPO = "Qwen/Qwen2.5-1.5B-Instruct-GGUF"
_DEFAULT_FILE = "*q4_k_m.gguf"



class LocalSLMJudge:
    """A small instruct model as judge, running on CPU via llama.cpp (GGUF).

    Two things make a sub-2B model trustworthy here rather than flaky:

      Constrained output   grammar forces a single 'A'/'B' token, so parsing is
                           exact and the model cannot waffle.

      Position debiasing   small models have a real bias toward whichever
                           response is shown first. Every pick is run in both
                           orders; only an answer that survives the swap counts
                           as confident. Disagreement is treated as "too close
                           to call", which is exactly the pair you do not want
                           in training data anyway.

    The model file downloads once, then inference is fully local and offline.
    No GPU: n_gpu_layers is pinned to 0.
    """

    def __init__(self, spec: dict[str, Any], *, model_path: str | None = None,
                 repo: str | None = None, filename: str | None = None,
                 n_threads: int | None = None, n_ctx: int = 2048, _llama: Any = None):
        self.spec = spec
        self.rubric = spec.get("rubric", "")
        self._grammar = None
        if _llama is not None:
            self._llm = _llama  # injected for tests; no real model loaded
            return

        from llama_cpp import Llama, LlamaGrammar  # imported lazily

        path = model_path or os.environ.get("PROFILE_LOOP_MODEL_PATH")
        repo = repo or os.environ.get("PROFILE_LOOP_MODEL_REPO", _DEFAULT_REPO)
        filename = filename or os.environ.get("PROFILE_LOOP_MODEL_FILE", _DEFAULT_FILE)
        n_threads = n_threads or int(os.environ.get("PROFILE_LOOP_THREADS", os.cpu_count() or 4))
        n_ctx = int(os.environ.get("PROFILE_LOOP_CTX", n_ctx))

        common = dict(n_ctx=n_ctx, n_threads=n_threads, n_gpu_layers=0, verbose=False)
        if path:
            self._llm = Llama(model_path=path, **common)
        else:
            self._llm = Llama.from_pretrained(repo_id=repo, filename=filename, **common)
        self._grammar = LlamaGrammar.from_string(_AB_GRAMMAR)

    def _prompt(self, prompt: str, first: str, second: str) -> str:
        return (f"You are a strict evaluator. Rubric: {self.rubric}\n\n"
                f"User prompt:\n{prompt}\n\n"
                f"Response A:\n{first}\n\nResponse B:\n{second}\n\n"
                "Which response better fits the rubric? Answer with only 'A' or 'B'.")

    def _ask(self, prompt: str) -> str:
        out = self._llm.create_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1, temperature=0.0, grammar=self._grammar)
        text = out["choices"][0]["message"]["content"].strip().upper()
        return "A" if text.startswith("A") else "B"

    def _vote(self, prompt: str, a: str, b: str) -> tuple[str, str]:
        # order 1: a shown as A, b as B  ->  'A' means a wins
        pick1 = "a" if self._ask(self._prompt(prompt, a, b)) == "A" else "b"
        # order 2: b shown as A, a as B  ->  'A' means b wins
        pick2 = "b" if self._ask(self._prompt(prompt, b, a)) == "A" else "a"
        return pick1, pick2

    def pick(self, prompt: str, a: str, b: str) -> str:
        p1, p2 = self._vote(prompt, a, b)
        return p1 if p1 == p2 else "a"  # unresolved after debias -> default a

    def pick_confident(self, prompt: str, a: str, b: str) -> str | None:
        """Like pick, but returns None when the two orders disagree.

        The loop uses this to drop low-confidence comparisons instead of
        feeding coin-flips into the buffer.
        """
        p1, p2 = self._vote(prompt, a, b)
        return p1 if p1 == p2 else None


def build_judge(spec: dict[str, Any]) -> LocalSLMJudge:
    """Build and return a LocalSLMJudge.

    Uses a small instruct model (default: Qwen2.5-1.5B-Instruct) run locally via
    llama.cpp, with no GPU required. Model file downloads once and is cached.

    Requires:
      - llama-cpp-python (or compatible)
      - Model GGUF file (auto-downloaded unless PROFILE_LOOP_MODEL_PATH is set)

    Environment variables:
      PROFILE_LOOP_MODEL_PATH       Override model path (local GGUF file)
      PROFILE_LOOP_MODEL_REPO       HuggingFace repo (default: Qwen/Qwen2.5-1.5B-Instruct-GGUF)
      PROFILE_LOOP_MODEL_FILE       Filename pattern (default: *q4_k_m.gguf)
      PROFILE_LOOP_THREADS          CPU threads (default: CPU count)
      PROFILE_LOOP_CTX              Context size (default: 2048)
    """
    try:
        return LocalSLMJudge(spec)
    except ImportError as e:
        raise RuntimeError(
            "LocalSLMJudge requires llama-cpp-python. Install with:\n"
            "  pip install 'profile-loop-mcp[local]'\n"
            "Then re-run. The model (~1GB) downloads once and is cached.") from e
