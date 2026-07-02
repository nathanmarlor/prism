# Profile Loop

Personalization as online RL over an editable text profile.

You cannot fine-tune a closed frontier model per user. You can change what it
reads. The Profile Loop treats the assistant's profile (the short block of
natural-language instructions it reads before answering) as the thing that
learns. A small judge watches how you actually react to responses and proposes
small, reversible edits to that profile over time.

Map it onto reinforcement learning and it lines up cleanly: the profile is the
policy, the judge is the reward model, and each edit is a step. The difference
from real RL is that the gradient is a diff you can read and undo.

## How it works

1. **Init** — describe the style you want ("warm and concise support replies")
   and the plugin picks out target dimensions: length, tone, format, directness.
2. **Validate the judge** — this is the step that matters most. The judge is run
   over 20-30 of your own labelled pairs and only trusted if it agrees with you.
   Until then the loop observes but never edits.
3. **Observe** — record real interactions. A correction like "shorter" or "just
   the code" is the strongest signal. Rephrases are weak. Silence barely counts.
4. **Review and apply** — when a preference is corroborated across sessions, the
   loop proposes a one-line edit as a diff. You approve or reject it.

## Why it behaves conservatively

Three protections are built into the arithmetic, not bolted on:

- **No update on one turn.** Evidence has to accumulate before anything moves.
- **One session cannot win.** Each session is capped below the firing threshold,
  so a change needs support from at least two sessions. One bad day cannot
  overwrite a stable preference.
- **Mixed signal holds.** Conflicting evidence is a reason to wait, not guess.

Old evidence also decays, so the profile follows your taste as it drifts rather
than freezing on the first thing it saw.

## Install

```
/plugin marketplace add <owner>/profile-loop-plugin
/plugin install profile-loop@profile-loop-plugins
/reload-plugins
```

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/). The MCP server
starts automatically.

## Use

```
/profile-loop:init warm and concise support replies
/profile-loop:validate            # label a handful of pairs; trust the judge
/profile-loop:validate-from-transcripts  # mine correction pairs from session history; trust the judge
/profile-loop:observe             # feed it real interactions as they happen
/profile-loop:status              # see what's learned, watched, and ready
/profile-loop:review              # approve or reject proposed edits
/profile-loop:profile             # view the current learned profile
```

`pl_validate_from_transcripts` is the fastest path: it walks your Claude Code
session transcripts, mines user→response→correction→improved-response sequences,
infers a judge spec, and validates automatically. Pass a custom `root` if your
transcripts are in a non-default location.

## The judge

The judge is the reward model of the loop and is critical to trustworthiness.
It is a small instruct model (Qwen2.5-1.5B by default) run locally via llama.cpp,
no GPU required.

```
pip install "profile-loop-mcp[local]"       # installs llama-cpp-python + huggingface-hub
```

The model file downloads once (~1 GB at 4-bit quantization), then all judging
is fully local and offline. Two design choices make a sub-2B model reliable:

1. **Constrained output** — grammar forces a single `A` or `B` token, so the
   model cannot waffle or refuse.
2. **Position debiasing** — every comparison is run in both orders; only a
   decision that survives the swap counts as confident. A comparison that
   flips when you swap the order is treated as "too close to call" rather
   than a noisy coin-flip.

Override the default model by setting environment variables:

```
export PROFILE_LOOP_MODEL_PATH=/path/to/your-model-q4_k_m.gguf
# or use a HuggingFace repo:
export PROFILE_LOOP_MODEL_REPO=bartowski/Llama-3.2-1B-Instruct-GGUF
export PROFILE_LOOP_MODEL_FILE="*Q4_K_M.gguf"
export PROFILE_LOOP_THREADS=8               # optional; defaults to CPU count
```

### Distilling your own judge

The off-the-shelf SLM works, but a judge distilled for your specific rubric is
sharper and can be smaller and faster still. `scripts/distill_judge.py` builds a
training set from your preference pairs, LoRA-fine-tunes a tiny base to emit
`A`/`B`, and gives you the llama.cpp steps to produce a quantised GGUF:

```
pip install "profile-loop-mcp[distill]"
python scripts/distill_judge.py pairs.jsonl --rubric "concise on length; warmer on tone" --train
export PROFILE_LOOP_MODEL_PATH=./judge-q4_k_m.gguf
```

Then re-run `/profile-loop:validate` to confirm the distilled judge still agrees
with your labels before trusting it.

## What this is not

A profile holds coarse, statable preferences. It is not a substitute for
weight-level preference training like DPO when you need deep, tacit style that
the context window cannot hold. The two are complementary: run the loop live for
fast per-user personalization, and distill a settled preference into weights
when it is worth baking in. See `CONCEPT.md` for the full design rationale and
trade-offs.

## Development

```
uv run pytest        # 22 tests over the deterministic core
```

The engine (signal extraction, the threshold buffer, diff proposal, validation)
has no third-party dependencies and is unit-tested independently of the MCP
layer.

## Layout

```
profile-loop/
  .claude-plugin/plugin.json     manifest
  .mcp.json                      MCP server declaration
  commands/                      slash commands
  skills/profile-loop/           orchestration skill
  profile_loop_mcp/
    store.py                     local state + profile files
    signals.py                   interaction -> weighted evidence
    buffer.py                    threshold engine + guardrails
    diff.py                      proposal / apply / revert
    evaluator.py                 spec builder + validation harness
    judge.py                     judge interface + offline + CPU SLM + hosted
    server.py                    MCP tools
  scripts/
    distill_judge.py             train + quantise a specialised CPU judge
  tests/
```
