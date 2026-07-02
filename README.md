# Prism

**Personalize Claude with online preference learning.**

Prism learns your conversational style from corrections and proposes reversible edits to your assistant profile. No fine-tuning, no API changes—just a growing list of natural-language preferences Claude reads before responding.

## The idea

You cannot fine-tune a closed frontier model like Claude. But you can change what it reads. Prism treats your assistant's profile (the short block of instructions it gets before each conversation) as a learnable policy. A small SLM judge watches how you actually respond to replies and proposes one-line edits to the profile over time—exactly the shape of online reinforcement learning, except the gradient is human-readable and reversible.

## How it works

1. **Init** — describe your style ("warm and concise")
2. **Validate the judge** — run it against 20–30 of your labelled pairs; must reach ≥80% agreement
3. **Observe** — record corrections as you work ("shorter", "less formal")
4. **Buffer** — evidence accumulates across sessions; single session cannot win
5. **Review** — when signal crosses threshold, system proposes a one-line edit
6. **Apply** — you approve the diff; profile is updated and versioned

## Key guarantees

- **No single session dominates** — evidence from 2+ conversations required to change the profile
- **Mixed signal holds** — conflicting preferences (some want shorter, some longer) block edits until one clearly leads
- **Evidence decays** — old preferences fade (14-day half-life), so the profile tracks taste drift
- **Judge must validate** — system refuses to act until the judge reaches 80%+ agreement on labelled pairs
- **All edits are reversible** — full version history; revert to any prior profile

## Install

```bash
# Clone the repo
git clone https://github.com/nathanmarlor/prism.git
cd prism/plugins/profile-loop

# Install the SLM judge and plugin
pip install -e ".[local]"
```

Then in Claude Code:
```
/plugin install /path/to/prism/plugins/profile-loop
/reload-plugins
/profile-loop:init warm and concise
```

## Quick start

```bash
# Describe your preference
/profile-loop:init warm and concise

# Validate the judge (mines from your transcript history)
/profile-loop:validate-from-transcripts

# Record corrections as you work
/profile-loop:observe <prompt> <response> <correction> [session_id]

# Check status
/profile-loop:status

# Apply proposed edits
/profile-loop:review
/profile-loop:apply all

# View the learned profile
/profile-loop:profile
```

## The judge

**LocalSLMJudge**: Qwen2.5-1.5B-Instruct running on CPU via llama.cpp.

Two design choices make a sub-2B model reliable:

1. **Constrained output** — grammar forces single `A` or `B` token (no waffling)
2. **Position debiasing** — every pair is compared in both orders; unstable answers are rejected

After validation, all judging is fully local and offline. The model (~1 GB at 4-bit) downloads once and is cached.

**Custom models:**
```bash
export PROFILE_LOOP_MODEL_PATH=/path/to/judge-q4_k_m.gguf
export PROFILE_LOOP_THREADS=8
```

**Distill your own judge** for maximum sharpness:
```bash
python scripts/distill_judge.py pairs.jsonl --rubric "concise; warm on tone" --train
export PROFILE_LOOP_MODEL_PATH=./judge-q4_k_m.gguf
```

## What this is not

- A substitute for weight-level preference training (DPO) when you need deep, tacit style
- A way to guarantee compliance—the model still has to obey the profile at inference
- A magic bullet—a 300-token profile cannot encode everything fine-tuning weights can

Profile Loop handles coarse, statable preferences (length, tone, format, directness) live and reversibly. Use it for fast per-user personalization; distill settled preferences into weights later with DPO if needed.

## Architecture

```
prism/
  plugins/profile-loop/
    .claude-plugin/       Claude Code plugin manifest
    .mcp.json             MCP server config
    commands/             Slash command docs
    skills/               Orchestration skill
    profile_loop_mcp/
      buffer.py           Threshold engine + safeguards
      signals.py          Correction → weighted evidence
      judge.py            SLM judge interface
      evaluator.py        Spec builder + validation
      diff.py             Proposal / apply / revert
      store.py            Local state + profile files
      server.py           MCP tools
      transcripts.py      Mine labels from session history
    scripts/
      distill_judge.py    Train + quantise a specialised judge
    tests/                41 unit tests (all passing)
```

## Tests

```bash
cd plugins/profile-loop
uv run pytest -v
# 41 tests covering buffer, signals, judge, diff, transcripts
```

## Tools

`pl_init`, `pl_validate`, `pl_validate_from_transcripts`, `pl_observe`, `pl_status`, `pl_review`, `pl_apply`, `pl_revert`, `pl_show_profile`, `pl_reset`.

State lives locally in `~/.config/profile-loop/`.

## Why Prism

A prism refracts light through its structure, separating it into spectrum. Prism learns to refract your assistant's responses through your stated preferences, breaking them into dimensions (length, tone, format, directness) and proposing adjustments. Your profile becomes a learnable lens.

## Roadmap

- [ ] Integrate profile injection into Claude Code context (auto-inject at inference)
- [ ] Support for hosted judges (API-based)
- [ ] Preference distillation pipeline (settle preferences → DPO weights)
- [ ] Cross-conversation preference mining (detect patterns without explicit correction)
- [ ] Preference sharing (export/import profiles)

## License

MIT

## References

- Concept: [CONCEPT.md](plugins/profile-loop/CONCEPT.md) — design rationale and trade-offs
- Judge: [LocalSLMJudge](plugins/profile-loop/profile_loop_mcp/judge.py) — position-debiased SLM
- Buffer: [Threshold engine](plugins/profile-loop/profile_loop_mcp/buffer.py) — session caps, decay, direction margin
- Validation: [Evaluator](plugins/profile-loop/profile_loop_mcp/evaluator.py) — judge validation + spec building

Built by [Version 1](https://www.version1.com/).
