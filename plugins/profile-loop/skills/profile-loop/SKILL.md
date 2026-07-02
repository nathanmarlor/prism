---
name: profile-loop
description: >
  Use when a user wants their assistant to learn their style preferences over
  time instead of being told the same thing every session. Covers setting up a
  preference loop, validating the judge, recording live interactions, and
  reviewing and applying small edits to a persistent profile. Trigger on
  requests like "make it remember I prefer short answers", "stop being so
  formal with me", "learn how I like responses", or any mention of the Profile
  Loop, preference learning, or personalising tone, length, or format.
---

# Profile Loop

Personalization done as online reinforcement learning, except the policy is a
block of natural-language text (the profile) rather than model weights. A small
judge watches how the user actually reacts and proposes small, reversible edits
to the profile. Because the model's weights are closed, this is the way to
personalise a frontier assistant: change what it reads, not what it is.

## The mental model

- The **profile** is the policy: a short list of guidance lines the model reads.
- The **judge** is the reward model: it scores or compares responses.
- Each **edit** is a gradient step, shown as a readable diff the user can veto.

## The loop, and the order that matters

1. **Init** (`pl_init`) — turn a one-line description into target dimensions
   (length, tone, format, directness). An empty description cold starts from
   defaults.
2. **Validate the judge** (`pl_validate`) — the step that must not be skipped.
   Run the judge over 20-30 of the user's own labelled pairs. If agreement is
   below the trust threshold, fix the labels or the description before going on.
   The loop refuses to apply any edit while the judge is untrusted.
   Alternatively, use `pl_validate_from_transcripts` to mine correction pairs
   directly from Claude Code session transcripts — no manual labelling required.
3. **Observe** (`pl_observe`) — record real interactions. Corrections ("shorter",
   "just the code", "less formal") are the strongest signal. A rephrase is a weak
   "miss". Silence is barely a whisper. Use a stable session id per conversation.
4. **Status** (`pl_status`) — see what has crossed the threshold, what is being
   watched, and why anything is being held.
5. **Review and apply** (`pl_review`, then `pl_apply`) — look at each proposed
   edit as a diff and approve or reject it. Approval is deliberate on purpose.

## The rules that keep it stable

State these to the user when relevant, because they explain why the loop
sometimes does nothing:

- **No update on one turn.** A single interaction is almost all noise. Evidence
  has to accumulate.
- **One session cannot win.** Each session's contribution is capped below the
  firing threshold, so a change needs corroboration across at least two
  sessions. A single frustrated session cannot rewrite a stable preference.
- **Mixed signal means hold.** If some evidence says shorter and some says
  longer, the loop waits for a clear direction rather than guessing.
- **Old evidence fades.** Recency decay means the profile tracks how the user's
  taste drifts instead of freezing on early signal.

## Honest limits to surface

Do not oversell this. A profile holds coarse, statable preferences (length,
tone, format), not the deep tacit "sounds exactly like me" behaviour that needs
weight-level training such as DPO. A perfect profile still has to be obeyed at
inference, so learning a preference is not the same as guaranteeing it. And
because the judge grades the same live traffic it learns from, keep an
independent check now and then so the model is not just learning to please the
judge.

## Tools

`pl_init`, `pl_validate`, `pl_validate_from_transcripts`, `pl_observe`,
`pl_status`, `pl_review`, `pl_apply`, `pl_revert`, `pl_show_profile`,
`pl_reset`. State lives locally under `~/.config/profile-loop/`.

## The judge

The judge is the reward model and must pass validation before the loop is
allowed to act. It is a small instruct model (default: Qwen2.5-1.5B-Instruct,
4-bit quantized) run locally via llama.cpp.

**Setup:**

```
pip install "profile-loop-mcp[local]"
```

After installation, the model (~1 GB) downloads once on first use and is cached
locally. All inference is then offline and CPU-only. Two design choices make
a sub-2B model trustworthy:

- **Constrained output** — grammar restricts the model to `A` or `B` only.
- **Position debiasing** — every pair is compared in both orders; only a
  stable decision is treated as confident.

**Custom models:** Point it at any GGUF you like with environment variables:

```
export PROFILE_LOOP_MODEL_PATH=/path/to/judge-q4_k_m.gguf
export PROFILE_LOOP_MODEL_REPO=bartowski/Llama-3.2-1B-Instruct-GGUF
export PROFILE_LOOP_MODEL_FILE="*Q4_K_M.gguf"
export PROFILE_LOOP_THREADS=8
```

**Distilling your own judge:** For the sharpest judge, distil a specialized
one for your rubric with `scripts/distill_judge.py`, then set
`PROFILE_LOOP_MODEL_PATH` to the resulting GGUF.
