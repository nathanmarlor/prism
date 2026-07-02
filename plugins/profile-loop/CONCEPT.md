# Profile Loop — design rationale

## The problem

Frontier assistants reset to a generic voice every session, and the only lever a
user has is to keep repeating themselves. You can fine-tune an open model to fix
this, but you cannot fine-tune a closed one like Claude or GPT. So on the models
most people actually use, personalization has to happen in the profile: the short
block of natural-language instructions the model reads before it answers.

## The move

Stop writing the profile by hand and start learning it. A small judge watches how
the user reacts to responses and proposes edits to the profile over time. It is
the same shape as reinforcement learning:

- the profile is the policy
- the judge is the reward model
- each edit is a step

What you gain over real RL is that the gradient is human-readable and reversible.
What you give up is bandwidth: a profile holds far less than weights can.

## Why context-space, not weight-space (DPO)

DPO is the right tool when you own the model and need depth. It bakes tacit
behaviour into weights from preference pairs. But it needs pairs, an open model,
and a training run, none of which fit a consumer on a closed model who just wants
shorter answers. These are complementary, not rivals. The loop handles the coarse
statable style layer live; DPO handles depth once you own weights. A mature system
could run the loop and occasionally distill a settled preference into weights.

## Going live means online RL

Recording live interactions is the version that needs zero user effort, but it
inherits online RL's hard parts:

- **No counterfactual.** You only see the one response you served, so the clean
  chosen-vs-rejected pair DPO relies on is gone. You fall back toward a scalar
  signal on the taken action.
- **Confounding.** Every interaction differs in topic and mood, so a low score is
  hard to attribute to style versus content versus a bad day.
- **Exploration on real users.** The assistant serves its single best answer, so
  it rarely observes what a different choice would have done, and every
  exploratory reply lands on a real person.

## What makes it tractable

You do not solve general online RL. You narrow it hard:

- **Corrections are the primary signal.** "Too long", "just the code" is an
  unambiguous, naturally contrastive statement about the response given. Treat
  silence as weak evidence, never as reward.
- **Buffer, threshold, corroborate.** Update only when a dimension clears a
  threshold across at least two sessions. This is the learning rate, and it is
  what stops the profile thrashing.
- **One bad session cannot win.** Per-session contribution is capped below the
  threshold, so a single frustrated user cannot overwrite a stable preference.
- **Do not grade your own homework.** Keep an independent check the training
  judge does not score, or you will reward-hack the judge.

## Honest limits

- A profile is a few hundred tokens. It cannot encode the fine gradients weights
  can, which is exactly the deep "sounds like us" behaviour DPO exists for.
- Text edits are discrete and noisy, so you inherit a memory-management problem:
  rules interfere and have to be consolidated.
- Stated is not followed. Even a perfect profile still has to be obeyed at
  inference, which models do imperfectly. That is a second lossy step weights do
  not have.

## Where the novelty is

The mechanism has precedent. Self-rewarding language models and iterative DPO use
a model's own judgments to build preference data. Reflexion frames verbal
feedback as a stand-in for weight updates. TextGrad and DSPy optimise text as the
target. Con-J trains the judge itself with DPO. The Plurai plugin supplies the
describe-to-deployed-judge piece.

The novel combination is a **validated** judge governing **live** edits to a
**persistent, per-user** profile as a deliberate closed loop, sitting on the
memory/profile surface real assistants already expose. That is a productization
and integration novelty rather than a new algorithm, which is fine. Nobody has
closed exactly this loop, and it is buildable now.

## What would need proving

- Does the judge stay trustworthy on live, messy traffic, not just the tidy
  validation set?
- Do buffered text-diffs improve steadily enough to feel like learning, or do
  they thrash?
- Does an independent eval confirm real gains rather than judge-pleasing?
- Is the right unit of reward corrections only, or weak implicit signals too?

Build order follows from this: the judge plus its validation harness is the
centre of gravity, so build and trust that first, then the loop around it.
