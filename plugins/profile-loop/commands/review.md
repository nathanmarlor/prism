---
description: Review proposed profile edits as diffs, then apply the ones you approve
---

Call `pl_review` to generate edit proposals for every dimension that has crossed
the threshold. Each proposal is shown as a diff: the line being removed (if any)
and the line being added.

Present each diff to the user and let them decide. This approval step is a
feature, not a formality: the whole point of learning in text rather than
weights is that a human can read and veto each change. When they approve, call
`pl_apply` with the proposal id (or "all"). If they want to undo a past change,
use `pl_revert`. Never apply edits the user has not seen.
