---
description: Record one live interaction as evidence for the loop
argument-hint: [paste the exchange, and any user follow-up]
---

Record a real interaction so the loop can learn from it.

From what the user pastes, identify the original prompt, the response that was
given, and the user's follow-up message if there was one. A follow-up like
"shorter" or "just the code" is the strongest possible signal, so capture it
carefully. Use a stable `session` id per conversation so the loop's
per-session guardrail works.

Call `pl_observe` with `prompt`, `response`, `followup`, and `session`. Report
back what signal (if any) was extracted. Remind the user that no single
interaction changes anything; evidence has to build up across sessions.
