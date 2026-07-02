---
description: Validate the judge against your own labelled examples before trusting it
argument-hint: [optional path to a labels JSON file]
---

Validating the judge is the step that decides whether everything downstream is
trustworthy. Do not skip it and do not rush it.

**Two ways to validate:**

1. **Manual labels** — Help the user produce 20-30 labelled pairs (prompt + two
   candidate responses + which one they prefer). Assemble as a JSON array of
   `{"prompt","a","b","pick"}` objects where `pick` is "a" or "b". Then call
   `pl_validate` with `labels_json` containing that array.

2. **From transcript history** — Call `pl_validate_from_transcripts` to mine
   correction pairs directly from your Claude Code session transcripts. This is
   the fastest path: it walks sessions, extracts user→response→correction→
   improved-response sequences, infers a judge spec, and validates automatically.
   Pass a custom `root` if your transcripts are in a non-default location, and
   use `limit` to cap how many pairs to mine.

If the judge is not trusted, say so plainly and help them add or fix labels
(rather than pushing on). A confident judge that grades the wrong thing poisons
the whole loop.
