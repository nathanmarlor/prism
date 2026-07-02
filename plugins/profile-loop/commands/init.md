---
description: Start a Profile Loop from a one-line description of the style you want
argument-hint: [description of preferred style]
---

The user wants to start personalising their assistant's responses.

Call the `pl_init` tool with their description as `description` (for example
"warm and concise support replies"). If they gave no description, call it with
an empty string to cold start from default dimensions.

Then relay the target dimensions back to them in plain language, and explain the
important next step: the judge has to be validated against their own labelled
examples (`/profile-loop:validate`) before the loop is allowed to change
anything. Offer to help them assemble those labels.
