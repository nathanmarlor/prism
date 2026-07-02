# profile-loop-plugin

A Claude Code plugin marketplace containing **Profile Loop**: personalization as
online RL over an editable text profile.

## Install

```
/plugin marketplace add <owner>/profile-loop-plugin
/plugin install profile-loop@profile-loop-plugins
/reload-plugins
```

Then install the judge (a small local model):

```
pip install "profile-loop-mcp[local]"       # llama-cpp-python + huggingface-hub
```

Then start with `/profile-loop:init warm and concise support replies`.

See [`plugins/profile-loop/README.md`](plugins/profile-loop/README.md) for how
it works and [`plugins/profile-loop/CONCEPT.md`](plugins/profile-loop/CONCEPT.md)
for the design rationale.
