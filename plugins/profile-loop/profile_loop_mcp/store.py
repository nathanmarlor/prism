"""Local, per-machine storage for the Profile Loop.

Everything a user's loop needs lives under a single config directory:
  profile.md        the current learned profile (what the model reads)
  state.json        dimensions, evidence buffer, version history, judge spec

Nothing here is sent anywhere. The MCP server reads and writes these files;
the deterministic engine (buffer/diff/signals) operates on the state dict.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any


def config_dir() -> Path:
    """Resolve the config directory, honouring an override for tests."""
    override = os.environ.get("PROFILE_LOOP_HOME")
    base = Path(override) if override else Path.home() / ".config" / "profile-loop"
    base.mkdir(parents=True, exist_ok=True)
    return base


def _state_path() -> Path:
    return config_dir() / "state.json"


def _profile_path() -> Path:
    return config_dir() / "profile.md"


def _empty_state() -> dict[str, Any]:
    return {
        "created_at": time.time(),
        "target": None,          # the user's one-line description, or None
        "dimensions": {},        # name -> {"description", "direction_hint"}
        "judge": None,           # evaluator spec once built
        "judge_validated": False,
        "buffer": [],            # list of evidence events (see signals/buffer)
        "versions": [],          # list of {"version", "text", "applied_at", "reason"}
        "proposals": {},         # id -> proposal dict awaiting apply
        "counters": {"interactions": 0, "proposal_seq": 0},
    }


def load_state() -> dict[str, Any]:
    path = _state_path()
    if not path.exists():
        state = _empty_state()
        save_state(state)
        return state
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def save_state(state: dict[str, Any]) -> None:
    tmp = _state_path().with_suffix(".json.tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(state, fh, indent=2, ensure_ascii=False)
    tmp.replace(_state_path())


def read_profile() -> str:
    path = _profile_path()
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def write_profile(text: str) -> None:
    _profile_path().write_text(text, encoding="utf-8")


def reset() -> None:
    """Wipe everything. Used by tests and the /profile-loop:reset flow."""
    for p in (_state_path(), _profile_path()):
        if p.exists():
            p.unlink()
