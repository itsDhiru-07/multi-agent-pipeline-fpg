"""Local content-addressed cache for LLM agent responses.

The Insight and Narrative agents are the only paid/non-deterministic stages
in the pipeline. Re-running the pipeline against unchanged KPI data (e.g.
re-generating a report after a UI restart, or resuming a checkpointed run)
should not re-spend tokens on an LLM call that would produce the same
input->output mapping. This cache is keyed on a hash of (agent name, model,
system prompt, user prompt) - identical inputs always hit the same entry,
so cache correctness is content-based, not time-based.

Deliberately a flat JSON file rather than a database: cache entries are
small, reads/writes are infrequent (once per agent per run), and a JSON
file is trivial to inspect/clear by hand during development.
"""
from __future__ import annotations

import json
import os
from typing import Optional

CACHE_DIR = ".cache"
CACHE_FILE = os.path.join(CACHE_DIR, "llm_cache.json")


def _load() -> dict:
    if not os.path.exists(CACHE_FILE):
        return {}
    try:
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def _save(cache: dict) -> None:
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2)


def get(cache_key: str) -> Optional[str]:
    return _load().get(cache_key)


def put(cache_key: str, value: str) -> None:
    cache = _load()
    cache[cache_key] = value
    _save(cache)
