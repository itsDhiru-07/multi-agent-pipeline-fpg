"""Deterministic hashing helpers used for idempotency, caching, and run IDs.

Centralized here so every place that needs a stable key (row-level dedup,
LLM response cache, run/thread ID for checkpointing) derives it the same
way - all content-addressed, never wall-clock-based, which is what makes
re-running the pipeline against the same input safe and repeatable.
"""
from __future__ import annotations

import hashlib
import json
import os

import pandas as pd

# Columns that make a transaction row unique in the source feed. This is the
# natural key FPG's PMS integration would already use for reconciliation -
# same idea as the "missing/unmatched" transaction matching described in
# their real audit-trail feature.
NATURAL_KEY_COLUMNS = [
    "hotel", "property_id", "agent", "arrival_date_year", "arrival_date_month",
    "stays_in_weekend_nights", "stays_in_week_nights", "adr", "reservation_status",
]


def row_transaction_id(row: pd.Series) -> str:
    """Stable content hash for one transaction row, used as an idempotent
    dedup key - the same underlying transaction always hashes to the same
    ID, regardless of how many times the source file is re-ingested."""
    parts = [str(row.get(c, "")) for c in NATURAL_KEY_COLUMNS]
    raw = "|".join(parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def file_content_hash(path: str, block_size: int = 65536) -> str:
    """Hash of a file's bytes, used to derive a deterministic run_id so that
    re-running against an unchanged source file reuses the same run/output
    artifacts instead of minting a new timestamped run each time."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(block_size):
            h.update(chunk)
    return h.hexdigest()[:16]


def stable_json_hash(obj) -> str:
    """Hash of a JSON-serializable object with sorted keys, used as the LLM
    response cache key (same input KPI data + same model -> same cache
    entry, regardless of dict ordering)."""
    encoded = json.dumps(obj, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:24]


def derive_run_id(source_path: str) -> str:
    """Deterministic run_id: content hash of the source file. Two runs
    against byte-identical input always get the same run_id, which is what
    makes output overwrite (not duplicate) and checkpoint resume line up."""
    if os.path.exists(source_path):
        return f"run-{file_content_hash(source_path)}"
    return f"run-{hashlib.sha256(source_path.encode()).hexdigest()[:16]}"
