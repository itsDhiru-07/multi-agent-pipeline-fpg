"""Spill/load helper for DataFrames that flow through checkpointed state.

LangGraph's SQLite checkpointer serializes PipelineState after every node so
a run can resume from its last completed step. Its serializer only handles
JSON/msgpack-safe types, and real checkpoint backends (Redis, Postgres, a
managed workflow store) never want multi-megabyte DataFrames inlined into a
state blob anyway - that bloats every checkpoint write, not just the final
one.

The pattern here: each agent that produces a DataFrame writes it to a
per-run parquet file under `.cache/<run_id>/` and puts only the *path*
(a string) into PipelineState. Any node that needs the DataFrame back reads
it from disk. This keeps checkpointed state small and msgpack-serializable,
and doubles as the memory-efficiency story - intermediate results live on
disk, not held redundantly in process memory across every checkpoint.
"""
from __future__ import annotations

import os

import pandas as pd

STORE_DIR = ".cache/dataframes"


def run_dir(run_id: str) -> str:
    d = os.path.join(STORE_DIR, run_id or "default")
    os.makedirs(d, exist_ok=True)
    return d


def spill(run_id: str, name: str, df: pd.DataFrame) -> str:
    """Writes df to a parquet file scoped to this run and returns its path."""
    path = os.path.join(run_dir(run_id), f"{name}.parquet")
    df.to_parquet(path, index=False)
    return path


def load(path: str | None) -> pd.DataFrame | None:
    if not path or not os.path.exists(path):
        return None
    return pd.read_parquet(path)
