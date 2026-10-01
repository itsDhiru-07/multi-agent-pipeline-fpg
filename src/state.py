"""Shared pipeline state passed between LangGraph nodes.

Each agent reads the fields it needs from PipelineState and returns a dict
of updated fields; LangGraph merges those into the running state. Keeping
this as an explicit pydantic model (rather than a loose dict) is the data
contract between agents - every agent's input/output shape is documented
and validated here in one place.

DataFrames are NOT stored directly on this model. The graph is checkpointed
(src/graph.py, LangGraph SqliteSaver) so state is serialized to disk after
every node; a state carrying multi-megabyte DataFrames would bloat every
single checkpoint write, and LangGraph's msgpack serializer can't encode a
DataFrame at all. Instead, each DataFrame-producing agent spills its result
to a per-run parquet file (src/dataframe_store.py) and stores only the
*path* here. `*_df` properties below load from disk on access, so the rest
of the codebase reads `state.clean_df` exactly as before - the spill/load
indirection is contained to this one file.
"""
from __future__ import annotations

from typing import Any, Optional

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from src.dataframe_store import load


class PipelineState(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # --- config / run metadata ---
    source_path: str = "data/raw/hotel_bookings.csv"
    run_id: str = ""
    chunk_size: int = 5000

    # --- ingestion output (parquet paths, not DataFrames - see module docstring) ---
    raw_df_path: Optional[str] = None
    source_tags: dict[str, str] = Field(default_factory=dict)
    rows_ingested: int = 0
    chunks_processed: int = 0

    # --- data quality output ---
    clean_df_path: Optional[str] = None
    quarantined_df_path: Optional[str] = None
    quality_report: dict[str, Any] = Field(default_factory=dict)
    duplicate_rows_skipped: int = 0

    # --- KPI output ---
    kpi_by_agent_path: Optional[str] = None
    kpi_by_property_path: Optional[str] = None
    kpi_overall: dict[str, Any] = Field(default_factory=dict)

    # --- insight agent output ---
    insights: list[dict[str, Any]] = Field(default_factory=list)
    insight_cache_hit: bool = False

    # --- narrative agent output ---
    executive_report_md: str = ""
    narrative_cache_hit: bool = False

    # --- supervisor / observability ---
    run_log: list[dict[str, Any]] = Field(default_factory=list)
    degraded_mode: bool = False

    # --- control flow ---
    errors: list[str] = Field(default_factory=list)
    quarantine_rate: float = 0.0

    # --- convenience accessors: load-on-access from the parquet spill store ---
    @property
    def raw_df(self) -> Optional[pd.DataFrame]:
        return load(self.raw_df_path)

    @property
    def clean_df(self) -> Optional[pd.DataFrame]:
        return load(self.clean_df_path)

    @property
    def quarantined_df(self) -> Optional[pd.DataFrame]:
        return load(self.quarantined_df_path)

    @property
    def kpi_by_agent(self) -> Optional[pd.DataFrame]:
        return load(self.kpi_by_agent_path)

    @property
    def kpi_by_property(self) -> Optional[pd.DataFrame]:
        return load(self.kpi_by_property_path)
