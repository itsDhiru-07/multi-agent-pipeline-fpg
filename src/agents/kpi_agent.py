"""KPI Transformation Agent - rule-based, no LLM (pure pandas).

Computes the same category of metrics FPG's IN-Sights layer reports on:
upsell conversion rate, incremental revenue, ADR lift from upgrades, attach
rate for ancillary add-ons, and average guest CSAT - rolled up per agent
and per property.
"""
from __future__ import annotations

import pandas as pd

from src.dataframe_store import spill
from src.state import PipelineState


def _add_upsell_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["was_upgraded"] = df["assigned_room_type"] != df["reserved_room_type"]
    df["had_addon"] = (df["required_car_parking_spaces"] > 0) | (df["total_of_special_requests"] > 0)
    df["upsold"] = df["was_upgraded"] | df["had_addon"]
    # baseline ADR estimated as mean ADR for that reserved_room_type/meal combo among non-upgraded stays
    baseline = (
        df[~df["was_upgraded"]]
        .groupby(["reserved_room_type", "meal"])["adr"]
        .mean()
        .rename("baseline_adr")
    )
    df = df.merge(baseline, on=["reserved_room_type", "meal"], how="left")
    df["baseline_adr"] = df["baseline_adr"].fillna(df["adr"].mean())
    df["incremental_revenue"] = (df["adr"] - df["baseline_adr"]).clip(lower=0)
    return df


def _rollup(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    g = df.groupby(group_col, dropna=True)
    out = g.agg(
        transactions=("adr", "count"),
        upsell_conversion_rate=("upsold", "mean"),
        avg_adr=("adr", "mean"),
        total_incremental_revenue=("incremental_revenue", "sum"),
        avg_incremental_revenue=("incremental_revenue", "mean"),
        attach_rate=("had_addon", "mean"),
        avg_csat=("csat_score", "mean"),
        cancellation_rate=("is_canceled", "mean"),
    ).reset_index()

    out["upsell_conversion_rate"] = (out["upsell_conversion_rate"] * 100).round(2)
    out["attach_rate"] = (out["attach_rate"] * 100).round(2)
    out["cancellation_rate"] = (out["cancellation_rate"] * 100).round(2)
    out["avg_adr"] = out["avg_adr"].round(2)
    out["total_incremental_revenue"] = out["total_incremental_revenue"].round(2)
    out["avg_incremental_revenue"] = out["avg_incremental_revenue"].round(2)
    out["avg_csat"] = out["avg_csat"].round(2)

    return out.sort_values("total_incremental_revenue", ascending=False).reset_index(drop=True)


def run(state: PipelineState) -> dict:
    df = _add_upsell_columns(state.clean_df)

    kpi_by_agent = _rollup(df, "agent").rename(columns={"agent": "agent_id"})
    kpi_by_property = _rollup(df, "property_id")

    kpi_overall = {
        "total_transactions": int(len(df)),
        "overall_upsell_conversion_rate": round(float(df["upsold"].mean()) * 100, 2),
        "overall_incremental_revenue": round(float(df["incremental_revenue"].sum()), 2),
        "overall_avg_csat": round(float(df["csat_score"].mean()), 2),
        "overall_cancellation_rate": round(float(df["is_canceled"].mean()) * 100, 2),
        "active_agents": int(df["agent"].nunique()),
        "active_properties": int(df["property_id"].nunique()),
    }

    return {
        "kpi_by_agent_path": spill(state.run_id, "kpi_by_agent", kpi_by_agent),
        "kpi_by_property_path": spill(state.run_id, "kpi_by_property", kpi_by_property),
        "kpi_overall": kpi_overall,
    }
