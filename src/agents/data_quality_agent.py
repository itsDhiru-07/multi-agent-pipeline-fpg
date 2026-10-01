"""Data Quality / Reconciliation Agent - rule-based, no LLM.

Mirrors FPG's real audit-trail states (approved / denied / error / locked)
and their "missing/unmatched PMS data" reconciliation problem: rows with a
null frontline-agent ID can't be attributed to anyone for coaching, rows
with impossible values (negative ADR) are data errors, and duplicate
transactions indicate a double-sync from the PMS feed.

Deduplication here is **idempotency-safe by construction**: it keys off the
content-derived `transaction_id` from the Ingestion Agent (src/hashing.py)
rather than a full-row equality check. That distinction matters - if the
same PMS batch gets re-delivered (a very real occurrence with webhook
retries), every row hashes to the same transaction_id and is caught as a
duplicate even if column ordering or a non-key field differs slightly. This
is what lets the pipeline be re-run against the same or overlapping source
data without ever double-counting revenue.

Deterministic validation is intentionally kept LLM-free - it's cheaper,
faster, and fully auditable, which matters more than "AI" here.
"""
from __future__ import annotations

import pandas as pd

from src.dataframe_store import spill
from src.state import PipelineState

REQUIRED_COLUMNS = ["agent", "adr", "hotel", "property_id", "reservation_status"]


def run(state: PipelineState) -> dict:
    df = state.raw_df.copy()

    reasons = pd.Series([""] * len(df), index=df.index)

    missing_agent = df["agent"].isna()
    reasons.loc[missing_agent] += "missing_agent;"

    bad_adr = df["adr"] < 0
    reasons.loc[bad_adr] += "negative_adr;"

    missing_required = df[REQUIRED_COLUMNS].isna().any(axis=1)
    reasons.loc[missing_required & ~missing_agent] += "missing_required_field;"

    # idempotent dedup: same transaction_id => same underlying transaction,
    # regardless of how many times the source batch was re-delivered
    is_duplicate = df.duplicated(subset=["transaction_id"], keep="first")
    reasons.loc[is_duplicate] += "duplicate_transaction;"

    is_bad = reasons != ""

    quarantined_df = df[is_bad].copy()
    quarantined_df["quarantine_reason"] = reasons[is_bad]
    quarantined_df["audit_status"] = "ERROR"

    clean_df = df[~is_bad].copy()
    # negative-ADR rows already excluded; fix any remaining non-fatal issues
    clean_df = clean_df.dropna(subset=["adr", "hotel", "property_id"])
    clean_df["audit_status"] = "APPROVED"

    quarantine_rate = len(quarantined_df) / len(df) if len(df) else 0.0

    quality_report = {
        "total_rows": len(df),
        "clean_rows": len(clean_df),
        "quarantined_rows": len(quarantined_df),
        "quarantine_rate": round(quarantine_rate, 4),
        "duplicate_rows_skipped": int(is_duplicate.sum()),
        "reasons_breakdown": (
            quarantined_df["quarantine_reason"].value_counts().to_dict()
            if not quarantined_df.empty
            else {}
        ),
    }

    clean_path = spill(state.run_id, "clean", clean_df)
    quarantined_path = spill(state.run_id, "quarantined", quarantined_df)

    return {
        "clean_df_path": clean_path,
        "quarantined_df_path": quarantined_path,
        "quality_report": quality_report,
        "quarantine_rate": quarantine_rate,
        "duplicate_rows_skipped": int(is_duplicate.sum()),
    }
