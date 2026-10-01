"""CLI entrypoint for the FPG-style multi-agent frontline performance pipeline.

Usage:
    python -m src.main
    python -m src.main --source data/raw/hotel_bookings.csv --outdir reports
    python -m src.main --force-new-run     # bypass idempotent run_id reuse
    python -m src.main --chunk-size 2000   # tune ingestion batch size
    python -m src.main --resume run-96eb638d15ce8ef6   # resume a crashed run

Idempotency: run_id defaults to a deterministic hash of the source file's
contents (src/hashing.py). Re-running against an unchanged source file
reuses the same run_id, which means:
  - Output files in --outdir are overwritten in place, not duplicated -
    running the pipeline twice against the same, successfully-completed
    input is a no-op replay, not two reports.
  - If a run crashes partway through (an agent with no safe degraded
    fallback - ingestion/data_quality/kpi - exhausts its retries), the
    error message tells you the run_id to pass to --resume. Resuming
    reuses LangGraph's SQLite checkpoint (checkpoints.sqlite) and continues
    from the last successfully completed node, instead of re-running
    ingestion and data quality validation from scratch.
Pass --force-new-run to opt out of idempotent reuse and mint a fresh,
timestamp-based run_id instead.
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone

from src.dataframe_store import load
from src.graph import build_graph
from src.hashing import derive_run_id
from src.state import PipelineState


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the FPG frontline performance pipeline")
    parser.add_argument("--source", default="data/raw/hotel_bookings.csv")
    parser.add_argument("--outdir", default="reports")
    parser.add_argument("--chunk-size", type=int, default=5000, help="Rows per ingestion chunk")
    parser.add_argument(
        "--force-new-run", action="store_true",
        help="Mint a fresh timestamp-based run_id instead of reusing the deterministic one",
    )
    parser.add_argument(
        "--resume", metavar="RUN_ID", default=None,
        help="Resume a previously crashed run by its run_id, continuing from its last checkpoint",
    )
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    app = build_graph()

    if args.resume:
        run_id = args.resume
        config = {"configurable": {"thread_id": run_id}}
        print(f"[pipeline] resuming run_id={run_id} from last checkpoint")
        try:
            result = app.invoke(None, config=config)
        except Exception as exc:
            print(f"[pipeline] FATAL: resume failed: {exc}")
            print(f"[pipeline] run_id={run_id} is still resumable - fix the underlying issue and retry "
                  f"--resume {run_id}")
            sys.exit(1)
    else:
        if args.force_new_run:
            run_id = "run-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        else:
            run_id = derive_run_id(args.source)

        initial_state = PipelineState(source_path=args.source, run_id=run_id, chunk_size=args.chunk_size)
        config = {"configurable": {"thread_id": run_id}}

        print(f"[pipeline] run_id={run_id} source={args.source}")
        try:
            result = app.invoke(initial_state, config=config)
        except Exception as exc:
            print(f"[pipeline] FATAL: {exc}")
            print(f"[pipeline] run_id={run_id} is resumable - fix the underlying issue and run:")
            print(f"[pipeline]   python -m src.main --resume {run_id}")
            sys.exit(1)

    quality = result.get("quality_report", {})
    print(f"[pipeline] ingested={quality.get('total_rows')} "
          f"clean={quality.get('clean_rows')} "
          f"quarantined={quality.get('quarantined_rows')} "
          f"({quality.get('quarantine_rate', 0) * 100:.1f}%) "
          f"duplicates_skipped={quality.get('duplicate_rows_skipped', 0)}")

    if result.get("degraded_mode"):
        print("[pipeline] WARNING: one or more LLM agents ran in degraded (fallback) mode - see run_log")

    run_log = result.get("run_log", [])
    for entry in run_log:
        status = entry.get("status")
        marker = "OK" if status in ("ok", "ok_after_retry") else "FAIL"
        print(f"[supervisor] [{marker}] {entry.get('agent')}: {status} "
              f"({entry.get('duration_seconds', '?')}s, attempt {entry.get('attempt', entry.get('attempts', '?'))})")

    errors = result.get("errors", [])
    if errors:
        for e in errors:
            print(f"[pipeline] ERROR: {e}")

    kpi_by_agent = load(result.get("kpi_by_agent_path"))
    kpi_by_property = load(result.get("kpi_by_property_path"))
    quarantined_df = load(result.get("quarantined_df_path"))
    report_md = result.get("executive_report_md", "")

    if kpi_by_agent is not None:
        kpi_by_agent.to_csv(os.path.join(args.outdir, "kpi_by_agent.csv"), index=False)
        print(f"[pipeline] wrote {args.outdir}/kpi_by_agent.csv ({len(kpi_by_agent)} agents)")

    if kpi_by_property is not None:
        kpi_by_property.to_csv(os.path.join(args.outdir, "kpi_by_property.csv"), index=False)
        print(f"[pipeline] wrote {args.outdir}/kpi_by_property.csv ({len(kpi_by_property)} properties)")

    if quarantined_df is not None:
        quarantined_df.to_csv(os.path.join(args.outdir, "quarantined.csv"), index=False)
        print(f"[pipeline] wrote {args.outdir}/quarantined.csv ({len(quarantined_df)} rows)")

    if report_md:
        report_path = os.path.join(args.outdir, "executive_report.md")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_md)
        print(f"[pipeline] wrote {report_path}")

    print(f"[pipeline] done. run_id={run_id} (re-run with the same source to resume/replay this run)")


if __name__ == "__main__":
    main()
