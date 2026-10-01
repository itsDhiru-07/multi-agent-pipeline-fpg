"""Streamlit dashboard for the FPG-style multi-agent frontline performance pipeline.

Run with:
    streamlit run app.py

This is a thin presentation layer over the same LangGraph pipeline used by
`src/main.py` - it calls `build_graph()` and streams node-by-node progress,
then renders the same outputs (KPI tables, quarantine detail, executive
report, supervisor run log) that the CLI writes to `reports/`. No pipeline
logic lives here.
"""
from __future__ import annotations

import os
import sys

import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.dataframe_store import load
from src.graph import QUARANTINE_RATE_THRESHOLD, build_graph
from src.hashing import derive_run_id
from src.llm_client import load_config
from src.state import PipelineState

st.set_page_config(page_title="FPG Frontline Performance Pipeline", layout="wide")

AGENT_LABELS = {
    "ingestion": "1. Ingestion Agent",
    "data_quality": "2. Data Quality / Reconciliation Agent",
    "kpi": "3. KPI Transformation Agent",
    "insight": "4. Insight & Anomaly Agent (LLM)",
    "narrative": "5. Narrative / Executive Report Agent (LLM)",
    "halt": "Quality Gate Halt",
}


def render_sidebar() -> tuple[str, int, bool, bool]:
    st.sidebar.title("Pipeline Controls")

    default_path = "data/raw/hotel_bookings.csv"
    uploaded = st.sidebar.file_uploader("Upload a transaction CSV (optional)", type=["csv"])

    source_path = default_path
    if uploaded is not None:
        tmp_path = os.path.join("data", "raw", "_uploaded.csv")
        with open(tmp_path, "wb") as f:
            f.write(uploaded.getbuffer())
        source_path = tmp_path
        st.sidebar.success(f"Using uploaded file ({uploaded.name})")
    else:
        st.sidebar.caption(f"Using default dataset: `{default_path}`")

    chunk_size = st.sidebar.number_input(
        "Ingestion chunk size", min_value=500, max_value=50000, value=5000, step=500,
        help="Rows read per batch during ingestion - bounds peak memory use regardless of file size.",
    )

    st.sidebar.divider()
    st.sidebar.subheader("LLM Agent Configuration")
    try:
        cfg = load_config()
        for name in ("insight", "narrative"):
            a = cfg.get("agents", {}).get(name, {})
            if a.get("use_llm"):
                st.sidebar.caption(f"**{name}**: {a.get('provider')} / {a.get('model')} ({a.get('tier')})")
            else:
                st.sidebar.caption(f"**{name}**: LLM disabled")
    except Exception as e:
        st.sidebar.warning(f"Could not read config/agents.yaml: {e}")
    st.sidebar.caption("Edit `config/agents.yaml` to change providers/models per agent.")

    st.sidebar.divider()
    st.sidebar.subheader("Run Identity")
    if os.path.exists(source_path):
        run_id_preview = derive_run_id(source_path)
        st.sidebar.caption(f"Deterministic run_id: `{run_id_preview}`")
        st.sidebar.caption(
            "Same source file → same run_id → checkpoint resume on failure, "
            "output overwrite (not duplication) on success."
        )
    force_new_run = st.sidebar.checkbox(
        "Force new run_id", value=False,
        help="Bypass idempotent reuse and mint a fresh run - use this to re-run from scratch "
             "even though the source file hasn't changed.",
    )

    st.sidebar.divider()
    run_clicked = st.sidebar.button("Run Pipeline", type="primary", use_container_width=True)

    return source_path, int(chunk_size), force_new_run, run_clicked


def run_pipeline_with_progress(source_path: str, chunk_size: int, force_new_run: bool) -> dict:
    import datetime as dt

    if force_new_run:
        run_id = "run-" + dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    else:
        run_id = derive_run_id(source_path)

    graph_app = build_graph()
    initial_state = PipelineState(source_path=source_path, run_id=run_id, chunk_size=chunk_size)
    config = {"configurable": {"thread_id": run_id}}

    status_area = st.status(f"Running pipeline (run_id={run_id})...", expanded=True)
    final_state: dict = {}

    for step in graph_app.stream(initial_state, config=config, stream_mode="updates"):
        for node_name, node_output in step.items():
            label = AGENT_LABELS.get(node_name, node_name)
            marker = "⚠️" if node_output.get("degraded_mode") else "✅"
            status_area.write(f"{marker} {label}")
            final_state.update(node_output)

    final_state["run_id"] = run_id
    status_area.update(label=f"Pipeline complete (run_id={run_id})", state="complete", expanded=False)
    return final_state


def render_overview(result: dict) -> None:
    quality = result.get("quality_report", {})
    kpi_overall = result.get("kpi_overall", {})

    errors = result.get("errors", [])
    if errors:
        for e in errors:
            (st.warning if result.get("degraded_mode") else st.error)(e)
        if not result.get("kpi_overall"):
            st.warning(
                f"Pipeline halted at the quality gate (threshold: {QUARANTINE_RATE_THRESHOLD:.0%} "
                "quarantined). Insight and Narrative agents did not run."
            )
        elif result.get("degraded_mode"):
            st.info("Pipeline completed in **degraded mode** - one or more LLM agents fell back "
                     "to a heuristic/templated result after repeated failures. See Run Log tab.")

    cols = st.columns(7)
    cols[0].metric("Total Transactions", quality.get("total_rows", "—"))
    cols[1].metric("Quarantined", f"{quality.get('quarantined_rows', 0)} ({quality.get('quarantine_rate', 0)*100:.1f}%)")
    cols[2].metric("Duplicates Skipped", quality.get("duplicate_rows_skipped", 0))
    cols[3].metric("Upsell Conversion", f"{kpi_overall.get('overall_upsell_conversion_rate', '—')}%")
    cols[4].metric("Incremental Revenue", f"${kpi_overall.get('overall_incremental_revenue', 0):,.0f}")
    cols[5].metric("Avg CSAT", f"{kpi_overall.get('overall_avg_csat', '—')}/5")
    cols[6].metric("Active Agents", kpi_overall.get("active_agents", "—"))


def render_tabs(result: dict) -> None:
    tab_report, tab_agents, tab_props, tab_insights, tab_quarantine, tab_runlog = st.tabs(
        ["Executive Report", "Agent KPIs", "Property KPIs", "Insights", "Quarantined Records", "Run Log (Supervisor)"]
    )

    with tab_report:
        report_md = result.get("executive_report_md", "")
        if report_md:
            st.markdown(report_md)
        else:
            st.info("No report generated.")

    with tab_agents:
        kpi_by_agent = load(result.get("kpi_by_agent_path"))
        if kpi_by_agent is not None and not kpi_by_agent.empty:
            st.dataframe(kpi_by_agent, use_container_width=True, hide_index=True)
            c1, c2 = st.columns(2)
            with c1:
                st.caption("Top 10 agents by incremental revenue")
                st.bar_chart(kpi_by_agent.head(10).set_index("agent_id")["total_incremental_revenue"])
            with c2:
                st.caption("Upsell conversion rate vs. avg CSAT (top 15 by volume)")
                top_vol = kpi_by_agent.sort_values("transactions", ascending=False).head(15)
                st.scatter_chart(top_vol, x="upsell_conversion_rate", y="avg_csat", size="transactions")
        else:
            st.info("Pipeline did not reach the KPI stage.")

    with tab_props:
        kpi_by_property = load(result.get("kpi_by_property_path"))
        if kpi_by_property is not None and not kpi_by_property.empty:
            st.dataframe(kpi_by_property, use_container_width=True, hide_index=True)
            st.caption("Incremental revenue by property")
            st.bar_chart(kpi_by_property.set_index("property_id")["total_incremental_revenue"])
        else:
            st.info("Pipeline did not reach the KPI stage.")

    with tab_insights:
        insights = result.get("insights", [])
        if result.get("insight_cache_hit"):
            st.caption("♻️ Served from LLM response cache (unchanged KPI input) - no API call made.")
        if insights:
            for ins in insights:
                icon = {"risk": "🔴", "opportunity": "🟡", "top_performer": "🟢", "anomaly": "🔵"}.get(ins.get("type"), "⚪")
                with st.container(border=True):
                    st.markdown(f"{icon} **[{ins.get('type', 'insight').upper()}] {ins.get('subject', '')}**")
                    st.write(ins.get("summary", ""))
                    st.caption(f"Recommended action: {ins.get('recommended_action', '')}")
        else:
            st.info("No insights generated (pipeline may have halted before this stage).")

    with tab_quarantine:
        quarantined_df = load(result.get("quarantined_df_path"))
        quality = result.get("quality_report", {})
        if quarantined_df is not None and not quarantined_df.empty:
            reasons = quality.get("reasons_breakdown", {})
            if reasons:
                st.caption("Quarantine reasons breakdown")
                st.bar_chart(pd.Series(reasons))
            st.dataframe(quarantined_df, use_container_width=True, hide_index=True)
        else:
            st.info("No records were quarantined.")

    with tab_runlog:
        run_log = result.get("run_log", [])
        st.caption(
            f"run_id: `{result.get('run_id', '—')}` · "
            f"degraded_mode: {'yes ⚠️' if result.get('degraded_mode') else 'no'} · "
            f"insight cache hit: {result.get('insight_cache_hit', False)} · "
            f"narrative cache hit: {result.get('narrative_cache_hit', False)}"
        )
        if run_log:
            log_df = pd.DataFrame(run_log)
            st.dataframe(log_df, use_container_width=True, hide_index=True)
        else:
            st.info("No supervisor log entries yet.")
        st.caption(
            "Per-run logs are also persisted to `run_logs/<run_id>.jsonl` and survive process "
            "restarts - the Supervisor Agent writes to both the in-memory pipeline state and disk."
        )


def main() -> None:
    st.title("FPG Frontline Performance Pipeline")
    st.caption(
        "Multi-agent pipeline: Ingestion → Data Quality → KPI Transformation → "
        "Insight & Anomaly (LLM) → Narrative Report (LLM), orchestrated with LangGraph, "
        "supervised for retries/idempotency/checkpointed resume."
    )

    source_path, chunk_size, force_new_run, run_clicked = render_sidebar()

    if "pipeline_result" not in st.session_state:
        st.session_state.pipeline_result = None

    if run_clicked:
        st.session_state.pipeline_result = run_pipeline_with_progress(source_path, chunk_size, force_new_run)

    result = st.session_state.pipeline_result
    if result is None:
        st.info("Configure a dataset in the sidebar and click **Run Pipeline** to start.")
        return

    render_overview(result)
    st.divider()
    render_tabs(result)


if __name__ == "__main__":
    main()
