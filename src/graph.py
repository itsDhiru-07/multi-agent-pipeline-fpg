"""LangGraph wiring for the frontline performance pipeline.

Linear flow with one conditional edge: if the Data Quality agent quarantines
too large a share of the batch, the pipeline halts before spending money on
the heavy LLM agents, rather than generating a report from a mostly-broken
dataset.

    ingestion -> data_quality -> [quality_gate] -> kpi -> insight -> narrative
                                        |
                                        +--(quarantine_rate too high)--> halt

Every node is wrapped by the Supervisor Agent (src/agents/supervisor_agent.py)
for timing, retry-with-backoff, and degraded-mode fallback - this is why
graph.py never calls `ingestion_agent.run` etc. directly, always through
`supervised(...)`.

The graph is compiled with a SQLite checkpointer
(langgraph-checkpoint-sqlite). Every node's state is persisted to
`checkpoints.sqlite` keyed by `thread_id` (== run_id, itself a deterministic
hash of the source file - see src/hashing.py). This means:
  - Re-invoking the graph with the same run_id after a mid-run crash resumes
    from the last successfully completed node instead of restarting from
    ingestion (real resume-on-failure, not just "run it again").
  - Re-invoking with the same run_id after a *successful* prior run is a
    no-op replay of the final state, not a duplicate run - the other half
    of the idempotency story alongside row-level dedup in the Data Quality
    agent.
"""
from __future__ import annotations

import sqlite3

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph

from src.agents import (
    data_quality_agent,
    ingestion_agent,
    insight_agent,
    kpi_agent,
    narrative_agent,
)
from src.agents.supervisor_agent import supervised
from src.state import PipelineState

QUARANTINE_RATE_THRESHOLD = 0.40
CHECKPOINT_DB_PATH = "checkpoints.sqlite"


def _halt_node(state: PipelineState) -> dict:
    return {
        "errors": state.errors + [
            f"Halted: quarantine rate {state.quarantine_rate:.1%} exceeds "
            f"{QUARANTINE_RATE_THRESHOLD:.0%} threshold. Check upstream PMS feed."
        ],
        "executive_report_md": (
            "# Frontline Performance Report\n\n"
            "**Pipeline halted before analysis.**\n\n"
            f"Data quality gate failed: {state.quarantine_rate:.1%} of records were quarantined "
            f"(threshold: {QUARANTINE_RATE_THRESHOLD:.0%}). Investigate the source PMS feed before re-running.\n"
        ),
    }


def _quality_gate(state: PipelineState) -> str:
    return "halt" if state.quarantine_rate > QUARANTINE_RATE_THRESHOLD else "continue"


def _get_checkpointer(db_path: str = CHECKPOINT_DB_PATH) -> SqliteSaver:
    conn = sqlite3.connect(db_path, check_same_thread=False)
    return SqliteSaver(conn)


def build_graph(checkpointer: SqliteSaver | None = None):
    """Builds and compiles the graph. Pass an explicit `checkpointer` to
    reuse a connection (e.g. across Streamlit reruns); otherwise a default
    SQLite-backed one is created against CHECKPOINT_DB_PATH."""
    graph = StateGraph(PipelineState)

    graph.add_node("ingestion", supervised("ingestion", ingestion_agent.run))
    graph.add_node("data_quality", supervised("data_quality", data_quality_agent.run))
    graph.add_node("kpi", supervised("kpi", kpi_agent.run))
    graph.add_node("insight", supervised("insight", insight_agent.run, insight_agent.degraded_fallback))
    graph.add_node("narrative", supervised("narrative", narrative_agent.run, narrative_agent.degraded_fallback))
    graph.add_node("halt", _halt_node)

    graph.set_entry_point("ingestion")
    graph.add_edge("ingestion", "data_quality")
    graph.add_conditional_edges(
        "data_quality",
        _quality_gate,
        {"continue": "kpi", "halt": "halt"},
    )
    graph.add_edge("kpi", "insight")
    graph.add_edge("insight", "narrative")
    graph.add_edge("narrative", END)
    graph.add_edge("halt", END)

    checkpointer = checkpointer or _get_checkpointer()
    return graph.compile(checkpointer=checkpointer)
