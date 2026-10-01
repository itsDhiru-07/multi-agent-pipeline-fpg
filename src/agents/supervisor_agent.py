"""Supervisor / Monitoring Agent.

Not a graph node in its own right - a wrapper that every other agent's
`run` function is passed through before being registered with LangGraph
(see src/graph.py). This is a deliberate architectural choice: a supervisor
that itself sat "in-line" in the graph as just another node wouldn't be
able to time, retry, or catch failures *in* the other nodes - it needs to
wrap them.

Responsibilities:
  1. **Observability** - records start/end time, duration, and status
     (ok / error / degraded) for every agent invocation into
     `PipelineState.run_log`, and persists the same record to
     `run_logs/<run_id>.jsonl` so a run's history survives process restarts
     (this is what a real on-call engineer would tail).
  2. **Retry with backoff** - every agent gets some retry budget (network
     blips, transient disk I/O errors, momentary API rate limits), but the
     budget differs by risk profile: LLM-backed agents (insight, narrative)
     get more retries since network calls are the expected failure mode;
     deterministic agents get fewer, since a repeated failure there usually
     means a real data/logic bug, not a transient condition.
  3. **Two different outcomes when retries are exhausted, by design**:
       - Agents with a `degraded_fallback` (insight, narrative) don't crash
         the run - the supervisor flips `PipelineState.degraded_mode = True`
         and calls the agent's own fallback (heuristic insights / templated
         report) so a usable, slightly-less-nuanced result still ships.
       - Agents WITHOUT a `degraded_fallback` (ingestion, data_quality, kpi)
         re-raise after exhausting retries instead of silently continuing.
         There is no safe fallback for "we don't actually know the KPI
         numbers" - fabricating one would be worse than stopping. Because
         the graph is checkpointed, a re-raise here does not lose the work
         already done: re-invoking with the same run_id/thread_id resumes
         from the last *successful* checkpoint rather than re-running
         ingestion from scratch. This is what makes "resume on failure" a
         real guarantee instead of "retry a few times and hope."
"""
from __future__ import annotations

import json
import os
import time
import traceback
from typing import Callable

from src.state import PipelineState

MAX_RETRIES = {
    "ingestion": 1,
    "data_quality": 1,
    "kpi": 1,
    "insight": 2,
    "narrative": 2,
}
BACKOFF_BASE_SECONDS = 1.5

RUN_LOG_DIR = "run_logs"


def _append_run_log(run_id: str, record: dict) -> None:
    if not run_id:
        return
    os.makedirs(RUN_LOG_DIR, exist_ok=True)
    path = os.path.join(RUN_LOG_DIR, f"{run_id}.jsonl")
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str) + "\n")


def supervised(
    agent_name: str,
    fn: Callable[[PipelineState], dict],
    degraded_fallback: Callable[[PipelineState], dict] | None = None,
) -> Callable[[PipelineState], dict]:
    """Wraps an agent's `run` function with timing, retry, and degraded-mode
    fallback. Returned callable has the same signature LangGraph expects
    for a node, so wrapping is transparent to graph.py.

    `degraded_fallback`, when provided, is the agent's own no-LLM fallback
    (e.g. insight_agent.degraded_fallback) - called once retries are
    exhausted, so the *agent* decides what a degraded result looks like,
    not the supervisor."""

    retries_allowed = MAX_RETRIES.get(agent_name, 0)

    def wrapped(state: PipelineState) -> dict:
        attempt = 0
        last_error: Exception | None = None
        start = time.monotonic()

        while attempt <= retries_allowed:
            try:
                result = fn(state)
                duration = time.monotonic() - start
                log_record = {
                    "agent": agent_name,
                    "status": "ok" if attempt == 0 else "ok_after_retry",
                    "attempt": attempt + 1,
                    "duration_seconds": round(duration, 3),
                }
                _append_run_log(state.run_id, log_record)
                new_run_log = state.run_log + [log_record]
                return {**result, "run_log": new_run_log}
            except Exception as exc:  # noqa: BLE001 - supervisor must catch broadly to protect the run
                last_error = exc
                attempt += 1
                if attempt <= retries_allowed:
                    time.sleep(BACKOFF_BASE_SECONDS * (2 ** (attempt - 1)))

        # Exhausted retries.
        duration = time.monotonic() - start

        if degraded_fallback is None:
            # No safe fallback exists for this agent (ingestion/data_quality/
            # kpi) - log the failure and re-raise so the checkpointer records
            # this node as incomplete. A subsequent invoke() with the same
            # thread_id resumes from the last successful checkpoint instead
            # of restarting the whole pipeline - that resume guarantee is
            # exactly why we don't paper over this failure here.
            log_record = {
                "agent": agent_name,
                "status": "failed_halted",
                "attempts": attempt,
                "duration_seconds": round(duration, 3),
                "error": str(last_error),
                "traceback": traceback.format_exc(limit=3),
            }
            _append_run_log(state.run_id, log_record)
            raise RuntimeError(
                f"{agent_name} failed after {attempt} attempt(s) with no safe fallback: {last_error}"
            ) from last_error

        # This agent has a degraded_fallback (insight/narrative) - don't
        # crash the run, flip degraded mode, and let the agent's own
        # fallback produce a usable (if less nuanced) result.
        log_record = {
            "agent": agent_name,
            "status": "failed_degraded",
            "attempts": attempt,
            "duration_seconds": round(duration, 3),
            "error": str(last_error),
            "traceback": traceback.format_exc(limit=3),
        }
        _append_run_log(state.run_id, log_record)
        new_run_log = state.run_log + [log_record]
        new_errors = state.errors + [
            f"{agent_name} failed after {attempt} attempt(s): {last_error}. Continuing in degraded mode."
        ]

        fallback_result: dict = {}
        try:
            fallback_result = degraded_fallback(state)
        except Exception as fallback_exc:  # noqa: BLE001
            new_errors.append(f"{agent_name} degraded_fallback also failed: {fallback_exc}")

        return {
            **fallback_result,
            "run_log": new_run_log,
            "errors": new_errors,
            "degraded_mode": True,
        }

    return wrapped
