"""Narrative / Executive Report Agent - heavy LLM tier.

Turns the KPI rollups + Insight Agent findings into a natural-language
executive summary and per-agent coaching notes - the output a regional
performance manager would actually read, mirroring the kind of report
FPG's IN-Sights reporting layer would surface to a portfolio owner.

Falls back to a templated (non-LLM) summary if no LLM is configured, so
the pipeline still produces a complete report offline.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

from src import llm_cache
from src.hashing import stable_json_hash
from src.llm_client import LLMConfigError, build_llm_client, get_agent_config
from src.state import PipelineState

SYSTEM_PROMPT = """You are writing a concise executive report for a hospitality frontline \
performance program (FPG-style: front-desk upsell coaching across multiple properties). \
You are given overall KPIs, per-agent and per-property rollups, and a list of AI-generated \
insights/anomalies.

Write a Markdown report with these sections:
## Executive Summary
2-4 sentences: overall performance, standout number, one key risk.

## Key Insights
Bullet each insight provided, in plain business language.

## Agent Coaching Notes
For each insight tagged "risk" or "opportunity", write one short, specific, actionable \
coaching note addressed to that agent's manager.

## Top Performers
Briefly call out the top performer(s) and what to replicate.

Keep the whole report under 400 words. Use real numbers from the data provided. Do not \
invent metrics that weren't given to you."""


def degraded_fallback(state: PipelineState) -> dict:
    """Entry point the Supervisor Agent calls when the LLM call keeps
    failing after retries - produces the same templated report as the
    no-API-key path, just reached from a different trigger."""
    body = _fallback_report(state.kpi_overall, state.insights)
    header = (
        f"# Frontline Performance Report\n\n"
        f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}\n\n"
        f"Data quality: {state.quality_report.get('clean_rows', 'N/A')} clean / "
        f"{state.quality_report.get('total_rows', 'N/A')} total transactions "
        f"({state.quality_report.get('quarantine_rate', 0) * 100:.1f}% quarantined)\n\n---\n\n"
    )
    return {"executive_report_md": header + body, "narrative_cache_hit": False}


def _fallback_report(kpi_overall: dict, insights: list[dict]) -> str:
    lines = [
        "## Executive Summary",
        f"This run processed {kpi_overall.get('total_transactions', 'N/A')} transactions "
        f"across {kpi_overall.get('active_properties', 'N/A')} properties and "
        f"{kpi_overall.get('active_agents', 'N/A')} agents. Overall upsell conversion rate was "
        f"{kpi_overall.get('overall_upsell_conversion_rate', 'N/A')}%, generating "
        f"${kpi_overall.get('overall_incremental_revenue', 'N/A')} in incremental revenue. "
        f"Average guest CSAT was {kpi_overall.get('overall_avg_csat', 'N/A')}/5.",
        "",
        "## Key Insights",
    ]
    for ins in insights:
        lines.append(f"- **[{ins.get('type', 'insight')}] {ins.get('subject', '')}**: {ins.get('summary', '')}")
    lines += ["", "## Agent Coaching Notes"]
    for ins in insights:
        if ins.get("type") in ("risk", "opportunity"):
            lines.append(f"- {ins.get('subject', '')}: {ins.get('recommended_action', '')}")
    lines += ["", "## Top Performers"]
    for ins in insights:
        if ins.get("type") == "top_performer":
            lines.append(f"- {ins.get('subject', '')}: {ins.get('summary', '')}")
    lines.append("\n*(Generated via templated fallback - no LLM configured for narrative agent.)*")
    return "\n".join(lines)


def run(state: PipelineState) -> dict:
    cache_hit = False
    try:
        complete = build_llm_client("narrative")
        agent_cfg = get_agent_config("narrative")
    except LLMConfigError:
        body = _fallback_report(state.kpi_overall, state.insights)
    else:
        user_prompt = (
            "Overall KPIs:\n" + json.dumps(state.kpi_overall, default=str) +
            "\n\nInsights:\n" + json.dumps(state.insights, default=str) +
            "\n\nTop 5 agents by incremental revenue:\n" +
            json.dumps(
                state.kpi_by_agent.head(5).to_dict(orient="records") if state.kpi_by_agent is not None else [],
                default=str,
            )
        )
        cache_key = "narrative:" + stable_json_hash({
            "model": agent_cfg.model, "provider": agent_cfg.provider,
            "system": SYSTEM_PROMPT, "user": user_prompt,
        })
        cached = llm_cache.get(cache_key)
        if cached is not None:
            body = cached
            cache_hit = True
        else:
            body = complete(SYSTEM_PROMPT, user_prompt)
            llm_cache.put(cache_key, body)

    header = (
        f"# Frontline Performance Report\n\n"
        f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}\n\n"
        f"Data quality: {state.quality_report.get('clean_rows', 'N/A')} clean / "
        f"{state.quality_report.get('total_rows', 'N/A')} total transactions "
        f"({state.quality_report.get('quarantine_rate', 0) * 100:.1f}% quarantined)\n\n---\n\n"
    )

    return {"executive_report_md": header + body, "narrative_cache_hit": cache_hit}
