"""Insight & Anomaly Agent - heavy LLM tier.

This is where agentic reasoning earns its keep over plain BI: given the
per-agent and per-property KPI rollups, the LLM is asked to identify
underperforming agents/properties, unusual patterns a rules engine would
miss (e.g. high upsell rate but low CSAT - a "pushy agent" signal), and
concrete coaching opportunities - the kind of judgment call FPG's own
coaching product line (Service 360 / Champion Certification) is built
around.

Falls back to a small deterministic heuristic if no LLM is configured, so
the pipeline still produces a usable (if less nuanced) result offline.
"""
from __future__ import annotations

import json

from src import llm_cache
from src.hashing import stable_json_hash
from src.llm_client import LLMConfigError, build_llm_client, get_agent_config
from src.state import PipelineState

SYSTEM_PROMPT = """You are a performance-analytics assistant for a hospitality frontline-agent \
coaching platform (in the style of FPG's IN-Gauge/IN-Sights products). You are given \
aggregated KPI tables for front-desk agents and properties: upsell conversion rate, \
average ADR, total/average incremental revenue, ancillary attach rate, average guest CSAT, \
and cancellation rate.

Identify the most important, non-obvious insights a regional performance manager should act \
on this week. Look specifically for:
- Agents with high upsell conversion but low CSAT (possible pushy/aggressive selling)
- Agents or properties with low upsell conversion despite high transaction volume (missed \
  revenue opportunity, good coaching candidates)
- Properties with unusually high cancellation rates
- Standout top performers worth studying/replicating

Return ONLY a JSON array of 4-8 objects, each with keys:
"type" (one of: "risk", "opportunity", "top_performer", "anomaly"),
"subject" (agent_id or property_id),
"summary" (one sentence, specific, with numbers),
"recommended_action" (one sentence, concrete and actionable).
No prose outside the JSON array."""


def degraded_fallback(state: PipelineState) -> dict:
    """Entry point the Supervisor Agent calls when the LLM call keeps
    failing after retries - produces the same heuristic result as the
    no-API-key path, just reached from a different trigger."""
    return {"insights": _heuristic_fallback(state.kpi_by_agent, state.kpi_by_property)}


def _heuristic_fallback(kpi_by_agent, kpi_by_property) -> list[dict]:
    insights = []
    if kpi_by_agent is not None and not kpi_by_agent.empty:
        top = kpi_by_agent.iloc[0]
        insights.append({
            "type": "top_performer",
            "subject": str(top["agent_id"]),
            "summary": f"Highest incremental revenue generator: ${top['total_incremental_revenue']:.2f} "
                       f"across {int(top['transactions'])} transactions.",
            "recommended_action": "Document this agent's upsell approach as a coaching case study.",
        })
        low_conv = kpi_by_agent[kpi_by_agent["transactions"] >= kpi_by_agent["transactions"].median()]
        if not low_conv.empty:
            worst = low_conv.sort_values("upsell_conversion_rate").iloc[0]
            insights.append({
                "type": "opportunity",
                "subject": str(worst["agent_id"]),
                "summary": f"Upsell conversion rate of only {worst['upsell_conversion_rate']}% "
                           f"despite {int(worst['transactions'])} transactions.",
                "recommended_action": "Enroll in a targeted upsell-technique coaching module.",
            })
    return insights


def run(state: PipelineState) -> dict:
    kpi_agent_records = state.kpi_by_agent.to_dict(orient="records") if state.kpi_by_agent is not None else []
    kpi_property_records = state.kpi_by_property.to_dict(orient="records") if state.kpi_by_property is not None else []

    try:
        complete = build_llm_client("insight")
        agent_cfg = get_agent_config("insight")
    except LLMConfigError:
        return {"insights": _heuristic_fallback(state.kpi_by_agent, state.kpi_by_property)}

    user_prompt = (
        "Per-agent KPIs:\n" + json.dumps(kpi_agent_records, default=str) +
        "\n\nPer-property KPIs:\n" + json.dumps(kpi_property_records, default=str)
    )

    cache_key = "insight:" + stable_json_hash({
        "model": agent_cfg.model, "provider": agent_cfg.provider,
        "system": SYSTEM_PROMPT, "user": user_prompt,
    })
    cached = llm_cache.get(cache_key)
    if cached is not None:
        raw = cached
        cache_hit = True
    else:
        raw = complete(SYSTEM_PROMPT, user_prompt)
        llm_cache.put(cache_key, raw)
        cache_hit = False

    try:
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned
        insights = json.loads(cleaned)
        if not isinstance(insights, list):
            raise ValueError("expected a JSON array")
    except (json.JSONDecodeError, ValueError):
        insights = _heuristic_fallback(state.kpi_by_agent, state.kpi_by_property)
        insights.append({
            "type": "anomaly",
            "subject": "insight_agent",
            "summary": "LLM response could not be parsed as JSON; used heuristic fallback instead.",
            "recommended_action": "Check the insight agent's model/provider configuration.",
        })

    return {"insights": insights, "insight_cache_hit": cache_hit}
