# FPG-Style Multi-Agent Frontline Performance Pipeline

A multi-agent data pipeline that ingests hotel front-desk transaction data,
validates/reconciles it, computes upsell & revenue KPIs, and uses LLM agents
to generate coaching insights and an executive report — modeled on how
Frontline Performance Group's **IN-Gauge** (PMS-connected transaction
tracking) and **IN-Sights** (analytics/reporting) products work conceptually.

See [`docs/architecture.md`](docs/architecture.md) for the full design
rationale, diagrams, and production-scaling notes.

## Architecture at a Glance

```
ingestion -> data_quality -> [quality gate] -> kpi -> insight (LLM) -> narrative (LLM)
                                    |
                                    +-- halts here if >40% of data is quarantined
```

Five agents, orchestrated with **LangGraph**, plus a **Supervisor Agent**
that wraps all of them for retries, timing, and degraded-mode fallback:

| Agent | Uses an LLM? | Job |
|---|---|---|
| Ingestion | No | Loads & tags the transaction feed (chunked, streamed to disk) |
| Data Quality | No | Validates, idempotently dedupes, quarantines bad records with a reason code |
| KPI Transformation | No | Computes upsell conversion, incremental revenue, attach rate, CSAT per agent/property |
| Insight & Anomaly | **Yes (heavy tier)** | Finds coaching-worthy patterns across agents/properties |
| Narrative / Executive Report | **Yes (heavy tier)** | Writes the final Markdown report |
| Supervisor | No | Times every agent, retries on failure, routes to degraded mode or a resumable halt |

Every LLM-using agent's **provider and model are configured independently**
in [`config/agents.yaml`](config/agents.yaml) — swap an agent between
Anthropic, OpenAI, NVIDIA NIM, or a local Ollama server without touching code.

**Production-grade properties, not just a happy-path demo:**
- **Idempotent** at three levels — row-level dedup (content-hashed
  `transaction_id`), run-level (deterministic `run_id` from the source
  file's hash, so re-running never duplicates output), and LLM-call level
  (cached by input hash, so unchanged data never re-spends tokens).
- **Resumable** — the graph is checkpointed to SQLite; if a deterministic
  agent (Ingestion/Data Quality/KPI) fails after retries, the CLI prints a
  `--resume <run_id>` command that continues from the last successful
  checkpoint instead of restarting from scratch.
- **Memory-efficient** — ingestion streams the source in configurable
  chunks straight to a parquet spill file; no stage holds the full dataset
  in memory at once.
- **Degradable, not fragile** — if an LLM call keeps failing, Insight and
  Narrative fall back to a deterministic heuristic / templated report
  rather than crashing the run.

See [`docs/architecture.md` §7](docs/architecture.md) for the full design
rationale and verification notes behind each of these.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` and set whichever API key(s) match the providers referenced in
`config/agents.yaml` (defaults to Anthropic for both LLM agents — set
`ANTHROPIC_API_KEY`).

**No API key?** The pipeline still runs end-to-end — the Insight and
Narrative agents each fall back to a deterministic (non-LLM) result so you
always get a complete report.

## Dataset

The pipeline runs against `data/raw/hotel_bookings.csv`, whose schema mirrors
the public **Hotel Booking Demand** dataset (Antonio, Almeida & Nunes, 2019 —
Kaggle: `jessemostipak/hotel-booking-demand`), remapped for the frontline-agent
upsell scenario (see column mapping in `docs/architecture.md`).

A ready-to-run synthetic copy of this dataset is already included/generated
via:

```bash
python src/generate_sample_data.py
```

This creates ~12,000 realistic rows with a synthesized guest CSAT score and
deliberately injected data-quality issues (missing agent IDs, negative ADR,
duplicate transactions) so the Data Quality agent has something real to catch.

To use the **real** Kaggle dataset instead, download `hotel_bookings.csv` from
the link above and drop it into `data/raw/hotel_bookings.csv` — same column
names, no code changes needed (satisfaction score would need to be added
separately since it isn't part of the original dataset).

## Running the Pipeline

```bash
python -m src.main
```

Outputs land in `reports/`:
- `kpi_by_agent.csv` — per-agent upsell/revenue/CSAT rollup
- `kpi_by_property.csv` — per-property rollup
- `quarantined.csv` — records that failed validation, with reason codes and audit status
- `executive_report.md` — the generated narrative report

Optional flags:

```bash
python -m src.main --source data/raw/hotel_bookings.csv --outdir reports
python -m src.main --chunk-size 2000       # tune ingestion batch size
python -m src.main --force-new-run         # bypass idempotent run_id reuse
python -m src.main --resume run-<hash>     # resume a crashed run from its last checkpoint
```

Re-running with the same source file is safe and cheap: it reuses the same
`run_id`, so output files are overwritten (not duplicated) and cached LLM
responses are reused rather than re-billed. If a run crashes partway
through, the error message tells you the exact `--resume` command to run.

## Interactive Dashboard (Streamlit)

```bash
streamlit run app.py
```

Same pipeline, presented as a live dashboard: upload a CSV (or use the
bundled dataset), watch each agent complete in real time, then browse the
executive report, KPI tables/charts, insights, quarantined records, and a
**Run Log** tab showing the Supervisor's per-agent timing, retries, cache
hits, and degraded-mode status.

## Configuring / Swapping LLM Providers Per Agent

Edit `config/agents.yaml`. Example — move the Insight agent to a cheap local
model and keep Narrative on Claude:

```yaml
agents:
  insight:
    use_llm: true
    tier: light
    provider: local          # points at a local Ollama server
    model: llama3.1:8b
    temperature: 0.3
    max_tokens: 2000

  narrative:
    use_llm: true
    tier: heavy
    provider: anthropic
    model: claude-sonnet-5
    temperature: 0.4
    max_tokens: 2500
```

To add NVIDIA NIM or any other OpenAI-compatible host, add a `providers`
entry with its `base_url` and the env var holding its API key — no code
changes required (see `src/llm_client.py`).

## Project Structure

```
multi_agent_pipeline_FPG/
  app.py                         # Streamlit dashboard
  data/raw/hotel_bookings.csv    # dataset (generated or real)
  config/agents.yaml             # per-agent provider/model config
  src/
    state.py                     # shared pipeline state (pydantic)
    llm_client.py                # provider-agnostic LLM factory
    llm_cache.py                 # LLM response cache (idempotent LLM calls)
    hashing.py                   # deterministic transaction_id / run_id derivation
    dataframe_store.py           # parquet spill/load for checkpoint-safe state
    generate_sample_data.py      # synthetic dataset generator
    graph.py                     # LangGraph wiring + SQLite checkpointer
    main.py                      # CLI entrypoint (run / resume)
    agents/
      supervisor_agent.py        # retry, timing, degraded-mode wrapper
      ingestion_agent.py
      data_quality_agent.py
      kpi_agent.py
      insight_agent.py
      narrative_agent.py
  reports/                       # generated output (gitignored)
  run_logs/                      # per-run supervisor logs (gitignored)
  checkpoints.sqlite             # LangGraph checkpoint store (gitignored)
  docs/architecture.md           # full design doc
```
