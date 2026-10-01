"""Ingestion Agent - rule-based, no LLM.

Simulates a multi-source pull the way FPG's IN-Gauge would: a PMS
transaction feed (the booking/stay records) tagged with its source system.
In a production build this node would instead poll a PMS webhook or read
from a Kafka topic; here it loads the local CSV to keep the demo
self-contained.

Reads the source in chunks (configurable via PipelineState.chunk_size) and
writes each chunk straight to the run's parquet spill file via a
ParquetWriter, rather than accumulating chunks in a Python list and
concatenating at the end - at no point does this agent hold the full source
file in memory at once. This is what makes the "chunked ingestion" claim
real rather than cosmetic: peak memory is bounded by chunk_size, not by
total row count, so it scales to source files far larger than available
RAM. Each row also gets a stable, content-derived `transaction_id` here
(src/hashing.py) - the idempotency key every downstream stage relies on to
detect re-ingested duplicates safely.
"""
from __future__ import annotations

import os

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.dataframe_store import run_dir
from src.hashing import row_transaction_id
from src.state import PipelineState


def run(state: PipelineState) -> dict:
    out_path = os.path.join(run_dir(state.run_id), "raw.parquet")

    writer: pq.ParquetWriter | None = None
    chunks_processed = 0
    rows_ingested = 0

    try:
        for chunk in pd.read_csv(state.source_path, chunksize=state.chunk_size):
            chunk = chunk.copy()
            chunk["_source_system"] = "PMS_TRANSACTION_FEED"
            chunk["transaction_id"] = chunk.apply(row_transaction_id, axis=1)

            table = pa.Table.from_pandas(chunk, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(out_path, table.schema)
            writer.write_table(table)

            chunks_processed += 1
            rows_ingested += len(chunk)
    finally:
        if writer is not None:
            writer.close()

    return {
        "raw_df_path": out_path if rows_ingested else None,
        "source_tags": {"PMS_TRANSACTION_FEED": state.source_path},
        "rows_ingested": rows_ingested,
        "chunks_processed": chunks_processed,
    }
