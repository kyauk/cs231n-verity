"""Load Waymo embedding JSONL records into PostgreSQL pgvector.

Mirrors ``pipeline/populate_pgvector.py``. Targets a ``waymo_window_embeddings``
table so Waymo vectors live alongside (but separate from) the nuPlan
``window_embeddings`` table. Waymo embeddings are 5 x 256 = 1280-d.

Usage:
  python -m waymo_pipeline.waymo_populate_pgvector \
      --input-jsonl outputs/waymo_window_embeddings.jsonl \
      --embedding-dim 1280
"""

from __future__ import annotations

import argparse
import json
import os
import re
from typing import Iterable

import psycopg2
import psycopg2.extras
from dotenv import load_dotenv

TABLE_NAME = "waymo_window_embeddings"


def parse_args() -> argparse.Namespace:
    """Parse CLI options for the Waymo pgvector population stage."""
    p = argparse.ArgumentParser(description="Populate pgvector from Waymo embedding JSONL.")
    p.add_argument("--input-jsonl", required=True)
    p.add_argument("--database-url", default=os.getenv("DATABASE_URL", ""))
    p.add_argument("--embedding-dim", type=int, default=1280)
    p.add_argument("--drop-and-recreate", action="store_true")
    return p.parse_args()


def iter_embedding_rows(path: str) -> Iterable[dict]:
    """Stream-parse embedding rows from JSONL."""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def get_embedding_dim_for_table(
    conn: "psycopg2.extensions.connection", table_name: str
) -> int | None:
    """Read the current vector dimension from a table's embedding column."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT format_type(a.atttypid, a.atttypmod)
            FROM pg_attribute a
            JOIN pg_class c ON a.attrelid = c.oid
            WHERE c.relname = %s AND a.attname = 'embedding'
            """,
            (table_name,),
        )
        row = cur.fetchone()
    if not row or not row[0]:
        return None
    match = re.search(r"vector\((\d+)\)", row[0])
    return int(match.group(1)) if match else None


def normalize_embedding_dim(embedding: list[float], target_dim: int | None) -> list[float]:
    """Pad or truncate an embedding to match the target vector column dimension."""
    if target_dim is None or len(embedding) == target_dim:
        return embedding
    if len(embedding) < target_dim:
        return embedding + [0.0] * (target_dim - len(embedding))
    return embedding[:target_dim]


def ensure_table(
    conn: "psycopg2.extensions.connection", embedding_dim: int, drop_and_recreate: bool
) -> None:
    """Ensure the waymo_window_embeddings table exists with a pgvector column."""
    ddl_create = f"""
    CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
        id              SERIAL PRIMARY KEY,
        window_id       TEXT NOT NULL UNIQUE,
        scene_token     TEXT NOT NULL,
        log_id          TEXT NOT NULL,
        scenario_tags   TEXT[],
        window_start_ts BIGINT NOT NULL,
        window_end_ts   BIGINT NOT NULL,
        camera_set      TEXT[],
        embedding       vector({embedding_dim}) NOT NULL,
        quality_stats   JSONB,
        metadata        JSONB,
        created_at      TIMESTAMPTZ DEFAULT NOW()
    );
    """
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        cur.execute(f"SELECT to_regclass('public.{TABLE_NAME}')")
        existing = cur.fetchone()[0]
        if drop_and_recreate:
            cur.execute(f"DROP TABLE IF EXISTS {TABLE_NAME};")
            existing = None
        if not existing:
            cur.execute(ddl_create)

    table_dim = get_embedding_dim_for_table(conn, TABLE_NAME)
    if table_dim is not None and table_dim > 2000:
        print(
            f"Skipping HNSW index for {TABLE_NAME} (dim={table_dim} > 2000 "
            f"pgvector limit). Brute-force cosine scan will be used."
        )
        conn.commit()
        return

    with conn.cursor() as cur:
        cur.execute(
            f"""
            CREATE INDEX IF NOT EXISTS idx_{TABLE_NAME}_hnsw
                ON {TABLE_NAME} USING hnsw (embedding vector_cosine_ops);
            """
        )
    conn.commit()


def insert_rows(conn: "psycopg2.extensions.connection", input_jsonl: str) -> int:
    """Insert Waymo window embedding records into the pgvector table."""
    table_dim = get_embedding_dim_for_table(conn, TABLE_NAME)
    rows = []
    for rec in iter_embedding_rows(input_jsonl):
        embedding = normalize_embedding_dim(rec["embedding"], table_dim)
        rows.append((
            rec["window_id"],
            rec["scene_token_hex"],
            rec["log_id"],
            rec.get("scenario_tags", []),
            rec["window_start_ts"],
            rec["window_end_ts"],
            rec.get("camera_set", []),
            embedding,
            json.dumps(rec.get("quality", {})),
            json.dumps(rec.get("metadata", {})),
        ))
    if not rows:
        return 0
    with conn.cursor() as cur:
        psycopg2.extras.execute_values(
            cur,
            f"""
            INSERT INTO {TABLE_NAME} (
                window_id, scene_token, log_id, scenario_tags,
                window_start_ts, window_end_ts, camera_set,
                embedding, quality_stats, metadata
            ) VALUES %s
            ON CONFLICT (window_id) DO NOTHING
            """,
            rows,
            template="(%s, %s, %s, %s, %s, %s, %s, %s::vector, %s::jsonb, %s::jsonb)",
            page_size=500,
        )
    conn.commit()
    return len(rows)


def main() -> None:
    """End-to-end table prep and Waymo embedding load into pgvector."""
    load_dotenv(dotenv_path=".env")
    args = parse_args()
    db_url = args.database_url or os.getenv("DATABASE_URL")
    if not db_url:
        raise ValueError("Provide --database-url or set DATABASE_URL.")

    with psycopg2.connect(db_url) as conn:
        ensure_table(conn, args.embedding_dim, args.drop_and_recreate)
        inserted = insert_rows(conn, args.input_jsonl)
        print(f"Inserted {inserted} rows into {TABLE_NAME}")


if __name__ == "__main__":
    main()
