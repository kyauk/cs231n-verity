"""Load scene embedding JSONL records into PostgreSQL pgvector."""

from __future__ import annotations

import argparse
import json
import os
import re
import uuid
from typing import Iterable

import psycopg2
import psycopg2.extras
from dotenv import load_dotenv


def parse_args() -> argparse.Namespace:
    """Purpose: Parse CLI options for pgvector population stage.
    Parameters: None.
    Returns: argparse.Namespace with input file and DB options.
    Called by: main().
    Calls: argparse.ArgumentParser().
    """
    parser = argparse.ArgumentParser(description="Populate pgvector from embedding JSONL.")
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--database-url", default=os.getenv("DATABASE_URL", ""))
    parser.add_argument("--embedding-dim", type=int, default=256)
    parser.add_argument("--drop-and-recreate", action="store_true")
    return parser.parse_args()


def iter_embedding_rows(path: str) -> Iterable[dict]:
    """Purpose: Stream parse embedding rows from JSONL.
    Parameters:
        path (str): JSONL path output by embed_scenes.py.
    Returns:
        Iterable[dict]: Parsed row dictionaries.
    Called by: main().
    Calls: json.loads().
    """
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            yield json.loads(line)


def get_table_columns(conn: psycopg2.extensions.connection, table_name: str) -> set[str]:
    """Purpose: Fetch column names for a given table in public schema.
    Parameters:
        conn (psycopg2 connection): Active DB connection.
        table_name (str): Target table name.
    Returns:
        set[str]: Column names present in the table.
    Called by: ensure_table(), insert_rows().
    Calls: information_schema.columns query.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema = 'public' AND table_name = %s
            """,
            (table_name,),
        )
        return {row[0] for row in cur.fetchall()}


def get_embedding_dim(conn: psycopg2.extensions.connection) -> int | None:
    """Purpose: Read current vector dimension from scene_embeddings.embedding.
    Parameters:
        conn (psycopg2 connection): Active DB connection.
    Returns:
        int | None: Vector dimension if detectable, else None.
    Called by: ensure_table(), insert_rows().
    Calls: pg_attribute/pg_class metadata query.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT format_type(a.atttypid, a.atttypmod)
            FROM pg_attribute a
            JOIN pg_class c ON a.attrelid = c.oid
            WHERE c.relname = 'scene_embeddings' AND a.attname = 'embedding'
            """
        )
        row = cur.fetchone()
    if not row or not row[0]:
        return None
    match = re.search(r"vector\((\d+)\)", row[0])
    return int(match.group(1)) if match else None


def normalize_embedding_dim(embedding: list[float], target_dim: int | None) -> list[float]:
    """Purpose: Match embedding dimensionality to target vector column dim.
    Parameters:
        embedding (list[float]): Source embedding values.
        target_dim (int | None): Desired dimension from DB schema.
    Returns:
        list[float]: Padded or truncated embedding.
    Called by: insert_rows().
    Calls: None.
    """
    if target_dim is None:
        return embedding
    if len(embedding) == target_dim:
        return embedding
    if len(embedding) < target_dim:
        return embedding + [0.0] * (target_dim - len(embedding))
    return embedding[:target_dim]


def ensure_table(
    conn: psycopg2.extensions.connection, embedding_dim: int, drop_and_recreate: bool
) -> None:
    """Purpose: Ensure scene_embeddings table exists with pgvector index.
    Parameters:
        conn (psycopg2 connection): Active DB connection.
        embedding_dim (int): Embedding dimension for vector column.
        drop_and_recreate (bool): Whether to rebuild table schema.
    Returns:
        None.
    Called by: main().
    Calls: SQL DDL statements.
    """
    ddl_drop = "DROP TABLE IF EXISTS scene_embeddings;"
    ddl_create = f"""
    CREATE TABLE IF NOT EXISTS scene_embeddings (
        id              SERIAL PRIMARY KEY,
        scene_token     TEXT NOT NULL,
        log_id          TEXT NOT NULL,
        scenario_tags   TEXT[],
        embedding       vector({embedding_dim}) NOT NULL,
        quality_stats   JSONB,
        created_at      TIMESTAMPTZ DEFAULT NOW()
    );
    """
    ddl_index = """
    CREATE INDEX IF NOT EXISTS idx_scene_embeddings_hnsw
        ON scene_embeddings USING hnsw (embedding vector_cosine_ops);
    """
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        cur.execute("SELECT to_regclass('public.scene_embeddings')")
        existing = cur.fetchone()[0]
        if drop_and_recreate:
            cur.execute(ddl_drop)
            existing = None
        if not existing:
            cur.execute(ddl_create)

    table_dim = get_embedding_dim(conn)
    if table_dim is not None and table_dim > 2000:
        print(
            "Skipping HNSW index creation for scene_embeddings.embedding because "
            f"vector dimension is {table_dim} (>2000 pgvector HNSW limit)."
        )
        conn.commit()
        return

    with conn.cursor() as cur:
        cur.execute(ddl_index)
    conn.commit()


def insert_rows(conn: psycopg2.extensions.connection, input_jsonl: str) -> int:
    """Purpose: Insert embedding records into scene_embeddings table.
    Parameters:
        conn (psycopg2 connection): Active DB connection.
        input_jsonl (str): Source JSONL path of embeddings.
    Returns:
        int: Number of rows inserted.
    Called by: main().
    Calls: psycopg2.extras.execute_values().
    """
    table_columns = get_table_columns(conn, "scene_embeddings")
    table_dim = get_embedding_dim(conn)
    rows = []

    is_new_schema = {"scene_token", "log_id", "embedding"}.issubset(table_columns)
    is_legacy_schema = {"run_id", "token", "embedding"}.issubset(table_columns)
    if not (is_new_schema or is_legacy_schema):
        raise RuntimeError(
            "scene_embeddings schema is not recognized by populate_pgvector.py. "
            "Expected either new schema (scene_token/log_id) or legacy schema (run_id/token)."
        )

    legacy_run_id = str(uuid.uuid4())
    for rec in iter_embedding_rows(input_jsonl):
        embedding = normalize_embedding_dim(rec["embedding"], table_dim)
        # Support both new and legacy table layouts so smoke tests can append
        # without destructive reset in mixed-migration environments.
        if is_new_schema:
            rows.append(
                (
                    rec["scene_token_hex"],
                    rec["log_id"],
                    rec.get("scenario_tags", []),
                    embedding,
                    json.dumps(rec.get("quality", {})),
                )
            )
            continue
        rows.append(
            (
                legacy_run_id,
                rec["scene_token_hex"],
                embedding,
            )
        )
    if not rows:
        return 0
    with conn.cursor() as cur:
        if is_new_schema:
            psycopg2.extras.execute_values(
                cur,
                """
                INSERT INTO scene_embeddings (
                    scene_token, log_id, scenario_tags, embedding, quality_stats
                ) VALUES %s
                """,
                rows,
                template="(%s, %s, %s, %s::vector, %s::jsonb)",
                page_size=500,
            )
        else:
            cur.execute(
                """
                INSERT INTO runs (run_id, status, params)
                VALUES (%s::uuid, %s, %s::jsonb)
                ON CONFLICT (run_id) DO NOTHING
                """,
                (legacy_run_id, "smoke_test", json.dumps({"source": "populate_pgvector"})),
            )
            psycopg2.extras.execute_values(
                cur,
                """
                INSERT INTO scene_embeddings (
                    run_id, token, embedding
                ) VALUES %s
                """,
                rows,
                template="(%s::uuid, %s, %s::vector)",
                page_size=500,
            )
    conn.commit()
    return len(rows)


def main() -> None:
    """Purpose: End-to-end table prep and data load into pgvector.
    Parameters: None.
    Returns: None.
    Called by: CLI invocation.
    Calls: ensure_table(), insert_rows().
    """
    load_dotenv()
    args = parse_args()
    db_url = args.database_url or os.getenv("DATABASE_URL")
    if not db_url:
        raise ValueError("Provide --database-url or set DATABASE_URL.")
    with psycopg2.connect(db_url) as conn:
        ensure_table(
            conn=conn,
            embedding_dim=args.embedding_dim,
            drop_and_recreate=args.drop_and_recreate,
        )
        inserted = insert_rows(conn=conn, input_jsonl=args.input_jsonl)
    print(f"Inserted {inserted} rows into scene_embeddings")


if __name__ == "__main__":
    main()

