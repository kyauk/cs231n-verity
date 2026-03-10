"""Load embedding JSONL records into PostgreSQL pgvector.

Supports two target tables:
  - scene_embeddings: legacy per-scene vectors
  - window_embeddings: temporal-window vectors (default)

Usage:
  python -m pipeline.populate_pgvector \\
      --input-jsonl outputs/window_embeddings_cosmos.jsonl \\
      --table window_embeddings \\
      --embedding-dim 2048
"""

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


# -------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    '''
    Purpose: Parse CLI options for pgvector population stage.
    Parameters: None.
    Returns: argparse.Namespace with input file, table, and DB
        options.
    Called by: main().
    Calls: argparse.ArgumentParser().
    '''
    p = argparse.ArgumentParser(
        description="Populate pgvector from embedding JSONL.",
    )
    p.add_argument("--input-jsonl", required=True)
    p.add_argument(
        "--database-url",
        default=os.getenv("DATABASE_URL", ""),
    )
    p.add_argument("--embedding-dim", type=int, default=2048)
    p.add_argument("--drop-and-recreate", action="store_true")
    p.add_argument(
        "--table",
        default="window_embeddings",
        choices=["window_embeddings", "scene_embeddings"],
        help="Target table (default: window_embeddings).",
    )
    return p.parse_args()


# -------------------------------------------------------------------
# JSONL reader
# -------------------------------------------------------------------

def iter_embedding_rows(path: str) -> Iterable[dict]:
    '''
    Purpose: Stream parse embedding rows from JSONL.
    Parameters:
        path (str): JSONL path output by embed_scenes.py.
    Returns:
        Iterable[dict]: Parsed row dictionaries.
    Called by: main().
    Calls: json.loads().
    '''
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            yield json.loads(line)


# -------------------------------------------------------------------
# Shared helpers
# -------------------------------------------------------------------

def get_table_columns(
    conn: psycopg2.extensions.connection,
    table_name: str,
) -> set[str]:
    '''
    Purpose: Fetch column names for a given table in public schema.
    Parameters:
        conn (psycopg2 connection): Active DB connection.
        table_name (str): Target table name.
    Returns:
        set[str]: Column names present in the table.
    Called by: ensure_table(), insert_rows(),
        insert_window_rows().
    Calls: information_schema.columns query.
    '''
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


def get_embedding_dim_for_table(
    conn: psycopg2.extensions.connection,
    table_name: str,
) -> int | None:
    '''
    Purpose: Read current vector dimension from a table's embedding
        column.
    Parameters:
        conn (psycopg2 connection): Active DB connection.
        table_name (str): Table containing embedding column.
    Returns:
        int | None: Vector dimension if detectable, else None.
    Called by: ensure_table(), ensure_window_table(),
        insert_rows(), insert_window_rows().
    Calls: pg_attribute/pg_class metadata query.
    '''
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


def normalize_embedding_dim(
    embedding: list[float],
    target_dim: int | None,
) -> list[float]:
    '''
    Purpose: Match embedding dimensionality to target vector column
        dim.
    Parameters:
        embedding (list[float]): Source embedding values.
        target_dim (int | None): Desired dimension from DB schema.
    Returns:
        list[float]: Padded or truncated embedding.
    Called by: insert_rows(), insert_window_rows().
    Calls: None.
    '''
    if target_dim is None:
        return embedding
    if len(embedding) == target_dim:
        return embedding
    if len(embedding) < target_dim:
        return embedding + [0.0] * (target_dim - len(embedding))
    return embedding[:target_dim]


# -------------------------------------------------------------------
# window_embeddings table (new)
# -------------------------------------------------------------------

def ensure_window_table(
    conn: psycopg2.extensions.connection,
    embedding_dim: int,
    drop_and_recreate: bool,
) -> None:
    '''
    Purpose: Ensure window_embeddings table exists with pgvector.
    Parameters:
        conn (psycopg2 connection): Active DB connection.
        embedding_dim (int): Embedding dimension for vector column.
        drop_and_recreate (bool): Whether to rebuild table schema.
    Returns: None.
    Called by: main().
    Calls: SQL DDL statements.
    '''
    ddl_drop = "DROP TABLE IF EXISTS window_embeddings;"
    ddl_create = f"""
    CREATE TABLE IF NOT EXISTS window_embeddings (
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
        cur.execute(
            "SELECT to_regclass('public.window_embeddings')",
        )
        existing = cur.fetchone()[0]
        if drop_and_recreate:
            cur.execute(ddl_drop)
            existing = None
        if not existing:
            cur.execute(ddl_create)

    table_dim = get_embedding_dim_for_table(
        conn, "window_embeddings",
    )
    if table_dim is not None and table_dim > 2000:
        print(
            f"Skipping HNSW index for window_embeddings "
            f"(dim={table_dim} > 2000 pgvector limit). "
            f"Brute-force cosine scan will be used."
        )
        conn.commit()
        return

    ddl_index = """
    CREATE INDEX IF NOT EXISTS idx_window_embeddings_hnsw
        ON window_embeddings
        USING hnsw (embedding vector_cosine_ops);
    """
    with conn.cursor() as cur:
        cur.execute(ddl_index)
    conn.commit()


def insert_window_rows(
    conn: psycopg2.extensions.connection,
    input_jsonl: str,
) -> int:
    '''
    Purpose: Insert window embedding records into
        window_embeddings table.
    Parameters:
        conn (psycopg2 connection): Active DB connection.
        input_jsonl (str): Source JSONL path of window embeddings.
    Returns:
        int: Number of rows inserted.
    Called by: main().
    Calls: psycopg2.extras.execute_values().
    '''
    table_dim = get_embedding_dim_for_table(
        conn, "window_embeddings",
    )
    rows = []
    for rec in iter_embedding_rows(input_jsonl):
        embedding = normalize_embedding_dim(
            rec["embedding"], table_dim,
        )
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
            """
            INSERT INTO window_embeddings (
                window_id, scene_token, log_id, scenario_tags,
                window_start_ts, window_end_ts, camera_set,
                embedding, quality_stats, metadata
            ) VALUES %s
            ON CONFLICT (window_id) DO NOTHING
            """,
            rows,
            template=(
                "(%s, %s, %s, %s, %s, %s, %s, "
                "%s::vector, %s::jsonb, %s::jsonb)"
            ),
            page_size=500,
        )
    conn.commit()
    return len(rows)


# -------------------------------------------------------------------
# scene_embeddings table (legacy, preserved)
# -------------------------------------------------------------------

def get_embedding_dim(
    conn: psycopg2.extensions.connection,
) -> int | None:
    '''
    Purpose: Read current vector dimension from
        scene_embeddings.embedding (legacy wrapper).
    Parameters:
        conn (psycopg2 connection): Active DB connection.
    Returns:
        int | None: Vector dimension if detectable, else None.
    Called by: ensure_table(), insert_rows().
    Calls: get_embedding_dim_for_table().
    '''
    return get_embedding_dim_for_table(conn, "scene_embeddings")


def ensure_table(
    conn: psycopg2.extensions.connection,
    embedding_dim: int,
    drop_and_recreate: bool,
) -> None:
    '''
    Purpose: Ensure scene_embeddings table exists with pgvector
        index.
    Parameters:
        conn (psycopg2 connection): Active DB connection.
        embedding_dim (int): Embedding dimension for vector column.
        drop_and_recreate (bool): Whether to rebuild table schema.
    Returns: None.
    Called by: main().
    Calls: SQL DDL statements.
    '''
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
        ON scene_embeddings
        USING hnsw (embedding vector_cosine_ops);
    """
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        cur.execute(
            "SELECT to_regclass('public.scene_embeddings')",
        )
        existing = cur.fetchone()[0]
        if drop_and_recreate:
            cur.execute(ddl_drop)
            existing = None
        if not existing:
            cur.execute(ddl_create)

    table_dim = get_embedding_dim(conn)
    if table_dim is not None and table_dim > 2000:
        print(
            "Skipping HNSW index creation for "
            "scene_embeddings.embedding because vector "
            f"dimension is {table_dim} (>2000 pgvector HNSW "
            "limit)."
        )
        conn.commit()
        return

    with conn.cursor() as cur:
        cur.execute(ddl_index)
    conn.commit()


def insert_rows(
    conn: psycopg2.extensions.connection,
    input_jsonl: str,
) -> int:
    '''
    Purpose: Insert embedding records into scene_embeddings table.
    Parameters:
        conn (psycopg2 connection): Active DB connection.
        input_jsonl (str): Source JSONL path of embeddings.
    Returns:
        int: Number of rows inserted.
    Called by: main().
    Calls: psycopg2.extras.execute_values().
    '''
    table_columns = get_table_columns(conn, "scene_embeddings")
    table_dim = get_embedding_dim(conn)
    rows = []

    is_new_schema = {
        "scene_token", "log_id", "embedding",
    }.issubset(table_columns)
    is_legacy_schema = {
        "run_id", "token", "embedding",
    }.issubset(table_columns)
    if not (is_new_schema or is_legacy_schema):
        raise RuntimeError(
            "scene_embeddings schema is not recognized by "
            "populate_pgvector.py. Expected either new schema "
            "(scene_token/log_id) or legacy schema (run_id/token)."
        )

    legacy_run_id = str(uuid.uuid4())
    for rec in iter_embedding_rows(input_jsonl):
        embedding = normalize_embedding_dim(
            rec["embedding"], table_dim,
        )
        if is_new_schema:
            rows.append((
                rec["scene_token_hex"],
                rec["log_id"],
                rec.get("scenario_tags", []),
                embedding,
                json.dumps(rec.get("quality", {})),
            ))
            continue
        rows.append((
            legacy_run_id,
            rec["scene_token_hex"],
            embedding,
        ))
    if not rows:
        return 0
    with conn.cursor() as cur:
        if is_new_schema:
            psycopg2.extras.execute_values(
                cur,
                """
                INSERT INTO scene_embeddings (
                    scene_token, log_id, scenario_tags,
                    embedding, quality_stats
                ) VALUES %s
                """,
                rows,
                template=(
                    "(%s, %s, %s, %s::vector, %s::jsonb)"
                ),
                page_size=500,
            )
        else:
            cur.execute(
                """
                INSERT INTO runs (run_id, status, params)
                VALUES (%s::uuid, %s, %s::jsonb)
                ON CONFLICT (run_id) DO NOTHING
                """,
                (
                    legacy_run_id,
                    "smoke_test",
                    json.dumps(
                        {"source": "populate_pgvector"},
                    ),
                ),
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


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------

def main() -> None:
    '''
    Purpose: End-to-end table prep and data load into pgvector.
        Dispatches to window_embeddings or scene_embeddings based
        on --table flag.
    Parameters: None.
    Returns: None.
    Called by: CLI invocation.
    Calls: ensure_window_table(), insert_window_rows(),
        ensure_table(), insert_rows().
    '''
    load_dotenv(dotenv_path=".env")
    args = parse_args()
    db_url = args.database_url or os.getenv("DATABASE_URL")
    if not db_url:
        raise ValueError(
            "Provide --database-url or set DATABASE_URL.",
        )

    with psycopg2.connect(db_url) as conn:
        if args.table == "window_embeddings":
            ensure_window_table(
                conn=conn,
                embedding_dim=args.embedding_dim,
                drop_and_recreate=args.drop_and_recreate,
            )
            inserted = insert_window_rows(
                conn=conn,
                input_jsonl=args.input_jsonl,
            )
            print(
                f"Inserted {inserted} rows into "
                "window_embeddings"
            )
        else:
            ensure_table(
                conn=conn,
                embedding_dim=args.embedding_dim,
                drop_and_recreate=args.drop_and_recreate,
            )
            inserted = insert_rows(
                conn=conn,
                input_jsonl=args.input_jsonl,
            )
            print(
                f"Inserted {inserted} rows into "
                "scene_embeddings"
            )


if __name__ == "__main__":
    main()
